#include <algorithm>
#include <cuda_runtime.h>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "assign_cells.cuh"
#include "check_cuda.h"
#include "domain.h"
#include "energy_diagnostics.cuh"
#include "force_kernels.cuh"
#include "halo_exchange.h"
#include "init_neighborhood.h"
#include "input.h"
#include "integration.cuh"
#include "migration.h"
#include "particle_device.cuh"
#include "particle_host.h"
#include "vtk_output.h"

// Upload the cell neighborhood table for a domain to the given GPU.
// Returns the device pointer (caller must cudaFree it).
static int *uploadNeighborhood(const Domain &dom) {
  std::vector<int> nb(dom.total_cells * 27, -1);
  initCellNeighborhood(dom.num_cells, nb);
  int *d_nb;
  CHECK_CUDA(cudaMalloc(&d_nb, dom.total_cells * 27 * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(d_nb, nb.data(), dom.total_cells * 27 * sizeof(int),
                        cudaMemcpyHostToDevice));
  return d_nb;
}

// Derive the base name from a file path (strips directory and extension).
static std::string sceneName(const char *path) {
  std::string s = path;
  if (auto p = s.find_last_of("/\\"); p != std::string::npos)
    s = s.substr(p + 1);
  if (auto p = s.rfind('.'); p != std::string::npos)
    s = s.substr(0, p);
  return s;
}

// Derive the project root directory from argv[0] so VTK output lands in
// ss26-project/ instead of inside the build tree.
static std::string projectRoot(const char *exe_path) {
  namespace fs = std::filesystem;
  fs::path p = fs::absolute(exe_path);
  p = p.parent_path(); // md2d.exe     → build/Debug
  p = p.parent_path(); // build/Debug  → build
  p = p.parent_path(); // build        → ss26-project
  std::string s = p.string();
  if (!s.empty() && s.back() != '/' && s.back() != '\\')
    s += '/';
  return s;
}

// Convert ParticleData → ParticleHost for upload.
static ParticleHost toHost(const ParticleData &src) {
  ParticleHost h;
  for (size_t i = 0; i < src.n; ++i)
    h.push(src.positions[i], src.velocities[i], src.masses[i], src.radii[i],
           src.kn[i], src.gamma_n[i], src.gamma_t[i], src.mu[i]);
  return h;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0]
              << " <scene.json> [max_steps] [num_gpus]\n";
    return 1;
  }
  const long max_steps = (argc > 2) ? std::stol(argv[2]) : 100000;
  const int steps_per_frame = 10; // write VTK every N steps

  // ------------------------------------------------------------------ //
  // 1.  Load scene
  // ------------------------------------------------------------------ //
  SceneConfig cfg;
  ParticleData global;
  loadScene(argv[1], cfg, global);
  printf("Loaded %zu particles  dt=%.2e  cell_size=%.3f\n", global.n, cfg.dt,
         cfg.cell_size);

  // ------------------------------------------------------------------ //
  // 2.  Determine GPU count
  // ------------------------------------------------------------------ //
  int device_count = 0;
  CHECK_CUDA(cudaGetDeviceCount(&device_count));
  if (device_count < 1) {
    fprintf(stderr, "Error: no CUDA-capable GPUs found.\n");
    return 1;
  }

  int num_gpus = device_count;
  if (argc > 3)
    num_gpus = std::min(static_cast<int>(std::stol(argv[3])), device_count);
  printf("Using %d of %d available GPUs\n", num_gpus, device_count);

  // ------------------------------------------------------------------ //
  // 3.  Build domain decomposition (N equal X-slices)
  // ------------------------------------------------------------------ //
  const float max_r =
      *std::max_element(global.radii.begin(), global.radii.end());
  const float halo_w = 2.0f * max_r; // capture all contacts across boundaries
  const float cell_size = cfg.cell_size > 0.0f ? cfg.cell_size : halo_w;

  std::vector<Domain> doms =
      buildDomains(cfg.domain_min, cfg.domain_max, halo_w, cell_size, num_gpus);

  printf("Domain split into %d slices (halo_width=%.3f):\n", num_gpus, halo_w);
  for (int g = 0; g < num_gpus; ++g) {
    const Domain &d = doms[g];
    printf("  GPU%d: cells %dx%d  owned [%.2f, %.2f]  local [%.2f, %.2f]"
           "  neighbors: L=%d R=%d\n",
           g, d.num_cells.x, d.num_cells.y, d.owned_min.x, d.owned_max.x,
           d.local_min.x, d.local_max.x, d.left_neighbor, d.right_neighbor);
  }

  // ------------------------------------------------------------------ //
  // 4.  Enable peer access between all GPU pairs
  // ------------------------------------------------------------------ //
  for (int g = 0; g < num_gpus; ++g) {
    CHECK_CUDA(cudaSetDevice(g));
    for (int peer = 0; peer < num_gpus; ++peer) {
      if (peer == g)
        continue;
      int canAccess = 0;
      CHECK_CUDA(cudaDeviceCanAccessPeer(&canAccess, g, peer));
      if (canAccess) {
        cudaError_t peerErr = cudaDeviceEnablePeerAccess(peer, 0);
        if (peerErr != cudaSuccess &&
            peerErr != cudaErrorPeerAccessAlreadyEnabled) {
          fprintf(
              stderr,
              "Warning: cudaDeviceEnablePeerAccess GPU%d -> GPU%d failed: %s\n",
              g, peer, cudaGetErrorString(peerErr));
        }
      }
    }
  }

  // ------------------------------------------------------------------ //
  // 5.  Distribute particles and upload to GPUs
  // ------------------------------------------------------------------ //
  std::vector<ParticleData> per_gpu;
  splitIntoN(global, doms, per_gpu);

  std::vector<ParticleDevice> pds(num_gpus);
  for (int g = 0; g < num_gpus; ++g) {
    ParticleHost h = toHost(per_gpu[g]);
    CHECK_CUDA(cudaSetDevice(g));
    h.upload(pds[g], doms[g].total_cells, global.n);
    printf("  GPU%d: %zu owned particles uploaded\n", g, pds[g].n);
  }

  // ------------------------------------------------------------------ //
  // 6.  Build cell neighbor tables (fixed for the lifetime of the run)
  // ------------------------------------------------------------------ //
  std::vector<int *> d_nb(num_gpus, nullptr);
  for (int g = 0; g < num_gpus; ++g) {
    CHECK_CUDA(cudaSetDevice(g));
    d_nb[g] = uploadNeighborhood(doms[g]);
  }

  // ------------------------------------------------------------------ //
  // 7.  Main simulation loop
  // ------------------------------------------------------------------ //
  constexpr dim3 BLOCK(256);
  const std::string scene = sceneName(argv[1]);
  const std::string output_root = projectRoot(argv[0]) + "output/";
  printf("VTK output → %sout_vtk_%s_%ld/\n", output_root.c_str(), scene.c_str(),
         max_steps);
  int frame = 0;

  for (long step = 0; step < max_steps; ++step) {

    // --- Halo exchange: populate ghost particles on each GPU ---
    exchangeHalos(pds, doms);

    // --- Assign cells (owned + halo) on each GPU ---
    for (int g = 0; g < num_gpus; ++g) {
      const Domain &dom = doms[g];
      CHECK_CUDA(cudaSetDevice(g));
      CHECK_CUDA(
          cudaMemset(pds[g].d_cellHeads, -1, dom.total_cells * sizeof(int)));
      if (pds[g].n_total > 0) {
        dim3 grid((pds[g].n_total + BLOCK.x - 1) / BLOCK.x);
        assignCell<<<grid, BLOCK>>>(pds[g].n_total, pds[g].d_positions,
                                    dom.num_cells, dom.local_min, dom.cell_size,
                                    pds[g].d_cellHeads, pds[g].d_cellTails,
                                    pds[g].d_cellIndexes);
        CHECK_LAST_CUDA();
      }
    }

    // --- Compute contact forces (owned particles only) ---
    for (int g = 0; g < num_gpus; ++g) {
      CHECK_CUDA(cudaSetDevice(g));
      if (pds[g].n > 0) {
        dim3 grid((pds[g].n + BLOCK.x - 1) / BLOCK.x);
        computeContactForces<<<grid, BLOCK>>>(
            pds[g].n, pds[g].n_total, pds[g].d_positions, pds[g].d_velocities,
            pds[g].d_forces, pds[g].d_masses, pds[g].d_radii, pds[g].d_kn,
            pds[g].d_gamma_n, pds[g].d_gamma_t, pds[g].d_mu, pds[g].d_cellHeads,
            pds[g].d_cellTails, pds[g].d_cellIndexes, d_nb[g], cfg.gravity);
        CHECK_LAST_CUDA();
      }
    }

    // --- Integrate (owned particles only) ---
    for (int g = 0; g < num_gpus; ++g) {
      CHECK_CUDA(cudaSetDevice(g));
      if (pds[g].n > 0) {
        dim3 grid((pds[g].n + BLOCK.x - 1) / BLOCK.x);
        integrate<<<grid, BLOCK>>>(cfg.dt, pds[g].n, pds[g].d_positions,
                                   pds[g].d_velocities, pds[g].d_forces,
                                   pds[g].d_masses, pds[g].d_radii,
                                   cfg.domain_min, cfg.domain_max);
        CHECK_LAST_CUDA();
      }
    }

    // Synchronize all GPUs before diagnostics / migration / output
    for (int g = 0; g < num_gpus; ++g) {
      CHECK_CUDA(cudaSetDevice(g));
      CHECK_CUDA(cudaDeviceSynchronize());
    }

    // --- Energy & momentum diagnostics (every frame) ---
    float total_ke = 0.0f, total_px = 0.0f, total_py = 0.0f, total_pz = 0.0f;
    if (step % steps_per_frame == 0) {
      for (int g = 0; g < num_gpus; ++g) {
        CHECK_CUDA(cudaSetDevice(g));
        if (pds[g].n > 0) {
          dim3 grid((pds[g].n + BLOCK.x - 1) / BLOCK.x);
          float ke_g = 0, px_g = 0, py_g = 0, pz_g = 0;
          computeDiagnostics(pds[g], BLOCK, grid, ke_g, px_g, py_g, pz_g);
          total_ke += ke_g;
          total_px += px_g;
          total_py += py_g;
          total_pz += pz_g;
        }
      }
    }

    // --- Particle migration ---
    migrateParticles(pds, doms, global.n);

    // --- VTK output ---
    if (step % steps_per_frame == 0) {
      std::vector<Vec3> pos, vel;
      std::vector<float> rad, gpu_id, halo_blend;

      for (int g = 0; g < num_gpus; ++g) {
        CHECK_CUDA(cudaSetDevice(g));
        ParticleHost h;
        h.download(pds[g]);
        const Domain &dom = doms[g];
        for (size_t i = 0; i < h.n; ++i) {
          pos.push_back(h.positions[i]);
          vel.push_back(h.velocities[i]);
          rad.push_back(h.radii[i]);

          // GPU ownership ID (integer tag)
          gpu_id.push_back(static_cast<float>(g));

          // Halo blend: interpolate GPU IDs near domain boundaries.
          // Inside owned region: value = g.
          // Within halo_width of a boundary, smoothly blend toward neighbor.
          float blend = static_cast<float>(g);
          const float x = h.positions[i].x;

          // Blend toward right neighbor (GPU g+1)
          if (dom.right_neighbor >= 0) {
            float dist = dom.owned_max.x - x;
            if (dist < dom.halo_width) {
              float t = 1.0f - dist / dom.halo_width;
              blend = static_cast<float>(g) + t;
            }
          }
          // Blend toward left neighbor (GPU g-1); only if not already
          // blending toward right neighbor.
          if (dom.left_neighbor >= 0 && blend == static_cast<float>(g)) {
            float dist = x - dom.owned_min.x;
            if (dist < dom.halo_width) {
              float t = 1.0f - dist / dom.halo_width;
              blend = static_cast<float>(g) - t;
            }
          }
          halo_blend.push_back(blend);
        }
      }

      writeParticlesVTK(frame++, pos, vel, rad, gpu_id, halo_blend, scene,
                        max_steps, output_root);

      printf("step %6ld  frame %4d  KE=%.4e  |p|=(%.2e,%.2e)", step, frame - 1,
             total_ke, total_px, total_py);
      for (int g = 0; g < num_gpus; ++g)
        printf("  GPU%d:%zu", g, pds[g].n);
      printf("\n");
    }
  }

  // ------------------------------------------------------------------ //
  // 8.  Cleanup
  // ------------------------------------------------------------------ //
  for (int g = 0; g < num_gpus; ++g) {
    CHECK_CUDA(cudaSetDevice(g));
    freeParticleDevice(pds[g]);
    CHECK_CUDA(cudaFree(d_nb[g]));
  }

  return 0;
}
