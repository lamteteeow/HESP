#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
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

// Convert ParticleData → ParticleHost for upload.
// Assigns sequential particle IDs starting from id_offset.
static ParticleHost toHost(const ParticleData &src, int &id_offset) {
  ParticleHost h;
  for (size_t i = 0; i < src.n; ++i)
    h.push(src.positions[i], src.velocities[i], src.masses[i], src.radii[i],
           src.kn[i], src.gamma_n[i], src.gamma_t[i], src.mu[i], id_offset++);
  return h;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0]
              << " <scene.json> [max_steps] [num_gpus]\n";
    return 1;
  }
  const long max_steps = (argc > 2) ? std::stol(argv[2]) : 100000;
  const int steps_per_frame = 50; // write VTK every N steps

  try {

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

    // Enable peer access between all GPU pairs (prerequisite for
    // cudaMemcpyPeer). Not all pairs may support it; skip those that don't.
    for (int i = 0; i < num_gpus; ++i) {
      CHECK_CUDA(cudaSetDevice(i));
      for (int j = 0; j < num_gpus; ++j) {
        if (i == j)
          continue;
        int can_access = 0;
        CHECK_CUDA(cudaDeviceCanAccessPeer(&can_access, i, j));
        if (can_access) {
          CHECK_CUDA(cudaDeviceEnablePeerAccess(j, 0));
          printf("  Peer access enabled: GPU%d -> GPU%d\n", i, j);
        }
      }
    }

    // ------------------------------------------------------------------ //
    // 3.  Build domain decomposition (N equal X-slices)
    // ------------------------------------------------------------------ //
    const float max_r =
        *std::max_element(global.radii.begin(), global.radii.end());
    const float halo_w = 2.0f * max_r; // capture all contacts across boundaries
    const float cell_size = cfg.cell_size > 0.0f ? cfg.cell_size : halo_w;

    std::vector<Domain> doms = buildDomains(cfg.domain_min, cfg.domain_max,
                                            halo_w, cell_size, num_gpus);

    printf("Domain split into %d slices (halo_width=%.3f):\n", num_gpus,
           halo_w);
    for (int g = 0; g < num_gpus; ++g) {
      const Domain &d = doms[g];
      printf("  GPU%d: grid(%d,%d) cells %dx%d  owned [%.2f-%.2f, %.2f-%.2f]"
             "  local [%.2f-%.2f, %.2f-%.2f]"
             "  neighbors: L=%d R=%d B=%d T=%d\n",
             g, d.grid_x, d.grid_y, d.num_cells.x, d.num_cells.y,
             d.owned_min.x, d.owned_max.x, d.owned_min.y, d.owned_max.y,
             d.local_min.x, d.local_max.x, d.local_min.y, d.local_max.y,
             d.left_neighbor, d.right_neighbor,
             d.bottom_neighbor, d.top_neighbor);
    }

    // ------------------------------------------------------------------ //
    // 4.  Distribute particles and upload to GPUs
    // ------------------------------------------------------------------ //
    std::vector<ParticleData> per_gpu;
    splitIntoN(global, doms, per_gpu);

    std::vector<ParticleDevice> pds(num_gpus);
    int next_id = 0;
    for (int g = 0; g < num_gpus; ++g) {
      ParticleHost h = toHost(per_gpu[g], next_id);
      CHECK_CUDA(cudaSetDevice(g));
      h.upload(pds[g], doms[g].total_cells, global.n);
      printf("  GPU%d: %zu owned particles uploaded\n", g, pds[g].n);
    }

    // ------------------------------------------------------------------ //
    // 5.  Build cell neighbor tables (fixed for the lifetime of the run)
    // ------------------------------------------------------------------ //
    std::vector<int *> d_nb(num_gpus, nullptr);
    for (int g = 0; g < num_gpus; ++g) {
      CHECK_CUDA(cudaSetDevice(g));
      d_nb[g] = uploadNeighborhood(doms[g]);
    }

    // ------------------------------------------------------------------ //
    // 6.  Main simulation loop
    // ------------------------------------------------------------------ //
    constexpr dim3 BLOCK(256);
    const std::string scene = sceneName(argv[1]);
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
                                      dom.num_cells, dom.local_min,
                                      dom.cell_size, pds[g].d_cellHeads,
                                      pds[g].d_cellTails, pds[g].d_cellIndexes);
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
              pds[g].d_gamma_n, pds[g].d_gamma_t, pds[g].d_mu,
              pds[g].d_cellHeads, pds[g].d_cellTails, pds[g].d_cellIndexes,
              d_nb[g], cfg.gravity);
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

      // Synchronize all GPUs before migration / output
      for (int g = 0; g < num_gpus; ++g) {
        CHECK_CUDA(cudaSetDevice(g));
        CHECK_CUDA(cudaDeviceSynchronize());
      }

      // --- Particle migration ---
      migrateParticles(pds, doms, global.n);

      // --- VTK output ---
      if (step % steps_per_frame == 0) {
        std::vector<Vec3> pos, vel;
        std::vector<float> rad;
        std::vector<int> pid, gpu_owner;

        for (int g = 0; g < num_gpus; ++g) {
          CHECK_CUDA(cudaSetDevice(g));
          ParticleHost h;
          h.download(pds[g]);
          for (size_t i = 0; i < h.n; ++i) {
            pos.push_back(h.positions[i]);
            vel.push_back(h.velocities[i]);
            rad.push_back(h.radii[i]);
            pid.push_back(h.ids[i]);
            gpu_owner.push_back(g);
          }
        }

        // Stable sort by particle ID (unique, invariant)
        {
          std::vector<size_t> idx(pos.size());
          for (size_t i = 0; i < idx.size(); ++i)
            idx[i] = i;
          std::sort(idx.begin(), idx.end(),
                    [&](size_t a, size_t b) { return pid[a] < pid[b]; });
          auto reorder = [&](auto &v) {
            auto v2 = v;
            for (size_t i = 0; i < idx.size(); ++i)
              v[i] = v2[idx[i]];
          };
          reorder(pos);
          reorder(vel);
          reorder(rad);
          reorder(gpu_owner);
        }

        // Compute border fraction: 0 = interior, 1 = at halo edge
        std::vector<float> border(pos.size(), 0.0f);
        for (size_t i = 0; i < pos.size(); ++i) {
          const Domain &d = doms[gpu_owner[i]];
          const float x = pos[i].x, y = pos[i].y;
          // Distance from each owned boundary, normalised to halo_width
          auto edge = [&](float coord, float own_lo, float own_hi) -> float {
            float d_lo = (coord - own_lo) / d.halo_width;
            float d_hi = (own_hi - coord) / d.halo_width;
            if (d_lo >= 0 && d_lo < 1) return 1.0f - d_lo;
            if (d_hi >= 0 && d_hi < 1) return 1.0f - d_hi;
            return 0.0f;
          };
          float bx = edge(x, d.owned_min.x, d.owned_max.x);
          float by = edge(y, d.owned_min.y, d.owned_max.y);
          border[i] = std::max(bx, by);
        }

        writeParticlesVTK(frame, pos, vel, rad, gpu_owner, border, scene,
                          max_steps);

        // Write domain decomposition lines once (first frame only)
        if (frame == 0) {
          writeDomainBoundaryVTK(cfg.domain_min, cfg.domain_max, doms, scene,
                                 max_steps);
        }

        ++frame;

        // Accumulate energy diagnostics across all GPUs
        float total_ke = 0, total_px = 0, total_py = 0, total_pz = 0;
        for (int g = 0; g < num_gpus; ++g) {
          CHECK_CUDA(cudaSetDevice(g));
          if (pds[g].n > 0) {
            dim3 grid((pds[g].n + BLOCK.x - 1) / BLOCK.x);
            float g_ke, g_px, g_py, g_pz;
            computeDiagnostics(pds[g], BLOCK, grid, g_ke, g_px, g_py, g_pz);
            total_ke += g_ke;
            total_px += g_px;
            total_py += g_py;
            total_pz += g_pz;
          }
        }

        printf("step %6ld  frame %4d  KE=%.4e  P=(%.3e,%.3e,%.3e)", step,
               frame - 1, total_ke, total_px, total_py, total_pz);
        for (int g = 0; g < num_gpus; ++g)
          printf("  GPU%d:%zu", g, pds[g].n);
        printf("\n");
      }
    }

    // ------------------------------------------------------------------ //
    // 7.  Cleanup
    // ------------------------------------------------------------------ //
    for (int g = 0; g < num_gpus; ++g) {
      CHECK_CUDA(cudaSetDevice(g));
      freeParticleDevice(pds[g]);
      CHECK_CUDA(cudaFree(d_nb[g]));
    }

  } catch (const std::exception &e) {
    fprintf(stderr, "Fatal error: %s\n", e.what());
    return 1;
  }

  return 0;
}
