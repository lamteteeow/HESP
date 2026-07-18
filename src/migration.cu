#include "migration.h"
#include "benchmark.h"
#include "check_cuda.h"
#include "pack_migrate.cuh"
#include "particle_host.h"
#include "vec3.cuh"
#include <cuda_runtime.h>
#include <cstring>

// ── Buffer management ────────────────────────────────────────────────────────

void allocMigPackBuf(MigPackBuf &b, size_t total_n) {
  b.max_owned = total_n;
  b.capacity  = total_n * 2;

  CHECK_CUDA(cudaMalloc(&b.d_send_count, sizeof(int)));
  CHECK_CUDA(cudaMalloc(&b.d_send_pos,   total_n * sizeof(Vec3)));
  CHECK_CUDA(cudaMalloc(&b.d_send_vel,   total_n * sizeof(Vec3)));
  CHECK_CUDA(cudaMalloc(&b.d_send_mass,  total_n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&b.d_send_rad,   total_n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&b.d_send_kn,    total_n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&b.d_send_gn,    total_n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&b.d_send_gt,    total_n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&b.d_send_mu,    total_n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&b.d_send_ids,   total_n * sizeof(int)));

  CHECK_CUDA(cudaMalloc(&b.d_tmp_count, sizeof(int)));
  CHECK_CUDA(cudaMalloc(&b.d_tmp_pos,   b.capacity * sizeof(Vec3)));
  CHECK_CUDA(cudaMalloc(&b.d_tmp_vel,   b.capacity * sizeof(Vec3)));
  CHECK_CUDA(cudaMalloc(&b.d_tmp_mass,  b.capacity * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&b.d_tmp_rad,   b.capacity * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&b.d_tmp_kn,    b.capacity * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&b.d_tmp_gn,    b.capacity * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&b.d_tmp_gt,    b.max_owned * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&b.d_tmp_mu,    b.max_owned * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&b.d_tmp_ids,   b.capacity * sizeof(int)));
}

void freeMigPackBuf(MigPackBuf &b) {
  CHECK_CUDA(cudaFree(b.d_send_count)); b.d_send_count = nullptr;
  CHECK_CUDA(cudaFree(b.d_send_pos));   b.d_send_pos   = nullptr;
  CHECK_CUDA(cudaFree(b.d_send_vel));   b.d_send_vel   = nullptr;
  CHECK_CUDA(cudaFree(b.d_send_mass));  b.d_send_mass  = nullptr;
  CHECK_CUDA(cudaFree(b.d_send_rad));   b.d_send_rad   = nullptr;
  CHECK_CUDA(cudaFree(b.d_send_kn));    b.d_send_kn    = nullptr;
  CHECK_CUDA(cudaFree(b.d_send_gn));    b.d_send_gn    = nullptr;
  CHECK_CUDA(cudaFree(b.d_send_gt));    b.d_send_gt    = nullptr;
  CHECK_CUDA(cudaFree(b.d_send_mu));    b.d_send_mu    = nullptr;
  CHECK_CUDA(cudaFree(b.d_send_ids));   b.d_send_ids   = nullptr;

  CHECK_CUDA(cudaFree(b.d_tmp_count));  b.d_tmp_count  = nullptr;
  CHECK_CUDA(cudaFree(b.d_tmp_pos));    b.d_tmp_pos    = nullptr;
  CHECK_CUDA(cudaFree(b.d_tmp_vel));    b.d_tmp_vel    = nullptr;
  CHECK_CUDA(cudaFree(b.d_tmp_mass));   b.d_tmp_mass   = nullptr;
  CHECK_CUDA(cudaFree(b.d_tmp_rad));    b.d_tmp_rad    = nullptr;
  CHECK_CUDA(cudaFree(b.d_tmp_kn));     b.d_tmp_kn     = nullptr;
  CHECK_CUDA(cudaFree(b.d_tmp_gn));     b.d_tmp_gn     = nullptr;
  CHECK_CUDA(cudaFree(b.d_tmp_gt));     b.d_tmp_gt     = nullptr;
  CHECK_CUDA(cudaFree(b.d_tmp_mu));     b.d_tmp_mu     = nullptr;
  CHECK_CUDA(cudaFree(b.d_tmp_ids));    b.d_tmp_ids    = nullptr;
}

// ── Helper: copy device→device within the same GPU ──────────────────────────

static void copyDevToDev(void *dst, const void *src, size_t bytes) {
  CHECK_CUDA(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
}

// ── CPU migration path (original) ───────────────────────────────────────────

static void migrateParticlesCPU(std::vector<ParticleDevice> &pds,
                                const std::vector<Domain> &doms,
                                size_t total_n, Benchmark *bench) {
  const int num_gpus = static_cast<int>(pds.size());
  if (num_gpus <= 1) return;

  // Download owned particles from all GPUs
  if (bench) bench->startHost(Benchmark::MIG_DOWNLOAD);
  std::vector<ParticleHost> hosts(num_gpus);
  for (int g = 0; g < num_gpus; ++g) {
    cudaSetDevice(g);
    hosts[g].download(pds[g]);
  }
  if (bench) bench->stopHost(Benchmark::MIG_DOWNLOAD);

  // Merge all particles
  if (bench) bench->startHost(Benchmark::MIG_MERGE);
  ParticleHost merged;
  for (int g = 0; g < num_gpus; ++g) {
    const ParticleHost &src = hosts[g];
    for (size_t i = 0; i < src.n; ++i)
      merged.push(src.positions[i], src.velocities[i], src.masses[i],
                  src.radii[i], src.kn[i], src.gamma_n[i], src.gamma_t[i],
                  src.mu[i], src.ids[i]);
  }

  // Re-split into nx×ny×nz grid
  const Domain &d0 = doms[0];
  const int nx = d0.grid_nx, ny = d0.grid_ny, nz = d0.grid_nz;
  const float gmin_x = d0.global_min.x, gmin_y = d0.global_min.y,
              gmin_z = d0.global_min.z;
  const float gmax_x = d0.global_max.x, gmax_y = d0.global_max.y,
              gmax_z = d0.global_max.z;
  const float dx = (gmax_x - gmin_x) / nx;
  const float dy = (gmax_y - gmin_y) / ny;
  const float dz = (gmax_z - gmin_z) / nz;

  std::vector<ParticleHost> new_hosts(num_gpus);
  for (size_t i = 0; i < merged.n; ++i) {
    const float x = merged.positions[i].x;
    const float y = merged.positions[i].y;
    const float z = merged.positions[i].z;
    int gx = std::min(std::max(static_cast<int>((x - gmin_x) / dx), 0),
                      nx - 1);
    int gy = std::min(std::max(static_cast<int>((y - gmin_y) / dy), 0),
                      ny - 1);
    int gz = std::min(std::max(static_cast<int>((z - gmin_z) / dz), 0),
                      nz - 1);
    int g = (gz * ny + gy) * nx + gx;
    new_hosts[g].push(merged.positions[i], merged.velocities[i],
                      merged.masses[i], merged.radii[i], merged.kn[i],
                      merged.gamma_n[i], merged.gamma_t[i], merged.mu[i],
                      merged.ids[i]);
  }
  if (bench) bench->stopHost(Benchmark::MIG_MERGE);

  // Re-upload
  if (bench) bench->startHost(Benchmark::MIG_UPLOAD);
  for (int g = 0; g < num_gpus; ++g) {
    cudaSetDevice(g);
    freeParticleDevice(pds[g]);
    new_hosts[g].upload(pds[g], doms[g].total_cells, total_n);
  }
  if (bench) bench->stopHost(Benchmark::MIG_UPLOAD);
}

// ── GPU migration path ──────────────────────────────────────────────────────

static void migrateParticlesGPU(std::vector<ParticleDevice> &pds,
                                const std::vector<Domain> &doms,
                                std::vector<MigPackBuf> &bufs,
                                Benchmark * /*bench*/) {
  const int num_gpus = static_cast<int>(pds.size());
  if (num_gpus <= 1) return;

  constexpr dim3 BLOCK(256);

  // ── Step 1: Compact stayers on each GPU → temp buffer ──────────────────
  // Each GPU compacts its owned particles that stayed in-region into the
  // temp buffer.  Original data is untouched (needed for Step 2 packing).

  std::vector<size_t> n_stay(num_gpus, 0);

  for (int g = 0; g < num_gpus; ++g) {
    cudaSetDevice(g);
    MigPackBuf &b = bufs[g];
    cudaMemset(b.d_tmp_count, 0, sizeof(int));

    if (pds[g].n == 0) continue;

    dim3 grid((pds[g].n + BLOCK.x - 1) / BLOCK.x);
    compactStayers<<<grid, BLOCK>>>(
        pds[g].n,
        pds[g].d_positions,   pds[g].d_velocities,
        pds[g].d_masses,      pds[g].d_radii,
        pds[g].d_kn,          pds[g].d_gamma_n,
        pds[g].d_gamma_t,     pds[g].d_mu,
        pds[g].d_ids,
        doms[g].owned_min,    doms[g].owned_max,
        b.d_tmp_count,
        b.d_tmp_pos,  b.d_tmp_vel,
        b.d_tmp_mass, b.d_tmp_rad,
        b.d_tmp_kn,   b.d_tmp_gn,
        b.d_tmp_gt,   b.d_tmp_mu,
        b.d_tmp_ids);

    cudaGetLastError();
    int cnt = 0;
    cudaMemcpy(&cnt, b.d_tmp_count, sizeof(int), cudaMemcpyDeviceToHost);
    n_stay[g] = static_cast<size_t>(cnt);
  }

  // ── Step 2: Pack migrants + cudaMemcpyPeer to destinations ─────────────
  // For each (src, dst) pair where src != dst, pack particles that moved
  // from src into dst's owned region, then peer-copy them to dst's temp
  // buffer (appended after dst's compacted stayers).

  // Track insertion offset per destination GPU
  std::vector<size_t> n_cur = n_stay; // starts at number of stayers

  for (int src = 0; src < num_gpus; ++src) {
    if (pds[src].n == 0) continue;
    MigPackBuf &sb = bufs[src];

    for (int dst = 0; dst < num_gpus; ++dst) {
      if (dst == src) continue;

      cudaSetDevice(src);
      cudaMemset(sb.d_send_count, 0, sizeof(int));

      dim3 grid((pds[src].n + BLOCK.x - 1) / BLOCK.x);
      packMigrants<<<grid, BLOCK>>>(
          pds[src].n,
          pds[src].d_positions,   pds[src].d_velocities,
          pds[src].d_masses,      pds[src].d_radii,
          pds[src].d_kn,          pds[src].d_gamma_n,
          pds[src].d_gamma_t,     pds[src].d_mu,
          pds[src].d_ids,
          doms[dst].owned_min,    doms[dst].owned_max,
          sb.d_send_count,
          sb.d_send_pos,  sb.d_send_vel,
          sb.d_send_mass, sb.d_send_rad,
          sb.d_send_kn,   sb.d_send_gn,
          sb.d_send_gt,   sb.d_send_mu,
          sb.d_send_ids);

      cudaGetLastError();
      int cnt = 0;
      cudaMemcpy(&cnt, sb.d_send_count, sizeof(int), cudaMemcpyDeviceToHost);
      if (cnt == 0) continue;

      size_t off = n_cur[dst];
      MigPackBuf &db = bufs[dst];

      // Peer-copy each array from src send buffer → dst temp buffer
      cudaMemcpyPeer(db.d_tmp_pos  + off, dst, sb.d_send_pos,  src,
                     cnt * sizeof(Vec3));
      cudaMemcpyPeer(db.d_tmp_vel  + off, dst, sb.d_send_vel,  src,
                     cnt * sizeof(Vec3));
      cudaMemcpyPeer(db.d_tmp_mass + off, dst, sb.d_send_mass, src,
                     cnt * sizeof(float));
      cudaMemcpyPeer(db.d_tmp_rad  + off, dst, sb.d_send_rad,  src,
                     cnt * sizeof(float));
      cudaMemcpyPeer(db.d_tmp_kn   + off, dst, sb.d_send_kn,   src,
                     cnt * sizeof(float));
      cudaMemcpyPeer(db.d_tmp_gn   + off, dst, sb.d_send_gn,   src,
                     cnt * sizeof(float));
      cudaMemcpyPeer(db.d_tmp_gt   + off, dst, sb.d_send_gt,   src,
                     cnt * sizeof(float));
      cudaMemcpyPeer(db.d_tmp_mu   + off, dst, sb.d_send_mu,   src,
                     cnt * sizeof(float));
      cudaMemcpyPeer(db.d_tmp_ids  + off, dst, sb.d_send_ids,  src,
                     cnt * sizeof(int));

      n_cur[dst] += cnt;
    }
  }

  // ── Step 3: Copy temp buffer → main particle arrays, update counts ─────

  for (int g = 0; g < num_gpus; ++g) {
    cudaSetDevice(g);
    MigPackBuf &b = bufs[g];
    size_t n_new = n_cur[g];
    if (n_new == 0) {
      pds[g].n = 0;
      pds[g].n_total = 0;
      continue;
    }

    // Copy shared arrays from temp buffer to main particle arrays
    copyDevToDev(pds[g].d_positions, b.d_tmp_pos,  n_new * sizeof(Vec3));
    copyDevToDev(pds[g].d_velocities,b.d_tmp_vel,  n_new * sizeof(Vec3));
    copyDevToDev(pds[g].d_masses,    b.d_tmp_mass, n_new * sizeof(float));
    copyDevToDev(pds[g].d_radii,     b.d_tmp_rad,  n_new * sizeof(float));
    copyDevToDev(pds[g].d_kn,        b.d_tmp_kn,   n_new * sizeof(float));
    copyDevToDev(pds[g].d_gamma_n,   b.d_tmp_gn,   n_new * sizeof(float));
    copyDevToDev(pds[g].d_ids,       b.d_tmp_ids,  n_new * sizeof(int));
    // Owned-only arrays
    copyDevToDev(pds[g].d_gamma_t,   b.d_tmp_gt,   n_new * sizeof(float));
    copyDevToDev(pds[g].d_mu,        b.d_tmp_mu,   n_new * sizeof(float));

    pds[g].n       = n_new;
    pds[g].n_total = n_new; // halo will be re-built on next step
  }
}

// ── Crossing check (shared by both paths) ───────────────────────────────────

static bool anyParticleCrossed(std::vector<ParticleDevice> &pds,
                                const std::vector<Domain> &doms) {
  const int num_gpus = static_cast<int>(pds.size());
  constexpr dim3 BLOCK(256);
  bool any = false;

  for (int g = 0; g < num_gpus; ++g) {
    cudaSetDevice(g);
    cudaMemset(pds[g].d_mig_flag, 0, sizeof(int));
    if (pds[g].n == 0) continue;

    dim3 grid((pds[g].n + BLOCK.x - 1) / BLOCK.x);
    checkMigration<<<grid, BLOCK>>>(pds[g].n, pds[g].d_positions,
                                    doms[g].owned_min, doms[g].owned_max,
                                    pds[g].d_mig_flag);
    cudaGetLastError();
    int flag = 0;
    cudaMemcpy(&flag, pds[g].d_mig_flag, sizeof(int), cudaMemcpyDeviceToHost);
    if (flag) any = true;
  }
  return any;
}

// ── Main dispatcher ─────────────────────────────────────────────────────────

void migrateParticles(std::vector<ParticleDevice> &pds,
                      const std::vector<Domain> &doms,
                      std::vector<MigPackBuf> &mig_bufs,
                      size_t total_n, Benchmark *bench) {
  const int num_gpus = static_cast<int>(pds.size());
  if (num_gpus <= 1) return;

  // ── GPU-side crossing check (cheap migration guard) ────────────────────
  if (!anyParticleCrossed(pds, doms)) return;

  if (bench) bench->recordMigCrossed();

  // ── Mode dispatch ──────────────────────────────────────────────────────
  const char *mode = getenv("MIGRATE");
  if (mode && strcmp(mode, "gpu") == 0) {
    // GPU path: one host timer wraps the entire operation.
    // MIG_MERGE holds the total; MIG_DOWNLOAD / MIG_UPLOAD are 0.
    if (bench) bench->startHost(Benchmark::MIG_MERGE);
    migrateParticlesGPU(pds, doms, mig_bufs, bench);
    if (bench) bench->stopHost(Benchmark::MIG_MERGE);
  } else {
    // CPU path: migrateParticlesCPU does its own internal timing
    // (MIG_DOWNLOAD / MIG_MERGE / MIG_UPLOAD).
    migrateParticlesCPU(pds, doms, total_n, bench);
  }
}
