#include "migration.h"
#include "benchmark.h"
#include "pack_migrate.cuh"
#include "particle_host.h"
#include "vec3.cuh"
#include <cuda_runtime.h>

void migrateParticles(std::vector<ParticleDevice> &pds,
                      const std::vector<Domain> &doms, size_t total_n,
                      Benchmark *bench) {
  const int num_gpus = static_cast<int>(pds.size());

  // Single GPU: no domain boundaries exist, migration is impossible
  if (num_gpus <= 1) return;

  // ── GPU-side crossing check (4-byte flag per GPU) ─────────────────────
  constexpr dim3 BLOCK(256);
  bool any_crossed = false;
  for (int g = 0; g < num_gpus; ++g) {
    cudaSetDevice(g);
    cudaMemset(pds[g].d_mig_flag, 0, sizeof(int));
    if (pds[g].n == 0)
      continue;
    dim3 grid((pds[g].n + BLOCK.x - 1) / BLOCK.x);
    checkMigration<<<grid, BLOCK>>>(pds[g].n, pds[g].d_positions,
                                    doms[g].owned_min, doms[g].owned_max,
                                    pds[g].d_mig_flag);
    cudaGetLastError();
    int flag = 0;
    cudaMemcpy(&flag, pds[g].d_mig_flag, sizeof(int), cudaMemcpyDeviceToHost);
    if (flag) any_crossed = true;
  }
  if (!any_crossed) return;

  // ── Crossing detected — fall through to full CPU round-trip ───────────
  if (bench) bench->recordMigCrossed();

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
  const float gmin_x = d0.global_min.x, gmin_y = d0.global_min.y, gmin_z = d0.global_min.z;
  const float gmax_x = d0.global_max.x, gmax_y = d0.global_max.y, gmax_z = d0.global_max.z;
  const float dx = (gmax_x - gmin_x) / nx;
  const float dy = (gmax_y - gmin_y) / ny;
  const float dz = (gmax_z - gmin_z) / nz;

  std::vector<ParticleHost> new_hosts(num_gpus);
  for (size_t i = 0; i < merged.n; ++i) {
    const float x = merged.positions[i].x;
    const float y = merged.positions[i].y;
    const float z = merged.positions[i].z;
    int gx = std::min(std::max(static_cast<int>((x - gmin_x) / dx), 0), nx - 1);
    int gy = std::min(std::max(static_cast<int>((y - gmin_y) / dy), 0), ny - 1);
    int gz = std::min(std::max(static_cast<int>((z - gmin_z) / dz), 0), nz - 1);
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
