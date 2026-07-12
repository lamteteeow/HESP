#ifndef MIGRATION_H
#define MIGRATION_H

#include "domain.h"
#include "particle_device.cuh"
#include "particle_host.h"
#include "vec3.cuh"
#include <cuda_runtime.h>
#include <vector>

// Particle migration: move particles that have crossed domain boundaries
// to the GPU that now owns them.
//
// Strategy: full CPU round-trip (simple, correct).
//   1. Download all owned particles from all GPUs.
//   2. Detect particles that crossed their owned region.
//   3. If any crossed, merge + re-split across N domains + re-upload.
//
// This is simple and correct but O(N) transfers per migration.
// TODO: replace with in-GPU compaction + cudaMemcpyPeer for efficiency.
inline void migrateParticles(std::vector<ParticleDevice> &pds,
                             const std::vector<Domain> &doms, size_t total_n) {
  const int num_gpus = static_cast<int>(pds.size());

  // Download owned particles from all GPUs
  std::vector<ParticleHost> hosts(num_gpus);
  for (int g = 0; g < num_gpus; ++g) {
    cudaSetDevice(g);
    hosts[g].download(pds[g]);
  }

  // Quick check: any particle crossed out of its owning domain?
  bool crossed = false;
  for (int g = 0; g < num_gpus && !crossed; ++g) {
    const Domain &dom = doms[g];
    for (size_t i = 0; i < hosts[g].n && !crossed; ++i) {
      const float x = hosts[g].positions[i].x;
      const float y = hosts[g].positions[i].y;
      if (x < dom.owned_min.x || x >= dom.owned_max.x ||
          y < dom.owned_min.y || y >= dom.owned_max.y)
        crossed = true;
    }
  }
  if (!crossed)
    return;

  // Merge all particles from all GPUs
  ParticleHost merged;
  for (int g = 0; g < num_gpus; ++g) {
    const ParticleHost &src = hosts[g];
    for (size_t i = 0; i < src.n; ++i)
      merged.push(src.positions[i], src.velocities[i], src.masses[i],
                  src.radii[i], src.kn[i], src.gamma_n[i], src.gamma_t[i],
                  src.mu[i], src.ids[i]);
  }

  // Re-split into nx × ny grid by (x, y) position
  const Domain &d0 = doms[0];
  const int nx = d0.grid_nx, ny = d0.grid_ny;
  const float gmin_x = d0.global_min.x, gmin_y = d0.global_min.y;
  const float gmax_x = d0.global_max.x, gmax_y = d0.global_max.y;
  const float dx = (gmax_x - gmin_x) / nx;
  const float dy = (gmax_y - gmin_y) / ny;

  std::vector<ParticleHost> new_hosts(num_gpus);
  for (size_t i = 0; i < merged.n; ++i) {
    const float x = merged.positions[i].x;
    const float y = merged.positions[i].y;
    int gx = static_cast<int>((x - gmin_x) / dx);
    int gy = static_cast<int>((y - gmin_y) / dy);
    gx = std::min(std::max(gx, 0), nx - 1);
    gy = std::min(std::max(gy, 0), ny - 1);
    int g = gy * nx + gx;
    new_hosts[g].push(merged.positions[i], merged.velocities[i],
                      merged.masses[i], merged.radii[i], merged.kn[i],
                      merged.gamma_n[i], merged.gamma_t[i], merged.mu[i],
                      merged.ids[i]);
  }

  // Re-upload to GPUs (free old arrays, allocate fresh)
  for (int g = 0; g < num_gpus; ++g) {
    cudaSetDevice(g);
    freeParticleDevice(pds[g]);
    new_hosts[g].upload(pds[g], doms[g].total_cells, total_n);
  }
}

#endif // MIGRATION_H
