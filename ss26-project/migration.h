#ifndef MIGRATION_H
#define MIGRATION_H

#include "Domain.h"
#include "ParticleDevice.cuh"
#include "ParticleHost.h"
#include "Vec3.cuh"
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

  // Quick check: any particle that crossed out of its owning domain?
  bool crossed = false;
  for (int g = 0; g < num_gpus && !crossed; ++g) {
    const Domain &dom = doms[g];
    for (size_t i = 0; i < hosts[g].n && !crossed; ++i) {
      const float x = hosts[g].positions[i].x;
      if (x < dom.owned_min.x || x >= dom.owned_max.x)
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
                  src.mu[i]);
  }

  // Re-split into N bins by x-coordinate
  // Domain g owns particles with x in [owned_min.x, owned_max.x)
  std::vector<ParticleHost> new_hosts(num_gpus);
  for (size_t i = 0; i < merged.n; ++i) {
    const float x = merged.positions[i].x;
    // Find the owning GPU: linear scan is fine for small N
    int g = 0;
    while (g < num_gpus - 1 && x >= doms[g].owned_max.x)
      ++g;
    new_hosts[g].push(merged.positions[i], merged.velocities[i],
                      merged.masses[i], merged.radii[i], merged.kn[i],
                      merged.gamma_n[i], merged.gamma_t[i], merged.mu[i]);
  }

  // Re-upload to GPUs (free old arrays, allocate fresh)
  for (int g = 0; g < num_gpus; ++g) {
    cudaSetDevice(g);
    freeParticleDevice(pds[g]);
    new_hosts[g].upload(pds[g], doms[g].total_cells, total_n);
  }
}

#endif // MIGRATION_H
