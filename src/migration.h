#ifndef MIGRATION_H
#define MIGRATION_H

#include "domain.h"
#include "particle_device.cuh"
#include "particle_host.h"
#include "vec3.cuh"
#include <cuda_runtime.h>
#include <vector>

// Particle migration across an nx×ny×nz GPU grid.
// Downloads all owned particles, checks for crossing, merges,
// re-splits by (x, y, z) position, and re-uploads.
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
      const float z = hosts[g].positions[i].z;
      if (x < dom.owned_min.x || x >= dom.owned_max.x ||
          y < dom.owned_min.y || y >= dom.owned_max.y ||
          z < dom.owned_min.z || z >= dom.owned_max.z)
        crossed = true;
    }
  }
  if (!crossed) return;

  // Merge all particles
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

  // Re-upload
  for (int g = 0; g < num_gpus; ++g) {
    cudaSetDevice(g);
    freeParticleDevice(pds[g]);
    new_hosts[g].upload(pds[g], doms[g].total_cells, total_n);
  }
}

#endif // MIGRATION_H
