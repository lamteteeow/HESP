#ifndef PARTICLE_HOST_H
#define PARTICLE_HOST_H

#include "particle_device.cuh"
#include "vec3.cuh"
#include <cstddef>
#include <vector>

// CPU-side particle buffer for one GPU's sub-domain.
// Handles allocation, upload, and download of GPU particle arrays.
//
// The GPU arrays are allocated with 'capacity' total slots:
//   [0,      n)        — owned particles (upload/download)
//   [n,      capacity) — halo slots (written by halo_exchange.h)
//
// capacity = total_n * 2 ensures each GPU can receive all N particles
// in the worst-case scenario where all particles migrate to one side.
struct ParticleHost {
  size_t n = 0;
  std::vector<Vec3> positions, velocities;
  std::vector<float> masses, radii, kn, gamma_n, gamma_t, mu;
  std::vector<int> ids;

  void push(Vec3 pos, Vec3 vel, float mass, float r, float _kn, float _gn,
            float _gt, float _mu, int id = -1) {
    positions.push_back(pos);
    velocities.push_back(vel);
    masses.push_back(mass);
    radii.push_back(r);
    kn.push_back(_kn);
    gamma_n.push_back(_gn);
    gamma_t.push_back(_gt);
    mu.push_back(_mu);
    ids.push_back(id);
    ++n;
  }

  // O(1) removal by swap-with-last (does not preserve order).
  void removeAt(size_t i) {
    const size_t last = n - 1;
    if (i != last) {
      positions[i] = positions[last];
      velocities[i] = velocities[last];
      masses[i] = masses[last];
      radii[i] = radii[last];
      kn[i] = kn[last];
      gamma_n[i] = gamma_n[last];
      gamma_t[i] = gamma_t[last];
      mu[i] = mu[last];
      ids[i] = ids[last];
    }
    positions.pop_back();
    velocities.pop_back();
    masses.pop_back();
    radii.pop_back();
    kn.pop_back();
    gamma_n.pop_back();
    gamma_t.pop_back();
    mu.pop_back();
    ids.pop_back();
    --n;
  }

  // Allocate GPU arrays and upload owned particles.
  // capacity = total_n * 2 (enough for all N particles + halo).
  // Call cudaSetDevice(gpu_id) before this.
  void upload(ParticleDevice &pd, int total_cells, size_t total_n);

  // Download owned particles (indices [0, pd.n)) from GPU.
  // Call cudaSetDevice(gpu_id) before this.
  void download(const ParticleDevice &pd);
};

// Free all device arrays in a ParticleDevice.
void freeParticleDevice(ParticleDevice &pd);

#endif // PARTICLE_HOST_H
