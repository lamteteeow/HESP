#ifndef PARTICLE_HOST_H
#define PARTICLE_HOST_H

#include "particle_device.cuh"
#include "vec3.cuh"
#include <algorithm>
#include <cuda_runtime.h>
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

  void push(Vec3 pos, Vec3 vel, float mass, float r, float _kn, float _gn,
            float _gt, float _mu) {
    positions.push_back(pos);
    velocities.push_back(vel);
    masses.push_back(mass);
    radii.push_back(r);
    kn.push_back(_kn);
    gamma_n.push_back(_gn);
    gamma_t.push_back(_gt);
    mu.push_back(_mu);
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
    }
    positions.pop_back();
    velocities.pop_back();
    masses.pop_back();
    radii.pop_back();
    kn.pop_back();
    gamma_n.pop_back();
    gamma_t.pop_back();
    mu.pop_back();
    --n;
  }

  // Allocate GPU arrays and upload owned particles.
  // capacity = total_n * 2 (enough for all N particles + halo).
  // Call cudaSetDevice(gpu_id) before this.
  void upload(ParticleDevice &pd, int total_cells, size_t total_n) {
    const size_t cap = total_n * 2; // halo + worst-case migration
    pd.n = n;
    pd.n_total = n;
    pd.capacity = cap;

    cudaMalloc(&pd.d_positions, cap * sizeof(Vec3));
    cudaMalloc(&pd.d_velocities, cap * sizeof(Vec3));
    cudaMalloc(&pd.d_masses, cap * sizeof(float));
    cudaMalloc(&pd.d_radii, cap * sizeof(float));
    cudaMalloc(&pd.d_kn, cap * sizeof(float));
    cudaMalloc(&pd.d_gamma_n, cap * sizeof(float));
    cudaMalloc(&pd.d_gamma_t, (cap / 2) * sizeof(float));
    cudaMalloc(&pd.d_mu, (cap / 2) * sizeof(float));
    cudaMalloc(&pd.d_forces, (cap / 2) * sizeof(Vec3));
    cudaMalloc(&pd.d_cellHeads, total_cells * sizeof(int));
    cudaMalloc(&pd.d_cellTails, cap * sizeof(int));
    cudaMalloc(&pd.d_cellIndexes, cap * sizeof(int));

    if (n == 0)
      return;
    auto cp = [](void *d, const void *h, size_t bytes) {
      cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice);
    };
    cp(pd.d_positions, positions.data(), n * sizeof(Vec3));
    cp(pd.d_velocities, velocities.data(), n * sizeof(Vec3));
    cp(pd.d_masses, masses.data(), n * sizeof(float));
    cp(pd.d_radii, radii.data(), n * sizeof(float));
    cp(pd.d_kn, kn.data(), n * sizeof(float));
    cp(pd.d_gamma_n, gamma_n.data(), n * sizeof(float));
    cp(pd.d_gamma_t, gamma_t.data(), n * sizeof(float));
    cp(pd.d_mu, mu.data(), n * sizeof(float));
  }

  // Download owned particles (indices [0, pd.n)) from GPU.
  // Call cudaSetDevice(gpu_id) before this.
  void download(const ParticleDevice &pd) {
    n = pd.n;
    positions.resize(n);
    velocities.resize(n);
    masses.resize(n);
    radii.resize(n);
    kn.resize(n);
    gamma_n.resize(n);
    gamma_t.resize(n);
    mu.resize(n);
    if (n == 0)
      return;
    auto cp = [](void *h, const void *d, size_t bytes) {
      cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost);
    };
    cp(positions.data(), pd.d_positions, n * sizeof(Vec3));
    cp(velocities.data(), pd.d_velocities, n * sizeof(Vec3));
    cp(masses.data(), pd.d_masses, n * sizeof(float));
    cp(radii.data(), pd.d_radii, n * sizeof(float));
    cp(kn.data(), pd.d_kn, n * sizeof(float));
    cp(gamma_n.data(), pd.d_gamma_n, n * sizeof(float));
    cp(gamma_t.data(), pd.d_gamma_t, n * sizeof(float));
    cp(mu.data(), pd.d_mu, n * sizeof(float));
  }
};

// Free all device arrays in a ParticleDevice.
inline void freeParticleDevice(ParticleDevice &pd) {
  cudaFree(pd.d_positions);
  pd.d_positions = nullptr;
  cudaFree(pd.d_velocities);
  pd.d_velocities = nullptr;
  cudaFree(pd.d_masses);
  pd.d_masses = nullptr;
  cudaFree(pd.d_radii);
  pd.d_radii = nullptr;
  cudaFree(pd.d_kn);
  pd.d_kn = nullptr;
  cudaFree(pd.d_gamma_n);
  pd.d_gamma_n = nullptr;
  cudaFree(pd.d_gamma_t);
  pd.d_gamma_t = nullptr;
  cudaFree(pd.d_mu);
  pd.d_mu = nullptr;
  cudaFree(pd.d_forces);
  pd.d_forces = nullptr;
  cudaFree(pd.d_cellHeads);
  pd.d_cellHeads = nullptr;
  cudaFree(pd.d_cellTails);
  pd.d_cellTails = nullptr;
  cudaFree(pd.d_cellIndexes);
  pd.d_cellIndexes = nullptr;
  pd.n = pd.n_total = pd.capacity = 0;
}

#endif // PARTICLE_HOST_H
