#ifndef HALO_EXCHANGE_H
#define HALO_EXCHANGE_H

#include "domain.h"
#include "particle_device.cuh"
#include "vec3.cuh"
#include <cuda_runtime.h>
#include <vector>

// Download particles in a boundary strip from a GPU into host buffers.
// strip_lo / strip_hi define the x-range of the strip to collect.
// E.g.: for a right neighbor, collect [owned_max.x - halo_width, owned_max.x).
//       for a left neighbor,  collect [owned_min.x, owned_min.x + halo_width).
//
// Call cudaSetDevice(dom.gpu_id) before this function.
inline void collectHalo(const ParticleDevice &pd, float strip_lo,
                        float strip_hi, std::vector<Vec3> &h_pos,
                        std::vector<Vec3> &h_vel, std::vector<float> &h_rad,
                        std::vector<float> &h_kn, std::vector<float> &h_gn) {
  const size_t n = pd.n;
  std::vector<Vec3> pos(n), vel(n);
  std::vector<float> rad(n), kn(n), gn(n);

  cudaMemcpy(pos.data(), pd.d_positions, n * sizeof(Vec3),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(vel.data(), pd.d_velocities, n * sizeof(Vec3),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(rad.data(), pd.d_radii, n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(kn.data(), pd.d_kn, n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(gn.data(), pd.d_gamma_n, n * sizeof(float),
             cudaMemcpyDeviceToHost);

  h_pos.clear();
  h_vel.clear();
  h_rad.clear();
  h_kn.clear();
  h_gn.clear();

  for (size_t i = 0; i < n; ++i) {
    const float x = pos[i].x;
    if (x >= strip_lo && x < strip_hi) {
      h_pos.push_back(pos[i]);
      h_vel.push_back(vel[i]);
      h_rad.push_back(rad[i]);
      h_kn.push_back(kn[i]);
      h_gn.push_back(gn[i]);
    }
  }
}

// Upload halo particles to a GPU, appending them after existing particles.
// Accumulates: pd.n_total += n_halo so multiple calls stack correctly.
// Call cudaSetDevice(target gpu_id) before this function.
inline void uploadHalo(ParticleDevice &pd, const std::vector<Vec3> &h_pos,
                       const std::vector<Vec3> &h_vel,
                       const std::vector<float> &h_rad,
                       const std::vector<float> &h_kn,
                       const std::vector<float> &h_gn) {
  const size_t nh = h_pos.size();
  if (nh == 0)
    return;
  const size_t off = pd.n_total; // append after current end
  pd.n_total += nh;

  cudaMemcpy(pd.d_positions + off, h_pos.data(), nh * sizeof(Vec3),
             cudaMemcpyHostToDevice);
  cudaMemcpy(pd.d_velocities + off, h_vel.data(), nh * sizeof(Vec3),
             cudaMemcpyHostToDevice);
  cudaMemcpy(pd.d_radii + off, h_rad.data(), nh * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(pd.d_kn + off, h_kn.data(), nh * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(pd.d_gamma_n + off, h_gn.data(), nh * sizeof(float),
             cudaMemcpyHostToDevice);
}

// Halo exchange across N GPU sub-domains split along the X axis.
//
// For each GPU:
//   1. Collect particles in the right boundary strip → send to right neighbor.
//   2. Collect particles in the left boundary strip  → send to left neighbor.
//   3. Upload particles received from left neighbor (its right strip).
//   4. Upload particles received from right neighbor (its left strip).
//
// The host acts as an intermediary (CPU-mediated exchange).
// TODO: replace with cudaMemcpyPeer for direct GPU-to-GPU transfer.
inline void exchangeHalos(std::vector<ParticleDevice> &pds,
                          const std::vector<Domain> &doms) {
  const int num_gpus = static_cast<int>(pds.size());

  // Per-GPU: halo strips to send to left/right neighbors.
  // right_strip[g] = particles from GPU g destined for its right neighbor.
  // left_strip[g]  = particles from GPU g destined for its left neighbor.
  struct HaloBuf {
    std::vector<Vec3> pos, vel;
    std::vector<float> rad, kn, gn;
  };
  std::vector<HaloBuf> right_strip(num_gpus), left_strip(num_gpus);

  // --- Phase 1: collect boundary strips from each GPU ---
  for (int g = 0; g < num_gpus; ++g) {
    const Domain &dom = doms[g];
    cudaSetDevice(g);

    // Collect strip for right neighbor (if any)
    if (dom.right_neighbor >= 0) {
      const float lo = dom.owned_max.x - dom.halo_width;
      const float hi = dom.owned_max.x;
      collectHalo(pds[g], lo, hi, right_strip[g].pos, right_strip[g].vel,
                  right_strip[g].rad, right_strip[g].kn, right_strip[g].gn);
    }

    // Collect strip for left neighbor (if any)
    if (dom.left_neighbor >= 0) {
      const float lo = dom.owned_min.x;
      const float hi = dom.owned_min.x + dom.halo_width;
      collectHalo(pds[g], lo, hi, left_strip[g].pos, left_strip[g].vel,
                  left_strip[g].rad, left_strip[g].kn, left_strip[g].gn);
    }
  }

  // --- Phase 2: upload received halo to each GPU ---
  for (int g = 0; g < num_gpus; ++g) {
    const Domain &dom = doms[g];
    cudaSetDevice(g);

    // Reset halo count; owned count stays fixed
    pds[g].n_total = pds[g].n;

    // Receive from left neighbor → its right_strip
    if (dom.left_neighbor >= 0) {
      const HaloBuf &src = right_strip[dom.left_neighbor];
      uploadHalo(pds[g], src.pos, src.vel, src.rad, src.kn, src.gn);
    }

    // Receive from right neighbor → its left_strip
    if (dom.right_neighbor >= 0) {
      const HaloBuf &src = left_strip[dom.right_neighbor];
      uploadHalo(pds[g], src.pos, src.vel, src.rad, src.kn, src.gn);
    }
  }
}

#endif // HALO_EXCHANGE_H
