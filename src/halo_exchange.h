#ifndef HALO_EXCHANGE_H
#define HALO_EXCHANGE_H

#include "domain.h"
#include "particle_device.cuh"
#include "vec3.cuh"
#include <cuda_runtime.h>
#include <vector>

// Download particles in a boundary strip from a GPU.
// filter_axis: 0=x, 1=y, 2=z — which coordinate to filter on.
inline void collectHalo(const ParticleDevice &pd, float strip_lo,
                        float strip_hi, std::vector<Vec3> &h_pos,
                        std::vector<Vec3> &h_vel, std::vector<float> &h_rad,
                        std::vector<float> &h_kn, std::vector<float> &h_gn,
                        int filter_axis = 0) {
  const size_t n = pd.n;
  std::vector<Vec3> pos(n), vel(n);
  std::vector<float> rad(n), kn(n), gn(n);

  cudaMemcpy(pos.data(), pd.d_positions, n * sizeof(Vec3), cudaMemcpyDeviceToHost);
  cudaMemcpy(vel.data(), pd.d_velocities, n * sizeof(Vec3), cudaMemcpyDeviceToHost);
  cudaMemcpy(rad.data(), pd.d_radii, n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(kn.data(), pd.d_kn, n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(gn.data(), pd.d_gamma_n, n * sizeof(float), cudaMemcpyDeviceToHost);

  h_pos.clear(); h_vel.clear(); h_rad.clear(); h_kn.clear(); h_gn.clear();
  for (size_t i = 0; i < n; ++i) {
    float coord = (filter_axis == 0) ? pos[i].x
                : (filter_axis == 1) ? pos[i].y
                : pos[i].z;
    if (coord >= strip_lo && coord < strip_hi) {
      h_pos.push_back(pos[i]);
      h_vel.push_back(vel[i]);
      h_rad.push_back(rad[i]);
      h_kn.push_back(kn[i]);
      h_gn.push_back(gn[i]);
    }
  }
}

inline void uploadHalo(ParticleDevice &pd, const std::vector<Vec3> &h_pos,
                       const std::vector<Vec3> &h_vel,
                       const std::vector<float> &h_rad,
                       const std::vector<float> &h_kn,
                       const std::vector<float> &h_gn) {
  const size_t nh = h_pos.size();
  if (nh == 0) return;
  const size_t off = pd.n_total;
  pd.n_total += nh;
  cudaMemcpy(pd.d_positions + off, h_pos.data(), nh * sizeof(Vec3), cudaMemcpyHostToDevice);
  cudaMemcpy(pd.d_velocities + off, h_vel.data(), nh * sizeof(Vec3), cudaMemcpyHostToDevice);
  cudaMemcpy(pd.d_radii + off, h_rad.data(), nh * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(pd.d_kn + off, h_kn.data(), nh * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(pd.d_gamma_n + off, h_gn.data(), nh * sizeof(float), cudaMemcpyHostToDevice);
}

// Halo exchange across an nx×ny×nz GPU grid.
// Each GPU exchanges with up to 6 neighbors (±X, ±Y, ±Z).
inline void exchangeHalos(std::vector<ParticleDevice> &pds,
                          const std::vector<Domain> &doms) {
  const int num_gpus = static_cast<int>(pds.size());

  struct HaloBuf {
    std::vector<Vec3> pos, vel;
    std::vector<float> rad, kn, gn;
  };
  std::vector<HaloBuf> right_strip(num_gpus), left_strip(num_gpus);
  std::vector<HaloBuf> top_strip(num_gpus), bottom_strip(num_gpus);
  std::vector<HaloBuf> front_strip(num_gpus), back_strip(num_gpus);

  // --- Phase 1: collect boundary strips from each GPU ---
  for (int g = 0; g < num_gpus; ++g) {
    const Domain &dom = doms[g];
    cudaSetDevice(g);

    // X direction
    if (dom.right_neighbor >= 0)
      collectHalo(pds[g], dom.owned_max.x - dom.halo_width, dom.owned_max.x,
                  right_strip[g].pos, right_strip[g].vel, right_strip[g].rad,
                  right_strip[g].kn, right_strip[g].gn, 0);
    if (dom.left_neighbor >= 0)
      collectHalo(pds[g], dom.owned_min.x, dom.owned_min.x + dom.halo_width,
                  left_strip[g].pos, left_strip[g].vel, left_strip[g].rad,
                  left_strip[g].kn, left_strip[g].gn, 0);

    // Y direction
    if (dom.top_neighbor >= 0)
      collectHalo(pds[g], dom.owned_max.y - dom.halo_width, dom.owned_max.y,
                  top_strip[g].pos, top_strip[g].vel, top_strip[g].rad,
                  top_strip[g].kn, top_strip[g].gn, 1);
    if (dom.bottom_neighbor >= 0)
      collectHalo(pds[g], dom.owned_min.y, dom.owned_min.y + dom.halo_width,
                  bottom_strip[g].pos, bottom_strip[g].vel, bottom_strip[g].rad,
                  bottom_strip[g].kn, bottom_strip[g].gn, 1);

    // Z direction
    if (dom.front_neighbor >= 0)
      collectHalo(pds[g], dom.owned_max.z - dom.halo_width, dom.owned_max.z,
                  front_strip[g].pos, front_strip[g].vel, front_strip[g].rad,
                  front_strip[g].kn, front_strip[g].gn, 2);
    if (dom.back_neighbor >= 0)
      collectHalo(pds[g], dom.owned_min.z, dom.owned_min.z + dom.halo_width,
                  back_strip[g].pos, back_strip[g].vel, back_strip[g].rad,
                  back_strip[g].kn, back_strip[g].gn, 2);
  }

  // --- Phase 2: upload received halos to each GPU ---
  for (int g = 0; g < num_gpus; ++g) {
    const Domain &dom = doms[g];
    cudaSetDevice(g);
    pds[g].n_total = pds[g].n;

    // X
    if (dom.left_neighbor >= 0) {
      const HaloBuf &src = right_strip[dom.left_neighbor];
      uploadHalo(pds[g], src.pos, src.vel, src.rad, src.kn, src.gn);
    }
    if (dom.right_neighbor >= 0) {
      const HaloBuf &src = left_strip[dom.right_neighbor];
      uploadHalo(pds[g], src.pos, src.vel, src.rad, src.kn, src.gn);
    }
    // Y
    if (dom.bottom_neighbor >= 0) {
      const HaloBuf &src = top_strip[dom.bottom_neighbor];
      uploadHalo(pds[g], src.pos, src.vel, src.rad, src.kn, src.gn);
    }
    if (dom.top_neighbor >= 0) {
      const HaloBuf &src = bottom_strip[dom.top_neighbor];
      uploadHalo(pds[g], src.pos, src.vel, src.rad, src.kn, src.gn);
    }
    // Z
    if (dom.back_neighbor >= 0) {
      const HaloBuf &src = front_strip[dom.back_neighbor];
      uploadHalo(pds[g], src.pos, src.vel, src.rad, src.kn, src.gn);
    }
    if (dom.front_neighbor >= 0) {
      const HaloBuf &src = back_strip[dom.front_neighbor];
      uploadHalo(pds[g], src.pos, src.vel, src.rad, src.kn, src.gn);
    }
  }
}

#endif // HALO_EXCHANGE_H
