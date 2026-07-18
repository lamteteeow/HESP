#include "halo_exchange.h"
#include "pack_halo.cuh"
#include "particle_host.h"
#include <cstring>
#include <cuda_runtime.h>

void allocHaloPackBuf(HaloPackBuf &b, size_t max_n) {
  b.cap = max_n;
  cudaMalloc(&b.d_count, sizeof(int));
  cudaMalloc(&b.d_pos, max_n * sizeof(Vec3));
  cudaMalloc(&b.d_vel, max_n * sizeof(Vec3));
  cudaMalloc(&b.d_rad, max_n * sizeof(float));
  cudaMalloc(&b.d_kn,  max_n * sizeof(float));
  cudaMalloc(&b.d_gn,  max_n * sizeof(float));
}

void freeHaloPackBuf(HaloPackBuf &b) {
  cudaFree(b.d_count); cudaFree(b.d_pos); cudaFree(b.d_vel);
  cudaFree(b.d_rad);   cudaFree(b.d_kn);  cudaFree(b.d_gn);
}

int packStrip(const ParticleDevice &pd, float lo, float hi,
              int axis, HaloPackBuf &buf, dim3 block, size_t buf_off) {
  cudaMemset(buf.d_count, 0, sizeof(int));
  if (pd.n == 0) return 0;
  dim3 grid((pd.n + block.x - 1) / block.x);
  packHaloParticles<<<grid, block>>>(pd.n,
      pd.d_positions, pd.d_velocities, pd.d_radii, pd.d_kn, pd.d_gamma_n,
      lo, hi, axis,
      buf.d_count,
      buf.d_pos + buf_off, buf.d_vel + buf_off,
      buf.d_rad + buf_off, buf.d_kn + buf_off, buf.d_gn + buf_off);
  cudaGetLastError();
  int count = 0;
  cudaMemcpy(&count, buf.d_count, sizeof(int), cudaMemcpyDeviceToHost);
  return count;
}

// ── GPU halo exchange (original) ──────────────────────────────────────────

static int exchangeHalosGPU(std::vector<ParticleDevice> &pds,
                             const std::vector<Domain> &doms,
                             std::vector<HaloPackBuf> &halo_bufs,
                             dim3 block) {
  const int num_gpus = static_cast<int>(pds.size());

  struct Strip { int count; size_t off; };
  std::vector<Strip> right(num_gpus), left(num_gpus);
  std::vector<Strip> top(num_gpus), bottom(num_gpus);
  std::vector<Strip> front(num_gpus), back(num_gpus);

  for (int g = 0; g < num_gpus; ++g) {
    const Domain &dom = doms[g];
    cudaSetDevice(g);
    size_t off = 0;

    if (dom.right_neighbor >= 0) {
      right[g].off = off;
      right[g].count = packStrip(pds[g], dom.owned_max.x - dom.halo_width,
                                  dom.owned_max.x, 0, halo_bufs[g], block, off);
      off += right[g].count;
    }
    if (dom.left_neighbor >= 0) {
      left[g].off = off;
      left[g].count = packStrip(pds[g], dom.owned_min.x,
                                 dom.owned_min.x + dom.halo_width, 0, halo_bufs[g], block, off);
      off += left[g].count;
    }
    if (dom.top_neighbor >= 0) {
      top[g].off = off;
      top[g].count = packStrip(pds[g], dom.owned_max.y - dom.halo_width,
                                dom.owned_max.y, 1, halo_bufs[g], block, off);
      off += top[g].count;
    }
    if (dom.bottom_neighbor >= 0) {
      bottom[g].off = off;
      bottom[g].count = packStrip(pds[g], dom.owned_min.y,
                                   dom.owned_min.y + dom.halo_width, 1, halo_bufs[g], block, off);
      off += bottom[g].count;
    }
    if (dom.front_neighbor >= 0) {
      front[g].off = off;
      front[g].count = packStrip(pds[g], dom.owned_max.z - dom.halo_width,
                                  dom.owned_max.z, 2, halo_bufs[g], block, off);
      off += front[g].count;
    }
    if (dom.back_neighbor >= 0) {
      back[g].off = off;
      back[g].count = packStrip(pds[g], dom.owned_min.z,
                                 dom.owned_min.z + dom.halo_width, 2, halo_bufs[g], block, off);
      off += back[g].count;
    }
  }

  auto copyPeer = [&](int dst_gpu, int src_gpu, const Strip &s, size_t &off) {
    if (s.count == 0) return;
    int nh = s.count;
    size_t src_off = s.off;
    cudaMemcpyPeer(pds[dst_gpu].d_positions + off, dst_gpu,
                   halo_bufs[src_gpu].d_pos + src_off, src_gpu, nh * sizeof(Vec3));
    cudaMemcpyPeer(pds[dst_gpu].d_velocities + off, dst_gpu,
                   halo_bufs[src_gpu].d_vel + src_off, src_gpu, nh * sizeof(Vec3));
    cudaMemcpyPeer(pds[dst_gpu].d_radii + off, dst_gpu,
                   halo_bufs[src_gpu].d_rad + src_off, src_gpu, nh * sizeof(float));
    cudaMemcpyPeer(pds[dst_gpu].d_kn + off, dst_gpu,
                   halo_bufs[src_gpu].d_kn + src_off, src_gpu, nh * sizeof(float));
    cudaMemcpyPeer(pds[dst_gpu].d_gamma_n + off, dst_gpu,
                   halo_bufs[src_gpu].d_gn + src_off, src_gpu, nh * sizeof(float));
    off += nh;
  };

  for (int g = 0; g < num_gpus; ++g) {
    cudaSetDevice(g);
    size_t off = pds[g].n;
    pds[g].n_total = off;
    const Domain &dom = doms[g];

    if (dom.left_neighbor >= 0)
      copyPeer(g, dom.left_neighbor, right[dom.left_neighbor], off);
    if (dom.right_neighbor >= 0)
      copyPeer(g, dom.right_neighbor, left[dom.right_neighbor], off);
    if (dom.bottom_neighbor >= 0)
      copyPeer(g, dom.bottom_neighbor, top[dom.bottom_neighbor], off);
    if (dom.top_neighbor >= 0)
      copyPeer(g, dom.top_neighbor, bottom[dom.top_neighbor], off);
    if (dom.back_neighbor >= 0)
      copyPeer(g, dom.back_neighbor, front[dom.back_neighbor], off);
    if (dom.front_neighbor >= 0)
      copyPeer(g, dom.front_neighbor, back[dom.front_neighbor], off);

    pds[g].n_total = off;
  }

  int total_ghosts = 0;
  for (int g = 0; g < num_gpus; ++g) {
    total_ghosts += right[g].count + left[g].count;
    total_ghosts += top[g].count + bottom[g].count;
    total_ghosts += front[g].count + back[g].count;
  }
  return total_ghosts;
}

// ── CPU halo exchange (baseline) ──────────────────────────────────────────

static int exchangeHalosCPU(std::vector<ParticleDevice> &pds,
                             const std::vector<Domain> &doms,
                             std::vector<HaloPackBuf> &halo_bufs,
                             dim3 block) {
  const int num_gpus = static_cast<int>(pds.size());

  // Download owned particles from every GPU
  std::vector<ParticleHost> hosts(num_gpus);
  for (int g = 0; g < num_gpus; ++g) {
    cudaSetDevice(g);
    hosts[g].download(pds[g]);
  }

  // Per-direction host-side strip buffers
  struct HostStrip {
    std::vector<Vec3> pos, vel;
    std::vector<float> rad, kn, gn;
    int count = 0;
  };
  std::vector<HostStrip> right(num_gpus), left(num_gpus);
  std::vector<HostStrip> top(num_gpus), bottom(num_gpus);
  std::vector<HostStrip> front(num_gpus), back(num_gpus);

  // Pack strips on CPU
  for (int g = 0; g < num_gpus; ++g) {
    const Domain &dom = doms[g];
    const ParticleHost &h = hosts[g];

    auto pack = [&](HostStrip &s, float lo, float hi, int axis) {
      for (size_t i = 0; i < h.n; ++i) {
        float coord = (axis == 0) ? h.positions[i].x
                    : (axis == 1) ? h.positions[i].y
                    : h.positions[i].z;
        if (coord >= lo && coord < hi) {
          s.pos.push_back(h.positions[i]);
          s.vel.push_back(h.velocities[i]);
          s.rad.push_back(h.radii[i]);
          s.kn.push_back(h.kn[i]);
          s.gn.push_back(h.gamma_n[i]);
          s.count++;
        }
      }
    };

    if (dom.right_neighbor >= 0)
      pack(right[g], dom.owned_max.x - dom.halo_width, dom.owned_max.x, 0);
    if (dom.left_neighbor >= 0)
      pack(left[g],  dom.owned_min.x, dom.owned_min.x + dom.halo_width, 0);
    if (dom.top_neighbor >= 0)
      pack(top[g],    dom.owned_max.y - dom.halo_width, dom.owned_max.y, 1);
    if (dom.bottom_neighbor >= 0)
      pack(bottom[g], dom.owned_min.y, dom.owned_min.y + dom.halo_width, 1);
    if (dom.front_neighbor >= 0)
      pack(front[g],  dom.owned_max.z - dom.halo_width, dom.owned_max.z, 2);
    if (dom.back_neighbor >= 0)
      pack(back[g],   dom.owned_min.z, dom.owned_min.z + dom.halo_width, 2);
  }

  // Upload strips to target GPUs
  for (int g = 0; g < num_gpus; ++g) {
    cudaSetDevice(g);
    size_t off = pds[g].n;
    pds[g].n_total = off;
    const Domain &dom = doms[g];

    auto upload = [&](const HostStrip &s) {
      if (s.count == 0) return;
      int nh = s.count;
      cudaMemcpy(pds[g].d_positions + off, s.pos.data(), nh * sizeof(Vec3), cudaMemcpyHostToDevice);
      cudaMemcpy(pds[g].d_velocities + off, s.vel.data(), nh * sizeof(Vec3), cudaMemcpyHostToDevice);
      cudaMemcpy(pds[g].d_radii     + off, s.rad.data(), nh * sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(pds[g].d_kn        + off, s.kn.data(),  nh * sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(pds[g].d_gamma_n   + off, s.gn.data(),  nh * sizeof(float), cudaMemcpyHostToDevice);
      off += nh;
    };

    if (dom.left_neighbor   >= 0) upload(right[dom.left_neighbor]);
    if (dom.right_neighbor  >= 0) upload(left[dom.right_neighbor]);
    if (dom.bottom_neighbor >= 0) upload(top[dom.bottom_neighbor]);
    if (dom.top_neighbor    >= 0) upload(bottom[dom.top_neighbor]);
    if (dom.back_neighbor   >= 0) upload(front[dom.back_neighbor]);
    if (dom.front_neighbor  >= 0) upload(back[dom.front_neighbor]);

    pds[g].n_total = off;
  }

  int total_ghosts = 0;
  for (int g = 0; g < num_gpus; ++g) {
    total_ghosts += right[g].count + left[g].count;
    total_ghosts += top[g].count + bottom[g].count;
    total_ghosts += front[g].count + back[g].count;
  }
  return total_ghosts;
}

// ── dispatcher ────────────────────────────────────────────────────────────

int exchangeHalos(std::vector<ParticleDevice> &pds,
                  const std::vector<Domain> &doms,
                  std::vector<HaloPackBuf> &halo_bufs,
                  dim3 block) {
  const char *mode = getenv("HALO");
  if (mode && strcmp(mode, "gpu") == 0)
    return exchangeHalosGPU(pds, doms, halo_bufs, block);
  return exchangeHalosCPU(pds, doms, halo_bufs, block);
}
