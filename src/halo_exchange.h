#ifndef HALO_EXCHANGE_H
#define HALO_EXCHANGE_H

#include "domain.h"
#include "pack_halo.cuh"
#include "particle_device.cuh"
#include "vec3.cuh"
#include <cuda_runtime.h>

// Pre-allocated GPU buffers for halo packing (one per GPU).
struct HaloPackBuf {
  int   *d_count;
  Vec3  *d_pos, *d_vel;
  float *d_rad, *d_kn, *d_gn;
  size_t cap;
};

inline void allocHaloPackBuf(HaloPackBuf &b, size_t max_n) {
  b.cap = max_n;
  cudaMalloc(&b.d_count, sizeof(int));
  cudaMalloc(&b.d_pos, max_n * sizeof(Vec3));
  cudaMalloc(&b.d_vel, max_n * sizeof(Vec3));
  cudaMalloc(&b.d_rad, max_n * sizeof(float));
  cudaMalloc(&b.d_kn,  max_n * sizeof(float));
  cudaMalloc(&b.d_gn,  max_n * sizeof(float));
}

inline void freeHaloPackBuf(HaloPackBuf &b) {
  cudaFree(b.d_count); cudaFree(b.d_pos); cudaFree(b.d_vel);
  cudaFree(b.d_rad);   cudaFree(b.d_kn);  cudaFree(b.d_gn);
}

// GPU-side pack kernel + count retrieval.  Only 4 bytes come back to CPU.
// 'buf_off' is the starting index in the output buffer (cumulative offset).
inline int packStrip(const ParticleDevice &pd, float lo, float hi,
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

// Halo exchange using GPU-side packing + cudaMemcpyPeer.
// No CPU staging of particle data — only 4-byte counts cross the bus.
inline void exchangeHalos(std::vector<ParticleDevice> &pds,
                          const std::vector<Domain> &doms,
                          std::vector<HaloPackBuf> &halo_bufs,
                          dim3 block) {
  const int num_gpus = static_cast<int>(pds.size());

  // Per-GPU strip: packed data sits in that GPU's halo_bufs[g]
  struct Strip { int count; size_t off; };
  std::vector<Strip> right(num_gpus), left(num_gpus);
  std::vector<Strip> top(num_gpus), bottom(num_gpus);
  std::vector<Strip> front(num_gpus), back(num_gpus);

  // --- Phase 1: pack strips on each GPU with cumulative offsets ---
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

  // Helper: copy packed strip from src GPU's buf → dst GPU's particle arrays
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

  // --- Phase 2: cudaMemcpyPeer strips to target GPUs ---
  for (int g = 0; g < num_gpus; ++g) {
    cudaSetDevice(g);
    size_t off = pds[g].n; // reset halo count, append after owned
    pds[g].n_total = off;

    const Domain &dom = doms[g];

    // Receive from left neighbor → its right strip
    if (dom.left_neighbor >= 0)
      copyPeer(g, dom.left_neighbor, right[dom.left_neighbor], off);
    // Receive from right neighbor → its left strip
    if (dom.right_neighbor >= 0)
      copyPeer(g, dom.right_neighbor, left[dom.right_neighbor], off);
    // Receive from bottom neighbor → its top strip
    if (dom.bottom_neighbor >= 0)
      copyPeer(g, dom.bottom_neighbor, top[dom.bottom_neighbor], off);
    // Receive from top neighbor → its bottom strip
    if (dom.top_neighbor >= 0)
      copyPeer(g, dom.top_neighbor, bottom[dom.top_neighbor], off);
    // Receive from back neighbor → its front strip
    if (dom.back_neighbor >= 0)
      copyPeer(g, dom.back_neighbor, front[dom.back_neighbor], off);
    // Receive from front neighbor → its back strip
    if (dom.front_neighbor >= 0)
      copyPeer(g, dom.front_neighbor, back[dom.front_neighbor], off);

    pds[g].n_total = off;
  }
}

#endif // HALO_EXCHANGE_H
