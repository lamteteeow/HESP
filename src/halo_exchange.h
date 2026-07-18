#ifndef HALO_EXCHANGE_H
#define HALO_EXCHANGE_H

#include "domain.h"
#include "particle_device.cuh"
#include "vec3.cuh"
#include <cuda_runtime.h>
#include <vector>

// Pre-allocated GPU buffers for halo packing (one per GPU).
struct HaloPackBuf {
  int   *d_count;
  Vec3  *d_pos, *d_vel;
  float *d_rad, *d_kn, *d_gn;
  size_t cap;
};

void allocHaloPackBuf(HaloPackBuf &b, size_t max_n);
void freeHaloPackBuf(HaloPackBuf &b);

// GPU-side pack kernel + count retrieval.
int packStrip(const ParticleDevice &pd, float lo, float hi,
              int axis, HaloPackBuf &buf, dim3 block, size_t buf_off);

// Halo exchange — reads HALO env var:
//   HALO=gpu: GPU packing + cudaMemcpyPeer
//   HALO=cpu (default): download → CPU filter → upload
int exchangeHalos(std::vector<ParticleDevice> &pds,
                  const std::vector<Domain> &doms,
                  std::vector<HaloPackBuf> &halo_bufs,
                  dim3 block);

// Get current halo mode string (for benchmark labelling)
inline const char *haloMode() {
  const char *v = getenv("HALO");
  return (v && strcmp(v, "gpu") == 0) ? "gpu" : "cpu";
}

#endif // HALO_EXCHANGE_H
