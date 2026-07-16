#include "energy_diagnostics.cuh"
#include "check_cuda.h"

__global__ void computeEnergyAndMomentum(const size_t n,
                                         const Vec3 *d_velocities,
                                         const float *d_masses,
                                         float *d_accum) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  // Per-thread partials
  float ke = 0.0f;
  float px = 0.0f, py = 0.0f, pz = 0.0f;
  if (i < n) {
    const Vec3 v = d_velocities[i];
    const float m = d_masses[i];
    const float v2 = v.x * v.x + v.y * v.y + v.z * v.z;
    ke = 0.5f * m * v2;
    px = m * v.x;
    py = m * v.y;
    pz = m * v.z;
  }

  // ---------- shared-memory reduction ----------
  extern __shared__ float s_buf[]; // [blockDim.x][4] interleaved
  const unsigned int tid = threadIdx.x;
  s_buf[tid * 4 + 0] = ke;
  s_buf[tid * 4 + 1] = px;
  s_buf[tid * 4 + 2] = py;
  s_buf[tid * 4 + 3] = pz;
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_buf[tid * 4 + 0] += s_buf[(tid + stride) * 4 + 0];
      s_buf[tid * 4 + 1] += s_buf[(tid + stride) * 4 + 1];
      s_buf[tid * 4 + 2] += s_buf[(tid + stride) * 4 + 2];
      s_buf[tid * 4 + 3] += s_buf[(tid + stride) * 4 + 3];
    }
    __syncthreads();
  }

  // Thread 0 writes block partials to global accumulators
  if (tid == 0) {
    atomicAdd(&d_accum[0], s_buf[0]);
    atomicAdd(&d_accum[1], s_buf[1]);
    atomicAdd(&d_accum[2], s_buf[2]);
    atomicAdd(&d_accum[3], s_buf[3]);
  }
}

void computeDiagnostics(const ParticleDevice &pd, const dim3 &block,
                        const dim3 &grid, float &ke, float &px,
                        float &py, float &pz) {
  float *d_accum = nullptr;
  CHECK_CUDA(cudaMalloc(&d_accum, 4 * sizeof(float)));
  CHECK_CUDA(cudaMemset(d_accum, 0, 4 * sizeof(float)));

  const size_t shm_bytes = block.x * 4 * sizeof(float); // 4 values per thread
  computeEnergyAndMomentum<<<grid, block, shm_bytes>>>(pd.n, pd.d_velocities,
                                                       pd.d_masses, d_accum);
  CHECK_LAST_CUDA();

  float h[4] = {};
  CHECK_CUDA(cudaMemcpy(h, d_accum, 4 * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(d_accum));
  ke = h[0];
  px = h[1];
  py = h[2];
  pz = h[3];
}
