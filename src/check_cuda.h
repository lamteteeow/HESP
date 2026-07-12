#ifndef CHECK_CUDA_H
#define CHECK_CUDA_H

#include <cstdio>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

// CUDA error checking macro — wraps all CUDA API calls and kernel launches.
// Usage: CHECK_CUDA(cudaMalloc(&ptr, size));
//        kernel<<<grid, block>>>(...); CHECK_CUDA(cudaGetLastError());
//        CHECK_CUDA(cudaDeviceSynchronize());
#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t _err = (call);                                                 \
    if (_err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error at %s:%d  %s\n  → %s (code %d)\n", __FILE__, \
              __LINE__, #call, cudaGetErrorString(_err),                       \
              static_cast<int>(_err));                                         \
      throw std::runtime_error(std::string("CUDA error: ") +                   \
                               cudaGetErrorString(_err));                      \
    }                                                                          \
  } while (0)

// Check and reset last error after kernel launch.
// Use:  kernel<<<g,b>>>(...); CHECK_LAST_CUDA();
#define CHECK_LAST_CUDA()                                                      \
  do {                                                                         \
    cudaError_t _err = cudaGetLastError();                                     \
    if (_err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA kernel error at %s:%d  → %s (code %d)\n",          \
              __FILE__, __LINE__, cudaGetErrorString(_err),                    \
              static_cast<int>(_err));                                         \
      throw std::runtime_error(std::string("CUDA kernel error: ") +            \
                               cudaGetErrorString(_err));                      \
    }                                                                          \
  } while (0)

#endif // CHECK_CUDA_H
