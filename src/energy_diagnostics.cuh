#ifndef ENERGY_DIAGNOSTICS_CUH
#define ENERGY_DIAGNOSTICS_CUH

#include "particle_device.cuh"
#include "vec3.cuh"
#include <cuda_runtime.h>

// Block-level parallel reduction summing kinetic energy and momentum
// (3 components) for owned particles.  Each block computes partial sums
// via shared-memory reduction and then atomically adds them to global
// accumulators.
//
// All four global scalars must be zeroed before the kernel launch.
//
// Layout of d_accum:  [0]=KE, [1]=px, [2]=py, [3]=pz   (4 floats)
__global__ void computeEnergyAndMomentum(const size_t n,
                                         const Vec3 *d_velocities,
                                         const float *d_masses,
                                         float *d_accum);

// Host helper: allocate zero-initialised device array of 4 floats, launch the
// reduction, copy the result back, and free the device array.
// Call cudaSetDevice(gpu_id) before this.
void computeDiagnostics(const ParticleDevice &pd, const dim3 &block,
                        const dim3 &grid, float &ke, float &px,
                        float &py, float &pz);

#endif // ENERGY_DIAGNOSTICS_CUH
