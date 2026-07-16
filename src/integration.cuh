#ifndef INTEGRATION_CUH
#define INTEGRATION_CUH
#include "vec3.cuh"
#include <cuda_runtime.h>

// Elastic wall reflection for all axes.
// In 2D mode (default), z walls are at z=0 so z reflection is a no-op.
__device__ void reflectWalls(Vec3 &pos, Vec3 &vel, float r,
                             const Vec3 gmin, const Vec3 gmax);

// Symplectic Euler integration for owned particles.
// In 2D mode (default), z is clamped to 0.
// In 3D mode (-DMD3D), all three axes are free.
// Only runs for indices [0, n) — halo particles are never integrated.
__global__ void integrate(const float dt, const size_t n,
                          Vec3 *d_positions, Vec3 *d_velocities,
                          const Vec3 *d_forces, const float *d_masses,
                          const float *d_radii, const Vec3 global_min,
                          const Vec3 global_max);

#endif // INTEGRATION_CUH
