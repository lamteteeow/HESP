#ifndef INTEGRATION_CUH
#define INTEGRATION_CUH
#include "vec3.cuh"
#include <cuda_runtime.h>

// Elastic wall reflection for all axes.
// In 2D mode (default), z walls are at z=0 so z reflection is a no-op.
__device__ inline void reflectWalls(Vec3 &pos, Vec3 &vel, float r,
                                    const Vec3 gmin, const Vec3 gmax) {
  if (pos.x - r < gmin.x) {
    pos.x = gmin.x + r;
    vel.x = fabsf(vel.x);
  }
  if (pos.x + r > gmax.x) {
    pos.x = gmax.x - r;
    vel.x = -fabsf(vel.x);
  }
  if (pos.y - r < gmin.y) {
    pos.y = gmin.y + r;
    vel.y = fabsf(vel.y);
  }
  if (pos.y + r > gmax.y) {
    pos.y = gmax.y - r;
    vel.y = -fabsf(vel.y);
  }
  if (pos.z - r < gmin.z) {
    pos.z = gmin.z + r;
    vel.z = fabsf(vel.z);
  }
  if (pos.z + r > gmax.z) {
    pos.z = gmax.z - r;
    vel.z = -fabsf(vel.z);
  }
}

// Symplectic Euler integration for owned particles.
// In 2D mode (default), z is clamped to 0.
// In 3D mode (-DMD3D), all three axes are free.
// Only runs for indices [0, n) — halo particles are never integrated.
__global__ inline void integrate(const float dt, const size_t n,
                                 Vec3 *d_positions, Vec3 *d_velocities,
                                 const Vec3 *d_forces, const float *d_masses,
                                 const float *d_radii, const Vec3 global_min,
                                 const Vec3 global_max) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  // Velocity update (force already includes gravity)
  d_velocities[i] += dt * (d_forces[i] / d_masses[i]);

  // Position update
  d_positions[i] += dt * d_velocities[i];

#ifndef MD3D
  // Enforce 2D plane (compile with -DMD3D to disable)
  d_positions[i].z = 0.0f;
  d_velocities[i].z = 0.0f;
#endif

  // Reflective domain walls
  reflectWalls(d_positions[i], d_velocities[i], d_radii[i], global_min,
               global_max);
}

#endif // INTEGRATION_CUH
