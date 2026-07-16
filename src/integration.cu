#include "integration.cuh"

__device__ void reflectWalls(Vec3 &pos, Vec3 &vel, float r,
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

__global__ void integrate(const float dt, const size_t n,
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

  // Reflective domain walls
  reflectWalls(d_positions[i], d_velocities[i], d_radii[i], global_min,
               global_max);
}
