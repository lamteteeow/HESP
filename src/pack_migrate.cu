#include "pack_migrate.cuh"

__global__ void checkMigration(const size_t n, const Vec3 *d_positions,
                               const Vec3 owned_min, const Vec3 owned_max,
                               int *d_flag) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  const Vec3 p = d_positions[i];
  if (p.x < owned_min.x || p.x >= owned_max.x ||
      p.y < owned_min.y || p.y >= owned_max.y ||
      p.z < owned_min.z || p.z >= owned_max.z) {
    *d_flag = 1;
  }
}
