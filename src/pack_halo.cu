#include "pack_halo.cuh"

__global__ void packHaloParticles(
    const size_t n,
    const Vec3 *d_positions, const Vec3 *d_velocities,
    const float *d_radii, const float *d_kn, const float *d_gamma_n,
    float strip_lo, float strip_hi, int filter_axis,
    int *d_count,
    Vec3 *d_out_pos, Vec3 *d_out_vel,
    float *d_out_rad, float *d_out_kn, float *d_out_gn)
{
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  float coord = (filter_axis == 0) ? d_positions[i].x
              : (filter_axis == 1) ? d_positions[i].y
              : d_positions[i].z;

  if (coord >= strip_lo && coord < strip_hi) {
    int out = atomicAdd(d_count, 1);
    d_out_pos[out] = d_positions[i];
    d_out_vel[out] = d_velocities[i];
    d_out_rad[out] = d_radii[i];
    d_out_kn[out]  = d_kn[i];
    d_out_gn[out]  = d_gamma_n[i];
  }
}
