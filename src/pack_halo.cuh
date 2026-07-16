#ifndef PACK_HALO_CUH
#define PACK_HALO_CUH

#include "vec3.cuh"
#include <cuda_runtime.h>

// GPU-side particle packing kernel declaration.
// Filters owned particles by coordinate and writes matching ones to
// a contiguous output buffer. Uses an atomic counter for offsets.
//
// d_count must be zeroed before launch. After launch, copy d_count to
// host to know how many particles were packed.
__global__ void packHaloParticles(
    const size_t n,
    const Vec3 *d_positions, const Vec3 *d_velocities,
    const float *d_radii, const float *d_kn, const float *d_gamma_n,
    float strip_lo, float strip_hi, int filter_axis,
    int *d_count,
    Vec3 *d_out_pos, Vec3 *d_out_vel,
    float *d_out_rad, float *d_out_kn, float *d_out_gn);

#endif // PACK_HALO_CUH
