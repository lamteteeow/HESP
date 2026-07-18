#ifndef PACK_MIGRATE_CUH
#define PACK_MIGRATE_CUH

#include "vec3.cuh"
#include <cuda_runtime.h>

// GPU-side crossing check: each thread checks whether its owned particle
// (index < n) has left the domain's owned region.  If any particle crossed,
// writes 1 to d_flag (idempotent — no atomics needed).
//
// d_flag must be memset to 0 before launch.  After launch, copy 4 bytes
// to host: if 0, no particle crossed → skip the full CPU round-trip.
__global__ void checkMigration(const size_t n, const Vec3 *d_positions,
                               const Vec3 owned_min, const Vec3 owned_max,
                               int *d_flag);

// Pack particles that left src_gpu's owned region and now belong to dst_gpu.
// Each thread checks whether its owned particle (index < n) lies inside
// dst_owned_min/dst_owned_max.  If so, atomically increments d_count and
// writes all particle data to the output buffer at that index.
//
// d_count must be memset to 0 before launch.
__global__ void packMigrants(
    const size_t n,
    const Vec3 *d_positions, const Vec3 *d_velocities,
    const float *d_masses, const float *d_radii,
    const float *d_kn, const float *d_gamma_n,
    const float *d_gamma_t, const float *d_mu,
    const int *d_ids,
    Vec3 dst_owned_min, Vec3 dst_owned_max,
    int *d_count,
    Vec3 *d_out_pos, Vec3 *d_out_vel,
    float *d_out_mass, float *d_out_rad,
    float *d_out_kn, float *d_out_gn,
    float *d_out_gt, float *d_out_mu,
    int *d_out_ids);

// Compact particles that stayed inside their owned region into a contiguous
// prefix of the output (temp) buffer.  Each thread checks whether its owned
// particle (index < n) is still inside owned_min/owned_max.  If so, atomically
// increments d_count and writes all data to the output buffer.
//
// d_count must be memset to 0 before launch.
__global__ void compactStayers(
    const size_t n,
    const Vec3 *d_positions, const Vec3 *d_velocities,
    const float *d_masses, const float *d_radii,
    const float *d_kn, const float *d_gamma_n,
    const float *d_gamma_t, const float *d_mu,
    const int *d_ids,
    Vec3 owned_min, Vec3 owned_max,
    int *d_count,
    Vec3 *d_out_pos, Vec3 *d_out_vel,
    float *d_out_mass, float *d_out_rad,
    float *d_out_kn, float *d_out_gn,
    float *d_out_gt, float *d_out_mu,
    int *d_out_ids);

#endif // PACK_MIGRATE_CUH
