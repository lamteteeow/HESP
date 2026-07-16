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

#endif // PACK_MIGRATE_CUH
