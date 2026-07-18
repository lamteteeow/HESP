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

// ── packMigrants: particles that left src GPU → pack for specific dst GPU ───

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
    int *d_out_ids)
{
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  const Vec3 p = d_positions[i];
  // Check if this particle is inside the destination's owned region
  if (p.x <  dst_owned_min.x || p.x >= dst_owned_max.x ||
      p.y <  dst_owned_min.y || p.y >= dst_owned_max.y ||
      p.z <  dst_owned_min.z || p.z >= dst_owned_max.z)
    return;

  int out = atomicAdd(d_count, 1);
  d_out_pos[out]  = p;
  d_out_vel[out]  = d_velocities[i];
  d_out_mass[out] = d_masses[i];
  d_out_rad[out]  = d_radii[i];
  d_out_kn[out]   = d_kn[i];
  d_out_gn[out]   = d_gamma_n[i];
  d_out_gt[out]   = d_gamma_t[i];
  d_out_mu[out]   = d_mu[i];
  d_out_ids[out]  = d_ids[i];
}

// ── compactStayers: particles that stayed → contiguous prefix of temp buffer ─

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
    int *d_out_ids)
{
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  const Vec3 p = d_positions[i];
  // Check if this particle is still inside its own GPU's owned region
  if (p.x <  owned_min.x || p.x >= owned_max.x ||
      p.y <  owned_min.y || p.y >= owned_max.y ||
      p.z <  owned_min.z || p.z >= owned_max.z)
    return;

  int out = atomicAdd(d_count, 1);
  d_out_pos[out]  = p;
  d_out_vel[out]  = d_velocities[i];
  d_out_mass[out] = d_masses[i];
  d_out_rad[out]  = d_radii[i];
  d_out_kn[out]   = d_kn[i];
  d_out_gn[out]   = d_gamma_n[i];
  d_out_gt[out]   = d_gamma_t[i];
  d_out_mu[out]   = d_mu[i];
  d_out_ids[out]  = d_ids[i];
}
