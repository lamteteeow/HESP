#ifndef FORCE_KERNELS_CUH
#define FORCE_KERNELS_CUH
#include "vec3.cuh"
#include <cuda_runtime.h>

// Spring-dashpot (DEM) contact force kernel.
//
// Iterates over cell neighbors and computes normal + tangential contact forces
// for every owned particle (i < n) against all particles in neighboring cells,
// including halo particles (i in [n, n_total)).
//
// Material properties: harmonic-mean effective stiffness for mixed materials.
// kn and gamma_n arrays are sized for all particles (owned + halo);
// gamma_t and mu are owned-only — fall back to particle i for halo contacts.
__global__ inline void computeContactForces(
    const size_t n,       // owned particle count
    const size_t n_total, // owned + halo
    const Vec3 *d_positions, const Vec3 *d_velocities,
    Vec3 *d_forces, // size n — written for owned particles only
    const float *d_masses, const float *d_radii, const float *d_kn,
    const float *d_gamma_n,
    const float *d_gamma_t, // capacity/2 — owned only
    const float *d_mu,      // capacity/2 — owned only
    const int *d_cellHeads, const int *d_cellTails, const int *d_cellIndexes,
    const int *d_neighbors_of_cell, const Vec3 gravity) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  const Vec3 pi = d_positions[i];
  const Vec3 vi = d_velocities[i];
  const float ri = d_radii[i];

  Vec3 fi = gravity * d_masses[i]; // gravity body force

  const int cell = d_cellIndexes[i];
  const int base = cell * 27;

  for (int k = 0; k < 27; ++k) {
    const int nbr = d_neighbors_of_cell[base + k];
    if (nbr < 0)
      continue;
    for (int j = d_cellHeads[nbr]; j >= 0; j = d_cellTails[j]) {
      if (static_cast<size_t>(j) == i)
        continue;

      const Vec3 pj = d_positions[j];
      const Vec3 vj = d_velocities[j];
      const Vec3 delta = pi - pj;
      const float dist = length(delta);
      if (dist < 1e-12f)
        continue;
      const float overlap = (ri + d_radii[j]) - dist;
      if (overlap <= 0.0f)
        continue;

      // Effective material properties (harmonic mean for mixed materials)
      const float kn_j = d_kn[j];
      const float kn_eff = 2.0f * d_kn[i] * kn_j / (d_kn[i] + kn_j);
      const float gn_j = d_gamma_n[j];
      const float gn_eff = 2.0f * d_gamma_n[i] * gn_j / (d_gamma_n[i] + gn_j);
      const float gt_eff = (static_cast<size_t>(j) < n)
                               ? 2.0f * d_gamma_t[i] * d_gamma_t[j] /
                                     (d_gamma_t[i] + d_gamma_t[j])
                               : d_gamma_t[i];
      const float mu_eff = (static_cast<size_t>(j) < n)
                               ? 2.0f * d_mu[i] * d_mu[j] / (d_mu[i] + d_mu[j])
                               : d_mu[i];

      // Normal force (spring + dashpot)
      const Vec3 n_hat = delta / dist;
      const Vec3 v_rel = vi - vj;
      const float vn = dot(v_rel, n_hat);
      const float fn_mag = kn_eff * overlap - gn_eff * vn;
      const Vec3 fn = fn_mag * n_hat;

      // Tangential force (viscous + Coulomb limit)
      const Vec3 vt = v_rel - n_hat * vn;
      const float vt_len = length(vt);
      Vec3 ft{0.0f, 0.0f, 0.0f};
      if (vt_len > 1e-8f) {
        const float coulomb = mu_eff * fabsf(fn_mag);
        const float ft_mag = fminf(gt_eff * vt_len, coulomb);
        ft = -(vt / vt_len) * ft_mag;
      }

      fi += fn + ft;
    }
  }

  d_forces[i] = fi;
}

#endif // FORCE_KERNELS_CUH
