#ifndef FORCE_KERNELS_CUH
#define FORCE_KERNELS_CUH
#include "Vec3.cuh"
#include <cuda_runtime.h>

// Spring-dashpot (DEM) contact force kernel.
//
// Iterates over cell neighbors and computes normal + tangential contact forces
// for every owned particle (i < n) against all particles in neighboring cells,
// including halo particles (i in [n, n_total)).
//
// Material properties (kn, gamma_n, gamma_t, mu) from the owned particle i are
// used for both sides of the contact — valid when all particles share the same
// material. TODO: use harmonic-mean effective stiffness for mixed materials.
__global__ inline void computeContactForces(
    const size_t n,       // owned particle count
    const size_t n_total, // owned + halo
    const Vec3 *d_positions, const Vec3 *d_velocities,
    Vec3 *d_forces, // size n — written for owned particles only
    const float *d_masses, const float *d_radii, const float *d_kn,
    const float *d_gamma_n,
    const float *d_gamma_t, // size n/2 — owned only
    const float *d_mu,      // size n/2 — owned only
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

      // Normal force (spring + dashpot)
      const Vec3 n_hat = delta / dist;
      const Vec3 v_rel = vi - vj;
      const float vn = dot(v_rel, n_hat);
      const float fn_mag = d_kn[i] * overlap - d_gamma_n[i] * vn;
      const Vec3 fn = fn_mag * n_hat;

      // Tangential force (viscous + Coulomb limit)
      const Vec3 vt = v_rel - n_hat * vn;
      const float vt_len = length(vt);
      Vec3 ft{0.0f, 0.0f, 0.0f};
      if (vt_len > 1e-8f) {
        const float coulomb = d_mu[i] * fabsf(fn_mag);
        const float ft_mag = fminf(d_gamma_t[i] * vt_len, coulomb);
        ft = -(vt / vt_len) * ft_mag;
      }

      fi += fn + ft;
    }
  }

  d_forces[i] = fi;
}

#endif // FORCE_KERNELS_CUH
