#ifndef FORCE_KERNELS_CUH
#define FORCE_KERNELS_CUH
#include "vec3.cuh"
#include <cuda_runtime.h>

// Spring-dashpot (DEM) contact force kernel declaration.
//
// Iterates over cell neighbors and computes normal + tangential contact forces
// for every owned particle (i < n) against all particles in neighboring cells,
// including halo particles (i in [n, n_total)).
//
// Material properties: harmonic-mean effective stiffness for mixed materials.
// kn and gamma_n arrays are sized for all particles (owned + halo);
// gamma_t and mu are owned-only — fall back to particle i for halo contacts.
__global__ void computeContactForces(
    const size_t n,       // owned particle count
    const size_t n_total, // owned + halo
    const Vec3 *d_positions, const Vec3 *d_velocities,
    Vec3 *d_forces, // size n — written for owned particles only
    const float *d_masses, const float *d_radii, const float *d_kn,
    const float *d_gamma_n,
    const float *d_gamma_t, // capacity/2 — owned only
    const float *d_mu,      // capacity/2 — owned only
    const int *d_cellHeads, const int *d_cellTails, const int *d_cellIndexes,
    const int *d_neighbors_of_cell, const Vec3 gravity,
    int *d_contact_count);

#endif // FORCE_KERNELS_CUH
