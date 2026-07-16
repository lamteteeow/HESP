#ifndef PARTICLE_DEVICE_CUH
#define PARTICLE_DEVICE_CUH
#include "vec3.cuh"
#include <cuda_runtime.h>

// GPU-side particle storage for one sub-domain.
//
// Memory layout:
//   indices [0,       n)        owned particles — updated by force + integrate
//   kernels indices [n,       n_total)  halo/ghost particles from the neighbor
//   GPU — read-only
//
// All arrays sized 'capacity' cover owned+halo combined.
// Owned-only arrays (d_forces, d_gamma_t, d_mu) are sized 'capacity/2' (= max
// owned).
struct ParticleDevice {
  size_t n;        // number of owned particles
  size_t n_total;  // owned + halo (updated after each halo exchange)
  size_t capacity; // max TOTAL slots allocated (owned + halo); capacity >=
                   // n_total always

  // Shared arrays (size = capacity): owned first, halo appended after
  Vec3 *d_positions;
  Vec3 *d_velocities;
  float *d_masses;
  float *d_radii;
  float *d_kn;
  float *d_gamma_n;

  // Owned-only arrays (size = capacity / 2)
  float *d_gamma_t;
  float *d_mu;
  Vec3 *d_forces;

  int *d_ids; // particle IDs (size = capacity, owned+halo), read-only

  // Persistent atomic counter for contact pairs (memset to 0 each step)
  int *d_contact_count;

  // Migration guard flag (1 int — written by checkMigration kernel)
  int *d_mig_flag;

  // Cell linked list (covers n_total particles; d_cellHeads covers total_cells)
  int *d_cellHeads;
  int *d_cellTails;   // size = capacity
  int *d_cellIndexes; // size = capacity
};

#endif // PARTICLE_DEVICE_CUH
