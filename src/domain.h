#ifndef DOMAIN_H
#define DOMAIN_H
#include "vec3.cuh"
#include <vector>

// Describes one GPU's portion of the simulation domain.
// Domains are split into an nx×ny×nz grid via factorGrid3D.
struct Domain {
  int gpu_id;
  int left_neighbor, right_neighbor;     // ±X
  int bottom_neighbor, top_neighbor;     // ±Y
  int back_neighbor, front_neighbor;     // ±Z  (-z neighbour, +z neighbour)

  int grid_x, grid_y, grid_z;
  int grid_nx, grid_ny, grid_nz;

  Vec3 global_min, global_max;
  Vec3 owned_min, owned_max;
  Vec3 local_min, local_max;

  float halo_width, cell_size;
  int3 num_cells;
  int total_cells;
};

// Factor n into nx×ny×nz, preferring near-cube decompositions.
// E.g. 8→2×2×2, 12→3×2×2, 6→3×2×1, 4→2×2×1.
void factorGrid3D(int n, int &nx, int &ny, int &nz);

// Construct domains split into nx×ny×nz 3D grid.
std::vector<Domain> buildDomains(Vec3 gmin, Vec3 gmax, float halo_width,
                                  float cell_size, int num_gpus);

// Recompute derived fields after owned_min/max change.
void recomputeDomain(Domain &d);

// Greedy boundary nudging: adjust domain boundaries to reduce load
// imbalance.  Returns true if any boundary moved.
// n_per_gpu: owned particle count per GPU (pds[g].n).
// imbalance_threshold: e.g. 0.15 = 15% triggers a nudge.
bool rebalanceDomains(std::vector<Domain> &doms,
                      const std::vector<size_t> &n_per_gpu,
                      float imbalance_threshold = 0.15f);

#endif // DOMAIN_H
