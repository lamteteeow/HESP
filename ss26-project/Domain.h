#ifndef DOMAIN_H
#define DOMAIN_H
#include "vec3.cuh"
#include <algorithm>
#include <cmath>
#include <vector>

// Describes one GPU's portion of the simulation domain.
// The domain is split along the X axis into N equal slices.
// For 2D: num_cells.z = 1 and all z-coordinates are 0.
struct Domain {
  int gpu_id;
  int left_neighbor;  // -1 if this is the leftmost domain
  int right_neighbor; // -1 if this is the rightmost domain

  Vec3 global_min, global_max; // full simulation bounding box

  Vec3 owned_min, owned_max; // this GPU's authoritative region
  Vec3 local_min, local_max; // owned region + halo padding on neighbor sides

  float halo_width; // >= largest particle diameter so all contacts are captured
  float cell_size;
  int3 num_cells; // cell grid dimensions for the local (padded) domain
  int total_cells;
};

// Construct N domains by splitting the X axis evenly.
// halo_width should be >= 2 * r_max (largest particle diameter).
inline std::vector<Domain> buildDomains(Vec3 gmin, Vec3 gmax, float halo_width,
                                        float cell_size, int num_gpus) {
  std::vector<Domain> domains(num_gpus);
  const float dx = (gmax.x - gmin.x) / num_gpus;

  for (int g = 0; g < num_gpus; ++g) {
    Domain &d = domains[g];
    d.gpu_id = g;
    d.left_neighbor = (g > 0) ? g - 1 : -1;
    d.right_neighbor = (g < num_gpus - 1) ? g + 1 : -1;

    d.global_min = gmin;
    d.global_max = gmax;
    d.halo_width = halo_width;
    d.cell_size = cell_size;

    // Owned region: [gmin.x + g*dx, gmin.x + (g+1)*dx)
    d.owned_min = {gmin.x + g * dx, gmin.y, gmin.z};
    d.owned_max = {gmin.x + (g + 1) * dx, gmax.y, gmax.z};

    // Local region (owned + halo padding towards neighbors)
    float lmin_x = d.owned_min.x - (d.left_neighbor >= 0 ? halo_width : 0.0f);
    float lmax_x = d.owned_max.x + (d.right_neighbor >= 0 ? halo_width : 0.0f);
    d.local_min = {std::max(lmin_x, gmin.x), gmin.y, gmin.z};
    d.local_max = {std::min(lmax_x, gmax.x), gmax.y, gmax.z};

    d.num_cells.x = static_cast<int>(
        std::ceil((d.local_max.x - d.local_min.x) / cell_size));
    d.num_cells.y = static_cast<int>(
        std::ceil((d.local_max.y - d.local_min.y) / cell_size));
    d.num_cells.z = 1; // 2D: single layer in z
    d.total_cells = d.num_cells.x * d.num_cells.y * d.num_cells.z;
  }

  return domains;
}

#endif // DOMAIN_H
