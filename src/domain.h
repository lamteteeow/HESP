#ifndef DOMAIN_H
#define DOMAIN_H
#include "vec3.cuh"
#include <algorithm>
#include <cmath>
#include <vector>

// Describes one GPU's portion of the simulation domain.
// For 2D (default): split along X only (ny=1).
// For 3D (-DMD3D): split into an nx × ny grid along X and Y.
struct Domain {
  int gpu_id;
  int left_neighbor;   // -1 if at left edge
  int right_neighbor;  // -1 if at right edge
  int bottom_neighbor; // -1 if at bottom edge
  int top_neighbor;    // -1 if at top edge

  int grid_x, grid_y;  // position in the nx × ny GPU grid
  int grid_nx, grid_ny; // grid dimensions

  Vec3 global_min, global_max; // full simulation bounding box

  Vec3 owned_min, owned_max; // this GPU's authoritative region
  Vec3 local_min, local_max; // owned region + halo padding on neighbor sides

  float halo_width;
  float cell_size;
  int3 num_cells;
  int total_cells;
};

// Factor n into a near-square grid (nx × ny) for 2D decomposition.
inline void factorGrid(int n, int &nx, int &ny) {
  nx = 1; ny = n;
  for (int i = static_cast<int>(std::sqrt(n)); i >= 1; --i) {
    if (n % i == 0) { nx = i; ny = n / i; break; }
  }
}

// Construct domains split into an nx × ny grid.
// In 2D mode (no MD3D), falls back to X-only: nx=n, ny=1.
inline std::vector<Domain> buildDomains(Vec3 gmin, Vec3 gmax, float halo_width,
                                        float cell_size, int num_gpus) {
  // Determine grid dimensions
  int nx, ny;
#ifdef MD3D
  factorGrid(num_gpus, nx, ny);  // 2D decomposition for 3D simulation
#else
  nx = num_gpus; ny = 1;         // X-only for 2D simulation
#endif

  const float dx = (gmax.x - gmin.x) / nx;
  const float dy = (gmax.y - gmin.y) / ny;

  std::vector<Domain> domains(num_gpus);
  for (int gy = 0; gy < ny; ++gy) {
    for (int gx = 0; gx < nx; ++gx) {
      int g = gy * nx + gx;
      Domain &d = domains[g];
      d.gpu_id = g;
      d.grid_x = gx; d.grid_y = gy;
      d.grid_nx = nx; d.grid_ny = ny;

      d.left_neighbor   = (gx > 0)          ? gy * nx + (gx - 1) : -1;
      d.right_neighbor  = (gx < nx - 1)     ? gy * nx + (gx + 1) : -1;
      d.bottom_neighbor = (gy > 0)          ? (gy - 1) * nx + gx : -1;
      d.top_neighbor    = (gy < ny - 1)     ? (gy + 1) * nx + gx : -1;

      d.global_min = gmin;
      d.global_max = gmax;
      d.halo_width = halo_width;
      d.cell_size  = cell_size;

      // Owned region
      d.owned_min = {gmin.x + gx * dx, gmin.y + gy * dy, gmin.z};
      d.owned_max = {gmin.x + (gx + 1) * dx, gmin.y + (gy + 1) * dy, gmax.z};

      // Local region (owned + halo towards neighbors)
      float lx_lo = d.owned_min.x - (d.left_neighbor   >= 0 ? halo_width : 0.0f);
      float lx_hi = d.owned_max.x + (d.right_neighbor  >= 0 ? halo_width : 0.0f);
      float ly_lo = d.owned_min.y - (d.bottom_neighbor >= 0 ? halo_width : 0.0f);
      float ly_hi = d.owned_max.y + (d.top_neighbor    >= 0 ? halo_width : 0.0f);
      d.local_min = {std::max(lx_lo, gmin.x), std::max(ly_lo, gmin.y), gmin.z};
      d.local_max = {std::min(lx_hi, gmax.x), std::min(ly_hi, gmax.y), gmax.z};

      // Cell grid
      d.num_cells.x = static_cast<int>(
          std::ceil((d.local_max.x - d.local_min.x) / cell_size));
      d.num_cells.y = static_cast<int>(
          std::ceil((d.local_max.y - d.local_min.y) / cell_size));
#ifdef MD3D
      d.num_cells.z = static_cast<int>(
          std::ceil((d.local_max.z - d.local_min.z) / cell_size));
#else
      d.num_cells.z = 1;
#endif
      d.total_cells = d.num_cells.x * d.num_cells.y * d.num_cells.z;
    }
  }

  return domains;
}

#endif // DOMAIN_H
