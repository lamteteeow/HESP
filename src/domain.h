#ifndef DOMAIN_H
#define DOMAIN_H
#include "vec3.cuh"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

// Describes one GPU's portion of the simulation domain.
// 2D mode: X-only split (nx=n, ny=nz=1).
// 3D mode: full nx×ny×nz grid decomposition.
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
inline void factorGrid3D(int n, int &nx, int &ny, int &nz) {
  nx = 1; ny = 1; nz = n;
  float best = std::numeric_limits<float>::max();
  for (int i = 1; i <= n; ++i) {
    if (n % i != 0) continue;
    int rem = n / i;
    // Factor 'rem' into j×k
    for (int j = static_cast<int>(std::sqrt(rem)); j >= 1; --j) {
      if (rem % j == 0) {
        int k = rem / j;
        // Score: variance of log sizes (prefers equal dimensions)
        float v = std::log(static_cast<float>(i))
                + std::log(static_cast<float>(j))
                + std::log(static_cast<float>(k));
        float mean = v / 3.0f;
        float score = (std::log(static_cast<float>(i)) - mean) * (std::log(static_cast<float>(i)) - mean)
                    + (std::log(static_cast<float>(j)) - mean) * (std::log(static_cast<float>(j)) - mean)
                    + (std::log(static_cast<float>(k)) - mean) * (std::log(static_cast<float>(k)) - mean);
        if (score < best) { best = score; nx = i; ny = j; nz = k; }
        break; // j loop picks largest divisor ≤ sqrt(rem), which is fine
      }
    }
  }
}

// Construct domains split into nx×ny×nz 3D grid.
// In 2D mode, falls back to X-only.
inline std::vector<Domain> buildDomains(Vec3 gmin, Vec3 gmax, float halo_width,
                                        float cell_size, int num_gpus) {
  int nx, ny, nz;
#ifdef MD3D
  factorGrid3D(num_gpus, nx, ny, nz);
#else
  nx = num_gpus; ny = 1; nz = 1;
#endif

  const float dx = (gmax.x - gmin.x) / nx;
  const float dy = (gmax.y - gmin.y) / ny;
  const float dz = (gmax.z - gmin.z) / nz;

  std::vector<Domain> domains(num_gpus);
  for (int gz = 0; gz < nz; ++gz) {
    for (int gy = 0; gy < ny; ++gy) {
      for (int gx = 0; gx < nx; ++gx) {
        int g = (gz * ny + gy) * nx + gx;
        Domain &d = domains[g];
        d.gpu_id = g;
        d.grid_x = gx; d.grid_y = gy; d.grid_z = gz;
        d.grid_nx = nx; d.grid_ny = ny; d.grid_nz = nz;

        d.left_neighbor   = (gx > 0)       ? (gz * ny + gy) * nx + (gx - 1) : -1;
        d.right_neighbor  = (gx < nx - 1)  ? (gz * ny + gy) * nx + (gx + 1) : -1;
        d.bottom_neighbor = (gy > 0)       ? (gz * ny + (gy - 1)) * nx + gx : -1;
        d.top_neighbor    = (gy < ny - 1)  ? (gz * ny + (gy + 1)) * nx + gx : -1;
        d.back_neighbor   = (gz > 0)       ? ((gz - 1) * ny + gy) * nx + gx : -1;
        d.front_neighbor  = (gz < nz - 1)  ? ((gz + 1) * ny + gy) * nx + gx : -1;

        d.global_min = gmin;
        d.global_max = gmax;
        d.halo_width = halo_width;
        d.cell_size  = cell_size;

        // Owned region
        d.owned_min = {gmin.x + gx * dx, gmin.y + gy * dy, gmin.z + gz * dz};
        d.owned_max = {gmin.x + (gx + 1) * dx, gmin.y + (gy + 1) * dy,
                       gmin.z + (gz + 1) * dz};

        // Local region (owned + halo towards neighbors)
        float lx_lo = d.owned_min.x - (d.left_neighbor   >= 0 ? halo_width : 0);
        float lx_hi = d.owned_max.x + (d.right_neighbor  >= 0 ? halo_width : 0);
        float ly_lo = d.owned_min.y - (d.bottom_neighbor >= 0 ? halo_width : 0);
        float ly_hi = d.owned_max.y + (d.top_neighbor    >= 0 ? halo_width : 0);
        float lz_lo = d.owned_min.z - (d.back_neighbor   >= 0 ? halo_width : 0);
        float lz_hi = d.owned_max.z + (d.front_neighbor  >= 0 ? halo_width : 0);
        d.local_min = {std::max(lx_lo, gmin.x), std::max(ly_lo, gmin.y), std::max(lz_lo, gmin.z)};
        d.local_max = {std::min(lx_hi, gmax.x), std::min(ly_hi, gmax.y), std::min(lz_hi, gmax.z)};

        d.num_cells.x = static_cast<int>(std::ceil((d.local_max.x - d.local_min.x) / cell_size));
        d.num_cells.y = static_cast<int>(std::ceil((d.local_max.y - d.local_min.y) / cell_size));
#ifdef MD3D
        d.num_cells.z = static_cast<int>(std::ceil((d.local_max.z - d.local_min.z) / cell_size));
#else
        d.num_cells.z = 1;
#endif
        d.total_cells = d.num_cells.x * d.num_cells.y * d.num_cells.z;
      }
    }
  }

  return domains;
}

#endif // DOMAIN_H
