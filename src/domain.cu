#include "domain.h"
#include <algorithm>
#include <cmath>
#include <limits>

// Factor n into nx×ny×nz, preferring near-cube decompositions.
void factorGrid3D(int n, int &nx, int &ny, int &nz) {
  nx = 1; ny = 1; nz = n;
  float best = std::numeric_limits<float>::max();
  for (int i = 1; i <= n; ++i) {
    if (n % i != 0) continue;
    int rem = n / i;
    for (int j = static_cast<int>(std::sqrt(rem)); j >= 1; --j) {
      if (rem % j == 0) {
        int k = rem / j;
        float v = std::log(static_cast<float>(i))
                + std::log(static_cast<float>(j))
                + std::log(static_cast<float>(k));
        float mean = v / 3.0f;
        float score = (std::log(static_cast<float>(i)) - mean) * (std::log(static_cast<float>(i)) - mean)
                    + (std::log(static_cast<float>(j)) - mean) * (std::log(static_cast<float>(j)) - mean)
                    + (std::log(static_cast<float>(k)) - mean) * (std::log(static_cast<float>(k)) - mean);
        if (score < best) { best = score; nx = i; ny = j; nz = k; }
        break;
      }
    }
  }
}

// Construct domains split into nx×ny×nz 3D grid.
std::vector<Domain> buildDomains(Vec3 gmin, Vec3 gmax, float halo_width,
                                  float cell_size, int num_gpus) {
  int nx, ny, nz;
  factorGrid3D(num_gpus, nx, ny, nz);

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

        d.owned_min = {gmin.x + gx * dx, gmin.y + gy * dy, gmin.z + gz * dz};
        d.owned_max = {gmin.x + (gx + 1) * dx, gmin.y + (gy + 1) * dy,
                       gmin.z + (gz + 1) * dz};

        float lx_lo = d.owned_min.x - (d.left_neighbor   >= 0 ? halo_width : 0);
        float lx_hi = d.owned_max.x + (d.right_neighbor  >= 0 ? halo_width : 0);
        float ly_lo = d.owned_min.y - (d.bottom_neighbor >= 0 ? halo_width : 0);
        float ly_hi = d.owned_max.y + (d.top_neighbor    >= 0 ? halo_width : 0);
        float lz_lo = d.owned_min.z - (d.back_neighbor   >= 0 ? halo_width : 0);
        float lz_hi = d.owned_max.z + (d.front_neighbor  >= 0 ? halo_width : 0);
        d.local_min = {std::max(lx_lo, gmin.x), std::max(ly_lo, gmin.y), std::max(lz_lo, gmin.z)};
        d.local_max = {std::min(lx_hi, gmax.x), std::min(ly_hi, gmax.y), std::min(lz_hi, gmax.z)};

        d.num_cells.x = std::max(1, static_cast<int>(std::ceil((d.local_max.x - d.local_min.x) / cell_size)));
        d.num_cells.y = std::max(1, static_cast<int>(std::ceil((d.local_max.y - d.local_min.y) / cell_size)));
        d.num_cells.z = std::max(1, static_cast<int>(std::ceil((d.local_max.z - d.local_min.z) / cell_size)));
        d.total_cells = d.num_cells.x * d.num_cells.y * d.num_cells.z;
      }
    }
  }

  return domains;
}
