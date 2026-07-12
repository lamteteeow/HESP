//
// Created by hans on 22.06.25.
//

#ifndef INIT_NEIGHBORHOOD_H
#define INIT_NEIGHBORHOOD_H
#include <vector>

inline bool axisIndicesOutOfBoundary(const int3 num_cells_per_axis,
                                     const int3 n) {
  return n.x < 0 || n.y < 0 || n.z < 0 || n.x >= num_cells_per_axis.x ||
         n.y >= num_cells_per_axis.y || n.z >= num_cells_per_axis.z;
}

inline void clampAxisIndices(const int3 num_cells_per_axis, int3 &n) {
  if (n.x < 0)
    n.x += num_cells_per_axis.x;
  else if (n.x >= num_cells_per_axis.x)
    n.x -= num_cells_per_axis.x;
  if (n.y < 0)
    n.y += num_cells_per_axis.y;
  else if (n.y >= num_cells_per_axis.y)
    n.y -= num_cells_per_axis.y;
  if (n.z < 0)
    n.z += num_cells_per_axis.z;
  else if (n.z >= num_cells_per_axis.z)
    n.z -= num_cells_per_axis.z;
}

inline int getCellIndexForPeriodicBoundary(const int3 num_cells_per_axis,
                                           int3 &n) {
  clampAxisIndices(num_cells_per_axis, n);
  return n.x + n.y * num_cells_per_axis.x +
         n.z * num_cells_per_axis.x * num_cells_per_axis.y;
}

inline int getCellIndexForFixedBoundary(const int3 num_cells_per_axis,
                                        const int3 &n) {
  if (axisIndicesOutOfBoundary(num_cells_per_axis, n))
    return -1;
  return n.x + n.y * num_cells_per_axis.x +
         n.z * num_cells_per_axis.x * num_cells_per_axis.y;
}

inline void initCellNeighborhood(const int3 num_cells_per_axis,
                                 std::vector<int> &neighborhood) {
  for (int iz = 0; iz < num_cells_per_axis.z; ++iz) {
    for (int iy = 0; iy < num_cells_per_axis.y; ++iy) {
      for (int ix = 0; ix < num_cells_per_axis.x; ++ix) {
        const int c = ix + iy * num_cells_per_axis.x +
                      iz * num_cells_per_axis.x * num_cells_per_axis.y;
        const int base = c * 27;
        int idx = 0;

        for (int dz = -1; dz <= 1; ++dz) {
          for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
              int3 n = {ix + dx, iy + dy, iz + dz};

              const int neighbor =
                  getCellIndexForFixedBoundary(num_cells_per_axis, n);
              neighborhood[base + idx] = neighbor;
              ++idx;
            }
          }
        }
      }
    }
  }
}

inline float sumOfLargestTwo(const std::vector<float> &vec) {
  if (vec.size() == 0)
    return 0.0;
  if (vec.size() == 1)
    return vec[0] * 2;

  float max1 = std::numeric_limits<float>::lowest();
  float max2 = std::numeric_limits<float>::lowest();

  for (const float v : vec) {
    if (v > max1) {
      max2 = max1;
      max1 = v;
    } else if (v > max2) {
      max2 = v;
    }
  }

  return max1 + max2;
}

#endif // INIT_NEIGHBORHOOD_H
