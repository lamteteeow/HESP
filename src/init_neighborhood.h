#ifndef INIT_NEIGHBORHOOD_H
#define INIT_NEIGHBORHOOD_H
#include "vec3.cuh"
#include <vector>

bool axisIndicesOutOfBoundary(const int3 num_cells_per_axis, const int3 n);

void clampAxisIndices(const int3 num_cells_per_axis, int3 &n);

int getCellIndexForPeriodicBoundary(const int3 num_cells_per_axis, int3 &n);

int getCellIndexForFixedBoundary(const int3 num_cells_per_axis, const int3 &n);

void initCellNeighborhood(const int3 num_cells_per_axis,
                          std::vector<int> &neighborhood);

float sumOfLargestTwo(const std::vector<float> &vec);

#endif // INIT_NEIGHBORHOOD_H
