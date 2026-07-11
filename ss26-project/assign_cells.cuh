//
// Created by hans on 22.06.25.
//

#ifndef ASSIGN_CELLS_CUH
#define ASSIGN_CELLS_CUH
#include <cuda_runtime.h>

#include "vec3.cuh"

__device__ inline int computeCellIndex(const int3 num_cells_per_axis,
                                       const Vec3 offset, const float cell_size,
                                       const Vec3 &pos) {
  int x = static_cast<int>((pos.x - offset.x) / cell_size);
  x = min(x, num_cells_per_axis.x - 1);
  x = max(x, 0);
  int y = static_cast<int>((pos.y - offset.y) / cell_size);
  y = min(y, num_cells_per_axis.y - 1);
  y = max(y, 0);
  int z = static_cast<int>((pos.z - offset.z) / cell_size);
  z = min(z, num_cells_per_axis.z - 1);
  z = max(z, 0);
  return x + y * num_cells_per_axis.x +
         z * num_cells_per_axis.x * num_cells_per_axis.y;
}

__global__ inline void assignCell(const size_t num_particles,
                                  const Vec3 *positions,
                                  const int3 num_cells_per_axis,
                                  const Vec3 offset, const float cell_size,
                                  int *d_cellHeads, int *d_cellTails,
                                  int *d_cellIndexes) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_particles)
    return;
  // Calculate cell index from position
  const int cell_idx =
      computeCellIndex(num_cells_per_axis, offset, cell_size, positions[i]);

  // set the index of the cell for particle i
  d_cellIndexes[i] = cell_idx;
  // prepend this particle's index i to the list of particles of the cell with
  // index cell_idx
  d_cellTails[i] = atomicExch(&d_cellHeads[cell_idx], i);
}

#endif // ASSIGN_CELLS_CUH
