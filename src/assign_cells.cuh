#ifndef ASSIGN_CELLS_CUH
#define ASSIGN_CELLS_CUH
#include "vec3.cuh"
#include <cuda_runtime.h>

__device__ int computeCellIndex(const int3 num_cells_per_axis,
                                const Vec3 offset, const float cell_size,
                                const Vec3 &pos);

__global__ void assignCell(const size_t num_particles,
                           const Vec3 *positions,
                           const int3 num_cells_per_axis,
                           const Vec3 offset, const float cell_size,
                           int *d_cellHeads, int *d_cellTails,
                           int *d_cellIndexes);

#endif // ASSIGN_CELLS_CUH
