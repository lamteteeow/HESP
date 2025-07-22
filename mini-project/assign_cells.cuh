//
// Created by hans on 22.06.25.
//

#ifndef NEIGHBORHOOD_CUH
#define NEIGHBORHOOD_CUH
#include <cstdio>
#include <cuda_runtime.h>

#include "Cells.cuh"
#include "Vec3.cuh"

__device__ inline int computeCellIndex(const int3 numOfCellsPerAxis,
                                       const Vec3 OFFSET,
                                       const float cellL,
                                       const Vec3& pos_i)
{
    int x = static_cast<int>((pos_i.x - OFFSET.x) / cellL);
    x = min(x, numOfCellsPerAxis.x - 1);
    x = max(x, 0);
    int y = static_cast<int>((pos_i.y - OFFSET.y) / cellL);
    y = min(y, numOfCellsPerAxis.y - 1);
    y = max(y, 0);
    int z = static_cast<int>((pos_i.z - OFFSET.z) / cellL);
    z = min(z, numOfCellsPerAxis.z - 1);
    z = max(z, 0);
    return x + y * numOfCellsPerAxis.x + z * numOfCellsPerAxis.x * numOfCellsPerAxis.y;
}

__global__ inline void assignCell(const size_t NUM_PARTICLES,
                                  const Vec3* positions,
                                  const int3 numOfCellsPerAxis,
                                  const Vec3 OFFSET,
                                  const float cellL,
                                  int *d_cellHeads,
                                  int *d_cellTails,
                                  int *d_cellIndexes)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NUM_PARTICLES) return;
    // Calculate cell index from position
    const int cellIdx = computeCellIndex(numOfCellsPerAxis, OFFSET, cellL, positions[i]);

    // set the index of the cell for particle i
    d_cellIndexes[i] = cellIdx;
    // prepend this particle's index i to the list of particles of the cell with index cellIdx
    d_cellTails[i] = atomicExch(&d_cellHeads[cellIdx], i);

}


#endif //NEIGHBORHOOD_CUH
