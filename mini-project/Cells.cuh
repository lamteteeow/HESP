//
// Created by hans on 13.07.25.
//

#ifndef CELLS_CUH
#define CELLS_CUH


struct Cells
{
    int *d_cellHeads, *d_cellTails, *d_cellIndexes;
    void initCells(const size_t NUM_OF_BODIES, const int NUM_OF_CELLS)
    {
        const auto sizeForNumCellsINT = NUM_OF_CELLS * sizeof(int);
        cudaMalloc(&d_cellHeads, sizeForNumCellsINT);
        const auto sizeForNumBodiesINT = NUM_OF_BODIES * sizeof(int);
        cudaMalloc(&d_cellTails, sizeForNumBodiesINT);
        cudaMalloc(&d_cellIndexes, sizeForNumBodiesINT);
    }
};
#endif //CELLS_CUH
