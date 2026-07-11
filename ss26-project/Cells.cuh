//
// Created by hans on 13.07.25.
//

#ifndef CELLS_CUH
#define CELLS_CUH

struct Cells {
  int *d_cellHeads, *d_cellTails, *d_cellIndexes;
  void initCells(const size_t num_bodies, const int num_cells) {
    const auto size_for_num_cells_int = num_cells * sizeof(int);
    cudaMalloc(&d_cellHeads, size_for_num_cells_int);
    const auto size_for_num_bodies_int = num_bodies * sizeof(int);
    cudaMalloc(&d_cellTails, size_for_num_bodies_int);
    cudaMalloc(&d_cellIndexes, size_for_num_bodies_int);
  }
};
#endif // CELLS_CUH
