#include <cuda_runtime.h>
#include <iostream>

#include "VTK.h"
#include "Vec3.cuh"
#include "assign_cells.cuh"
#include "debug.cuh"
#include "init_neighborhood.h"
#include "input_file.h"
#include "read_input.h"
#include "symplecticEuler.cuh"

int main(const int argc, char **argv) {
  printf("Running...\n");
  long maxSteps = 100000;
  if (argc < 2) {
    std::cerr << "No filename provided." << std::endl;
    return 1;
  }
  std::string filename = argv[1];
  char *endptr = nullptr;
  if (argc > 2)
    maxSteps = strtol(argv[2], &endptr, 10);
  if (endptr == argv[2])
    fprintf(stderr, "Fehler: '%s' ist keine gültige Zahl.\n", argv[2]);

  SceneHost host;
  host.loadFromJson(filename);
  SphereDevice sd;
  host.spheres.upload(sd);
  PlaneDevice pd;
  host.planes.upload(pd);
  MeshDevice md;
  host.complex_objects.upload(md);

  const Vec3 MAX_POS = {host.length.x + host.offset.x,
                        host.length.y + host.offset.y,
                        host.length.z + host.offset.z};
  const Vec3 MIN_POS = {host.offset.x, host.offset.y, host.offset.z};

  // Cells
  const float cellL = 4.0f; // sum_of_largest_two(host.spheres.radius);
  const int3 NUM_CELLS = toInt3ceil(host.length / cellL);
  const int TOTAL_NUM_CELLS = NUM_CELLS.x * NUM_CELLS.y * NUM_CELLS.z;
  std::vector<int> h_neighborhood(TOTAL_NUM_CELLS * 27, -1);
  initCellNeighborhood(NUM_CELLS, h_neighborhood);

  const auto sizeForNumCellsINT = TOTAL_NUM_CELLS * sizeof(int);
  int *d_neighborsOfCell;
  cudaMalloc(&d_neighborsOfCell, sizeForNumCellsINT * 27);
  cudaMemcpy(d_neighborsOfCell, h_neighborhood.data(), sizeForNumCellsINT * 27,
             cudaMemcpyHostToDevice);
  cudaMalloc(&md.d_cellHeads, sizeForNumCellsINT);
  cudaMalloc(&sd.d_cellHeads, sizeForNumCellsINT);

  printf("Number of cells: %d\n", TOTAL_NUM_CELLS);
  printf("Number of cells per axis: %d/%d/%d\n", NUM_CELLS.x, NUM_CELLS.y,
         NUM_CELLS.z);
  printf("cellL: %f\n", cellL);

  printf("set block, grid...\n");
  constexpr dim3 block(256);
  dim3 gridSpheres((host.spheres.n + block.x - 1) / block.x);
  dim3 gridComplexObj((host.complex_objects.n + block.x - 1) / block.x);

  int stepCount = 0;
  int framecount = 0;
  constexpr int stepsPerFrame = 1;
  int stepsToNextFrame = stepsPerFrame;
  printf("L: %f/%f/%f\n", host.length.x, host.length.y, host.length.z);
  printf("OFFSET: %f/%f/%f\n", host.offset.x, host.offset.y, host.offset.z);
  printf("Number of particles: %lu\n", host.spheres.n);
  printf("Min pos: %f/%f/%f\n", MIN_POS.x, MIN_POS.y, MIN_POS.z);
  printf("Max pos: %f/%f/%f\n", MAX_POS.x, MAX_POS.y, MAX_POS.z);

  writeBoundingBoxWallsVTK(filename, maxSteps, "bounding_box.vtk", MIN_POS,
                           MAX_POS);
  while (stepCount < maxSteps) {
    cudaMemset(sd.d_cellHeads, -1, sizeForNumCellsINT);
    cudaMemset(md.d_cellHeads, -1, sizeForNumCellsINT);
    cudaError_t err;
    if (sd.n > 0) {
      assignCell<<<gridSpheres, block>>>(sd.n, sd.d_positions, NUM_CELLS,
                                         host.offset, cellL, sd.d_cellHeads,
                                         sd.d_cellTails, sd.d_cellIndexes);
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        fprintf(stderr, "Assign spheres error: %s\n", cudaGetErrorString(err));
        exit(1);
      }
    }
    if (md.n > 0) {
      assignCell<<<gridComplexObj, block>>>(md.n, md.d_positions, NUM_CELLS,
                                            host.offset, cellL, md.d_cellHeads,
                                            md.d_cellTails, md.d_cellIndexes);
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        fprintf(stderr, "Assign boxes error: %s\n", cudaGetErrorString(err));
        exit(1);
      }
    }
    if (sd.n > 0) {
      cfSphereOnSphere<<<gridSpheres, block>>>(host.gravity, sd,
                                               d_neighborsOfCell);
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        fprintf(stderr, "SphereOnSphere Force error: %s\n",
                cudaGetErrorString(err));
        exit(1);
      }
      cfSphereOnPlane<<<gridSpheres, block>>>(sd, pd);
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        fprintf(stderr, "SphereOnPlane Force error: %s\n",
                cudaGetErrorString(err));
        exit(1);
      }
    }
    if (md.n > 0) {
      cfMeshOnMesh<<<gridComplexObj, block>>>(host.gravity, md,
                                              d_neighborsOfCell);
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        fprintf(stderr, "MeshOnMesh error: %s\n", cudaGetErrorString(err));
        exit(1);
      }
      cudaDeviceSynchronize();

      cfMeshOnPlane<<<gridComplexObj, block>>>(md, pd);
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        fprintf(stderr, "MeshOnPlane error: %s\n", cudaGetErrorString(err));
        exit(1);
      }
    }
    if (sd.n > 0) {
      IntegrateVelAndPos<<<gridSpheres, block>>>(host.dt, sd);
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        fprintf(stderr, "Integration Sphere error: %s\n",
                cudaGetErrorString(err));
        exit(1);
      }
    }
    if (md.n > 0) {
      IntegrateVelAndPosMesh<<<gridComplexObj, block>>>(host.dt, md);
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        fprintf(stderr, "Integration Mesh error: %s\n",
                cudaGetErrorString(err));
        exit(1);
      }
    }
    cudaDeviceSynchronize();

    stepCount++;
    stepsToNextFrame--;
    if (stepsToNextFrame > 0)
      continue;

    auto sphereVTK = host.spheres.download(sd);
    auto meshVTK = host.complex_objects.download(md);

    Vec3 total = {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < host.complex_objects.n; ++i)
      total += meshVTK.forces[i];

    writeVTK(framecount, sphereVTK, filename, maxSteps);
    writeMeshVTK(framecount, meshVTK, filename, maxSteps);

    stepsToNextFrame = stepsPerFrame;
    framecount++;
  }

  return 0;
}
