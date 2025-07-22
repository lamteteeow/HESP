//
// Created by hans on 15.07.25.
//

#ifndef PLANEDEVICE_CUH
#define PLANEDEVICE_CUH
#include "Vec3.cuh"

struct PlaneDevice
{
    size_t n;
    Vec3 *d_normals;
    float *d_distances;
};

#endif //PLANEDEVICE_CUH
