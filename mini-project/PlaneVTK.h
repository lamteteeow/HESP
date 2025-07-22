//
// Created by hans on 15.07.25.
//

#ifndef PLANEVTK_H
#define PLANEVTK_H
#include "Vec3.cuh"

struct PlaneVTK
{
    size_t n;
    Vec3 *normals;
    float *distances;

    void resizeAll(size_t h_n)
    {
        n = h_n;
        normals = new Vec3[n];
        distances = new float[n];
    }
};

#endif //PLANEVTK_H
