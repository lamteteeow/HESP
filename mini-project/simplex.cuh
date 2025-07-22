//
// Created by hans on 18.07.25.
//

#ifndef SIMPLEX_CUH
#define SIMPLEX_CUH
#include "Vec3.cuh"

struct Simplex
{
    Vec3 pts[4];
    int   size;

    __device__ __host__ __forceinline__
    void init() {
        size = 0;
    }

    __device__ __host__ __forceinline__
    void push(const Vec3& p) {
        if (size < 4) ++size;
        for (int i = size - 1; i > 0; --i)
            pts[i] = pts[i - 1];
        pts[0] = p;
    }
    __device__ __host__ __forceinline__
    Vec3& operator[](const int i) { return pts[i]; }
};
#endif //SIMPLEX_CUH
