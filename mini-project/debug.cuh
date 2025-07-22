//
// Created by hans on 18.07.25.
//

#ifndef DEBUG_CUH
#define DEBUG_CUH
#include <cstdio>

#include "Vec3.cuh"

__device__ inline void ausgeben(Vec3 v)
{
    printf("Vec3: %f/%f/%f\n",v.x,v.y,v.z);
}
__device__ inline void ausgeben(int3 v)
{
    printf("int3: %d/%d/%d\n",v.x,v.y,v.z);
}


#endif //DEBUG_CUH
