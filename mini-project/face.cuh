//
// Created by hans on 19.07.25.
//

#ifndef FACE_CUH
#define FACE_CUH
#include "Vec3.cuh"

struct Face {
    Vec3   verts[3];    // counter-clockwise winding as seen from outside
    Vec3   normal;      // unit outward normal
    float  dist;        // = dot(normal, verts[0])
    bool   obsolete;    // mark for deletion when it “sees” the new point
};

#endif //FACE_CUH
