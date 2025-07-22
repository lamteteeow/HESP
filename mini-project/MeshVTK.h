//
// Created by hans on 15.07.25.
//

#ifndef BOXVTK_H
#define BOXVTK_H
#include "Quaternion.cuh"
#include "Vec3.cuh"

struct MeshVTK
{
    size_t n, m, vertices_n, faces_n;
    int *mesh_index, *meshes_offset, *faces_offset;
    int3* faces;
    Vec3 *positions, *velocities, *forces, *vertices, *scales;
    Quaternion* orientations;


    void resizeAll(const size_t h_n, const size_t h_m)
    {
        n = h_n;
        m = h_m;
        scales = new Vec3[n];
        positions = new Vec3[n];
        velocities = new Vec3[n];
        forces = new Vec3[n];
        orientations = new Quaternion[n];
        mesh_index = new int[n];
        meshes_offset = new int[m];
        faces_offset = new int[m];
    }
};

#endif //BOXVTK_H
