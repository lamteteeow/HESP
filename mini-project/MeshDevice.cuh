//
// Created by hans on 15.07.25.
//

#ifndef MESDEVICE_CUH
#define MESDEVICE_CUH
#include "Quaternion.cuh"
#include "Vec3.cuh"

struct MeshDevice
{
    size_t n;
    float* d_masses, *d_kn, *d_gamma_n, *d_gamma_t, *d_mu;
    Vec3 *d_forces, *d_velocities, *d_positions,
         *d_torques, *d_angularVelocities, *d_inertia, *d_inertia_inv, *d_vertices, *d_scales;
    Quaternion* d_orientations;
    int *d_cellHeads, *d_cellTails, *d_cellIndexes, *d_meshes_offset, *d_mesh_ids, *d_faces_offset;
    int3 *d_faces;
};

#endif //MESDEVICE_CUH
