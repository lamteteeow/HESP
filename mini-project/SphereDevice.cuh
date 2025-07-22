//
// Created by hans on 13.07.25.
//

#ifndef SPHEREDEVICE_CUH
#define SPHEREDEVICE_CUH
#include "Quaternion.cuh"
#include "Vec3.cuh"

struct SphereDevice
{
    size_t n;
    Vec3 *d_forces, *d_velocities, *d_positions, *d_torques, *d_angularVelocities;
    Quaternion *d_orientations;
    float *d_masses, *d_inertia, *d_radii, *d_kn, *d_gamma_n, *d_gamma_t, *d_mu;
    int *d_cellHeads, *d_cellTails, *d_cellIndexes;
};

#endif //SPHEREDEVICE_CUH
