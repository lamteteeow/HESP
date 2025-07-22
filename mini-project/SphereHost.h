//
// Created by hans on 08.07.25.
//

#ifndef SPHEREHOST_H
#define SPHEREHOST_H
#include <vector>
#include "SphereDevice.cuh"
#include "Quaternion.cuh"
#include "SphereVTK.h"
#include "Vec3.cuh"

struct SphereHost
{
    unsigned long sizeOfVec3NUM, sizeOfIntNUM;
    unsigned long sizeOfFloatNUM;
    unsigned long sizeOfQuaternionNUM;
    size_t n; // number of particles
    std::vector<int> id;
    std::vector<float> mass, radius, inertia, kn, gamma_n, mu, gamma_t;
    std::vector<Vec3> position, velocity, force, angularVelocity, torque;
    std::vector<Quaternion> orientation;
    SphereVTK vtk;

    void load(const nlohmann::basic_json<>& json)
    {
        resizeAll(json.size());
        if (n == 0)
            return;
        read_json(json);
    }

    void resizeAll(const size_t n)
    {
        this->n = n;
        id.resize(n);
        mass.resize(n);
        radius.resize(n);
        inertia.resize(n);
        position.resize(n);
        velocity.resize(n);
        force.resize(n);
        orientation.resize(n);
        angularVelocity.resize(n);
        torque.resize(n);
        kn.resize(n);
        gamma_n.resize(n);
        mu.resize(n);
        gamma_t.resize(n);

        vtk.resizeAll(n);
    }

    void read_json(const nlohmann::basic_json<>& json)
    {
        for (size_t i = 0; i < n; ++i) {
            auto& p = json[i];
            id[i]       = p["id"].get<int>();
            mass[i]     = p["mass"].get<float>();
            radius[i]   = p["radius"].get<float>();
            inertia[i]  = p["inertia"].get<float>();

            position[i]        = Vec3FromJson(p["position"]);
            velocity[i]       = Vec3FromJson(p["velocity"]);
            force[i]           = Vec3FromJson(p["force"]);
            orientation[i]     = QuaternionFromJson(p["orientation"]);
            angularVelocity[i]= Vec3FromJson(p["angularVelocity"]);
            torque[i]          = Vec3FromJson(p["torque"]);

            kn[i]        = p["kn"].get<float>();
            gamma_n[i]   = p["gamma_n"].get<float>();
            mu[i]        = p["mu"].get<float>();
            gamma_t[i]   = p["gamma_t"].get<float>();
        }

        vtk.radii = radius;
    }

    void upload(SphereDevice& sd)
    {
        sd.n = n;

        sizeOfIntNUM = n * sizeof(int);
        sizeOfVec3NUM = n * sizeof(Vec3);
        sizeOfFloatNUM = n * sizeof(float);
        sizeOfQuaternionNUM = n * sizeof(Quaternion);

        cudaMalloc(&sd.d_forces, sizeOfVec3NUM);
        cudaMalloc(&sd.d_velocities, sizeOfVec3NUM);
        cudaMalloc(&sd.d_positions, sizeOfVec3NUM);
        cudaMalloc(&sd.d_masses, sizeOfFloatNUM);
        cudaMalloc(&sd.d_radii, sizeOfFloatNUM);
        cudaMalloc(&sd.d_torques, sizeOfVec3NUM);
        cudaMalloc(&sd.d_angularVelocities, sizeOfVec3NUM);
        cudaMalloc(&sd.d_orientations, sizeOfQuaternionNUM);
        cudaMalloc(&sd.d_inertia, sizeOfFloatNUM);
        cudaMalloc(&sd.d_kn, sizeOfFloatNUM);
        cudaMalloc(&sd.d_gamma_n, sizeOfFloatNUM);
        cudaMalloc(&sd.d_gamma_t, sizeOfFloatNUM);
        cudaMalloc(&sd.d_mu, sizeOfFloatNUM);
        cudaMalloc(&sd.d_cellTails, sizeOfIntNUM);
        cudaMalloc(&sd.d_cellIndexes, sizeOfIntNUM);

        cudaMemcpy(sd.d_forces, force.data(), sizeOfVec3NUM, cudaMemcpyHostToDevice);
        cudaMemcpy(sd.d_velocities, velocity.data(), sizeOfVec3NUM, cudaMemcpyHostToDevice);
        cudaMemcpy(sd.d_positions, position.data(), sizeOfVec3NUM, cudaMemcpyHostToDevice);
        cudaMemcpy(sd.d_masses, mass.data(), sizeOfFloatNUM, cudaMemcpyHostToDevice);
        cudaMemcpy(sd.d_radii, radius.data(), sizeOfFloatNUM, cudaMemcpyHostToDevice);
        cudaMemcpy(sd.d_torques, torque.data(), sizeOfVec3NUM, cudaMemcpyHostToDevice);
        cudaMemcpy(sd.d_angularVelocities, angularVelocity.data(),sizeOfVec3NUM, cudaMemcpyHostToDevice);
        cudaMemcpy(sd.d_orientations, orientation.data(), sizeOfQuaternionNUM, cudaMemcpyHostToDevice);
        cudaMemcpy(sd.d_inertia, inertia.data(), sizeOfFloatNUM, cudaMemcpyHostToDevice);
        cudaMemcpy(sd.d_kn, kn.data(), sizeOfFloatNUM, cudaMemcpyHostToDevice);
        cudaMemcpy(sd.d_gamma_n, gamma_n.data(), sizeOfFloatNUM, cudaMemcpyHostToDevice);
        cudaMemcpy(sd.d_gamma_t, gamma_t.data(), sizeOfFloatNUM, cudaMemcpyHostToDevice);
        cudaMemcpy(sd.d_mu, mu.data(), sizeOfFloatNUM, cudaMemcpyHostToDevice);
    }

    SphereVTK& download(const SphereDevice& sd)
    {
        cudaMemcpy(vtk.positions, sd.d_positions, sizeOfVec3NUM, cudaMemcpyDeviceToHost);
        cudaMemcpy(vtk.velocities, sd.d_velocities, sizeOfVec3NUM, cudaMemcpyDeviceToHost);
        cudaMemcpy(vtk.forces, sd.d_forces, sizeOfVec3NUM, cudaMemcpyDeviceToHost);
        cudaMemcpy(vtk.orientations, sd.d_orientations, sizeOfQuaternionNUM, cudaMemcpyDeviceToHost);

        return vtk;
    }
};
#endif //SPHEREHOST_H
