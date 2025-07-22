//
// Created by hans on 15.07.25.
//

#ifndef PLANEHOST_H
#define PLANEHOST_H
#include <vector>

#include "PlaneDevice.cuh"
#include "PlaneVTK.h"
#include "Vec3.cuh"

struct PlaneHost
{
    unsigned long sizeForNumSpheresVec3;
    unsigned long sizeForNumSpheresFLOAT;
    size_t n;
    std::vector<int> id;
    std::vector<Vec3> normal;
    std::vector<float> distance;
    PlaneVTK vtk;

    void load(const nlohmann::basic_json<>& json)
    {
        resizeAll(json.size());
        read_json(json);
    }

    void resizeAll(size_t n)
    {
        this->n = n;
        id.resize(n);
        normal.resize(n);
        distance.resize(n);

        vtk.resizeAll(n);
    }

    void read_json(const nlohmann::basic_json<>& json)
    {
        for (size_t i = 0; i < n; i++)
        {
            auto p = json[i];
            id[i] = p["id"].get<int>();
            normal[i] = Vec3FromJson(p["normal"]);
            distance[i] = p["distance"].get<float>();
        }
    }

    void upload(PlaneDevice& pd)
    {
        pd.n = n;
        sizeForNumSpheresVec3 = n * sizeof(Vec3);
        sizeForNumSpheresFLOAT = n * sizeof(float);

        cudaMalloc(&pd.d_normals, sizeForNumSpheresVec3);
        cudaMalloc(&pd.d_distances, sizeForNumSpheresFLOAT);

        cudaMemcpy(pd.d_normals, normal.data(), sizeForNumSpheresVec3, cudaMemcpyHostToDevice);
        cudaMemcpy(pd.d_distances, distance.data(), sizeForNumSpheresFLOAT, cudaMemcpyHostToDevice);
    }

    PlaneVTK& download(const PlaneDevice& pd)
    {
        cudaMemcpy(vtk.normals, pd.d_normals, sizeForNumSpheresVec3, cudaMemcpyDeviceToHost);
        cudaMemcpy(vtk.distances, pd.d_distances, sizeForNumSpheresFLOAT, cudaMemcpyDeviceToHost);

        return vtk;
    }
};

#endif //PLANEHOST_H
