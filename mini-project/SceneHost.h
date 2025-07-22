//
// Created by hans on 13.07.25.
//

#ifndef SCENEHOST_H
#define SCENEHOST_H
#include "MeshHost.h"
#include "PlaneHost.h"
#include "SphereHost.h"
#include "Vec3.cuh"

struct SceneHost
{
    float dt;
    Vec3 gravity;
    Vec3 length;
    Vec3 offset;

    PlaneHost planes;
    SphereHost spheres;
    MeshHost complex_objects;

    void loadFromJson(const std::string& filename)
    {
        const auto j = readJson(filename);
        load(j);
    }

    static nlohmann::json readJson(const std::string& filename)
    {
        std::ifstream in{filename};
        if (!in) throw std::runtime_error("Cannot open " + filename);
        nlohmann::json j;
        in >> j;
        return j;
    }

    void load(const nlohmann::json& j)
    {
        dt = j["dt"].get<float>();
        gravity = Vec3FromJson(j["gravity"]);
        length = Vec3FromJson(j["length"]);
        offset = Vec3FromJson(j["offset"]);
        planes.load(j["planes"]);
        spheres.load(j["spheres"]);
        complex_objects.load(j["complex_objects"], j["meshes"]);
    }
};
#endif //SCENEHOST_H
