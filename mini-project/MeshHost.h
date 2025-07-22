#ifndef MESHHOST_H
#define MESHHOST_H

#include <vector>
#include <unordered_map>
#include <stdexcept>
#include "json.hpp"
#include "MeshDevice.cuh"
#include "MeshVTK.h"
#include "Quaternion.cuh"
#include "Vec3.cuh"

struct MeshHost
{
    // Anzahl Instanzen
    size_t n;

    // Per-Instanz-Arrays
    std::vector<int> id; // JSON-ID der Instanz
    std::vector<int> mesh_index; // interner Mesh-Index (0..M-1)
    std::vector<float> mass, kn, gamma_n, mu, gamma_t;
    std::vector<Vec3> scale;
    std::vector<Vec3> position;
    std::vector<Vec3> velocity;
    std::vector<Vec3> force;
    std::vector<Quaternion> orientation;
    std::vector<Vec3> angularVelocity;
    std::vector<Vec3> torque;
    std::vector<Vec3> inertia;
    std::vector<Vec3> inertia_inv;

    // Shared-Geometry
    std::vector<int> meshes_offset; // Start-Offset ins `vertices` pro Mesh (M+1 Einträge)
    std::vector<int> faces_offset; // Start-Offset ins `faces`    pro Mesh (M+1 Einträge)
    std::vector<Vec3> vertices; // alle Vertices aller Meshes concatenated
    std::vector<int3> faces; // alle Dreiecke aller Meshes concatenated

    // Für VTK-Rendering
    MeshVTK vtk;

    // Resize aller internen Arrays (Instanzen + M+1 Offsets)
    void resizeAll(const size_t instanceCount, const size_t meshCount)
    {
        n = instanceCount;
        // Instanzen
        id.resize(n);
        mesh_index.resize(n);
        mass.resize(n);
        kn.resize(n);
        gamma_n.resize(n);
        mu.resize(n);
        gamma_t.resize(n);
        scale.resize(n);
        position.resize(n);
        velocity.resize(n);
        force.resize(n);
        orientation.resize(n);
        angularVelocity.resize(n);
        torque.resize(n);
        inertia.resize(n);
        inertia_inv.resize(n);

        // Mesh-Offsets, M+1


        // VTK-Instanzdaten
        vtk.resizeAll(n, meshCount);
    }

    // JSON-Loader: komplexe Objekte und Mesh-Definitionen
    void load(const nlohmann::json& complex_obj,
              const nlohmann::json& meshes_json)
    {
        const size_t N = complex_obj.size();
        const size_t M = meshes_json.size();
        resizeAll(N, M);
        std::unordered_map<int, int> id2idx;
        id2idx.reserve(M);

        vertices.clear();
        faces.clear();


        // 1) Mesh-Daten parsen
        for (size_t mi = 0; mi < M; ++mi)
        {
            const auto& mdef = meshes_json[mi];
            int mid = mdef["id"].get<int>();
            id2idx[mid] = static_cast<int>(mi);

            // Vertex-Offset
            meshes_offset.push_back(static_cast<int>(vertices.size()));
            for (const auto& v : mdef["vertices"])
            {
                vertices.emplace_back(
                    v[0].get<float>(), v[1].get<float>(), v[2].get<float>()
                );
            }

            // Face-Offset
            faces_offset.push_back(static_cast<int>(faces.size()));
            for (const auto& f : mdef["faces"])
            {
                faces.emplace_back(int3{f[0].get<int>(), f[1].get<int>(), f[2].get<int>()});
            }
        }
        // End-Offset anhängen (M+1)
        meshes_offset.push_back(static_cast<int>(vertices.size()));
        faces_offset.push_back(static_cast<int>(faces.size()));

        // 2) Instanzen parsen

        for (size_t i = 0; i < N; ++i)
        {
            const auto& p = complex_obj[i];
            id[i] = p["id"].get<int>();
            int json_mid = p["mesh_id"].get<int>();
            auto it = id2idx.find(json_mid);
            if (it == id2idx.end())
                throw std::runtime_error("Unknown mesh_id " + std::to_string(json_mid));
            mesh_index[i] = it->second;
            kn[i]        = p["kn"].get<float>();
            gamma_n[i]   = p["gamma_n"].get<float>();
            mu[i]        = p["mu"].get<float>();
            gamma_t[i]   = p["gamma_t"].get<float>();
            mass[i] = p["mass"].get<float>();
            scale[i] = Vec3FromJson(p["scale"]);
            position[i] = Vec3FromJson(p["position"]);
            velocity[i] = Vec3FromJson(p["velocity"]);
            force[i] = Vec3FromJson(p["force"]);
            orientation[i] = QuaternionFromJson(p["orientation"]);
            angularVelocity[i] = Vec3FromJson(p["angular_velocity"]);
            torque[i] = Vec3FromJson(p["torque"]);
            inertia[i] = Vec3FromJson(p["inertia"]);
            inertia_inv[i] = Vec3FromJson(p["inertia_inv"]);
        }

        // 3) VTK-Buffer initialisieren
        size_t nv = vertices.size();
        size_t nf = faces.size();
        vtk.vertices_n = nv;
        vtk.faces_n = nf;
        vtk.vertices = new Vec3[nv];
        vtk.faces = new int3[nf];
        vtk.m = M;
        for (size_t mi = 0; mi < M; ++mi)
        {
            vtk.meshes_offset[mi] = meshes_offset[mi];
            vtk.faces_offset[mi] = faces_offset[mi];
        }
        for (size_t vi = 0; vi < nv; ++vi) vtk.vertices[vi] = vertices[vi];
        for (size_t fi = 0; fi < nf; ++fi) vtk.faces[fi] = faces[fi];
        for (size_t i = 0; i < n; ++i)
        {
            vtk.scales[i] = scale[i];
            vtk.mesh_index[i] = mesh_index[i];
        }
    }

    // Upload aller Daten auf die GPU
    void upload(MeshDevice& md) const
    {
        md.n = n;
        size_t iInt = n * sizeof(int);
        size_t iFloat = n * sizeof(float);
        size_t iVec3 = n * sizeof(Vec3);
        size_t iQuat = n * sizeof(Quaternion);

        // Instanz-Daten
        cudaMalloc(&md.d_masses, iFloat);
        cudaMalloc(&md.d_scales, iVec3);
        cudaMalloc(&md.d_positions, iVec3);
        cudaMalloc(&md.d_velocities, iVec3);
        cudaMalloc(&md.d_forces, iVec3);
        cudaMalloc(&md.d_orientations, iQuat);
        cudaMalloc(&md.d_angularVelocities, iVec3);
        cudaMalloc(&md.d_torques, iVec3);
        cudaMalloc(&md.d_inertia, iVec3);
        cudaMalloc(&md.d_inertia_inv, iVec3);
        cudaMalloc(&md.d_mesh_ids, iInt);
        cudaMalloc(&md.d_kn, iFloat);
        cudaMalloc(&md.d_gamma_n, iFloat);
        cudaMalloc(&md.d_gamma_t, iFloat);
        cudaMalloc(&md.d_mu, iFloat);
        cudaMalloc(&md.d_cellTails, iInt);
        cudaMalloc(&md.d_cellIndexes, iInt);

        cudaMemcpy(md.d_masses, mass.data(), iFloat, cudaMemcpyHostToDevice);
        cudaMemcpy(md.d_scales, scale.data(), iVec3, cudaMemcpyHostToDevice);
        cudaMemcpy(md.d_positions, position.data(), iVec3, cudaMemcpyHostToDevice);
        cudaMemcpy(md.d_velocities, velocity.data(), iVec3, cudaMemcpyHostToDevice);
        cudaMemcpy(md.d_forces, force.data(), iVec3, cudaMemcpyHostToDevice);
        cudaMemcpy(md.d_orientations, orientation.data(), iQuat, cudaMemcpyHostToDevice);
        cudaMemcpy(md.d_angularVelocities, angularVelocity.data(), iVec3, cudaMemcpyHostToDevice);
        cudaMemcpy(md.d_torques, torque.data(), iVec3, cudaMemcpyHostToDevice);
        cudaMemcpy(md.d_inertia, inertia.data(), iVec3, cudaMemcpyHostToDevice);
        cudaMemcpy(md.d_inertia_inv, inertia_inv.data(), iVec3, cudaMemcpyHostToDevice);
        cudaMemcpy(md.d_mesh_ids, mesh_index.data(), iInt, cudaMemcpyHostToDevice);
        cudaMemcpy(md.d_kn, kn.data(), iFloat, cudaMemcpyHostToDevice);
        cudaMemcpy(md.d_gamma_n, gamma_n.data(), iFloat, cudaMemcpyHostToDevice);
        cudaMemcpy(md.d_gamma_t, gamma_t.data(), iFloat, cudaMemcpyHostToDevice);
        cudaMemcpy(md.d_mu, mu.data(), iFloat, cudaMemcpyHostToDevice);


        // Shared-Geometrie: vertices, faces, offsets
        size_t vertCount = vertices.size();
        size_t faceCount = faces.size();
        size_t offCount = meshes_offset.size(); // M+1

        cudaMalloc(&md.d_vertices, vertCount * sizeof(Vec3));
        cudaMalloc(&md.d_faces, faceCount * sizeof(int3));
        cudaMalloc(&md.d_meshes_offset, offCount * sizeof(int));
        cudaMalloc(&md.d_faces_offset, offCount * sizeof(int));

        cudaMemcpy(md.d_vertices, vertices.data(), vertCount * sizeof(Vec3), cudaMemcpyHostToDevice);
        cudaMemcpy(md.d_faces, faces.data(), faceCount * sizeof(int3), cudaMemcpyHostToDevice);
        cudaMemcpy(md.d_meshes_offset, meshes_offset.data(), offCount * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(md.d_faces_offset, faces_offset.data(), offCount * sizeof(int), cudaMemcpyHostToDevice);
    }

    // Download aktueller Zustände für VTK
    MeshVTK& download(const MeshDevice& md)
    {
        size_t iVec3 = n * sizeof(Vec3);
        size_t iQuat = n * sizeof(Quaternion);
        cudaMemcpy(vtk.positions, md.d_positions, iVec3, cudaMemcpyDeviceToHost);
        cudaMemcpy(vtk.velocities, md.d_velocities, iVec3, cudaMemcpyDeviceToHost);
        cudaMemcpy(vtk.forces, md.d_forces, iVec3, cudaMemcpyDeviceToHost);
        cudaMemcpy(vtk.orientations, md.d_orientations, iQuat, cudaMemcpyDeviceToHost);
        return vtk;
    }
};

#endif // MESHHOST_H
