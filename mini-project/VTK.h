#ifndef VTK_H
#define VTK_H

#include "MeshVTK.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

inline void writeVTK(int frame, const SphereVTK &vtk,
                     const std::string &postfix, long steps) {
  const std::string parent_prefix = "../";
  std::string cleaned_postfix = postfix;
  while (cleaned_postfix.rfind(parent_prefix, 0) != std::string::npos)
    cleaned_postfix.erase(0, parent_prefix.size());
  const auto pos = cleaned_postfix.find_last_of('.');
  const std::string scenario = cleaned_postfix.substr(0, pos);

  std::filesystem::path out_dir = "out_vtk_" + scenario + std::to_string(steps);
  if (!std::filesystem::exists(out_dir)) {
    if (std::error_code error;
        !std::filesystem::create_directory(out_dir, error)) {
      std::cerr << "Fehler: Konnte Verzeichnis " << out_dir
                << "nicht anlegen: " << error.message() << "\n";
      return;
    }
  } else if (!std::filesystem::is_directory(out_dir)) {
    std::cerr << "Fehler: " << out_dir
              << " existiert, ist aber kein Verzeichnis.\n";
    return;
  }

  std::stringstream filename;
  filename << "output_" << std::setw(6) << std::setfill('0') << frame << ".vtk";
  std::filesystem::path out_path = out_dir / filename.str();

  std::ofstream vtk_file(out_path);
  if (!vtk_file)
    throw std::runtime_error("Failed to open VTK file");

  vtk_file << "# vtk DataFile Version 3.0\n";
  vtk_file << "Molecular Dynamics Particles\n";
  vtk_file << "ASCII\n";
  vtk_file << "DATASET UNSTRUCTURED_GRID\n\n";

  vtk_file << "POINTS " << vtk.n << " float\n";
  for (size_t i = 0; i < vtk.n; ++i) {
    vtk_file << vtk.positions[i].x << " " << vtk.positions[i].y << " "
             << vtk.positions[i].z << "\n";
  }

  vtk_file << "\nCELLS " << vtk.n << " " << 2 * vtk.n << "\n";
  for (size_t i = 0; i < vtk.n; ++i) {
    vtk_file << "1 " << i << "\n";
  }

  vtk_file << "\nCELL_TYPES " << vtk.n << "\n";
  for (size_t i = 0; i < vtk.n; ++i) {
    vtk_file << "1\n"; // VTK_VERTEX
  }

  vtk_file << "\nPOINT_DATA " << vtk.n << "\n";

  vtk_file << "VECTORS velocity float\n";
  for (size_t i = 0; i < vtk.n; ++i) {
    vtk_file << vtk.velocities[i].x << " " << vtk.velocities[i].y << " "
             << vtk.velocities[i].z << "\n";
  }

  vtk_file << "VECTORS eulerAngles float\n";
  for (size_t i = 0; i < vtk.n; ++i) {
    auto [x, y, z] = vtk.orientations[i].toEulerAngles();
    vtk_file << z << " " << y << " " << x << "\n";
  }

  vtk_file << "\nVECTORS force float\n";
  for (size_t i = 0; i < vtk.n; ++i) {
    vtk_file << vtk.forces[i].x << " " << vtk.forces[i].y << " "
             << vtk.forces[i].z << "\n";
  }

  vtk_file << "SCALARS Radius float 1\n";
  vtk_file << "LOOKUP_TABLE default\n";
  for (auto r : vtk.radii) {
    vtk_file << r << "\n";
  }

  vtk_file.close();
}

inline void writeMeshVTK(int frame, const MeshVTK &vtk,
                         const std::string &postfix, long steps) {
  // same directory-setup as writeVTK
  const std::string parent_prefix = "../";
  std::string cleaned_postfix = postfix;
  while (cleaned_postfix.rfind(parent_prefix, 0) == 0)
    cleaned_postfix.erase(0, parent_prefix.size());
  auto pos = cleaned_postfix.find_last_of('.');
  std::string scenario = cleaned_postfix.substr(0, pos);

  std::filesystem::path out_dir = "out_vtk_" + scenario + std::to_string(steps);
  if (!std::filesystem::exists(out_dir)) {
    if (std::error_code ec; !std::filesystem::create_directory(out_dir, ec)) {
      std::cerr << "Fehler: Konnte Verzeichnis " << out_dir
                << " nicht anlegen: " << ec.message() << "\n";
      return;
    }
  } else if (!std::filesystem::is_directory(out_dir)) {
    std::cerr << "Fehler: " << out_dir
              << " existiert, ist aber kein Verzeichnis.\n";
    return;
  }

  std::ostringstream filename;
  filename << "mesh_" << std::setw(6) << std::setfill('0') << frame << ".vtk";
  auto out_path = out_dir / filename.str();

  std::ofstream vtk_file(out_path);
  if (!vtk_file)
    throw std::runtime_error("Failed to open VTK file");

  // — header —
  vtk_file << "# vtk DataFile Version 3.0\n";
  vtk_file << "Mesh Instances\nASCII\n";
  vtk_file << "DATASET UNSTRUCTURED_GRID\n\n";

  // 1) Count total points and cells
  int totalPts = 0, totalCells = 0;
  for (size_t i = 0; i < vtk.n; ++i) {
    int mi = vtk.mesh_index[i];
    int v0 = vtk.meshes_offset[mi];
    int v1 = (mi + 1 < vtk.m ? vtk.meshes_offset[mi + 1]
                             : static_cast<int>(vtk.vertices_n));
    totalPts += (v1 - v0);
    int f0 = vtk.faces_offset[mi];
    int f1 = (mi + 1 < vtk.m ? vtk.faces_offset[mi + 1]
                             : static_cast<int>(vtk.faces_n));
    totalCells += (f1 - f0);
  }

  // 2) Write points
  vtk_file << "POINTS " << totalPts << " float\n";
  int ptOffset = 0;
  for (size_t i = 0; i < vtk.n; ++i) {
    int mi = vtk.mesh_index[i];
    int v0 = vtk.meshes_offset[mi];
    int v1 = (mi + 1 < vtk.m ? vtk.meshes_offset[mi + 1]
                             : static_cast<int>(vtk.vertices_n));

    // for each base-vertex: apply S·vertex, rotate, translate
    for (int vi = v0; vi < v1; ++vi) {
      Vec3 v = vtk.vertices[vi];
      // scale
      v.x *= vtk.scales[i].x;
      v.y *= vtk.scales[i].y;
      v.z *= vtk.scales[i].z;
      // rotate
      v = vtk.orientations[i].rotate(v);
      // translate
      v += vtk.positions[i];
      vtk_file << v.x << " " << v.y << " " << v.z << "\n";
    }
    ptOffset += (v1 - v0);
  }

  // 3) Write cells
  // each triangle cell: "3 i j k"
  int connSize = totalCells * 4; // 3 + three indices
  vtk_file << "\nCELLS " << totalCells << " " << connSize << "\n";

  ptOffset = 0;
  for (size_t i = 0; i < vtk.n; ++i) {
    int mi = vtk.mesh_index[i];
    int v0 = vtk.meshes_offset[mi];
    int v1 = (mi + 1 < vtk.m ? vtk.meshes_offset[mi + 1]
                             : static_cast<int>(vtk.vertices_n));
    int f0 = vtk.faces_offset[mi];
    int f1 = (mi + 1 < vtk.m ? vtk.faces_offset[mi + 1]
                             : static_cast<int>(vtk.faces_n));

    int nv = v1 - v0; // number of verts of this mesh
    // for each face, output indices + ptOffset
    for (int fi = f0; fi < f1; ++fi) {
      auto f = vtk.faces[fi];
      vtk_file << "3 " << (ptOffset + (f.x - v0)) << " "
               << (ptOffset + (f.y - v0)) << " " << (ptOffset + (f.z - v0))
               << "\n";
    }
    ptOffset += nv;
  }

  // 4) Cell types (all triangles = 5)
  vtk_file << "\nCELL_TYPES " << totalCells << "\n";
  for (int i = 0; i < totalCells; ++i)
    vtk_file << "5\n";

  // 5) Point data
  vtk_file << "\nPOINT_DATA " << totalPts << "\n";

  // velocities
  vtk_file << "VECTORS velocity float\n";
  ptOffset = 0;
  for (size_t i = 0; i < vtk.n; ++i) {
    int mi = vtk.mesh_index[i];
    int v0 = vtk.meshes_offset[mi];
    int v1 = (mi + 1 < vtk.m ? vtk.meshes_offset[mi + 1]
                             : static_cast<int>(vtk.vertices_n));
    for (int k = 0; k < v1 - v0; ++k)
      vtk_file << vtk.velocities[i].x << " " << vtk.velocities[i].y << " "
               << vtk.velocities[i].z << "\n";
    ptOffset += (v1 - v0);
  }

  // forces
  vtk_file << "VECTORS force float\n";
  ptOffset = 0;
  for (size_t i = 0; i < vtk.n; ++i) {
    int mi = vtk.mesh_index[i];
    int v0 = vtk.meshes_offset[mi];
    int v1 = (mi + 1 < vtk.m ? vtk.meshes_offset[mi + 1]
                             : static_cast<int>(vtk.vertices_n));
    for (int k = 0; k < v1 - v0; ++k)
      vtk_file << vtk.forces[i].x << " " << vtk.forces[i].y << " "
               << vtk.forces[i].z << "\n";
    ptOffset += (v1 - v0);
  }

  // scale (as vector)
  vtk_file << "VECTORS scale float\n";
  ptOffset = 0;
  for (size_t i = 0; i < vtk.n; ++i) {
    int mi = vtk.mesh_index[i];
    int v0 = vtk.meshes_offset[mi];
    int v1 = (mi + 1 < vtk.m ? vtk.meshes_offset[mi + 1]
                             : static_cast<int>(vtk.vertices_n));
    for (int k = 0; k < v1 - v0; ++k)
      vtk_file << vtk.scales[i].x << " " << vtk.scales[i].y << " "
               << vtk.scales[i].z << "\n";
    ptOffset += (v1 - v0);
  }

  vtk_file.close();
}

inline void writeBoundingBoxWallsVTK(const std::string &postfix, long steps,
                                     const std::string &filename,
                                     const Vec3 minPos, const Vec3 maxPos) {
  const std::string parent_prefix = "../";
  std::string cleaned_postfix = postfix;
  while (cleaned_postfix.rfind(parent_prefix, 0) != std::string::npos)
    cleaned_postfix.erase(0, parent_prefix.size());
  const auto pos = cleaned_postfix.find_last_of('.');
  const std::string scenario = cleaned_postfix.substr(0, pos);

  std::filesystem::path out_dir = "out_vtk_" + scenario + std::to_string(steps);
  if (!std::filesystem::exists(out_dir)) {
    if (std::error_code error;
        !std::filesystem::create_directory(out_dir, error)) {
      std::cerr << "Fehler: Konnte Verzeichnis " << out_dir
                << "nicht anlegen: " << error.message() << "\n";
      return;
    }
  } else if (!std::filesystem::is_directory(out_dir)) {
    std::cerr << "Fehler: " << out_dir
              << " existiert, ist aber kein Verzeichnis.\n";
    return;
  }
  std::filesystem::path out_path = out_dir / filename;

  std::ofstream vtk_file(out_path);
  if (!vtk_file)
    throw std::runtime_error("Failed to open bounding box VTK file");

  vtk_file << "# vtk DataFile Version 3.0\n";
  vtk_file << "Bounding Box with Walls\n";
  vtk_file << "ASCII\n";
  vtk_file << "DATASET POLYDATA\n";

  vtk_file << "POINTS 8 float\n";
  vtk_file << minPos.x << " " << minPos.y << " " << minPos.z << "\n"; // 0
  vtk_file << maxPos.x << " " << minPos.y << " " << minPos.z << "\n"; // 1
  vtk_file << maxPos.x << " " << maxPos.y << " " << minPos.z << "\n"; // 2
  vtk_file << minPos.x << " " << maxPos.y << " " << minPos.z << "\n"; // 3
  vtk_file << minPos.x << " " << minPos.y << " " << maxPos.z << "\n"; // 4
  vtk_file << maxPos.x << " " << minPos.y << " " << maxPos.z << "\n"; // 5
  vtk_file << maxPos.x << " " << maxPos.y << " " << maxPos.z << "\n"; // 6
  vtk_file << minPos.x << " " << maxPos.y << " " << maxPos.z << "\n"; // 7

  vtk_file << "POLYGONS 6 30\n";
  auto writeFace = [&](int a, int b, int c, int d) {
    vtk_file << "4 " << a << " " << b << " " << c << " " << d << "\n";
  };
  // Faces
  writeFace(0, 1, 2, 3); // Bottom
  writeFace(4, 5, 6, 7); // Top
  writeFace(0, 1, 5, 4); // Front
  writeFace(2, 3, 7, 6); // Back
  writeFace(0, 3, 7, 4); // Left
  writeFace(1, 2, 6, 5); // Right

  vtk_file.close();
}

#endif // VTK_H
