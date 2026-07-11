#ifndef VTK_OUTPUT_H
#define VTK_OUTPUT_H

#include "vec3.cuh"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Write one VTK frame for a set of 2D particles.
// ParaView can render these as spheres using the Radius scalar field.
inline void writeParticlesVTK(int frame, const std::vector<Vec3> &positions,
                              const std::vector<Vec3> &velocities,
                              const std::vector<float> &radii,
                              const std::string &scenario, long steps) {
  namespace fs = std::filesystem;
  const size_t n = positions.size();

  const std::string dir_name =
      "out_vtk_" + scenario + "_" + std::to_string(steps);
  fs::create_directories(dir_name);

  std::ostringstream fname;
  fname << dir_name << "/output_" << std::setw(6) << std::setfill('0') << frame
        << ".vtk";

  std::ofstream f(fname.str());
  if (!f)
    throw std::runtime_error("Failed to open VTK file: " + fname.str());

  f << "# vtk DataFile Version 3.0\n";
  f << "MD2D Particles\nASCII\nDATASET UNSTRUCTURED_GRID\n\n";

  f << "POINTS " << n << " float\n";
  for (size_t i = 0; i < n; ++i)
    f << positions[i].x << " " << positions[i].y << " " << positions[i].z
      << "\n";

  f << "\nCELLS " << n << " " << 2 * n << "\n";
  for (size_t i = 0; i < n; ++i)
    f << "1 " << i << "\n";

  f << "\nCELL_TYPES " << n << "\n";
  for (size_t i = 0; i < n; ++i)
    f << "1\n"; // VTK_VERTEX

  f << "\nPOINT_DATA " << n << "\n";

  f << "VECTORS velocity float\n";
  for (size_t i = 0; i < n; ++i)
    f << velocities[i].x << " " << velocities[i].y << " " << velocities[i].z
      << "\n";

  f << "SCALARS radius float 1\nLOOKUP_TABLE default\n";
  for (size_t i = 0; i < n; ++i)
    f << radii[i] << "\n";

  f.close();
}

#endif // VTK_OUTPUT_H
