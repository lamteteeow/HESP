#ifndef VTK_OUTPUT_H
#define VTK_OUTPUT_H

#include "domain.h"
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
// gpu_owner is written as a scalar so you can color by GPU ownership.
// Output goes to output/<scenario>_<steps>/
inline void writeParticlesVTK(int frame, const std::vector<Vec3> &positions,
                              const std::vector<Vec3> &velocities,
                              const std::vector<float> &radii,
                              const std::vector<int> &gpu_owner,
                              const std::string &scenario, long steps) {
  namespace fs = std::filesystem;
  const size_t n = positions.size();

  const std::string dir_name =
      "output/" + scenario + "_" + std::to_string(steps);
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

  f << "SCALARS gpu_owner int 1\nLOOKUP_TABLE default\n";
  for (size_t i = 0; i < n; ++i)
    f << gpu_owner[i] << "\n";

  f.close();
}

// Write domain boundaries as a wireframe box + GPU split lines.
// Call once during the first frame — loaded alongside particle frames in
// ParaView to show the simulation domain and GPU decomposition.
inline void writeDomainBoundaryVTK(const Vec3 domain_min,
                                   const Vec3 domain_max,
                                   const std::vector<Domain> &doms,
                                   const std::string &scenario, long steps) {
  namespace fs = std::filesystem;

  const std::string dir_name =
      "output/" + scenario + "_" + std::to_string(steps);
  fs::create_directories(dir_name);

  std::ofstream f(dir_name + "/domain_boundary.vtk");
  if (!f)
    throw std::runtime_error("Failed to open domain boundary VTK file");

  // z is 0 for 2D
  const float z = 0.0f;
  const float min_x = domain_min.x, min_y = domain_min.y;
  const float max_x = domain_max.x, max_y = domain_max.y;

  // Number of split lines: one vertical line at each GPU-owned boundary
  const int num_gpus = static_cast<int>(doms.size());
  const int num_split_lines = num_gpus - 1;

  // Points: 4 for outer box + 2 * num_split_lines for verticals
  const int npts = 4 + 2 * num_split_lines;
  // Lines: 4 for outer box + num_split_lines verticals
  const int nlines = 4 + num_split_lines;

  f << "# vtk DataFile Version 3.0\n";
  f << "Domain boundary\nASCII\nDATASET POLYDATA\n\n";

  f << "POINTS " << npts << " float\n";
  // Outer box corners
  f << min_x << " " << min_y << " " << z << "\n";  // 0: bottom-left
  f << max_x << " " << min_y << " " << z << "\n";  // 1: bottom-right
  f << max_x << " " << max_y << " " << z << "\n";  // 2: top-right
  f << min_x << " " << max_y << " " << z << "\n";  // 3: top-left

  // Vertical split lines at owned_max.x for each GPU (except last)
  int pi = 4;
  for (int g = 0; g < num_gpus - 1; ++g) {
    const float sx = doms[g].owned_max.x;
    f << sx << " " << min_y << " " << z << "\n";  // bottom
    f << sx << " " << max_y << " " << z << "\n";  // top
  }

  // Each line entry: <npoints> <idx0> <idx1 ...>
  f << "\nLINES " << nlines << " " << (3 * 4 + 3 * num_split_lines) << "\n";
  f << "2 0 1\n";  // bottom
  f << "2 1 2\n";  // right
  f << "2 2 3\n";  // top
  f << "2 3 0\n";  // left

  pi = 4;
  for (int g = 0; g < num_gpus - 1; ++g) {
    f << "2 " << pi << " " << (pi + 1) << "\n";
    pi += 2;
  }

  f.close();
}

#endif // VTK_OUTPUT_H
