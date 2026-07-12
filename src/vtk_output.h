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
// gpu_owner: which GPU owns the particle (color by this for domain view).
// border: 0 = interior, 1 = at halo edge (gradient at GPU boundaries).
// Output goes to output/<scenario>_<steps>/
inline void writeParticlesVTK(int frame, const std::vector<Vec3> &positions,
                              const std::vector<Vec3> &velocities,
                              const std::vector<float> &radii,
                              const std::vector<int> &gpu_owner,
                              const std::vector<float> &border,
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

  f << "SCALARS border float 1\nLOOKUP_TABLE default\n";
  for (size_t i = 0; i < n; ++i)
    f << border[i] << "\n";

  f.close();
}

// Write domain boundaries + halo regions as a wireframe + filled strips.
// Call once during the first frame.
// In ParaView, color by 'region_type' to distinguish:
//   0 = domain boundary, 1 = owned split, 2 = halo strip
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

  // ---- Gather geometry ----
  const float min_x = domain_min.x, min_y = domain_min.y;
  const float max_x = domain_max.x, max_y = domain_max.y;
#ifdef MD3D
  const float min_z = domain_min.z, max_z = domain_max.z;
#else
  const float min_z = 0.0f, max_z = 0.0f;
#endif
  const int num_gpus = static_cast<int>(doms.size());

  // Build point list and cell lists
  struct Pt { float x, y, z; };
  std::vector<Pt> pts;
  std::vector<std::vector<int>> lines;   // each = {2, i0, i1}
  std::vector<std::vector<int>> polys;   // each = {4, i0, i1, i2, i3}
  std::vector<int> line_types, poly_types;  // region_type per cell
  auto addPt = [&](float x, float y, float z) {
    pts.push_back({x, y, z});
    return static_cast<int>(pts.size()) - 1;
  };

  // --- Outer domain box ---
#ifdef MD3D
  int b0 = addPt(min_x, min_y, min_z), b1 = addPt(max_x, min_y, min_z);
  int b2 = addPt(max_x, max_y, min_z), b3 = addPt(min_x, max_y, min_z);
  int t0 = addPt(min_x, min_y, max_z), t1 = addPt(max_x, min_y, max_z);
  int t2 = addPt(max_x, max_y, max_z), t3 = addPt(min_x, max_y, max_z);
  for (auto &l : std::vector<std::vector<int>>{
           {b0,b1},{b1,b2},{b2,b3},{b3,b0},
           {t0,t1},{t1,t2},{t2,t3},{t3,t0},
           {b0,t0},{b1,t1},{b2,t2},{b3,t3}})
    { lines.push_back({2, l[0], l[1]}); line_types.push_back(0); }
#else
  int bl = addPt(min_x, min_y, min_z), br = addPt(max_x, min_y, min_z);
  int tr = addPt(max_x, max_y, min_z), tl = addPt(min_x, max_y, min_z);
  for (auto &l : std::vector<std::vector<int>>{
           {bl,br},{br,tr},{tr,tl},{tl,bl}})
    { lines.push_back({2, l[0], l[1]}); line_types.push_back(0); }
#endif

  // --- Owned-region split lines (grid boundaries) ---
  const int nx = doms[0].grid_nx, ny = doms[0].grid_ny;
  const float dx = (max_x - min_x) / nx;
  const float dy = (max_y - min_y) / ny;
  // X splits (vertical lines)
  for (int gx = 1; gx < nx; ++gx) {
    float sx = min_x + gx * dx;
    int b = addPt(sx, min_y, min_z), t = addPt(sx, max_y, max_z);
    lines.push_back({2, b, t});
    line_types.push_back(1);
  }
  // Y splits (horizontal lines)
  for (int gy = 1; gy < ny; ++gy) {
    float sy = min_y + gy * dy;
    int l = addPt(min_x, sy, min_z), r = addPt(max_x, sy, max_z);
    lines.push_back({2, l, r});
    line_types.push_back(1);
  }

  // --- Halo strips (filled 3D boxes, or 2D rectangles) ---
  // Helper: add 8 corners of a box and return the 6 face quads
  auto addBox = [&](float x0, float x1, float y0, float y1, float z0, float z1) {
#ifdef MD3D
    int p000 = addPt(x0, y0, z0), p100 = addPt(x1, y0, z0);
    int p110 = addPt(x1, y1, z0), p010 = addPt(x0, y1, z0);
    int p001 = addPt(x0, y0, z1), p101 = addPt(x1, y0, z1);
    int p111 = addPt(x1, y1, z1), p011 = addPt(x0, y1, z1);
    // Each face: 4 points CCW when viewed from outside the box.
    // Outside = negative side of the face normal.
    polys.push_back({4, p000, p100, p110, p010}); // -z  bottom
    polys.push_back({4, p001, p101, p111, p011}); // +z  top
    polys.push_back({4, p000, p010, p011, p001}); // -x  left
    polys.push_back({4, p100, p110, p111, p101}); // +x  right
    polys.push_back({4, p000, p100, p101, p001}); // -y  front
    polys.push_back({4, p010, p110, p111, p011}); // +y  back
    for (int k = 0; k < 6; ++k) poly_types.push_back(2);
#else
    // 2D: single flat quad at z=0
    int a = addPt(x0, y0, 0), b = addPt(x1, y0, 0);
    int c = addPt(x1, y1, 0), d = addPt(x0, y1, 0);
    polys.push_back({4, a, b, c, d});
    poly_types.push_back(2);
#endif
  };

  for (int g = 0; g < num_gpus; ++g) {
    const Domain &d = doms[g];
#ifdef MD3D
    float lo_z = min_z, hi_z = max_z;
#else
    float lo_z = 0, hi_z = 0;
#endif
    if (d.left_neighbor >= 0)
      addBox(d.local_min.x, d.owned_min.x, d.owned_min.y, d.owned_max.y, lo_z, hi_z);
    if (d.right_neighbor >= 0)
      addBox(d.owned_max.x, d.local_max.x, d.owned_min.y, d.owned_max.y, lo_z, hi_z);
    if (d.bottom_neighbor >= 0)
      addBox(d.owned_min.x, d.owned_max.x, d.local_min.y, d.owned_min.y, lo_z, hi_z);
    if (d.top_neighbor >= 0)
      addBox(d.owned_min.x, d.owned_max.x, d.owned_max.y, d.local_max.y, lo_z, hi_z);
  }

  // ---- Write VTK ----
  f << "# vtk DataFile Version 3.0\n";
  f << "Domain boundary\nASCII\nDATASET POLYDATA\n\n";

  f << "POINTS " << pts.size() << " float\n";
  for (auto &p : pts)
    f << p.x << " " << p.y << " " << p.z << "\n";

  // Lines
  int line_bytes = 0;
  for (auto &l : lines) line_bytes += static_cast<int>(l.size());
  f << "\nLINES " << lines.size() << " " << line_bytes << "\n";
  for (auto &l : lines) {
    f << l[0];
    for (size_t i = 1; i < l.size(); ++i) f << " " << l[i];
    f << "\n";
  }

  // Halo polygons
  int poly_bytes = 0;
  for (auto &p : polys) poly_bytes += static_cast<int>(p.size());
  if (!polys.empty()) {
    f << "\nPOLYGONS " << polys.size() << " " << poly_bytes << "\n";
    for (auto &p : polys) {
      f << p[0];
      for (size_t i = 1; i < p.size(); ++i) f << " " << p[i];
      f << "\n";
    }
  }

  // Cell data: region_type
  int ncell = static_cast<int>(lines.size() + polys.size());
  f << "\nCELL_DATA " << ncell << "\n";
  f << "SCALARS region_type int 1\nLOOKUP_TABLE default\n";
  for (int t : line_types) f << t << "\n";
  for (int t : poly_types) f << t << "\n";

  f.close();
}

#endif // VTK_OUTPUT_H
