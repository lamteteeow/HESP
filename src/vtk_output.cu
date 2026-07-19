#include "vtk_output.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

void writeParticlesVTK(int frame, const std::vector<Vec3> &positions,
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
  f << "MD Particles\nASCII\nDATASET UNSTRUCTURED_GRID\n\n";

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

void writeDomainBoundaryVTK(const Vec3 domain_min, const Vec3 domain_max,
                            const std::vector<Domain> &doms,
                            const std::string &scenario, long steps,
                            int frame) {
  namespace fs = std::filesystem;

  const std::string dir_name =
      "output/" + scenario + "_" + std::to_string(steps);
  fs::create_directories(dir_name);

  std::ostringstream fname;
  fname << dir_name << "/domain_boundary";
  if (frame >= 0)
    fname << "_" << std::setw(6) << std::setfill('0') << frame;
  fname << ".vtk";

  std::ofstream f(fname.str());
  if (!f)
    throw std::runtime_error("Failed to open domain boundary VTK file");

  const float min_x = domain_min.x, min_y = domain_min.y, min_z = domain_min.z;
  const float max_x = domain_max.x, max_y = domain_max.y, max_z = domain_max.z;
  const int num_gpus = static_cast<int>(doms.size());

  struct Pt { float x, y, z; };
  std::vector<Pt> pts;
  std::vector<std::vector<int>> lines;
  std::vector<std::vector<int>> polys;
  std::vector<int> line_types, poly_types;
  auto addPt = [&](float x, float y, float z) {
    pts.push_back({x, y, z});
    return static_cast<int>(pts.size()) - 1;
  };

  // --- Outer domain box (3D wireframe) ---
  int b0 = addPt(min_x, min_y, min_z), b1 = addPt(max_x, min_y, min_z);
  int b2 = addPt(max_x, max_y, min_z), b3 = addPt(min_x, max_y, min_z);
  int t0 = addPt(min_x, min_y, max_z), t1 = addPt(max_x, min_y, max_z);
  int t2 = addPt(max_x, max_y, max_z), t3 = addPt(min_x, max_y, max_z);
  for (auto &l : std::vector<std::vector<int>>{
           {b0,b1},{b1,b2},{b2,b3},{b3,b0},
           {t0,t1},{t1,t2},{t2,t3},{t3,t0},
           {b0,t0},{b1,t1},{b2,t2},{b3,t3}})
    { lines.push_back({2, l[0], l[1]}); line_types.push_back(0); }

  // --- Owned-region split lines ---
  const int nx = doms[0].grid_nx, ny = doms[0].grid_ny, nz = doms[0].grid_nz;
  const float dx = (max_x - min_x) / nx;
  const float dy = (max_y - min_y) / ny;
  const float dz = (max_z - min_z) / nz;
  for (int gx = 1; gx < nx; ++gx) {
    float sx = min_x + gx * dx;
    int b = addPt(sx, min_y, min_z), t = addPt(sx, max_y, max_z);
    lines.push_back({2, b, t}); line_types.push_back(1);
  }
  for (int gy = 1; gy < ny; ++gy) {
    float sy = min_y + gy * dy;
    int l = addPt(min_x, sy, min_z), r = addPt(max_x, sy, max_z);
    lines.push_back({2, l, r}); line_types.push_back(1);
  }
  for (int gz = 1; gz < nz; ++gz) {
    float sz = min_z + gz * dz;
    int bl = addPt(min_x, min_y, sz), br = addPt(max_x, min_y, sz);
    int tr = addPt(max_x, max_y, sz), tl = addPt(min_x, max_y, sz);
    lines.push_back({2, bl, br}); line_types.push_back(1);
    lines.push_back({2, br, tr}); line_types.push_back(1);
    lines.push_back({2, tr, tl}); line_types.push_back(1);
    lines.push_back({2, tl, bl}); line_types.push_back(1);
  }

  // --- Halo strips (3D boxes, 6 quad faces each) ---
  auto addBox = [&](float x0, float x1, float y0, float y1, float z0, float z1) {
    int p000 = addPt(x0, y0, z0), p100 = addPt(x1, y0, z0);
    int p110 = addPt(x1, y1, z0), p010 = addPt(x0, y1, z0);
    int p001 = addPt(x0, y0, z1), p101 = addPt(x1, y0, z1);
    int p111 = addPt(x1, y1, z1), p011 = addPt(x0, y1, z1);
    polys.push_back({4, p000, p100, p110, p010});
    polys.push_back({4, p001, p101, p111, p011});
    polys.push_back({4, p000, p010, p011, p001});
    polys.push_back({4, p100, p110, p111, p101});
    polys.push_back({4, p000, p100, p101, p001});
    polys.push_back({4, p010, p110, p111, p011});
    for (int k = 0; k < 6; ++k) poly_types.push_back(2);
  };

  for (int g = 0; g < num_gpus; ++g) {
    const Domain &d = doms[g];
    float z0 = d.owned_min.z, z1 = d.owned_max.z;
    if (d.left_neighbor >= 0)
      addBox(d.local_min.x, d.owned_min.x, d.owned_min.y, d.owned_max.y, z0, z1);
    if (d.right_neighbor >= 0)
      addBox(d.owned_max.x, d.local_max.x, d.owned_min.y, d.owned_max.y, z0, z1);
    if (d.bottom_neighbor >= 0)
      addBox(d.owned_min.x, d.owned_max.x, d.local_min.y, d.owned_min.y, z0, z1);
    if (d.top_neighbor >= 0)
      addBox(d.owned_min.x, d.owned_max.x, d.owned_max.y, d.local_max.y, z0, z1);
    if (d.back_neighbor >= 0)
      addBox(d.owned_min.x, d.owned_max.x, d.owned_min.y, d.owned_max.y,
             d.local_min.z, d.owned_min.z);
    if (d.front_neighbor >= 0)
      addBox(d.owned_min.x, d.owned_max.x, d.owned_min.y, d.owned_max.y,
             d.owned_max.z, d.local_max.z);
  }

  // ---- Write VTK ----
  f << "# vtk DataFile Version 3.0\n";
  f << "Domain boundary\nASCII\nDATASET POLYDATA\n\n";

  f << "POINTS " << pts.size() << " float\n";
  for (auto &p : pts)
    f << p.x << " " << p.y << " " << p.z << "\n";

  int line_bytes = 0;
  for (auto &l : lines) line_bytes += static_cast<int>(l.size());
  f << "\nLINES " << lines.size() << " " << line_bytes << "\n";
  for (auto &l : lines) {
    f << l[0];
    for (size_t i = 1; i < l.size(); ++i) f << " " << l[i];
    f << "\n";
  }

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

  int ncell = static_cast<int>(lines.size() + polys.size());
  f << "\nCELL_DATA " << ncell << "\n";
  f << "SCALARS region_type int 1\nLOOKUP_TABLE default\n";
  for (int t : line_types) f << t << "\n";
  for (int t : poly_types) f << t << "\n";

  f.close();
}
