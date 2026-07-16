#include "input.h"
#include <cstdio>
#include <fstream>
#include <stdexcept>

Vec3 vec3FromJson(const nlohmann::json &a) {
  return Vec3{a[0].get<float>(), a[1].get<float>(), a[2].get<float>()};
}

void loadScene(const std::string &filename, SceneConfig &cfg, ParticleData &pd) {
  std::ifstream f(filename);
  if (!f)
    throw std::runtime_error("Cannot open scene file: " + filename);

  nlohmann::json j;
  f >> j;

  cfg.dt = j["dt"].get<float>();
  cfg.gravity = vec3FromJson(j["gravity"]);
  cfg.domain_min = vec3FromJson(j["domain"]["min"]);
  cfg.domain_max = vec3FromJson(j["domain"]["max"]);
  cfg.cell_size = j.value("cell_size", 0.0f);

  float max_r = 0.0f;
  for (const auto &p : j["particles"]) {
    Vec3 pos = vec3FromJson(p["position"]);
    Vec3 vel = vec3FromJson(p["velocity"]);
    float r = p["radius"].get<float>();
    max_r = std::max(max_r, r);
    pd.push(pos, vel, p["mass"].get<float>(), r, p["kn"].get<float>(),
            p["gamma_n"].get<float>(), p["gamma_t"].get<float>(),
            p["mu"].get<float>());
  }

  if (cfg.cell_size <= 0.0f)
    cfg.cell_size = 2.0f * max_r;
}

void splitAt(const ParticleData &src, float split_x, ParticleData &left,
             ParticleData &right) {
  for (size_t i = 0; i < src.n; ++i) {
    ParticleData &dst = (src.positions[i].x < split_x) ? left : right;
    dst.push(src.positions[i], src.velocities[i], src.masses[i], src.radii[i],
             src.kn[i], src.gamma_n[i], src.gamma_t[i], src.mu[i]);
  }
}

void splitIntoN(const ParticleData &src, const std::vector<Domain> &doms,
                std::vector<ParticleData> &out) {
  const int num_gpus = static_cast<int>(doms.size());
  out.resize(num_gpus);
  const Domain &d0 = doms[0];
  const int nx = d0.grid_nx, ny = d0.grid_ny, nz = d0.grid_nz;
  printf("  splitIntoN: grid=%dx%dx%d  doms=%zu  particles=%zu\n",
         nx, ny, nz, doms.size(), src.n);
  const float gmin_x = d0.global_min.x, gmin_y = d0.global_min.y, gmin_z = d0.global_min.z;
  const float gmax_x = d0.global_max.x, gmax_y = d0.global_max.y, gmax_z = d0.global_max.z;
  const float dx = (gmax_x - gmin_x) / nx;
  const float dy = (gmax_y - gmin_y) / ny;
  const float dz = (gmax_z - gmin_z) / nz;
  for (size_t i = 0; i < src.n; ++i) {
    const float x = src.positions[i].x;
    const float y = src.positions[i].y;
    const float z = src.positions[i].z;
    int gx = std::min(std::max(static_cast<int>((x - gmin_x) / dx), 0), nx - 1);
    int gy = std::min(std::max(static_cast<int>((y - gmin_y) / dy), 0), ny - 1);
    int gz = std::min(std::max(static_cast<int>((z - gmin_z) / dz), 0), nz - 1);
    int g = (gz * ny + gy) * nx + gx;
    out[g].push(src.positions[i], src.velocities[i], src.masses[i],
                src.radii[i], src.kn[i], src.gamma_n[i], src.gamma_t[i],
                src.mu[i]);
  }
  printf("  splitIntoN: per-gpu counts:");
  for (int g = 0; g < num_gpus; ++g)
    printf(" %d:%zu", g, out[g].n);
  printf("\n");
}
