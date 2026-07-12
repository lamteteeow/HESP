#ifndef INPUT_H
#define INPUT_H

#include "domain.h"
#include "json.hpp"
#include "vec3.cuh"
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// Parse a Vec3 from a JSON array (host-only utility).
inline Vec3 vec3FromJson(const nlohmann::json &a) {
  return Vec3{a[0].get<float>(), a[1].get<float>(), a[2].get<float>()};
}

struct SceneConfig {
  float dt;
  Vec3 gravity;
  Vec3 domain_min;
  Vec3 domain_max;
  float cell_size; // 0 = auto (= 2 * r_max)
};

// Simple host-side particle list used for loading and splitting.
struct ParticleData {
  size_t n = 0;
  std::vector<Vec3> positions, velocities;
  std::vector<float> masses, radii, kn, gamma_n, gamma_t, mu;

  void push(Vec3 pos, Vec3 vel, float mass, float r, float _kn, float _gn,
            float _gt, float _mu) {
    positions.push_back(pos);
    velocities.push_back(vel);
    masses.push_back(mass);
    radii.push_back(r);
    kn.push_back(_kn);
    gamma_n.push_back(_gn);
    gamma_t.push_back(_gt);
    mu.push_back(_mu);
    ++n;
  }

  // O(1) removal by swap-with-last.
  void removeAt(size_t i) {
    size_t last = n - 1;
    if (i != last) {
      positions[i] = positions[last];
      velocities[i] = velocities[last];
      masses[i] = masses[last];
      radii[i] = radii[last];
      kn[i] = kn[last];
      gamma_n[i] = gamma_n[last];
      gamma_t[i] = gamma_t[last];
      mu[i] = mu[last];
    }
    positions.pop_back();
    velocities.pop_back();
    masses.pop_back();
    radii.pop_back();
    kn.pop_back();
    gamma_n.pop_back();
    gamma_t.pop_back();
    mu.pop_back();
    --n;
  }
};

inline void loadScene(const std::string &filename, SceneConfig &cfg,
                      ParticleData &pd) {
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

// Split particles into two sets by x-coordinate (backward-compatible).
inline void splitAt(const ParticleData &src, float split_x, ParticleData &left,
                    ParticleData &right) {
  for (size_t i = 0; i < src.n; ++i) {
    ParticleData &dst = (src.positions[i].x < split_x) ? left : right;
    dst.push(src.positions[i], src.velocities[i], src.masses[i], src.radii[i],
             src.kn[i], src.gamma_n[i], src.gamma_t[i], src.mu[i]);
  }
}

// Split particles into N sets, one per domain, by (x, y, z) position.
inline void splitIntoN(const ParticleData &src, const std::vector<Domain> &doms,
                       std::vector<ParticleData> &out) {
  const int num_gpus = static_cast<int>(doms.size());
  out.resize(num_gpus);
  const Domain &d0 = doms[0];
  const int nx = d0.grid_nx, ny = d0.grid_ny, nz = d0.grid_nz;
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
}

#endif // INPUT_H
