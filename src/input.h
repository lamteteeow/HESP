#ifndef INPUT_H
#define INPUT_H

#include "domain.h"
#include "vec3.cuh"
#include "json.hpp"
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

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

// Split particles into N sets, one per domain, by x-coordinate.
inline void splitIntoN(const ParticleData &src, const std::vector<Domain> &doms,
                       std::vector<ParticleData> &out) {
  const int num_gpus = static_cast<int>(doms.size());
  out.resize(num_gpus);
  for (size_t i = 0; i < src.n; ++i) {
    const float x = src.positions[i].x;
    int g = 0;
    while (g < num_gpus - 1 && x >= doms[g].owned_max.x)
      ++g;
    out[g].push(src.positions[i], src.velocities[i], src.masses[i],
                src.radii[i], src.kn[i], src.gamma_n[i], src.gamma_t[i],
                src.mu[i]);
  }
}

#endif // INPUT_H
