#ifndef INPUT_H
#define INPUT_H

#include "domain.h"
#include "json.hpp"
#include "vec3.cuh"
#include <string>
#include <vector>

// Parse a Vec3 from a JSON array (host-only utility).
Vec3 vec3FromJson(const nlohmann::json &a);

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

void loadScene(const std::string &filename, SceneConfig &cfg, ParticleData &pd);

// Split particles into two sets by x-coordinate (backward-compatible).
void splitAt(const ParticleData &src, float split_x, ParticleData &left,
             ParticleData &right);

// Split particles into N sets, one per domain, by (x, y, z) position.
void splitIntoN(const ParticleData &src, const std::vector<Domain> &doms,
                std::vector<ParticleData> &out);

#endif // INPUT_H
