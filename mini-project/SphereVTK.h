//
// Created by hans on 13.07.25.
//

#ifndef SPHEREVTK_H
#define SPHEREVTK_H
#include "Quaternion.cuh"
#include "Vec3.cuh"
#include <vector>

struct SphereVTK {
  size_t n;
  Vec3 *positions;
  Vec3 *velocities;
  Vec3 *forces;
  Quaternion *orientations;
  std::vector<float> radii;

  void resizeAll(const size_t h_n) {
    n = h_n;
    positions = new Vec3[h_n];
    velocities = new Vec3[h_n];
    forces = new Vec3[h_n];
    orientations = new Quaternion[h_n];
  }
};
#endif // SPHEREVTK_H
