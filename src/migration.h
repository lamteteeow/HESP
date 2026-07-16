#ifndef MIGRATION_H
#define MIGRATION_H

#include "domain.h"
#include "particle_device.cuh"
#include <cstdlib>
#include <cstring>
#include <vector>

class Benchmark;

// Particle migration across GPU grid.
// Reads MIGRATE env var:
//   MIGRATE=cpu (default): CPU download → merge → split → upload
//   MIGRATE=gpu (future):  GPU packing + cudaMemcpyPeer
void migrateParticles(std::vector<ParticleDevice> &pds,
                      const std::vector<Domain> &doms, size_t total_n,
                      Benchmark *bench = nullptr);

inline const char *migrateMode() {
  const char *v = getenv("MIGRATE");
  return (v && strcmp(v, "gpu") == 0) ? "gpu" : "cpu";
}

#endif // MIGRATION_H
