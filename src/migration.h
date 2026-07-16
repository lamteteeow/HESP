#ifndef MIGRATION_H
#define MIGRATION_H

#include "domain.h"
#include "particle_device.cuh"
#include <vector>

class Benchmark;

// Particle migration across an nx×ny×nz GPU grid.
// If bench is non-null, records per-phase host timers.
void migrateParticles(std::vector<ParticleDevice> &pds,
                      const std::vector<Domain> &doms, size_t total_n,
                      Benchmark *bench = nullptr);

#endif // MIGRATION_H
