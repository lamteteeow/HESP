#ifndef MIGRATION_H
#define MIGRATION_H

#include "domain.h"
#include "particle_device.cuh"
#include "vec3.cuh"
#include <cstdlib>
#include <cstring>
#include <vector>

class Benchmark;

// ── GPU migration pack buffers (allocated once at startup) ─────────────────

// Per-GPU buffers for GPU-side migration.
// Send buffer: packs particles leaving this GPU (max_owned capacity).
// Temp buffer: workspace for compacting stayers + receiving migrants
//   (capacity-sized for shared arrays, capacity/2-sized for owned-only).
struct MigPackBuf {
  // Send buffer (max_owned capacity — at most all owned particles leave)
  int   *d_send_count;
  Vec3  *d_send_pos,  *d_send_vel;
  float *d_send_mass, *d_send_rad, *d_send_kn, *d_send_gn;
  float *d_send_gt,   *d_send_mu;
  int   *d_send_ids;

  // Temp workspace for compact + receive (capacity-sized for shared,
  // capacity/2-sized for owned-only arrays)
  int   *d_tmp_count;
  Vec3  *d_tmp_pos,  *d_tmp_vel;
  float *d_tmp_mass, *d_tmp_rad, *d_tmp_kn, *d_tmp_gn;
  float *d_tmp_gt,   *d_tmp_mu;
  int   *d_tmp_ids;

  size_t max_owned;  // = total_n (capacity / 2)
  size_t capacity;   // = total_n * 2
};

void allocMigPackBuf(MigPackBuf &b, size_t total_n);
void freeMigPackBuf(MigPackBuf &b);

// ── Migration entry point ──────────────────────────────────────────────────

// Particle migration across GPU grid.
// Reads MIGRATE env var:
//   MIGRATE=cpu (default): CPU download → merge → split → upload
//   MIGRATE=gpu:           GPU packing + cudaMemcpyPeer
void migrateParticles(std::vector<ParticleDevice> &pds,
                      const std::vector<Domain> &doms,
                      std::vector<MigPackBuf> &mig_bufs,
                      size_t total_n, Benchmark *bench = nullptr);

inline const char *migrateMode() {
  const char *v = getenv("MIGRATE");
  return (v && strcmp(v, "gpu") == 0) ? "gpu" : "cpu";
}

#endif // MIGRATION_H
