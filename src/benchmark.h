#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "particle_device.cuh"
#include <chrono>
#include <cuda_runtime.h>
#include <fstream>
#include <string>
#include <vector>

#define BENCH_MAX_GPUS 16

// Per-step timing for each pipeline stage, with per-GPU resolution.
// GPU metrics use cudaEvent pairs; host metrics use std::chrono.
// Aggregation: "max" across GPUs for compute kernels (critical path),
// "sum" across GPUs for transfers (PCIe contention).
class Benchmark {
public:
  enum Metric {
    WALL,          // host wall clock (end-to-end step)
    HALO_PACK,     // halo exchange (host timer: pack + peer)
    ASSIGN,        // assignCell kernel
    FORCE,         // computeContactForces kernel
    INTEGRATE,     // integrate kernel
    SYNC,          // cudaDeviceSynchronize loop
    MIG_DOWNLOAD,  // CPU migration: GPU→CPU download
    MIG_MERGE,     // CPU migration: merge+split / GPU migration: total
    MIG_UPLOAD,    // CPU migration: CPU→GPU upload
    VTK,           // VTK output (download + write)
    NUM_METRICS
  };

  // One-time init: create CUDA events and open CSV file.
  // Call once after GPU count is known.
  void init(int num_gpus, const std::string &scene, long max_steps);

  // Record wall-clock start. Call at top of step loop.
  void beginStep();

  // Record a CUDA event start for metric m on GPU g.
  // Caller must have called cudaSetDevice(g) first.
  void start(Metric m, int gpu);

  // Record a CUDA event stop for metric m on GPU g.
  // Caller must have called cudaSetDevice(g) first.
  void stop(Metric m, int gpu);

  // Record a host-timer measurement (for MIG_MERGE, VTK host part).
  void startHost(Metric m);
  void stopHost(Metric m);

  // Record auxiliary counters for this step.
  void recordMigCrossed() { mig_triggers_++; cur_mig_crossed_ = true; }
  void recordGhosts(int n) { cur_ghosts_ += n; }
  void recordContacts(int n) { cur_contacts_ += n; }

  // Call at end of each step.  Writes CSV row if step % bench_interval == 0.
  // Prints human-readable block.
  void endStep(long step, int bench_interval,
               const std::vector<ParticleDevice> &pds);

  // Print final summary to stdout.
  void printFinal() const;

private:
  static const char *metricName(Metric m);
  bool isSum(Metric m) const;   // sum across GPUs vs max

  struct Acc {
    double total = 0, minv = 1e30, maxv = 0;
    long samples = 0;
  };

  int num_gpus_ = 0;
  Acc acc_[NUM_METRICS];
  cudaEvent_t ev_start_[NUM_METRICS][BENCH_MAX_GPUS];
  cudaEvent_t ev_stop_[NUM_METRICS][BENCH_MAX_GPUS];

  // Host timers
  std::chrono::steady_clock::time_point host_t0_[NUM_METRICS];
  double cur_step_ms_[NUM_METRICS];  // per-step values for display/CSV

  // Wall clock
  std::chrono::steady_clock::time_point wall_t0_;

  // Counters
  long mig_triggers_ = 0;
  long total_contacts_ = 0, total_ghosts_ = 0;
  int cur_contacts_ = 0, cur_ghosts_ = 0;
  bool cur_mig_crossed_ = false;
  long steps_ = 0;

  // CSV output
  std::string csv_path_;
  std::ofstream csv_;
  bool csv_header_written_ = false;
};

#endif // BENCHMARK_H
