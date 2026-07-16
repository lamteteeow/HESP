#include "benchmark.h"
#include "check_cuda.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>

// ── helpers ─────────────────────────────────────────────────────────────────

const char *Benchmark::metricName(Metric m) {
  switch (m) {
  case WALL:         return "wall";
  case HALO_PACK:    return "halo_pack";
  case HALO_PEER:    return "halo_peer";
  case ASSIGN:       return "assign";
  case FORCE:        return "force";
  case INTEGRATE:    return "integrate";
  case SYNC:         return "sync";
  case MIG_DOWNLOAD: return "mig_dl";
  case MIG_MERGE:    return "mig_merge";
  case MIG_UPLOAD:   return "mig_ul";
  case VTK:          return "vtk";
  default:           return "???";
  }
}

bool Benchmark::isSum(Metric m) const {
  // Sum: transfers that compete for PCIe bandwidth, and sync (host loop).
  // Max: compute kernels (critical path is slowest GPU).
  switch (m) {
  case HALO_PEER:
  case SYNC:
  case MIG_DOWNLOAD:
  case MIG_UPLOAD:  return true;
  default:          return false;
  }
}

// ── init / cleanup ──────────────────────────────────────────────────────────

void Benchmark::init(int num_gpus, const std::string &scene, long max_steps) {
  num_gpus_ = num_gpus;

  // Create CUDA events — one pair per metric per GPU, on the correct device
  for (int m = 0; m < NUM_METRICS; ++m) {
    for (int g = 0; g < num_gpus; ++g) {
      CHECK_CUDA(cudaSetDevice(g));
      CHECK_CUDA(cudaEventCreate(&ev_start_[m][g]));
      CHECK_CUDA(cudaEventCreate(&ev_stop_[m][g]));
    }
  }

  // Open CSV (with mode tags so different configs don't collide)
  const char *hm = getenv("HALO");
  const char *mm = getenv("MIGRATE");
  std::string tag;
  if (hm && strcmp(hm, "cpu") == 0) tag += "_halocpu";
  if (mm && strcmp(mm, "gpu") == 0) tag += "_miggpu";
  csv_path_ = "benchmark/bench_" + scene + "_" + std::to_string(max_steps)
            + "_gpu" + std::to_string(num_gpus) + tag + ".csv";
  csv_.open(csv_path_);
  if (!csv_) {
    fprintf(stderr, "\n*** WARNING: cannot open %s for benchmark CSV\n",
            csv_path_.c_str());
    fprintf(stderr, "*** Benchmarks will only appear in terminal output.\n\n");
  } else {
    printf("Benchmark CSV → %s\n", csv_path_.c_str());
  }
}

// ── start / stop ────────────────────────────────────────────────────────────

void Benchmark::beginStep() {
  wall_t0_ = std::chrono::steady_clock::now();
  cur_contacts_ = 0;
  cur_ghosts_ = 0;
  cur_mig_crossed_ = false;
  for (int m = 0; m < NUM_METRICS; ++m) cur_step_ms_[m] = 0;
}

void Benchmark::start(Metric m, int gpu) {
  CHECK_CUDA(cudaEventRecord(ev_start_[m][gpu]));
}

void Benchmark::stop(Metric m, int gpu) {
  CHECK_CUDA(cudaEventRecord(ev_stop_[m][gpu]));
}

void Benchmark::startHost(Metric m) {
  host_t0_[m] = std::chrono::steady_clock::now();
}

void Benchmark::stopHost(Metric m) {
  auto t1 = std::chrono::steady_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - host_t0_[m]).count();
  cur_step_ms_[m] = ms;
  Acc &a = acc_[m];
  a.total += ms;
  a.minv = std::min(a.minv, ms);
  a.maxv = std::max(a.maxv, ms);
  a.samples++;
}

// ── end-of-step ─────────────────────────────────────────────────────────────

void Benchmark::endStep(long step, int bench_interval,
                        const std::vector<ParticleDevice> &pds) {
  steps_++;

  // Resolve pending CUDA events and accumulate.
  // Only query GPU-timer metrics (ASSIGN, FORCE, INTEGRATE); host-timer
  // metrics and HALO_PEER are skipped.
  double metric_ms[NUM_METRICS] = {};

  static const int gpu_metrics[] = {ASSIGN, FORCE, INTEGRATE};
  for (int mi = 0; mi < 3; ++mi) {
    int m = gpu_metrics[mi];

    double agg = 0;
    int samples = 0;
    for (int g = 0; g < num_gpus_; ++g) {
      // Check if event was recorded (by testing if the stop event exists and
      // has been recorded — crude but effective: query elapsed time and
      // ignore zero/negative).
      float ms = 0;
      cudaError_t err = cudaEventElapsedTime(&ms, ev_start_[m][g],
                                             ev_stop_[m][g]);
      if (err != cudaSuccess) continue;   // event pair not recorded

      samples++;
      if (isSum(static_cast<Metric>(m)))
        agg += ms;
      else
        agg = std::max(agg, static_cast<double>(ms));
    }

    if (samples > 0) {
      Acc &a = acc_[m];
      a.total += agg;
      a.minv = std::min(a.minv, agg);
      a.maxv = std::max(a.maxv, agg);
      a.samples++;
      metric_ms[m] = agg;
    }
  }

  // Wall clock
  auto t1 = std::chrono::steady_clock::now();
  double wall_ms = std::chrono::duration<double, std::milli>(t1 - wall_t0_).count();
  {
    Acc &a = acc_[WALL];
    a.total += wall_ms;
    a.minv = std::min(a.minv, wall_ms);
    a.maxv = std::max(a.maxv, wall_ms);
    a.samples++;
    metric_ms[WALL] = wall_ms;
  }

  // Accumulate counters
  total_contacts_ += cur_contacts_;
  total_ghosts_  += cur_ghosts_;

  // Copy host-timer per-step values into metric_ms
  metric_ms[HALO_PACK]    = cur_step_ms_[HALO_PACK];
  metric_ms[SYNC]         = cur_step_ms_[SYNC];
  metric_ms[MIG_DOWNLOAD] = cur_step_ms_[MIG_DOWNLOAD];
  metric_ms[MIG_MERGE]    = cur_step_ms_[MIG_MERGE];
  metric_ms[MIG_UPLOAD]   = cur_step_ms_[MIG_UPLOAD];
  metric_ms[VTK]          = cur_step_ms_[VTK];

  // Clear any CUDA errors left by cudaEventElapsedTime on unrecorded events
  cudaGetLastError();

  // ── print step block ──────────────────────────────────────────────────
  if (bench_interval > 0 && step % bench_interval == 0) {
    printf("\n=== step %ld =========================================\n", step);

    auto pr = [&](Metric m, const char *extra = nullptr) {
      if (acc_[m].samples == 0) return;
      printf("  %-10s %7.2f ms", metricName(m), metric_ms[m]);
      if (extra) printf("  (%s)", extra);
      printf("\n");
    };

    pr(WALL);
    {
      double halo_total = metric_ms[HALO_PACK] + metric_ms[HALO_PEER];
      printf("  %-10s %7.2f ms  (pack=%.2f  peer=%.2f  ghosts=%d)\n",
             "halo", halo_total, metric_ms[HALO_PACK],
             metric_ms[HALO_PEER], cur_ghosts_);
    }
    pr(ASSIGN);
    printf("  %-10s %7.2f ms  (contacts=%d)\n",
           metricName(FORCE), metric_ms[FORCE], cur_contacts_);
    pr(INTEGRATE);
    pr(SYNC);

    {
      double mig_total = metric_ms[MIG_DOWNLOAD] + metric_ms[MIG_MERGE]
                       + metric_ms[MIG_UPLOAD];
      const char *tag = cur_mig_crossed_ ? "CROSSED" : "idle";
      printf("  %-10s %7.2f ms  (dl=%.2f  merge=%.2f  ul=%.2f) %s\n",
             "migrate", mig_total, metric_ms[MIG_DOWNLOAD],
             metric_ms[MIG_MERGE], metric_ms[MIG_UPLOAD], tag);
    }

    if (metric_ms[VTK] > 0)
      pr(VTK);

    printf("  --------------------------\n");
    printf("  particles  ");
    for (int g = 0; g < num_gpus_; ++g)
      printf("GPU%d:%zu ", g, pds[g].n);
    // load variance
    if (num_gpus_ > 1) {
      double mean = 0, m2 = 0;
      for (int g = 0; g < num_gpus_; ++g) {
        double d = static_cast<double>(pds[g].n);
        mean += d;
        m2 += d * d;
      }
      mean /= num_gpus_;
      double var = m2 / num_gpus_ - mean * mean;
      printf(" var=%.1f", var);
    }
    printf("\n");
  }

  // ── write CSV row ─────────────────────────────────────────────────────
  if (csv_.is_open() && (bench_interval <= 0 || step % bench_interval == 0)) {
    if (!csv_header_written_) {
      csv_ << "step,wall_ms,halo_pack_ms,halo_peer_ms,halo_ghosts,"
              "assign_ms,force_ms,force_contacts,integrate_ms,"
              "sync_ms,mig_dl_ms,mig_merge_ms,mig_ul_ms,mig_crossed,"
              "vtk_ms";
      for (int g = 0; g < num_gpus_; ++g)
        csv_ << ",n_gpu" << g;
      csv_ << ",load_var\n";
      csv_header_written_ = true;
    }

    csv_ << step << ","
         << std::fixed << std::setprecision(3)
         << metric_ms[WALL] << ","
         << metric_ms[HALO_PACK] << ","
         << metric_ms[HALO_PEER] << ","
         << cur_ghosts_ << ","
         << metric_ms[ASSIGN] << ","
         << metric_ms[FORCE] << ","
         << cur_contacts_ << ","
         << metric_ms[INTEGRATE] << ","
         << metric_ms[SYNC] << ","
         << metric_ms[MIG_DOWNLOAD] << ","
         << metric_ms[MIG_MERGE] << ","
         << metric_ms[MIG_UPLOAD] << ","
         << (cur_mig_crossed_ ? 1 : 0) << ","
         << metric_ms[VTK];

    double mean = 0, m2 = 0;
    for (int g = 0; g < num_gpus_; ++g) {
      double d = static_cast<double>(pds[g].n);
      mean += d;
      m2 += d * d;
      csv_ << "," << pds[g].n;
    }
    mean /= num_gpus_;
    double var = (num_gpus_ > 1) ? (m2 / num_gpus_ - mean * mean) : 0.0;
    csv_ << "," << std::fixed << std::setprecision(1) << var << "\n";
    csv_.flush();  // ensure data hits disk even if crash later
  }
}

// ── final summary ───────────────────────────────────────────────────────────

void Benchmark::printFinal() const {
  printf("\n=== FINAL =============================================\n");
  printf("  steps:         %ld\n", steps_);

  auto pr = [&](Metric m, bool show_pct = true) {
    const Acc &a = acc_[m];
    if (a.samples == 0) return;
    double avg = a.total / a.samples;
    double pct = (acc_[WALL].samples > 0)
                   ? 100.0 * a.total / acc_[WALL].total : 0;
    printf("  avg %-10s %7.2f ms", metricName(m), avg);
    if (show_pct) printf("  (%4.1f%%)", pct);
    printf("  [min=%.3f  max=%.3f  n=%ld]\n", a.minv, a.maxv, a.samples);
  };

  pr(WALL, false);
  {
    double halo_total = acc_[HALO_PACK].total + acc_[HALO_PEER].total;
    long halo_n = acc_[HALO_PACK].samples;
    if (halo_n > 0) {
      double halo_avg = halo_total / halo_n;
      double pct = (acc_[WALL].total > 0) ? 100.0 * halo_total / acc_[WALL].total : 0;
      printf("  avg %-10s %7.2f ms  (%4.1f%%)  [pack=%.3f  peer=%.3f]\n",
             "halo", halo_avg, pct,
             acc_[HALO_PACK].total / halo_n,
             acc_[HALO_PEER].total / halo_n);
    }
  }
  pr(ASSIGN);
  pr(FORCE);
  pr(INTEGRATE);
  pr(SYNC);

  {
    double mig_total = acc_[MIG_DOWNLOAD].total + acc_[MIG_MERGE].total
                     + acc_[MIG_UPLOAD].total;
    long mig_n = std::max({acc_[MIG_DOWNLOAD].samples, acc_[MIG_MERGE].samples,
                           acc_[MIG_UPLOAD].samples});
    if (mig_n > 0) {
      double mig_avg = mig_total / mig_n;
      double pct = (acc_[WALL].total > 0) ? 100.0 * mig_total / acc_[WALL].total : 0;
      printf("  avg %-10s %7.2f ms  (%4.1f%%)  [dl=%.3f  merge=%.3f  ul=%.3f]",
             "migrate", mig_avg, pct,
             acc_[MIG_DOWNLOAD].samples > 0 ? acc_[MIG_DOWNLOAD].total / acc_[MIG_DOWNLOAD].samples : 0,
             acc_[MIG_MERGE].samples    > 0 ? acc_[MIG_MERGE].total    / acc_[MIG_MERGE].samples    : 0,
             acc_[MIG_UPLOAD].samples   > 0 ? acc_[MIG_UPLOAD].total   / acc_[MIG_UPLOAD].samples   : 0);
      printf("  [crossed %ld/%ld steps]\n", mig_triggers_, steps_);
    }
  }

  if (acc_[VTK].samples > 0) {
    double vtk_per_out = acc_[VTK].total / acc_[VTK].samples;
    double vtk_amort  = acc_[VTK].total / steps_;
    double pct_out = (acc_[WALL].total > 0) ? 100.0 * acc_[VTK].total / acc_[WALL].total : 0;
    double pct_amort = (acc_[WALL].total > 0) ? 100.0 * vtk_amort * steps_ / acc_[WALL].total : 0;
    printf("  avg %-10s %7.2f ms  (%4.1f%%)  [per-output=%.2f ms  amort=%.3f ms  frames=%ld]\n",
           "vtk", vtk_per_out, pct_out, vtk_per_out,
           acc_[VTK].total / steps_, acc_[VTK].samples);
  }

  if (steps_ > 0) {
    printf("  avg contacts/step: ~%ld\n",
           steps_ > 0 ? total_contacts_ / steps_ : 0);
    printf("  avg ghosts/step:  ~%ld\n",
           steps_ > 0 ? total_ghosts_ / steps_ : 0);
  }

  if (!csv_path_.empty())
    printf("\n  benchmark CSV → %s\n", csv_path_.c_str());
}
