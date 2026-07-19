# Implementation Plan

**Status:** Principle ✅ · Benchmarking ✅ · Crossing Guard ✅ · Domain Rebalance ✅ · GPU Migration ✅
**Remaining:** S2–S5 stress test scenes ⬜ · CUDA streams ⬜ · Async VTK ⬜ · Cell sorting ⬜

## 0. Principle

**Benchmark first, optimize second, verify third.** Every change must be
measured against a recorded baseline to prove its impact. No optimization
ships without a before/after comparison. Optimizations in this document are
**hypotheses** — the actual work is determined by benchmarks, not by this plan.

---

## 1. Implemented Optimizations & Infrastructure

### 1.1  Pre-Benchmarking Fixes ✅

| Fix | What | Status |
|---|---|---|
| Persistent contact counter | `int *d_contact_count` allocated once in `ParticleDevice`, `cudaMemset` to 0 each step — avoids per-step `cudaMalloc`/`cudaFree` | ✅ |
| VTK suppression | `vtk_interval` CLI arg (4th positional) — set > `max_steps` to disable VTK during benchmarks | ✅ |

### 1.2  Benchmarking Framework ✅

- `src/benchmark.h` / `src/benchmark.cu` — `Benchmark` class with per-GPU `cudaEvent` pairs, host wall-clock timers, running accumulators
- **Metrics tracked**: `WALL`, `HALO_PACK`, `ASSIGN`, `FORCE`, `INTEGRATE`, `SYNC`, `MIG_DOWNLOAD`, `MIG_MERGE`, `MIG_UPLOAD`, `VTK`
- **Per-step output** (every `bench_interval` steps) + **CSV logging** to `benchmark/bench_<scene>_...csv`
- **Final summary** with averages, percentages, min/max
- **Warm-up**: first 10 steps skipped to avoid cold-cache artefacts
- **Environment-variable toggles** for A/B comparison without recompiling:

| Variable | Values | Default | Effect |
|---|---|---|---|
| `HALO` | `cpu`, `gpu` | `cpu` | CPU download/filter/upload vs GPU packing + `cudaMemcpyPeer` |
| `MIGRATE` | `cpu`, `gpu` | `cpu` | CPU round-trip vs GPU packing + peer copy |
| `DYNAMIC` | `off`, `on` | `off` | Dynamic domain decomposition |

### 1.3  GPU-side Crossing Check (Cheap Migration Guard) ✅

A GPU kernel (`checkMigration`) checks per-particle whether any owned particle
has left its domain's owned region. If no particle crossed (common case),
migration is skipped entirely — avoiding the O(N) CPU download on idle steps.
Cost: ~4 µs per GPU (kernel launch + 4-byte D2H copy).

Implemented in `pack_migrate.cuh`/`.cu` and `migration.cu`.

### 1.4  Dynamic Domain Decomposition ✅

**Greedy boundary nudging** in X, Y, Z axes: every `REBALANCE_INTERVAL` (100)
steps, if `max(n_g) / mean(n_g) > 1.15`, boundaries nudge by ±`cell_size`
toward the heavier side.  Converges gradually, no oscillation.

- `domain.cu:rebalanceDomains()` — operates on all 3 axes
- Triggered by `DYNAMIC=on` env var in `main.cu`
- Cell arrays reallocated if `total_cells` exceeds pre-allocated capacity

### 1.5  Full GPU-side Migration ✅

When migration triggers, two code paths are available via `MIGRATE` env var:

- **`MIGRATE=cpu`** (default): download all → merge + split on CPU → re-upload
- **`MIGRATE=gpu`**: pack `compactStayers` + `packMigrants` kernels on GPU,
  `cudaMemcpyPeer` transfers only particles that actually moved between GPUs

Key components:
- `MigPackBuf` struct — per-GPU send + temp buffers, allocated once at startup
- `migrateParticlesGPU()` — 3-step algorithm: compact stayers → pack + peer-copy migrants → copy back to main arrays
- `migrateParticlesCPU()` — unchanged original path

---

## 2. Test Scenes

### 2.1  Existing Scenes

| Scene | Particles | Notes |
|---|---|---|
| `random20.json` | 20 | Sparse contacts; correctness only |
| `cube8.json` / `cube16.json` / `cube256.json` | 8–256 | Uniform, stationary; correctness |
| `crossing_freq.json` (S1) | 500 | Frequent boundary crossings ✅ |
| `scale10k.json` / `scale10k_crossing.json` (S6) | 10k | Scaling benchmarks ✅ |
| `scale100k_crossing.json` (S6) | 100k | Scaling benchmarks ✅ |
| `comet1k.json` / `comet5k.json` / `comet100k.json` | 1k–100k | Diagonal crossing with dense head + tail |

### 2.2  Remaining Stress Test Scenes ⬜

All generators use a fixed random seed for reproducibility.

#### S2 — All-on-One (`all_on_gpu0.json`)
**Goal**: Extreme load imbalance — all particles on GPU 0 at startup.

- Domain: 10.0 × 10.0
- Particles: 500, all placed in GPU 0's owned region (x < domain_width/N
  for an N-GPU split)
- Velocity: 5–10 m/s random (particles spread across domain over time)
- Radius: 0.3
- dt: 5e-5
- Gravity: [0, 0, 0]

Expected: GPU 0 heavily overloaded. Dynamic decomposition should
redistribute domains. Migration triggers on most early steps as particles
spread.

#### S3 — Dense Contacts (`dense_contacts.json`)
**Goal**: Maximum contact density — stress-test force kernel scaling.

- Domain: 8.0 × 8.0
- Particles: 1000, placed on a perturbed hexagonal lattice with spacing
  slightly less than 2×radius (guaranteed overlap at startup)
- Radius: 0.3
- Velocity: 0.1–0.5 m/s (small perturbations to break symmetry)
- dt: 5e-5
- Gravity: [0, -9.8, 0] (settling under gravity)
- kn: 1000

**Note**: 1000 particles of radius 0.3 in an 8×8 domain gives a packing
fraction of ~4.4 (particles overlap by ~4×). This produces very large
initial contact forces and is primarily a **numerical stability and
contact-density stress test**. Use a smaller dt or fewer particles if
the simulation diverges.

#### S4 — Large Halo (`large_halo.json`)
**Goal**: Large halo width → many ghost particles transferred.

- Domain: 20.0 × 20.0
- Particles: 200, evenly distributed
- Radius: 0.8–1.2 (halo_width = 2.4, nearly half a GPU's width with 4 GPUs)
- Velocity: 5–15 m/s random
- dt: 1e-4
- Gravity: [0, 0, 0]

Expected: Ghost particles = 30–50% of owned (vs. 5–15% for normal scenes).
Stress-tests halo exchange packing kernel and cudaMemcpyPeer bandwidth.

#### S5 — Combined Stress (`stress_all.json`)
**Goal**: All stressors simultaneously — many particles, large radius,
high velocity, clustered start.

- Domain: 20.0 × 20.0
- Particles: 1000, all in leftmost 25% of domain
- Radius: 0.5 (halo_width = 1.0)
- Velocity: 20–50 m/s random
- Mass: 0.1–2.0 (wide mass range tests harmonic-mean contact properties)
- dt: 2e-5
- Gravity: [0, -9.8, 0]
- Steps: 100,000 (longer run needed — 10k steps = 0.2s sim time, not
  enough for particles to cross the full domain)

Expected: Heavy migration early, high contacts, large ghost counts, load
imbalance. This is the "worst case" benchmark for validating all
optimisations.

---

## 3. Success Criteria

| Goal | Metric | Target |
|---|---|---|
| Migration overhead (idle steps) | avg migrate/step | < 0.1% of step time |
| Migration overhead (crossing steps) | migrate time on trigger | < 5% of step time |
| Load balance | `max(n_g) / mean(n_g)` | < 1.15 for 90% of steps |
| Halo exchange overhead | avg halo/step | < 10% of step time |
| Strong scaling efficiency | `T(1) / (N × T(N))` | > 0.7 for N=4 on cube256 |
| Benchmark accuracy | timer overhead | < 1 µs per event pair |

These targets are provisional and should be revised once baseline data exists.

---

## 4. Future Work (Deferred)

- **CUDA streams**: overlap halo peer copies with local force computation.
  Currently everything uses the default stream.
- **Async VTK**: download + write on a separate stream/thread so VTK cost
  doesn't block the simulation.
- **NVIDIA NSight profiling**: once benchmarks identify the bottleneck,
  use NSight Compute/Systems for deeper kernel-level analysis.
- **MPI backend**: for multi-node scaling beyond a single host's GPU count.
- **Cell-sorted particle layout**: replace the linked-list cell structure
  (`d_cellHeads` / `d_cellIndexes` pointer chase) with a sorted contiguous
  layout (`d_cellStart[cell]` / `d_cellEnd[cell]`).  The force kernel currently
  does random pointer-chasing through L2 cache; sorting particles by cell
  ID turns those into sequential reads that the GPU prefetcher can hide.
  Caveat: this touches every kernel (force, assign, halo pack, migration
  pack, upload) and effectively doubles the working set (sorted + unsorted
  copies must coexist during the sort pass).  Worth doing only if L2-cache
  pressure is confirmed as the single-GPU bottleneck via NSight profiling.
