# Implementation Plan

**Status:** Steps 0–2 ✅ · Step 3 ⬜ · Step 4 ✅ · S2–S5 scenes ⬜

## 0. Principle

**Benchmark first, optimize second, verify third.** Every change must be
measured against a recorded baseline to prove its impact. No optimization
ships without a before/after comparison. Optimizations in this document are
**hypotheses** — the actual work is determined by benchmarks, not by this plan.

---

## 1. Pre-Benchmarking Fixes  <- MUST COME FIRST

Before any benchmarking, fix a known problem in the hot path that would
contaminate all measurements:

### 1.1 Persistent contact counter

`main.cu` currently allocates and frees a 4-byte integer **every step**
(lines 203-216):

```cpp
int *d_cnt = nullptr;
cudaMalloc(&d_cnt, sizeof(int));   // ALLOC EVERY STEP
// ... kernel ...
cudaFree(d_cnt);                    // FREE EVERY STEP
```

`cudaMalloc` can trigger heavyweight driver synchronisation. Replace with a
persistent counter allocated once at startup (add `int *d_contact_count` to
`ParticleDevice`, allocate in `upload()`, memset to 0 each step, free in
`freeParticleDevice()`).

### 1.2 Separate VTK path from benchmarked path

VTK writes dominate step time (downloading all particles, file I/O). When
benchmarking, use a large `vtk_interval` (e.g., 10× `max_steps`) to suppress
VTK entirely. Benchmark output should report VTK cost **separately** from the
per-step average so it doesn't distort optimisation targets.

---

## 2. Benchmarking Framework

### 2.1 Why First

We need to know:
- **Where time is actually spent** (not where we think it's spent).
- **How often migration actually triggers** with real workloads.
- **What the GPU-GPU halo exchange actually costs**.
- **Load imbalance** across GPUs for real particle distributions.

Without this, we're optimising blind.

### 2.2 Multi-GPU Timing Strategy

CUDA events are per-device. Timing across multiple GPUs requires care:

- **Per-metric, per-GPU** event pairs: `cudaEvent_t start[METRIC][MAX_GPU]`,
  `stop[METRIC][MAX_GPU]`. Record on the device doing the work.
- **Wall-clock host timer** (`std::chrono::steady_clock`) around the entire
  step loop for end-to-end step time. This is the ground truth.
- **Critical-path semantics**: where GPUs must synchronise (before migration,
  before VTK), report the **max** across GPUs, not the sum. Where GPUs work
  independently, report per-GPU times.
- **`Sync` metric**: a `cudaDeviceSynchronize` on the straggler GPU measures
  idle time of the **host**, not sync overhead. Sync time is `max_gpu_finish -
  min_gpu_finish`. Track via host timer between the last kernel launch on each
  GPU and its synchronisation return.
- **Warm-up**: skip the first N steps (default 10) to avoid cold-cache and
  lazy-initialisation artefacts.

### 2.3 Metrics

| Metric | Source | Unit | Aggregation |
|---|---|---|---|
| Step time (total, end-to-end) | `std::chrono` host timer | ms | — |
| Halo - pack kernel | `cudaEvent` per GPU | ms | max across GPUs |
| Halo - `cudaMemcpyPeer` | `cudaEvent` per GPU | ms | max across GPUs |
| Halo - total | Sum of above | ms | — |
| Force kernel | `cudaEvent` per GPU | ms | max across GPUs |
| Cell assignment | `cudaEvent` per GPU | ms | max across GPUs |
| Integration | `cudaEvent` per GPU | ms | max across GPUs |
| Sync (GPU straggler wait) | host timer | ms | — |
| Migration - GPU→CPU download | `cudaEvent` per GPU | ms | sum across GPUs |
| Migration - CPU merge+split | `std::chrono` host timer | ms | — |
| Migration - CPU→GPU upload | `cudaEvent` per GPU | ms | sum across GPUs |
| Migration - total | Sum of above | ms | — |
| VTK output (when triggered) | `cudaEvent` + host timer | ms | — |
| Particles per GPU | `pds[g].n` | count | — |
| Migration triggered? | bool per step | yes/no | — |
| Ghost particles transferred | `exchangeHalos()` count × 2 (send+recv) | count | — |
| Contact pairs evaluated | persistent atomic counter | count | — |
| Load variance | `Var(n_g)` across GPUs | count² | — |

### 2.4 Output Format

**Per-step** (every `bench_interval` steps, human-readable):
```
=== step 1000 =========================================
  wall        2.55 ms
  halo        0.12 ms  (pack=0.08  peer=0.04  ghosts=234)
  assign      0.05 ms
  force       0.45 ms  (contacts=12345)
  integrate   0.03 ms
  sync        0.01 ms
  migrate     0.62 ms  (download=0.30  merge=0.02  upload=0.30) CROSSED
  vtk         2.10 ms  (frame=50)
  --------------------------
  particles   GPU0:1024 GPU1:1023  var=0.5
```

**Per-step machine-readable** (appended to `bench_<scene>.csv`):
```csv
step,wall_ms,halo_pack_ms,halo_peer_ms,halo_ghosts,assign_ms,force_ms,force_contacts,integrate_ms,sync_ms,mig_dl_ms,mig_merge_ms,mig_ul_ms,mig_crossed,vtk_ms,vtk_frame,n_gpu0,n_gpu1,n_gpu2,n_gpu3,load_var
1000,2.55,0.08,0.04,234,0.05,0.45,12345,0.03,0.01,0.30,0.02,0.30,1,2.10,50,1024,1023,1022,1025,0.5
```

**Final summary** at exit:
```
=== FINAL =============================================
  steps:              10000
  avg wall:           2.15 ms
  avg halo:           0.11 ms  (5.1%)
  avg force:          0.44 ms  (20.5%)
  avg migrate:        0.03 ms  (1.4%)   [triggered 12/10000]
  avg vtk (amort):    0.11 ms  (5.1%)   [triggered every 20 steps]
  avg vtk (per-out):  2.10 ms
  avg contacts/step:  ~12000
  particles/GPU:      mean=1023  min=980  max=1067  var=+/-43
```

**Note on percentages**: percentages in the final summary use the **amortised**
average (total time / total steps), so VTK cost is spread across all steps.
This gives an accurate picture of where total runtime goes. Per-output VTK cost
is also reported separately.

### 2.5 Implementation

- `src/benchmark.h` — `BenchStats` struct with per-GPU `cudaEvent_t` pairs,
  running min/max/avg accumulators, `start(metric, gpu)`, `stop(metric, gpu)`,
  `printStep()`, `printFinal()`. Host-wall timer independent of CUDA events.
  CSV output to `bench_<scene>.csv`.
- `src/benchmark.cu` — implementation (non-trivial methods).
- `main.cu` — wrap each logical block with `BenchStats::start` / `stop`.
  Call `printStep()` every `bench_interval` steps. Call `printFinal()` before
  `return 0`.
- CLI: `bench_interval` as 5th argument (default 100). Set `bench_interval=0`
  to disable step-level output (final summary only).

### 2.6 Files
- `src/benchmark.h` (new)
- `src/benchmark.cu` (new)
- `src/main.cu` (wrap sections)
- `src/particle_device.cuh` (add `d_contact_count` field)
- `src/particle_host.h` / `src/particle_host.cu` (allocate/free persistent counter)
- `Makefile` (add new `.cu` files)

### 2.7 Runtime Toggles

Environment variables control algorithm variants for A/B comparison without
recompiling:

| Variable | Values | Default | Effect |
|---|---|---|---|
| `HALO` | `gpu`, `cpu` | `gpu` | GPU packing + `cudaMemcpyPeer` vs CPU download/filter/upload |
| `MIGRATE` | `cpu`, `gpu` | `cpu` | CPU round-trip vs GPU packing (GPU path: Step 4) |
| `DYNAMIC` | `off`, `on` | `off` | Dynamic domain decomposition (Step 3) |

CSV filenames include mode tags (e.g., `_halocpu`) so different configurations
produce separate output files without overwriting. The startup banner prints
the active configuration: `Modes: halo=gpu  migrate=cpu  dynamic=off`.

---

## 3. GPU-side Crossing Check (Cheap Migration Guard)

### 3.1 Current Behavior

Every step:
```
download ALL particles GPU→CPU  [measured: X ms per GPU]
check for crossing on CPU        [measured: Y ms]
if none crossed → return         [most steps]
else → merge + split + upload   [rare, measured: Z ms]
```

The download is O(N) PCIe transfer and dominates idle-step migration cost.

### 3.2 Hypothesis

A GPU-side atomic-flag check reduces idle-step migration cost from O(N) PCIe
to O(1) PCIe (4 bytes per GPU). Since the download is already paid on crossing
steps, the check adds negligible overhead to those steps.

### 3.3 Expected Impact

| Metric | Before | After | Delta |
|---|---|---|---|
| Migration time (idle step) | X ms (download all) | ~4 µs (flag copy) | -99%+ |
| Migration time (crossing step) | X+Y+Z ms | X+Y+Z + ε ms | ~0% |
| Avg migration per step | ~X ms | ~(X × trigger_rate) ms | -99%+ for low trigger rates |

### 3.4 Implementation

1. Add `int *d_mig_flag` to `ParticleDevice` — one int per GPU, allocated at
   startup, zeroed each step.
2. Add `checkMigration` kernel: each thread checks whether its owned particle
   is outside `owned_min`/`owned_max`; if so, writes `1` to `d_mig_flag`
   (idempotent — no atomics needed since all threads write the same value).
3. Reset `d_mig_flag` to 0 via `cudaMemset` before each kernel launch.
4. In `migrateParticles()`:
   ```
   launch checkMigration on each GPU
   cudaMemcpy d_mig_flag → host flag (4 bytes per GPU)
   if all flags == 0 → return
   // else fall through to existing download + merge + upload
   ```
5. Pass `owned_min`/`owned_max` as kernel arguments (not `__constant__`
   memory) so they work correctly with dynamic decomposition (Section 4).

### 3.5 Decision: Always Do This

This optimisation costs ~20 lines of code, has negligible runtime overhead
(a few µs of kernel launch + 4-byte D2H copy per GPU), and its benefit scales
with N. There is no scenario where it makes things worse. **Always implement
after benchmarking baseline.**

### 3.6 Files
- `src/migration.h` / `src/migration.cu` — add GPU guard
- `src/particle_device.cuh` — add `d_mig_flag` field
- `src/particle_host.h` / `src/particle_host.cu` — allocate/free flag
- `src/pack_migrate.cuh` (new) — `checkMigration` kernel

---

## 4. Dynamic Domain Decomposition

### 4.1 Current Behavior (to be measured by Section 2)

- Domains are equal geometric slices.
- If particles cluster, some GPUs do more work → force kernel bottlenecked by
  the busiest GPU (all GPUs sync before migration + VTK).
- Load variance = `Var(n_g)` — measured by benchmarking.

### 4.2 Hypothesis

Dynamic rebalancing reduces `max(n_g)`, cutting force computation time by up
to `(max_n - mean_n) / max_n`. Impact is proportional to load imbalance.

### 4.3 Algorithm

**Trigger**: every `rebalance_interval` steps (default 100), if load variance
exceeds a threshold (e.g., `max_n > 1.2 × mean_n`). Use hysteresis: don't
rebalance again until variance has been above threshold for 2 consecutive
checks, and don't rebalance if the improvement would be < 5%.

**X-axis rebalancing**:
1. Collect `n_g` per GPU (already available).
2. Sort particle positions on each GPU and gather to host (required for
   precise boundary placement — this is the main cost).
3. Compute new boundaries via **binary search** over the global sorted
   position list: find X coordinates where the cumulative particle count
   crosses `g × target_per_gpu` for `g = 1..num_gpus-1`.
4. Apply to `Domain` structs: update `owned_min.x`/`owned_max.x`,
   recalculate `local_min`/`local_max`, `num_cells`, `total_cells`.
5. Allocate cell arrays for the **maximum possible** `total_cells` at startup
   to avoid mid-run `cudaFree`/`cudaMalloc`. If the new `total_cells` still
   fits within the pre-allocated size, reuse the buffers.
6. Recompute and re-upload the cell neighborhood table (`d_nb`) only if
   `total_cells` actually grew beyond the pre-allocated maximum.
7. Trigger migration to redistribute particles per new boundaries.

**3D (MD3D, deferred)**: adjust X boundaries within each YZ-column. Since
halo exchange assumes axis-aligned rectangular owned regions, boundaries
must be aligned across columns. This requires a global constraint that
limits flexibility. Full 3D repartitioning is future work.

### 4.4 Expected Impact

| Metric | Before | After | Delta |
|---|---|---|---|
| Max particles per GPU | mean + delta | ≈ mean | ~delta reduction |
| Force kernel time | limited by max | limited by avg | up to delta/N % |
| Load variance | measured | → near 0 | |

### 4.5 Decision Gate

**Only implement if** benchmarks from Section 2 show load variance
`max(n_g) / mean(n_g) > 1.15` for a significant fraction of steps.

### 4.6 Interaction with Section 3

Dynamic decomposition **increases** migration frequency (moving boundaries
forces particle redistribution). Section 3 (GPU crossing check) should be
implemented **before** Section 4 to keep idle-step migration cost low.

### 4.7 Files
- `src/domain.h` / `src/domain.cu` — `rebalanceDomains()` function
- `src/main.cu` — trigger + stats

---

## 5. Full GPU-side Migration (Pack + cudaMemcpyPeer) ✅ IMPLEMENTED

### 5.1 Prerequisites

- Section 2 (benchmarking) — to verify GPU migration is actually faster.
- Section 3 (crossing check) — already eliminates downloads on idle steps.
- Section 4 (dynamic decomposition) — may affect migration patterns.

### 5.2 Hypothesis

When migration DOES trigger, the current CPU round-trip still costs O(N)
PCIe transfers. A GPU-side approach using pack kernels + `cudaMemcpyPeer`
keeps data on-device, transferring only particles that actually crossed.

### 5.3 Expected Impact

| Metric | Before | After | Delta |
|---|---|---|---|
| Migration (crossing step) | O(N) PCIe | O(H) GPU-GPU | ~(1 − H/N)× reduction |
| GPU memory | capacity | capacity + pack bufs | + O(N) temp |

Where H = number of particles that actually crossed (usually << N).

### 5.4 Algorithm Outline

1. **Compact stayers**: each GPU packs particles that stayed in-region into a
   per-GPU temp buffer (contiguous prefix). Original data is preserved for
   step 2.
2. **Pack migrants + cudaMemcpyPeer**: for each (src, dst) pair, launch
   `packMigrants` kernel on src to pack particles that moved into dst's owned
   region, then `cudaMemcpyPeer` to append them after dst's compacted stayers
   in dst's temp buffer.
3. **Copy temp → main arrays**: each GPU copies its temp buffer back to the
   main particle arrays and updates `n`, `n_total`.
4. **All fields transferred**: positions, velocities, masses, radii, kn,
   gamma_n, gamma_t, mu, ids. Forces are recomputed each step and not transferred.

### 5.5 Implementation Notes

- Two new kernels in `pack_migrate.cuh`/`pack_migrate.cu`: `packMigrants`
  and `compactStayers` (in addition to the existing `checkMigration`).
- `MigPackBuf` struct holds per-GPU send buffer and temp workspace, allocated
  once at startup in `main.cu`.
- Dispatched behind `MIGRATE=gpu` env var. CPU path (`MIGRATE=cpu`) unchanged.
- Crossing check (Section 3) still runs first on both paths — idle steps skip
  migration entirely.

### 5.6 Decision Gate

**Only implement if** after Sections 3 and 4, migration cost on crossing steps
exceeds 5% of step time for relevant workloads.

### 5.7 Files
- `src/migration.h` / `src/migration.cu` — rewritten
- `src/pack_migrate.cuh` / `src/pack_migrate.cu` — pack + compact kernels
- `src/main.cu` — buffer allocation + updated call site

---

## 6. Implementation Order

```
                    ┌─────────────────────────────┐
                    │ Step 0: Pre-benchmark fixes │
                    │ (persistent contact counter,│
                    │  VTK suppression for bench) │
                    └─────────────┬───────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │ Step 1: Benchmarking        │
                    │ (baseline data collected)   │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────┼───────────────┐
                    │             │               │
                    ▼             ▼               ▼
          ┌──────────────┐ ┌───────────┐  ┌──────────────┐
          │ Migrate cost │ │ Load      │  │ Halo cost    │
          │ vs step time │ │ variance  │  │ vs step time │
          └──────┬───────┘ └─────┬─────┘  └──────────────┘
                 │               │
                 ▼               ▼
          ┌─────────────────────────────────────┐
          │ Step 2: GPU crossing check          │
          │ ALWAYS do this — no downside,       │
          │ O(1) overhead, massive saving on    │
          │ idle steps for any N > small        │
          └─────────────────┬───────────────────┘
                            │
                            ▼
          ┌─────────────────────────────────────┐
          │ REMEASURE                           │
          └─────────────────┬───────────────────┘
                            │
               ┌────────────┼─────────────┐
               ▼                          ▼
     ┌──────────────────┐     ┌──────────────────────┐
     │ Migrate on       │     │ Load imbalance       │
     │ crossing steps   │     │ > 15%?               │
     │ > 5% of step?    │     │ → Step 3: Dynamic    │
     │ → Step 4: GPU    │     │   decomposition      │
     │   migration?     │     └──────────┬───────────┘
     └──────────────────┘                │
                               ┌────────┼────────┐
                               ▼                 ▼
                    ┌────────────────┐  ┌──────────────────┐
                    │ REMEASURE      │  │ Migrate on       │
                    │                │  │ crossing steps   │
                    │                │  │ still > 5%?      │
                    │                │  │ → Step 4: GPU    │
                    │                │  │   migration      │
                    └────────────────┘  └──────────────────┘
```

**Summary of decision gates:**

| Check | Threshold | Action |
|---|---|---|
| Step 1 complete? | Baseline collected | → always proceed to Step 2 |
| Step 2 implement? | Always | No downside, unconditional |
| Step 3 implement? | Load variance > 15% | Dynamic decomposition |
| Step 4 implement? | Migration cost on crossing steps > 5% | GPU-side migration |

### 6.1 File Map

| Step | New Files | Modified Files |
|---|---|---|
| 0 | — | `src/main.cu`, `src/particle_device.cuh`, `src/particle_host.h`, `src/particle_host.cu` |
| 1 | `src/benchmark.h`, `src/benchmark.cu` | `src/main.cu`, `Makefile` |
| 2 | `src/pack_migrate.cuh`, `src/pack_migrate.cu` | `src/migration.h`, `src/migration.cu`, `src/particle_device.cuh`, `src/particle_host.h`, `src/particle_host.cu` |
| 3 | — | `src/domain.h`, `src/domain.cu`, `src/main.cu` |
| 4 | — | `src/migration.h`, `src/migration.cu`, `src/pack_migrate.cuh`, `src/pack_migrate.cu`, `src/main.cu` |

---

## 7. Test Cases

### 7.1 Existing Scenes (Limitations)

| Scene | Particles | Stress Type | Limitation |
|---|---|---|---|
| `random20.json` | 20 | Sparse contacts | No migration stress |
| `cube8/16/256.json` | 8–256 | Uniform, stationary | No load imbalance, rare crossings |

These are useful for correctness checks but insufficient for performance
benchmarking.

### 7.2 Stress Test Scenes

All generators use a fixed random seed for reproducibility.

#### S1 — Crossing Frequency (`crossing_freq.json`)
**Goal**: Frequent boundary crossings to stress-test migration guard.

- Domain: 20.0 × 10.0
- Particles: 500, placed in narrow strips straddling each domain boundary
  (e.g., for 4 GPUs, place ~125 particles in [4.8, 5.2] around each boundary)
- Velocity: 2–5 m/s in ±X (particles oscillate across boundaries)
- Radius: 0.15 (small — minimise contacts, maximise crossings)
- dt: 5e-4
- Gravity: [0, 0, 0]
- Steps: 50,000

Expected: migration triggers on 5–15% of steps. Particles near boundaries
cross back and forth continuously.

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

Expected: thousands of contacts per step. Benchmarks force kernel
performance under high contact load.

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

#### S6 — Scale Test
**Goal**: Measure scaling behaviour with particle count.

- Scenes with 1k, 10k, 100k particles, uniform random distribution
- Use existing `gen_random.py` / `gen_random3d.py` patterns
- Fixed domain size, fixed dt, fixed step count
- Run on 1, 2, 4 GPUs

Expected: Identifies the crossover where communication cost exceeds
computation benefit. Determines whether the current approach is
compute-bound or bandwidth-bound at scale.

### 7.3 Scene Generators

| Generator | Output Scene |
|---|---|
| `scripts/gen_crossing_freq.py` | `scenes/crossing_freq.json` |
| `scripts/gen_clustered.py` | `scenes/all_on_gpu0.json` |
| `scripts/gen_dense_contacts.py` | `scenes/dense_contacts.json` |
| `scripts/gen_large_halo.py` | `scenes/large_halo.json` |
| `scripts/gen_stress_all.py` | `scenes/stress_all.json` |

All generators accept a `--seed` flag (default: fixed constant) for
deterministic output.

### 7.4 Benchmark Protocol

For each test scene, run with 1, 2, 4 GPUs:
```
./build/md2d scenes/<scene>.json <steps> <num_gpus> <vtk_interval> [bench_interval]
```

| Configuration | Purpose |
|---|---|
| 1 GPU | Baseline (no migration, no halo, no comm) |
| 2 GPUs | Minimal multi-GPU (1 boundary) |
| 4 GPUs | Realistic multi-GPU (3 boundaries) |

For benchmarking runs, set `vtk_interval` larger than `max_steps` to
suppress VTK output entirely.

Collect from CSV output:
- Avg step time, halo time, migration time
- Migration trigger frequency
- Ghost particle count
- Contact count
- Load variance

Compare before/after each optimisation step to quantify improvement.

---

## 8. Success Criteria

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

## 9. Future Work (Deferred)

- **CUDA streams**: overlap halo peer copies with local force computation.
  Currently everything uses the default stream.
- **Async VTK**: download + write on a separate stream/thread so VTK cost
  doesn't block the simulation.
- **Full 3D dynamic decomposition**: repartition all three axes, not just X.
- **NVIDIA NSight profiling**: once benchmarks identify the bottleneck,
  use NSight Compute/Systems for deeper kernel-level analysis.
- **MPI backend**: for multi-node scaling beyond a single host's GPU count.
