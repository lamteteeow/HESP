# Implementation Plan

## 0. Principle

**Benchmark first, optimize second, verify third.** Every change must be
measured against a recorded baseline to prove its impact. No optimization
ships without a before/after comparison.

---

## 1. Benchmarking Framework  ← MUST COME FIRST

### Why First
We need to know:
- **Where time is actually spent** (not where we think it's spent).
- **How often migration actually triggers** with real workloads.
- **What the GPU-GPU halo exchange actually costs** vs. the old CPU approach.
- **Load imbalance** across GPUs for real particle distributions.

Without this, we're optimizing blind.

### Metrics

| Metric | Source | Unit |
|---|---|---|
| Step time (total) | `cudaEvent` wall clock | ms |
| Halo — pack kernel | `cudaEvent` around `packHaloParticles` | ms |
| Halo — `cudaMemcpyPeer` | `cudaEvent` around all peer copies | ms |
| Halo — total | Sum of above | ms |
| Migration — GPU→CPU download | `cudaEvent` around `cudaMemcpyDeviceToHost` | ms |
| Migration — CPU merge+split | `std::chrono` host timer | ms |
| Migration — CPU→GPU upload | `cudaEvent` around `cudaMemcpyHostToDevice` | ms |
| Migration — total | Sum of above | ms |
| Force kernel | `cudaEvent` around `computeContactForces` | ms |
| Cell assignment | `cudaEvent` around `assignCell` | ms |
| Integration | `cudaEvent` around `integrate` | ms |
| Sync | `cudaEvent` around `cudaDeviceSynchronize` | ms |
| VTK output (when triggered) | `cudaEvent` + host timer | ms |
| Particles per GPU | `pds[g].n` | count |
| Migration triggered? | bool per step | yes/no |
| Ghost particles transferred | `exchangeHalos()` return value | count |
| Contact pairs evaluated | atomic counter in force kernel | count |
| Load variance | `Var(n_g)` across GPUs | count² |

### Output Format (every `bench_interval` steps)
```
=== step 1000 =========================================
  halo        0.12 ms  (pack=0.08  peer=0.04  ghosts=234)
  assign      0.05 ms
  force       0.45 ms  (contacts=12345)
  integrate   0.03 ms
  sync        0.01 ms
  migrate     0.62 ms  (download=0.30  merge=0.02  upload=0.30) CROSSED
  vtk         2.10 ms  (frame=50)
  ──────────────────────────
  total       3.38 ms
  particles   GPU0:1024 GPU1:1023  var=0.5
```

And a **final summary** at exit:
```
=== FINAL =============================================
  steps:         10000
  avg step:      2.15 ms
  avg halo:      0.11 ms  (5.1%)
  avg force:     0.44 ms  (20.5%)
  avg migrate:   0.03 ms  (1.4%)   [triggered 12/10000 steps]
  avg vtk:       2.10 ms  (97.7%)  [every 20 steps]
  migrations:    12
  contacts/step: ~12000
  particles/GPU: mean=1023  min=980  max=1067  var=±43
```

### Implementation
- `src/benchmark.h` — `BenchStats` struct with `cudaEvent_t` pairs per metric,
  running min/max/avg accumulators, `printStep()` and `printFinal()`.
- `main.cu` — wrap each logical block with `BenchStats::start("halo")` /
  `BenchStats::stop("halo")`. Call `printStep()` every `bench_interval` steps.
  Call `printFinal()` before `return 0`.
- CLI flag: `bench_interval` via 5th argument or compile-time default (100).

### Files
- `src/benchmark.h` (new)
- `src/main.cu` (wrap sections)

---

## 2. GPU-side Crossing Check (Cheap Migration Guard)

### Current Behavior (to be measured by step 1)
Every step:
```
download ALL particles GPU→CPU  [measured: X ms]
check for crossing on CPU        [measured: Y ms]
if none crossed → return          [most steps]
else → merge + split + upload   [rare, measured: Z ms]
```

### Hypothesis
- Migration triggers on << 1% of steps for typical simulations.
- The download dominates migration cost (O(N) PCIe transfer).
- A GPU-side atomic-flag check costs O(1) PCIe transfer (4 bytes per GPU).

### Expected Impact
| Metric | Before | After | Delta |
|---|---|---|---|
| Migration time (idle step) | X ms (download) | ~4 μs (flag copy) | -99%+ |
| Migration time (crossing step) | X+Y+Z ms | X+Y+Z ms | 0% (unchanged) |
| Avg migration per step | ~X ms | ~0 ms | -99%+ |

### Implementation
1. Add `int *d_mig_flag` to `ParticleDevice` (allocated once at startup,
   capacity=1 int per GPU).
2. Add `checkMigration` kernel (no atomics — just write `1`; idempotent).
3. In `migrateParticles()`:
   ```
   launch checkMigration on each GPU
   cudaMemcpy d_mig_flag → host (4 bytes per GPU)
   if all flags == 0 → return (no download)
   // else fall through to existing code
   ```

### Files
- `src/migration.h`
- `src/particle_device.cuh` (add `d_mig_flag` field)

---

## 3. Dynamic Domain Decomposition

### Current Behavior (to be measured by step 1)
- Domains are equal geometric X-slices.
- If particles cluster (e.g., settling to bottom, grouping at center), some GPUs
  do more work.
- Load variance = `Var(n_g)` — measured by benchmarking.

### Hypothesis
- Load imbalance causes force kernel to be bottlenecked by the busiest GPU
  (all GPUs must sync before migration + VTK).
- Dynamic rebalancing reduces the max `n_g`, cutting force computation time
  by up to `(max_n - mean_n) / max_n`.

### Expected Impact
| Metric | Before | After | Delta |
|---|---|---|---|
| Max particles per GPU | N/num_gpus + δ | ≈ N/num_gpus | ~δ reduction |
| Force kernel time | limited by max GPU | limited by avg GPU | up to δ/N% reduction |
| Load variance | measured | → 0 | |

### Algorithm (2D X-only)
1. Every `rebalance_interval` steps (default 100):
   - Collect `n_g` per GPU (already available, no extra work).
   - Compute new boundaries:
     ```
     target_per_gpu = total_N / num_gpus
     cumulative = 0
     for g in 0..num_gpus-2:
         cumulative += n_g
         // Move boundary to equalize cumulative vs. g*target
         // Use linear interpolation between last particle in GPU g
         // and first particle in GPU g+1.
         new_boundary[g] = interpolate(...)
     ```
   - Apply to `Domain` structs: recalculate `owned_min/max`, `local_min/max`,
     `num_cells`, `total_cells`.
   - If `total_cells` changed: reallocate cell arrays.
2. Trigger migration to redistribute particles per new boundaries.

### Limitations
- 3D grid (MD3D): adjust X boundaries within each YZ-column only. Full 3D
  repartitioning is deferred.

### Files
- `src/domain.h` — `rebalanceDomains()` function
- `src/main.cu` — trigger + stats

---

## 4. Full GPU-side Migration (Pack + cudaMemcpyPeer)

### Prerequisites
- Steps 1 (benchmarking) — to verify GPU migration is actually faster.
- Step 2 (crossing check) — already eliminates downloads on idle steps.
- Step 3 (dynamic decomposition) — reduces migration frequency further.

### Hypothesis
When migration DOES trigger, the current CPU round-trip (download → merge →
upload) still costs O(N) PCIe transfers. A GPU-side approach using the same
pattern as `halo_exchange.h` (pack kernel + `cudaMemcpyPeer`) keeps data
on-device.

### Expected Impact
| Metric | Before | After | Delta |
|---|---|---|---|
| Migration (crossing step) | X+Y+Z ms (CPU) | ~H ms (GPU) | ~(1 - H/N)× reduction |
| GPU memory | capacity | capacity + pack bufs | + O(N) temp |

Where H = number of particles that actually crossed (usually << N).

### Algorithm (Double-Buffer Approach)
1. **Pack migrated-out** per GPU per direction → 6 output buffers (like
   `packHaloParticles` but checking `owned_min`/`owned_max`).
2. **cudaMemcpyPeer** each direction's packed strip → target GPU's receive buffer.
3. **Pack stayers** into a contiguous temp buffer (atomic counter, same kernel
   pattern).
4. **Append migrated-in** from each neighbor's receive buffer after stayers.
5. **Swap pointers** (double-buffer — already have `capacity = 2*total_N`,
   use the second half as temp).

### Complexity / Risk
- Most complex change in this plan. Introduces new packing kernels and pointer
  management.
- Only worth doing if step 2 + step 3 still leave measurable migration cost.
- Defer until benchmarks from step 1 prove it's needed.

### Files
- `src/migration.h` (rewrite)
- `src/pack_migrate.cuh` (new — pack kernels)

---

## 5. Implementation Order

```
Step 1 ──→ BASELINE MEASURED
              │
              ├─→ Step 2 (crossing check) ──→ MEASURE IMPACT
              │                                    │
              │                                    ├─→ Step 3 (dynamic decomp) ──→ MEASURE
              │                                    │                                    │
              │                                    │                                    └─→ Step 4 (GPU migration)?
              │                                    │                                          ↑
              │                                    │                                   ONLY IF step 2+3
              │                                    │                                   still leave measurable
              │                                    │                                   migration cost
              │                                    │
              │                                    └─→ Done (if migration cost ≈ 0)
              │
              └─→ Triage: migrate is X% of step time
                   if X < 1% → skip step 2
                   if X < 5% → step 2 only
                   if X > 5% → step 2 + 3
```

### File Map
| Step | New Files | Modified Files |
|---|---|---|
| 1 | `src/benchmark.h` | `src/main.cu` |
| 2 | — | `src/migration.h`, `src/particle_device.cuh` |
| 3 | — | `src/domain.h`, `src/main.cu` |
| 4 | `src/pack_migrate.cuh` | `src/migration.h` |
