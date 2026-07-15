# Implementation Plan

## 1. GPU-GPU Migration

### Current State
`migration.h` uses full CPU round-trip **every step**:
```
1. Download ALL owned particles GPU→CPU (every step, even if nothing crossed)
2. CPU-side check for particle crossing
3. If crossed: CPU merge + re-split + free+re-upload to GPUs
```

The halo exchange is already GPU-side (pack kernel + `cudaMemcpyPeer`), but
migration is the remaining bottleneck.

### Phase 1a: Cheap GPU-side Crossing Check
**Goal:** Eliminate the download on steps where nothing crossed (99.9%+ of steps).

Add a `checkMigration` kernel that uses a single atomic flag per GPU:
```cuda
__global__ void checkMigration(const size_t n, const Vec3 *positions,
                               Vec3 owned_min, Vec3 owned_max, int *d_flag) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const Vec3 p = positions[i];
  if (p.x < owned_min.x || p.x >= owned_max.x ||
      p.y < owned_min.y || p.y >= owned_max.y ||
      p.z < owned_min.z || p.z >= owned_max.z)
    *d_flag = 1;  // any thread sets it, no atomic needed (write 1 is idempotent)
}
```

In `migrateParticles()`:
```
1. Launch checkMigration kernel on each GPU (no download)
2. cudaMemcpy 4-byte flag GPU→CPU per GPU
3. If no GPU has flag set → return immediately (common case, zero particle data transferred)
4. Otherwise fall through to existing CPU round-trip (rare case)
```

**Files changed:** `migration.h` only (+ new `.cuh` or inline kernel).

**Benchmark impact:** Migration cost drops from O(N) per step to O(1) for >99.9% of steps.

---

### Phase 1b: Full GPU-side Migration
**Goal:** Eliminate CPU involvement entirely for migration (like halo exchange).

**Approach:** Pack + copy + reassemble, using the same pattern as `halo_exchange.h`.

#### Data Structures
Reuse `HaloPackBuf` from `halo_exchange.h`, or add a `MigratePackBuf`:
```cpp
struct MigratePackBuf {
  int   *d_count[6];  // one atomic counter per direction
  Vec3  *d_pos[6];    // packed output per direction
  Vec3  *d_vel[6];
  float *d_rad[6], *d_kn[6], *d_gn[6], *d_gt[6], *d_mu[6];
  int   *d_ids[6];
  size_t cap_per_dir; // max_n per direction
};
```

6 directions: `{right, left, top, bottom, front, back}`.

#### Step-by-step Algorithm

For each GPU g:

**A. Pack migrated-out particles** (GPU kernel, like `packHaloParticles` but
checking against owned bounds per-direction):
```cuda
__global__ void packMigrateOut(const size_t n, const Vec3 *positions, ...,
                               Vec3 owned_min, Vec3 owned_max,
                               float halo_w,
                               int *d_count[6], Vec3 *d_out_pos[6], ...);
```

Each thread checks if its particle is in a boundary strip and atomically adds
it to the appropriate direction's output buffer. A particle in the right strip
goes to `d_count[RIGHT]` / `d_out_pos[RIGHT]`, etc.

**B. cudaMemcpyPeer** packed buffers from source GPU to destination GPU:
- GPU g's RIGHT strip → GPU (g's right neighbor)'s migration receive buffer
- GPU g's LEFT strip → GPU (g's left neighbor)'s migration receive buffer
- Same for top/bottom/front/back

**C. Reassemble on target GPU:**
1. Compact stayers: use the same pack mechanism — pack particles that stayed
   into a contiguous buffer (atomic counter), then copy back over the owned region.
   Or use a prefix-sum (scan) to compact in-place without a temp buffer.
2. Append migrated-in particles from each neighbor's strip after the compacted
   stayers.
3. Update `n` and `n_total`.

**D. Alternative simpler approach (double-buffer):**
- Allocate a second set of particle arrays (`d_positions2`, etc.) same size as original.
- Pack stayers to the front of temp arrays (atomic counter).
- Append migrated-in from all neighbors.
- Swap pointers (`d_positions ↔ d_positions2`), free old arrays.
- No prefix-sum needed, no in-place compaction complexity.
- Cost: 2× GPU memory for particle arrays (already at `capacity = total_n * 2`).

**Files changed:** `migration.h`, new `pack_migrate.cuh`.

---

## 2. Dynamic Domain Decomposition

### Goal
Balance particle count across GPUs to minimize load variance. Currently, domains
are equal geometric X-slices, which causes imbalance if particles cluster.

### Approach: Greedy Equal-Area → Greedy Equal-Particle

#### Cost Model
- Particle count per GPU ≈ computational load (contact forces dominate).
- Target: minimize variance of `n_g` across GPUs.

#### Algorithm (2D X-only, default mode)

1. **Periodically** (every `rebalance_interval` steps, default 100):
   - Gather `n_g` = owned particle count from each GPU (4 bytes each, cheap).
   - Total N = sum(n_g).
   - Target per GPU = N / num_gpus.

2. **Compute new boundaries** using a cumulative-sum sweep:
   - Sort particle x-coordinates (or use histogram).
   - For g in 0..num_gpus-2:
     - Place `owned_max[g]` at the x-coordinate of the `(g+1) * target`-th particle.
   - Edge GPUs keep global boundaries.

3. **Apply new boundaries** to `Domain` structs:
   - Recalculate `owned_min`, `owned_max` for each GPU.
   - Recalculate `local_min`, `local_max` (with halo padding).
   - Recalculate `num_cells`, `total_cells`.

4. **Reallocate cell arrays** if `total_cells` changed:
   - Free old `d_cellHeads`, `d_cellTails`, `d_cellIndexes`.
   - Allocate new ones (capacity unchanged, only cell grid changes).

5. **Trigger migration** immediately after boundaries change to redistribute
   particles to new owners.

#### Statistics Tracking
Track per-step:
- `n_g` per GPU (particle count)
- Load variance: `sum((n_g - mean)^2) / num_gpus`
- Boundary positions
- Print summary at rebalance steps.

#### Limitations (3D mode)
- For 3D grid (nx×ny×nz), adjusting all boundaries dynamically is harder.
- Start with 2D-only adaptive decomposition.
- 3D can: a) keep fixed geometry, or b) adjust X-only within each YZ-column.

### Files changed
- `domain.h` — new `rebalanceDomains()` function.
- `main.cu` — periodic rebalance trigger + stats printing.

---

## 3. Benchmarking Framework

### Metrics to Track

| Metric | How | Unit |
|---|---|---|
| Step time (total) | `cudaEvent` elapsed | ms |
| Halo exchange time | `cudaEvent` around `exchangeHalos()` | ms |
| Migration time | `cudaEvent` around `migrateParticles()` | ms |
| Force kernel time | `cudaEvent` around `computeContactForces` | ms |
| Cell assignment time | `cudaEvent` around `assignCell` | ms |
| Integration time | `cudaEvent` around `integrate` | ms |
| Sync time | `cudaEvent` around `cudaDeviceSynchronize` | ms |
| VTK output time | `cudaEvent` around VTK block | ms |
| Particles per GPU | `pds[g].n` | count |
| Migration triggers | count steps where particles crossed | count |
| Ghost particles | return value of `exchangeHalos` | count |
| Contact count | atomic counter in force kernel | count |
| Load variance | `Var(n_g)` across GPUs | count² |

### Implementation

```cpp
struct BenchmarkStats {
  double total_step_ms, halo_ms, migrate_ms, force_ms, assign_ms, integrate_ms;
  double sync_ms, vtk_ms;
  long migrations_triggered, steps;
  // Running averages
  void update(...);
  void printSummary();  // called on exit or every N steps
};
```

- Use `cudaEvent_t` pairs for GPU kernel timing.
- Use `std::chrono::high_resolution_clock` for CPU-side timing (migration
  CPU round-trip).
- Print per-step timing every `bench_interval` steps (default 100).
- Print final summary on exit.

### Output Format
```
=== step 100 ===
  halo:      0.12 ms  (pack=0.08  peer=0.04)
  assign:    0.05 ms
  force:     0.45 ms  (contacts=12345)
  integrate: 0.03 ms
  sync:      0.01 ms
  migrate:   0.00 ms  (no crossing)
  vtk:       2.10 ms
  total:     2.76 ms
  particles: GPU0:1024 GPU1:1023  var=0.5
```

### Files changed
- New `benchmark.h` — timing utilities + stats struct.
- `main.cu` — wrap sections with timing calls.

---

## 4. Implementation Order

| Step | Task | Dependencies | Files |
|---|---|---|---|
| 1 | Add benchmark framework | none | `benchmark.h` (new), `main.cu` |
| 2 | GPU-side crossing check (Phase 1a) | none | `migration.h` |
| 3 | Dynamic domain decomposition | step 2 | `domain.h`, `main.cu` |
| 4 | Full GPU-side migration (Phase 1b) | step 2 | `migration.h`, `pack_migrate.cuh` (new) |
| 5 | Benchmark + validate | steps 1-4 | run tests, tune intervals |

Step 1 (benchmarking) should come first so we can measure the impact of each
subsequent change.

Step 2 (cheap crossing check) gives the biggest win with the least risk — it
keeps the existing fallback and just avoids downloading data on idle steps.

Step 4 (full GPU migration) is the logical conclusion but can be deferred if
step 2 + periodic rebalance already eliminate most migration overhead.
