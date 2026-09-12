# Implementation Plan

**Principle:** benchmark first, optimize second, verify third.

**Current status:**

| Area | Status | Meaning |
|---|---|---|
| Persistent contact counter | ✅ Implemented | Allocated once and reset per force step |
| Benchmarking and CSV output | ✅ Implemented | Needs one timing-validity cleanup before final publication |
| GPU-side crossing guard | ✅ Implemented | Needs race-safety cleanup and validation |
| GPU halo exchange | ✅ Implemented | Capacity and correctness stress tests still pending |
| GPU migration | ✅ Implemented prototype | CPU migration is not yet dynamic-domain safe |
| Dynamic decomposition | ⚠️ Prototype | Boundary continuity and cell-table updates must be fixed |
| Dynamic VTK output | ⚠️ Partial | Halo boxes use current bounds, internal split lines are still static |
| 4-GPU layout | ⚠️ Documentation/code mismatch | Current factorization must be made deterministic |
| S2–S5 stress scenes | ⬜ Not implemented | Only S1, S6, and comet scenes currently exist |

No feature should be described as production-ready until it passes particle
conservation, domain-continuity, multi-GPU, and benchmark-validity checks.

---

## 0. Principle and scope

The project is a CUDA multi-GPU 3D DEM simulator. The near-term goals are:

1. Preserve a correct CPU baseline.
2. Measure halo, migration, force, synchronization, and VTK costs separately.
3. Compare CPU and GPU halo/migration paths using reproducible CSV artifacts.
4. Validate dynamic decomposition on the intended 8-GPU `2×2×2` case.
5. Identify the communication/computation crossover instead of assuming that
   more GPUs must be faster.

All optimization claims in this document are hypotheses until supported by a
recorded before/after run.

---

## 1. Implemented infrastructure

### 1.1 Persistent contact counter ✅

`ParticleDevice` contains a persistent `d_contact_count` allocation. The force
loop clears it with `cudaMemset` before launching the contact kernel and copies
back the result afterward. This removes the old per-step `cudaMalloc`/`cudaFree`
overhead.

Relevant files:

- `src/particle_device.cuh`
- `src/particle_host.cu`
- `src/force_kernels.cu`
- `src/main.cu`

### 1.2 VTK and benchmark sentinels ✅

The command-line arguments are:

```text
./md3d <scene.json> [max_steps] [num_gpus] [vtk_interval] [bench_interval]
```

The actual sentinel behavior is:

- `vtk_interval=0`: disable VTK completely.
- `bench_interval=0`: suppress human-readable per-step blocks, but still write
  every warm-up-excluded step to the CSV.
- `bench_interval>0`: print and write only steps divisible by the interval.

Using a value greater than `max_steps` does **not** disable VTK because step 0
still satisfies `0 % interval == 0`.

`scripts/bench.sh` uses `vtk_interval=0` and `bench_interval=0` for normal
performance runs.

### 1.3 Benchmarking framework ✅, validation cleanup pending

Implemented in:

- `src/benchmark.h`
- `src/benchmark.cu`
- `src/main.cu`

Tracked metrics:

- end-to-end wall time,
- halo exchange time and ghost count,
- cell assignment,
- force kernel time and contact count,
- integration,
- synchronization,
- CPU migration download/merge/upload,
- GPU migration total in the migration-merge field,
- VTK output time,
- particles per GPU,
- migration-trigger flag,
- particle-count variance.

The first 10 steps are excluded from final averages. Human-readable progress is
printed at 0%, each 10%, and 100% of the run.

CSV filenames include:

```text
scene, max steps, GPU count, concise GPU name,
HALO mode, MIGRATE mode, and _dynon when dynamic mode is enabled
```

Example:

```text
benchmark/bench_scale100k_crossing_10000_gpu4_RTX3080_halogpu_miggpu.csv
```

Required benchmark cleanup:

- Track whether each CUDA event was recorded in the current step. At present,
  an event for a GPU with no work can retain its previous-step timing.
- Keep `load_var` documented as particle-count variance. It is not the same as
  `max(n_g)/mean(n_g)`.
- Measure and report dynamic rebalance time separately if it is included in
  wall time.

### 1.4 Runtime modes ✅

Environment variables are interpreted as follows:

| Variable | Values | Default | Effect |
|---|---|---|---|
| `HALO` | `cpu`, `gpu` | `cpu` | CPU download/filter/upload or GPU pack + peer copy |
| `MIGRATE` | `cpu`, `gpu` | `cpu` | CPU round-trip or GPU pack + peer copy |
| `DYNAMIC` | `off`, `on` | `off` | Enable dynamic boundary nudging |

The defaults are applied by `scripts/bench.sh`, and the binary also treats any
value other than the enabling value as the default CPU/off mode.

---

## 2. GPU-side crossing guard ✅, validation cleanup pending

`checkMigration` scans owned particles on each GPU and sets a host-visible flag
when a particle lies outside that GPU's owned region. If no flag is set,
`migrateParticles()` returns without downloading all particles.

This removes the old O(N) CPU download on idle migration steps. The measured
cost is approximately a kernel launch plus a 4-byte device-to-host copy per GPU.

Relevant files:

- `src/pack_migrate.cuh`
- `src/pack_migrate.cu`
- `src/migration.cu`
- `src/particle_device.cuh`

Before finalizing this item:

- replace the unsynchronized `*d_flag = 1` write with `atomicOr` or
  `atomicExch`,
- test empty GPUs and all-on-one distributions,
- verify that the guard uses the current dynamic bounds after every rebalance.

---

## 3. Halo exchange ✅, stress validation pending

### 3.1 CPU halo path

`HALO=cpu` downloads owned particles, filters directional strips on the CPU,
and uploads the required fields to neighboring GPUs. This remains the default
baseline path.

### 3.2 GPU halo path

`HALO=gpu` launches `packHaloParticles` on each source GPU and transfers packed
strips with `cudaMemcpyPeer`. Single-GPU runs return immediately because there
are no neighbors.

Relevant files:

- `src/halo_exchange.h`
- `src/halo_exchange.cu`
- `src/pack_halo.cuh`
- `src/pack_halo.cu`

### 3.3 Capacity validation required

A single `HaloPackBuf` is reused for all directional strips. Its current
allocation is based on `max_n`, while the sum of packed strips can exceed
`max_n` when particles lie near multiple boundaries or the halo width is large.

Before large-halo and comet stress runs:

- allocate enough capacity for all possible directional strips, or
- add explicit per-direction capacity checks and a safe growth path.

The main particle arrays and halo buffers must be tested with halos wider than
one cell and with particles near edges/corners.

---

## 4. GPU-side migration ✅ prototype

### 4.1 CPU path

`MIGRATE=cpu` performs:

1. download owned particles from every GPU,
2. merge on the host,
3. split by domain ownership,
4. free and re-upload each GPU's arrays.

This is the baseline path for comparisons.

### 4.2 GPU path

`MIGRATE=gpu` performs:

1. `compactStayers` on each source GPU,
2. `packMigrants` for destination ownership regions,
3. `cudaMemcpyPeer` for packed particle fields,
4. copy the compacted destination buffers back into the main arrays.

Buffers are allocated once in `main.cu` using `MigPackBuf`.

Relevant files:

- `src/migration.h`
- `src/migration.cu`
- `src/pack_migrate.cuh`
- `src/pack_migrate.cu`

### 4.3 Dynamic-domain incompatibility to fix

The CPU migration path currently computes ownership with fixed equal geometric
splits instead of `doms[g].owned_min`/`owned_max`. Therefore:

```text
DYNAMIC=on MIGRATE=cpu
```

is not yet a valid configuration after the first boundary movement.

The CPU split must use the current `Domain` bounds, with one unambiguous rule
for particles exactly on a shared boundary. Then test that CPU and GPU migration
produce the same owner assignment and particle ID set.

### 4.4 Migration validation

For every migration mode, verify after each migration event:

- total owned particle count remains exactly N,
- particle IDs are unique,
- no ID is lost,
- all particles lie inside exactly one owned region,
- the next halo exchange sees valid owned counts,
- CPU and GPU paths produce equivalent ownership for the same state.

---

## 5. Dynamic domain decomposition ⚠️ prototype

### 5.1 Current implementation

`DYNAMIC=on` calls `rebalanceDomains()` every 100 steps. It receives the current
owned particle count per GPU and examines aggregate counts across each internal
X, Y, and Z split.

A split moves by one `cell_size` toward the heavier side when:

```text
abs(left_count - right_count) / max(left_count, right_count) > 0.15
```

This is **not** the same as checking `max(n_g)/mean(n_g) > 1.15`.

Relevant files:

- `src/domain.h`
- `src/domain.cu`
- `src/main.cu`

### 5.2 Current limitations

The implementation currently changes `owned_min` and `owned_max` fields on
neighboring domains independently. Clamping uses `halo_width` as a minimum
owned width. With a large halo, one side can clamp before the other, producing
an overlap or gap at a shared boundary.

This is a correctness blocker. The implementation must instead maintain shared
split coordinates:

```text
x_split[0..nx]
y_split[0..ny]
z_split[0..nz]
```

Each domain should be reconstructed from the corresponding split coordinates,
then `local_min`, `local_max`, `num_cells`, and `total_cells` should be derived.
This guarantees exact coverage and shared-boundary equality.

### 5.3 Cell-table update requirements

`d_nb` depends on the complete `num_cells.x/y/z` shape. It must be rebuilt when
any cell dimension changes, even if `total_cells` does not grow. It must also be
re-uploaded after a shrink or a same-capacity shape change.

The current `main.cu` path only rebuilds when `total_cells > cell_caps[g]`.
This is insufficient for dynamic domains.

The safe update sequence is:

1. compare old and new `num_cells`,
2. grow or reuse `d_cellHeads` capacity as needed,
3. rebuild and upload `d_nb` whenever the shape changes,
4. clear the active `d_cellHeads` range before cell assignment,
5. migrate particles using the new shared bounds.

### 5.4 Dynamic VTK requirements

Dynamic VTK output currently writes a new boundary file per frame, but
`src/vtk_output.cu` still draws internal split lines from the original equal
geometric spacing. Halo boxes use current domain bounds, so the file is only
partly dynamic.

The boundary writer must derive all internal boundary geometry from the current
`Domain` objects or the shared split arrays. A valid dynamic visualization must
show:

- outer domain bounds,
- current owned-region boundaries,
- optional halo regions,
- current GPU ownership in particle files.

Static mode should write one `domain_boundary.vtk`; dynamic mode should write
numbered boundary files without overwriting previous frames.

### 5.5 Dynamic validation matrix

Before marking dynamic decomposition complete:

- run 2 GPUs with a small comet,
- run 4 GPUs with a small comet,
- run 8 GPUs with `comet1k.json` or `comet5k.json`,
- use `DYNAMIC=off` and `DYNAMIC=on`,
- check shared boundaries numerically after every rebalance,
- check particle conservation after migration,
- inspect numbered VTK boundary frames,
- compare CPU and GPU migration modes.

---

## 6. Domain layout and benchmark validity ⚠️

### 6.1 Required deterministic factorization

The intended layouts are:

| GPU count | Intended layout |
|---:|---|
| 1 | `1×1×1` |
| 2 | `1×1×2` or documented equivalent |
| 4 | `2×2×1` |
| 8 | `2×2×2` |

The current `factorGrid3D()` tie-breaking can produce `1×2×2` for four GPUs,
while the README and benchmark assumptions describe `2×2×1`. This changes which
axes contain boundaries and can invalidate the intended X/Y crossing workload.

Resolve this by either:

1. making factorization deterministic and preferring the documented axis order,
or
2. updating every scene, benchmark, and README statement to the actual layout.

The preferred option is to fix factorization and then verify the generated
4-GPU domain map at startup.

### 6.2 Domain invariants

Every domain update must satisfy:

- owned regions cover the global domain exactly,
- adjacent owned bounds are equal,
- no owned regions overlap,
- each owned width is at least the configured minimum,
- local bounds stay within global bounds,
- neighbor IDs remain reciprocal,
- `num_cells` and neighborhood tables match local bounds.

---

## 7. Measured benchmark status

The current artifact set contains the required 11 core CSVs plus two 8-GPU
comet CSVs (dynamic off/on). There is also an older VTK-enabled RTX 3080 CSV
that must not be used for performance comparisons.

Observed behavior on the 100k crossing workload:

- single RTX 3080: approximately `0.652 ms/step`,
- four RTX 3080: approximately `1.225 ms/step` with GPU halo/migration,
- single A100: approximately `0.24 ms/step`,
- four A100: approximately `1.25 ms/step` with GPU halo/migration,
- four-GPU force time falls substantially, but halo exchange is approximately
  `0.56 ms/step` and dominates the distributed run.

Current conclusion:

> The workload demonstrates force-kernel scaling but not end-to-end strong
> scaling. Halo packing, synchronization, and peer communication dominate the
> four-GPU runs.

The 8-GPU comet runs use VTK every 200 steps and are visualization/correctness
runs, not clean performance comparisons.

---

## 8. Test scenes

### 8.1 Existing scenes

| Scene | Particles | Use |
|---|---:|---|
| `random20.json` | 20 | Basic correctness |
| `cube8.json`, `cube16.json`, `cube256.json` | 8–256 | Small correctness checks |
| `crossing_freq.json` | 500 | Frequent crossings |
| `scale10k.json` | 10,000 | Scale baseline |
| `scale10k_crossing.json` | 10,000 | Scale plus crossings |
| `scale100k_crossing.json` | 100,000 | Main benchmark workload |
| `comet1k.json`, `comet5k.json`, `comet100k.json` | 1k–100k | Dynamic/VTK visualization |

### 8.2 S1 — Crossing frequency ✅

`crossing_freq.json` contains 500 particles in a 3D `20×10×10` domain with
small radius and high velocity. Its actual boundary axis depends on the
resolved GPU layout and must be checked after the factorization fix.

### 8.3 S2 — All-on-one ⬜

Generate a true 3D scene that places particles inside the actual initial
`GPU0` owned box, not merely inside `x < domain_width/N`. The generator must
query or reproduce the deterministic domain layout.

Purpose:

- severe initial load imbalance,
- early migration,
- dynamic rebalance response,
- empty-GPU benchmark timing.

### 8.4 S3 — Dense contacts ⬜

Use a 3D domain, for example `8×8×8`, and calculate packing fraction as:

```text
N × (4πr³/3) / domain_volume
```

The previous 2D-style packing estimate of approximately 4.4 was incorrect for
this 3D simulator. Overlap and numerical stability still need to be controlled
with `dt`, stiffness, damping, and particle spacing.

Purpose:

- force-kernel scaling,
- contact-counter validation,
- high-contact timing without confusing it with migration performance.

### 8.5 S4 — Large halo ⬜

Use a 3D domain such as `20×20×20`, with radii selected so the halo occupies a
substantial fraction of an owned width. Do not assume a fixed 30–50% ghost
ratio; record the actual ghost/owned ratio from the CSV.

Purpose:

- halo pack capacity,
- ghost count correctness,
- CPU versus GPU halo exchange,
- peer-copy behavior under large strips.

### 8.6 S5 — Combined stress ⬜

Use a 3D clustered scene with high velocity, large radius, mixed masses, and
controlled timestep. Record whether the scene remains numerically stable before
using it for performance claims.

Purpose:

- simultaneous migration, halo, force, and imbalance stress,
- end-to-end correctness under the worst workload.

### 8.7 S6 — Scale tests ✅ partially

Existing scale scenes cover 10k and 100k particles. The current recorded matrix
covers single-GPU and four-GPU comparisons on A100 and RTX 3080, plus one
RTX 2080 Ti-labelled single-GPU artifact whose actual hardware should be verified
from its Slurm log.

The 8-GPU comet run is a visualization test and should not be treated as a
scale-performance result because it uses only 5k particles and periodic VTK.

---

## 9. Validation and benchmark protocol

### 9.1 Build and run

Build once on the target GPU node before submitting parallel jobs:

```bash
module load cuda/12.8.0
make clean && make
```

After the binary exists, `scripts/bench.sh` skips rebuilding so multiple jobs do
not race through `make clean`/`make`.

### 9.2 Performance runs

Use:

```text
vtk_interval=0
bench_interval=0
```

Run all CPU/GPU halo and migration variants with identical scene, step count,
GPU count, and hardware. Exclude any CSV produced with VTK enabled from timing
comparisons.

### 9.3 Correctness checks

For small and stress scenes, verify:

- exact particle count in every VTK frame,
- unique and conserved particle IDs,
- no NaN/Inf positions or velocities,
- no particles outside the global domain unless wall behavior explicitly allows it,
- every owned particle belongs to exactly one domain,
- halo counts are non-negative and within allocated capacity,
- CPU and GPU halo/migration paths agree on owned particle IDs.

### 9.4 Dynamic-specific checks

At every rebalance:

- log the split coordinates,
- log old/new owned bounds,
- log whether any boundary moved,
- log particle counts before and after migration,
- assert shared-boundary equality,
- assert global coverage without gaps or overlaps,
- rebuild neighborhood tables when cell dimensions change.

### 9.5 Benchmark metrics to compare

Use `scripts/compare_bench.py` for:

- CPU versus GPU halo,
- CPU versus GPU migration,
- one GPU versus four GPUs,
- A100 versus RTX 3080,
- dynamic off versus dynamic on only when VTK settings are identical.

Dynamic VTK runs answer whether the visualization and boundary tracking work;
clean `vtk_interval=0` runs answer whether dynamic balancing improves runtime.

---

## 10. Success criteria

These targets remain provisional until the correctness fixes and timing cleanup
are complete.

| Goal | Metric | Target/status |
|---|---|---|
| Particle conservation | total unique IDs | exactly N every checked frame |
| Domain coverage | shared bounds/gaps/overlaps | no violations |
| Idle migration overhead | migration guard cost | measured separately; no full download |
| Crossing migration overhead | migration time on trigger | compare CPU and GPU paths |
| Load balance | max/mean and variance | report both; target <1.15 max/mean for 90% of steps |
| Halo overhead | halo time / wall time | reduce from current dominant ~0.56 ms component |
| Strong scaling | `T(1)/(N×T(N))` | measure; do not assume cube256 target is meaningful |
| Benchmark timing | per-event overhead | measure before claiming <1 µs |
| Dynamic VTK | boundary frame geometry | coordinates change with domain splits |

The current 100k results do **not** meet the intended end-to-end strong-scaling
outcome: four GPUs are slower than one because halo exchange dominates.

---

## 11. Immediate corrective work

These are correctness and measurement tasks, not optional optimizations:

1. Rework dynamic boundaries around shared X/Y/Z split coordinates.
2. Make CPU migration use current dynamic domain bounds.
3. Rebuild `d_nb` whenever any cell dimension changes.
4. Make dynamic VTK internal boundaries use current domain coordinates.
5. Make 4-GPU factorization deterministic and align scenes/docs with it.
6. Add per-step CUDA-event validity flags to the benchmark.
7. Add halo-buffer capacity checks and a safe growth path.
8. Use an atomic operation for the migration flag.
9. Run small 2/4/8-GPU conservation tests before new benchmark claims.

---

## 12. Future work (deferred)

### 12.1 GPU histogram / weighted quantile rebalancing

The current greedy nudge is intentionally conservative, but it only uses
aggregate particle counts and can require many intervals to respond.

A future superior approach is:

1. build a GPU-side per-cell load histogram,
2. optionally weight cells by contacts and halo cost rather than particle count,
3. use prefix sums to identify candidate load quantiles,
4. update shared X/Y/Z split coordinates,
5. cap movement to one cell per rebalance for stability,
6. migrate once after applying the new splits.

This is deferred until the current prototype is correct and measured. A host
sort of all particle positions is not the preferred implementation because its
transfer/sort cost can erase the balancing gain and it does not naturally solve
coordinated `2×2×2` partitioning.

### 12.2 Cell-sorted particle layout

Replace linked-list cell traversal (`d_cellHeads`, `d_cellTails`, and
`d_cellIndexes`) with a sorted contiguous layout (`cellStart`/`cellEnd`). This
may improve force-kernel locality and L2 behavior, but it is a hypothesis that
requires Nsight evidence. It affects force, assignment, halo packing, migration,
and upload, and temporarily increases working-set size.

### 12.3 CUDA streams

Overlap independent local work with halo peer copies after dependencies are
explicitly modeled. The current implementation uses the default stream and
mostly serial host orchestration.

### 12.4 Asynchronous VTK

Move particle downloads and file writes off the simulation critical path using
separate streams/threads. This is for visualization throughput, not clean
benchmark timing.

### 12.5 Nsight profiling

Use Nsight Systems/Compute after the correctness fixes to quantify:

- halo pack atomic contention,
- peer-copy bandwidth,
- synchronization gaps,
- force-kernel memory locality,
- event and host-timer overhead.

### 12.6 MPI backend

Support multi-node scaling beyond the GPUs in one host after the single-host
communication path is understood.
