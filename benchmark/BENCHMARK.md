# Benchmark Runbook

## Defaults

| Env var | Default | Override |
|---|---|---|
| `HALO` | `cpu` | `HALO=gpu` |
| `MIGRATE` | `cpu` | `MIGRATE=gpu` |
| `DYNAMIC` | `off` | `DYNAMIC=on` |

VTK output is disabled (`vtk_interval=0`) — benchmarks measure simulation,
not I/O.

## Scene

Generate once:
```bash
python3 scripts/gen_scale_crossing.py 100000 > scenes/scale100k_crossing.json
```

100,000 particles — 50% in narrow boundary strips with 60 m/s crossing
velocity, 50% lattice fill.  Radius 0.7 (halo width 1.4), ~10% packing
fraction.

## 1. Single-GPU baselines

```bash
# A100
sbatch.tinygpu --gres=gpu:a100:1 --partition=a100 scripts/bench.sh scale100k_crossing 10000 1 0 0

# RTX 3080
sbatch.tinygpu --gres=gpu:rtx3080:1 --partition=rtx3080 scripts/bench.sh scale100k_crossing 10000 1 0 0

# RTX 2080 Ti
sbatch.tinygpu --gres=gpu:rtx2080ti:1 --partition=work scripts/bench.sh scale100k_crossing 10000 1 0 0
```

## 2. Four-GPU NVLink (A100)

```bash
# A — default (CPU halo + CPU migration)
sbatch.tinygpu --gres=gpu:a100:4 --partition=a100 scripts/bench.sh scale100k_crossing 10000 4 0 0

# B — GPU halo
HALO=gpu sbatch.tinygpu --gres=gpu:a100:4 --partition=a100 scripts/bench.sh scale100k_crossing 10000 4 0 0

# C — GPU migration
MIGRATE=gpu sbatch.tinygpu --gres=gpu:a100:4 --partition=a100 scripts/bench.sh scale100k_crossing 10000 4 0 0

# D — Both
HALO=gpu MIGRATE=gpu sbatch.tinygpu --gres=gpu:a100:4 --partition=a100 scripts/bench.sh scale100k_crossing 10000 4 0 0
```

## 3. Four-GPU PCIe (RTX 3080)

```bash
# A — default (CPU halo + CPU migration)
sbatch.tinygpu --gres=gpu:rtx3080:4 --partition=rtx3080 scripts/bench.sh scale100k_crossing 10000 4 0 0

# B — GPU halo
HALO=gpu sbatch.tinygpu --gres=gpu:rtx3080:4 --partition=rtx3080 scripts/bench.sh scale100k_crossing 10000 4 0 0

# C — GPU migration
MIGRATE=gpu sbatch.tinygpu --gres=gpu:rtx3080:4 --partition=rtx3080 scripts/bench.sh scale100k_crossing 10000 4 0 0

# D — Both
HALO=gpu MIGRATE=gpu sbatch.tinygpu --gres=gpu:rtx3080:4 --partition=rtx3080 scripts/bench.sh scale100k_crossing 10000 4 0 0
```

## Compare

```bash
# Halo benefit — A100 (A→B)
python3 scripts/compare_bench.py \
  benchmark/bench_scale100k_crossing_10000_gpu4_A100_halocpu_migcpu.csv \
  benchmark/bench_scale100k_crossing_10000_gpu4_A100_halogpu_migcpu.csv

# Halo benefit — RTX 3080 (A→B)
python3 scripts/compare_bench.py \
  benchmark/bench_scale100k_crossing_10000_gpu4_RTX3080_halocpu_migcpu.csv \
  benchmark/bench_scale100k_crossing_10000_gpu4_RTX3080_halogpu_migcpu.csv

# Migration benefit — A100 (A→C)
python3 scripts/compare_bench.py \
  benchmark/bench_scale100k_crossing_10000_gpu4_A100_halocpu_migcpu.csv \
  benchmark/bench_scale100k_crossing_10000_gpu4_A100_halocpu_miggpu.csv

# Migration benefit — RTX 3080 (A→C)
python3 scripts/compare_bench.py \
  benchmark/bench_scale100k_crossing_10000_gpu4_RTX3080_halocpu_migcpu.csv \
  benchmark/bench_scale100k_crossing_10000_gpu4_RTX3080_halocpu_miggpu.csv

# Both optimisations — A100 (A→D)
python3 scripts/compare_bench.py \
  benchmark/bench_scale100k_crossing_10000_gpu4_A100_halocpu_migcpu.csv \
  benchmark/bench_scale100k_crossing_10000_gpu4_A100_halogpu_miggpu.csv

# Both optimisations — RTX 3080 (A→D)
python3 scripts/compare_bench.py \
  benchmark/bench_scale100k_crossing_10000_gpu4_RTX3080_halocpu_migcpu.csv \
  benchmark/bench_scale100k_crossing_10000_gpu4_RTX3080_halogpu_miggpu.csv

# NVLink vs PCIe (best config D vs D)
python3 scripts/compare_bench.py \
  benchmark/bench_scale100k_crossing_10000_gpu4_A100_halogpu_miggpu.csv \
  benchmark/bench_scale100k_crossing_10000_gpu4_RTX3080_halogpu_miggpu.csv

# 1-GPU vs 4-GPU scaling — A100
python3 scripts/compare_bench.py \
  benchmark/bench_scale100k_crossing_10000_gpu1_A100_halocpu_migcpu.csv \
  benchmark/bench_scale100k_crossing_10000_gpu4_A100_halogpu_miggpu.csv

# 1-GPU vs 4-GPU scaling — RTX 3080
python3 scripts/compare_bench.py \
  benchmark/bench_scale100k_crossing_10000_gpu1_RTX3080_halocpu_migcpu.csv \
  benchmark/bench_scale100k_crossing_10000_gpu4_RTX3080_halogpu_miggpu.csv
```

## Expected CSVs (11 files)

```
bench_scale100k_crossing_10000_gpu1_A100_halocpu_migcpu.csv
bench_scale100k_crossing_10000_gpu1_RTX3080_halocpu_migcpu.csv
bench_scale100k_crossing_10000_gpu1_RTX2080Ti_halocpu_migcpu.csv

bench_scale100k_crossing_10000_gpu4_A100_halocpu_migcpu.csv
bench_scale100k_crossing_10000_gpu4_A100_halogpu_migcpu.csv
bench_scale100k_crossing_10000_gpu4_A100_halocpu_miggpu.csv
bench_scale100k_crossing_10000_gpu4_A100_halogpu_miggpu.csv

bench_scale100k_crossing_10000_gpu4_RTX3080_halocpu_migcpu.csv
bench_scale100k_crossing_10000_gpu4_RTX3080_halogpu_migcpu.csv
bench_scale100k_crossing_10000_gpu4_RTX3080_halocpu_miggpu.csv
bench_scale100k_crossing_10000_gpu4_RTX3080_halogpu_miggpu.csv
```

## 4. Dynamic rebalancing (comet scene, 8× RTX 3080)

Uses `gen_comet.py` — dense cluster moving diagonally across the domain
with trailing tail, stressing load imbalance and boundary tracking.

```bash
# Generate a small comet scene for visualisation
python3 scripts/gen_comet.py 5000 > scenes/comet5k.json

# DYNAMIC=off — static equal-split domains
DYNAMIC=off HALO=gpu MIGRATE=gpu sbatch.tinygpu \
  --gres=gpu:rtx3080:8 --partition=rtx3080 \
  scripts/bench.sh comet5k 50000 8 200 0

# DYNAMIC=on — greedy domain balancing every 100 steps
DYNAMIC=on HALO=gpu MIGRATE=gpu sbatch.tinygpu \
  --gres=gpu:rtx3080:8 --partition=rtx3080 \
  scripts/bench.sh comet5k 50000 8 200 0
```

VTK output every 200 steps (250 frames) — domain boundary boxes update each
frame when `DYNAMIC=on`, showing the GPU regions tracking the comet.

```bash
# Compare dynamic vs static
python3 scripts/compare_bench.py \
  benchmark/bench_comet5k_50000_gpu8_RTX3080_halogpu_miggpu.csv \
  benchmark/bench_comet5k_50000_gpu8_RTX3080_halogpu_miggpu_dynon.csv
```

## 5. Results and current verdict

### 5.1 Coverage

The current benchmark artifacts contain:

- all 11 core CSVs listed above,
- both 8-GPU comet runs (`DYNAMIC=off` and `DYNAMIC=on`),
- one additional older RTX 3080 CSV produced with VTK enabled.

Do not use the older `-vtk.csv` artifact for performance comparisons. VTK
includes particle downloads and file I/O in the wall time.

### 5.2 Main measured result

On the 100,000-particle `scale100k_crossing` scene with VTK disabled:

| Configuration | Approx. wall time/step | Important observation |
|---|---:|---|
| 1× RTX 3080, CPU halo/migration | 0.652 ms | Force kernel dominates |
| 4× RTX 3080, GPU halo/migration | 1.225 ms | Halo exchange dominates |
| 1× A100, CPU halo/migration | 0.24 ms | Faster single-GPU force execution |
| 4× A100, GPU halo/migration | 1.25 ms | Similar distributed cost to RTX 3080 |

The 4-GPU force component falls from roughly 0.606 ms on one RTX 3080 to
roughly 0.08–0.10 ms per GPU, but the distributed run adds approximately
0.56 ms/step of halo exchange. Consequently, four GPUs do not outperform one
GPU for this workload.

### 5.3 Interpretation

The current bottleneck is no longer primarily the force kernel in the
multi-GPU runs. It is the communication path:

- halo packing and compaction,
- host/device and peer-copy orchestration,
- synchronization between GPUs,
- repeated ghost-particle movement.

The A100 and RTX 3080 four-GPU results are close because the measured cost is
dominated by this communication and packing path rather than by peak force
throughput or raw NVLink bandwidth. GPU migration is implemented, but it does
not remove the dominant halo cost.

The current scene is therefore useful for exposing communication overhead, but
it is not a workload on which the present implementation demonstrates strong
multi-GPU scaling.

### 5.4 Dynamic-decomposition caveat

The 8-GPU comet runs use VTK every 200 steps and are visualization/correctness
runs. They should not be treated as clean dynamic-performance comparisons.
Use `vtk_interval=0` for runtime comparisons between dynamic off and on.

Before relying on dynamic results, fix and validate:

- shared boundary coordinates and no gaps/overlaps,
- CPU migration with changed domain bounds,
- neighborhood-table rebuilds after cell-shape changes,
- dynamic internal VTK split geometry,
- particle-ID conservation after each rebalance.

### 5.5 Benchmark-validity caveats

The current domain factorization can produce a `1×2×2` layout for four GPUs,
while the runbook and crossing-scene assumptions describe `2×2×1`. If the
factorization is corrected, rerun the four-GPU matrix because the boundary axes
and crossing workload will change.

The RTX 2080 Ti-labelled CSV should also be verified against its Slurm log;
one earlier `work` allocation returned a different GPU model.

### 5.6 Next valid benchmark sequence

After the correctness fixes:

1. rebuild once on the target node,
2. rerun the 4-GPU A100 and RTX 3080 11-case matrix with `vtk_interval=0`,
3. run dynamic off/on with `vtk_interval=0` for performance,
4. separately run the 8-GPU comet with `vtk_interval=200` for visualization,
5. compare wall, halo, migration, ghost count, contacts, and load balance,
6. only then evaluate whether dynamic balancing improves end-to-end runtime.

The current conclusion is:

> Force computation scales across GPUs, but halo exchange dominates the
> distributed step. The next optimization target is halo packing and
> communication overlap, not further force-kernel tuning.
