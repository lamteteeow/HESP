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
