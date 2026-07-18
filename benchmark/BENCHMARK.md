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
fraction.  ~0.5-1.0 ms/step on 1 GPU depending on architecture.

## 1. Single-GPU baselines

```bash
sbatch.tinygpu --gres=gpu:a100:1 --partition=a100 scripts/bench.sh scale100k_crossing 5000 1 0 0
sbatch.tinygpu --gres=gpu:rtx3080:1 --partition=rtx3080 scripts/bench.sh scale100k_crossing 5000 1 0 0
```

## 2. Four-GPU NVLink (A100)

```bash
# A — default (CPU halo + CPU migration)
sbatch.tinygpu --gres=gpu:a100:4 --partition=a100 scripts/bench.sh scale100k_crossing 5000 4 0 0

# B — GPU halo
HALO=gpu sbatch.tinygpu --gres=gpu:a100:4 --partition=a100 scripts/bench.sh scale100k_crossing 5000 4 0 0

# C — GPU migration
MIGRATE=gpu sbatch.tinygpu --gres=gpu:a100:4 --partition=a100 scripts/bench.sh scale100k_crossing 5000 4 0 0

# D — Both
HALO=gpu MIGRATE=gpu sbatch.tinygpu --gres=gpu:a100:4 --partition=a100 scripts/bench.sh scale100k_crossing 5000 4 0 0
```

## 3. Four-GPU PCIe (RTX 3080)

Same four variants with `--gres=gpu:rtx3080:4 --partition=rtx3080`.

## Compare

```bash
# Halo benefit (cpu vs gpu)
python3 scripts/compare_bench.py \
  benchmark/bench_scale100k_crossing_5000_gpu4_A100_halocpu_migcpu.csv \
  benchmark/bench_scale100k_crossing_5000_gpu4_A100_halogpu_migcpu.csv

# Migration benefit
python3 scripts/compare_bench.py \
  benchmark/bench_scale100k_crossing_5000_gpu4_A100_halocpu_migcpu.csv \
  benchmark/bench_scale100k_crossing_5000_gpu4_A100_halocpu_miggpu.csv

# Both optimisations
python3 scripts/compare_bench.py \
  benchmark/bench_scale100k_crossing_5000_gpu4_A100_halocpu_migcpu.csv \
  benchmark/bench_scale100k_crossing_5000_gpu4_A100_halogpu_miggpu.csv

# NVLink vs PCIe
python3 scripts/compare_bench.py \
  benchmark/bench_scale100k_crossing_5000_gpu4_A100_halogpu_miggpu.csv \
  benchmark/bench_scale100k_crossing_5000_gpu4_RTX3080_halogpu_miggpu.csv

# 1-GPU vs 4-GPU scaling (A100)
python3 scripts/compare_bench.py \
  benchmark/bench_scale100k_crossing_5000_gpu1_A100_halocpu_migcpu.csv \
  benchmark/bench_scale100k_crossing_5000_gpu4_A100_halogpu_miggpu.csv

# 1-GPU vs 4-GPU scaling (RTX 3080)
python3 scripts/compare_bench.py \
  benchmark/bench_scale100k_crossing_5000_gpu1_RTX3080_halocpu_migcpu.csv \
  benchmark/bench_scale100k_crossing_5000_gpu4_RTX3080_halogpu_miggpu.csv
```

## Expected CSVs (10 files)

```
bench_scale100k_crossing_5000_gpu1_A100_halocpu_migcpu.csv
bench_scale100k_crossing_5000_gpu1_RTX3080_halocpu_migcpu.csv

bench_scale100k_crossing_5000_gpu4_A100_halocpu_migcpu.csv
bench_scale100k_crossing_5000_gpu4_A100_halogpu_migcpu.csv
bench_scale100k_crossing_5000_gpu4_A100_halocpu_miggpu.csv
bench_scale100k_crossing_5000_gpu4_A100_halogpu_miggpu.csv

bench_scale100k_crossing_5000_gpu4_RTX3080_halocpu_migcpu.csv
bench_scale100k_crossing_5000_gpu4_RTX3080_halogpu_migcpu.csv
bench_scale100k_crossing_5000_gpu4_RTX3080_halocpu_miggpu.csv
bench_scale100k_crossing_5000_gpu4_RTX3080_halogpu_miggpu.csv
```
