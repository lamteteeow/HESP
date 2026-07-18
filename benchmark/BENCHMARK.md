# Benchmark Runbook

## Defaults

| Env var | Default | Override |
|---|---|---|
| `HALO` | `cpu` | `HALO=gpu` |
| `MIGRATE` | `cpu` | `MIGRATE=gpu` |
| `DYNAMIC` | `off` | `DYNAMIC=on` |

Default is the slowest, most conservative path. Every `_halogpu` / `_miggpu`
tag in a CSV filename means an optimisation was explicitly enabled.

## CSVs produced

Each run writes one CSV to `benchmark/`.  The filename encodes:

```
bench_<scene>_<steps>_gpu<N>_<GPUmodel>_halo<H>_mig<M>.csv
```

## Test matrix — 4 GPUs

For each hardware target (`a100` NVLink and `rtx3080` PCIe), run all four
mode combinations:

| # | HALO | MIGRATE | CSV tag | Measures |
|---|---|---|---|---|
| **A** | cpu | cpu | `halocpu_migcpu` | Baseline — all CPU, slowest |
| **B** | gpu | cpu | `halogpu_migcpu` | GPU halo benefit only |
| **C** | cpu | gpu | `halocpu_miggpu` | GPU migration benefit only |
| **D** | gpu | gpu | `halogpu_miggpu` | Both optimisations combined |

Plus a single **1-GPU baseline** (no halo, no migration, no communication).

## 1. Single-GPU baseline (run once)

```bash
sbatch.tinygpu --gres=gpu:1 scripts/bench.sh crossing_freq 50000 1 100000 0
```

## 2. Four-GPU NVLink (A100)

```bash
# A — CPU halo + CPU migration (default — no env vars needed)
sbatch.tinygpu --gres=gpu:a100:4 --partition=a100 scripts/bench.sh \
  crossing_freq 50000 4 100000 0

# B — GPU halo + CPU migration
HALO=gpu sbatch.tinygpu --gres=gpu:a100:4 --partition=a100 scripts/bench.sh \
  crossing_freq 50000 4 100000 0

# C — CPU halo + GPU migration
MIGRATE=gpu sbatch.tinygpu --gres=gpu:a100:4 --partition=a100 scripts/bench.sh \
  crossing_freq 50000 4 100000 0

# D — GPU halo + GPU migration
HALO=gpu MIGRATE=gpu sbatch.tinygpu --gres=gpu:a100:4 --partition=a100 \
  scripts/bench.sh crossing_freq 50000 4 100000 0
```

## 3. Four-GPU PCIe (RTX 3080)

```bash
# A — CPU halo + CPU migration
sbatch.tinygpu --gres=gpu:rtx3080:4 --partition=rtx3080 scripts/bench.sh \
  crossing_freq 50000 4 100000 0

# B — GPU halo + CPU migration
HALO=gpu sbatch.tinygpu --gres=gpu:rtx3080:4 --partition=rtx3080 \
  scripts/bench.sh crossing_freq 50000 4 100000 0

# C — CPU halo + GPU migration
MIGRATE=gpu sbatch.tinygpu --gres=gpu:rtx3080:4 --partition=rtx3080 \
  scripts/bench.sh crossing_freq 50000 4 100000 0

# D — GPU halo + GPU migration
HALO=gpu MIGRATE=gpu sbatch.tinygpu --gres=gpu:rtx3080:4 --partition=rtx3080 \
  scripts/bench.sh crossing_freq 50000 4 100000 0
```

## 4. Compare results

After all jobs finish, diff the CSVs in pairs that isolate one variable:

### 4a. Halo: CPU vs GPU (A → B)

```bash
python3 scripts/compare_bench.py \
  benchmark/bench_crossing_freq_50000_gpu4_A100_halocpu_migcpu.csv \
  benchmark/bench_crossing_freq_50000_gpu4_A100_halogpu_migcpu.csv
```

### 4b. Migration: CPU vs GPU (A → C)

```bash
python3 scripts/compare_bench.py \
  benchmark/bench_crossing_freq_50000_gpu4_A100_halocpu_migcpu.csv \
  benchmark/bench_crossing_freq_50000_gpu4_A100_halocpu_miggpu.csv
```

### 4c. Both optimisations combined (A → D)

```bash
python3 scripts/compare_bench.py \
  benchmark/bench_crossing_freq_50000_gpu4_A100_halocpu_migcpu.csv \
  benchmark/bench_crossing_freq_50000_gpu4_A100_halogpu_miggpu.csv
```

### 4d. NVLink vs PCIe (same config, different hardware)

```bash
python3 scripts/compare_bench.py \
  benchmark/bench_crossing_freq_50000_gpu4_A100_halogpu_miggpu.csv \
  benchmark/bench_crossing_freq_50000_gpu4_RTX3080_halogpu_miggpu.csv
```

### 4e. 1-GPU vs 4-GPU scaling

```bash
python3 scripts/compare_bench.py \
  benchmark/bench_crossing_freq_50000_gpu1_RTX3080_halocpu_migcpu.csv \
  benchmark/bench_crossing_freq_50000_gpu4_RTX3080_halogpu_miggpu.csv
```

## Expected CSV outputs (9 files)

```
bench_crossing_freq_50000_gpu1_RTX3080_halocpu_migcpu.csv

bench_crossing_freq_50000_gpu4_A100_halocpu_migcpu.csv
bench_crossing_freq_50000_gpu4_A100_halogpu_migcpu.csv
bench_crossing_freq_50000_gpu4_A100_halocpu_miggpu.csv
bench_crossing_freq_50000_gpu4_A100_halogpu_miggpu.csv

bench_crossing_freq_50000_gpu4_RTX3080_halocpu_migcpu.csv
bench_crossing_freq_50000_gpu4_RTX3080_halogpu_migcpu.csv
bench_crossing_freq_50000_gpu4_RTX3080_halocpu_miggpu.csv
bench_crossing_freq_50000_gpu4_RTX3080_halogpu_miggpu.csv
```

## Target metrics

| Metric | 1-GPU | 4-GPU NVLink | 4-GPU PCIe |
|---|---|---|---|
| Wall clock (ms) | T₁ | T₄ | T₄ |
| Scaling efficiency | — | T₁/(4×T₄) | T₁/(4×T₄) |
| Halo cost (% step) | 0 | see A→B diff | see A→B diff |
| Migrate cost, idle (% step) | 0 | < 0.1% | < 0.1% |
| Migrate cost, crossing (ms) | 0 | see C→D diff | see C→D diff |
