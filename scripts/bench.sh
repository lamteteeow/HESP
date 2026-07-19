#!/bin/bash -l
# ============================================================================
# md3d benchmark batch job — parameterised for any GPU partition.
#
# Usage (from login node):
#   sbatch.tinygpu --gres=gpu:a100:4 --partition=a100 scripts/bench.sh \
#     <scene> <max_steps> <num_gpus> [vtk_interval] [bench_interval]
#
# Env vars HALO, MIGRATE, DYNAMIC are forwarded to the job:
#   HALO=gpu sbatch.tinygpu --gres=gpu:a100:4 scripts/bench.sh scale100k_crossing 10000 4
#   MIGRATE=gpu sbatch.tinygpu --gres=gpu:rtx3080:4 --partition=rtx3080 scripts/bench.sh scale100k_crossing 10000 4
#   DYNAMIC=on sbatch.tinygpu --gres=gpu:rtx3080:8 --partition=rtx3080 scripts/bench.sh comet5k 50000 8 200 0
#
# Examples:
#   # 1-GPU baseline
#   sbatch.tinygpu --gres=gpu:a100:1 --partition=a100 scripts/bench.sh scale100k_crossing 10000 1 0 0
#
#   # 4-GPU A100, GPU halo
#   HALO=gpu sbatch.tinygpu --gres=gpu:a100:4 --partition=a100 scripts/bench.sh scale100k_crossing 10000 4 0 0
#
#   # 8-GPU RTX 3080, dynamic rebalancing with VTK
#   HALO=gpu MIGRATE=gpu DYNAMIC=on sbatch.tinygpu \
#     --gres=gpu:rtx3080:8 --partition=rtx3080 scripts/bench.sh comet5k 50000 8 200 0
# ============================================================================

#SBATCH --time=6:00:00
#SBATCH --export=ALL
#SBATCH --job-name=md3d

unset SLURM_EXPORT_ENV

# ---- Apply defaults so stale env vars don't leak through ----
: "${HALO:=cpu}"
: "${MIGRATE:=cpu}"
: "${DYNAMIC:=off}"
export HALO MIGRATE DYNAMIC

# ---- Parse arguments ----
SCENE="${1:?Usage: bench.sh <scene> <max_steps> <num_gpus> [vtk_interval] [bench_interval]}"
MAX_STEPS="${2:?}"
NUM_GPUS="${3:?}"
VTK_INTERVAL="${4:-0}"
BENCH_INTERVAL="${5:-0}"

echo "========================================"
echo " md3d benchmark"
echo "========================================"
echo "Scene:          $SCENE"
echo "Max steps:      $MAX_STEPS"
echo "Num GPUs:       $NUM_GPUS"
echo "VTK interval:   $VTK_INTERVAL"
echo "Bench interval: $BENCH_INTERVAL"
echo "HALO:           ${HALO:-cpu}"
echo "MIGRATE:        ${MIGRATE:-cpu}"
echo "DYNAMIC:        ${DYNAMIC:-off}"
echo "========================================"

# ---- Load modules ----
module load cuda/12.8.0

# ---- Build (once, before submitting — skip if binary exists) ----
if [ ! -f md3d ]; then
  make clean && make
fi

# ---- Run ----
./md3d "scenes/${SCENE}.json" "$MAX_STEPS" "$NUM_GPUS" "$VTK_INTERVAL" "$BENCH_INTERVAL"
