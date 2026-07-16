#!/bin/bash -l
# ============================================================================
# md3d batch job — work partition (RTX 2080 Ti / 3080)
#
# Usage:
#   sbatch.tinygpu scripts/sbatch_work.sh <scene.json> [max_steps] [num_gpus]
#
# Override GPU count on the command line:
#   sbatch.tinygpu --gres=gpu:4 scripts/sbatch_work.sh scenes/cube256.json 50000 4
#
# Examples:
#   sbatch.tinygpu scripts/sbatch_work.sh scenes/cube256.json 50000 2
# ============================================================================

#SBATCH --gres=gpu:1
#SBATCH --partition=work
#SBATCH --time=2:00:00
#SBATCH --export=NONE
#SBATCH --job-name=md3d

unset SLURM_EXPORT_ENV

# ---- Parse arguments ----
SCENE="${1:-scenes/cube256.json}"
MAX_STEPS="${2:-50000}"
NUM_GPUS="${3:-1}"

echo "Scene:      $SCENE"
echo "Max steps:  $MAX_STEPS"
echo "Num GPUs:   $NUM_GPUS"

# ---- Load modules ----
module load cuda/12.8.0

# ---- Build ----
make

# ---- Run ----
./md3d "$SCENE" "$MAX_STEPS" "$NUM_GPUS"
