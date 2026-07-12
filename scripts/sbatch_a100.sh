#!/bin/bash -l
# ============================================================================
# md2d batch job — A100 partition (NVLink, best multi-GPU performance)
#
# Usage:
#   sbatch.tinygpu scripts/sbatch_a100.sh <scene.json> [max_steps] [num_gpus]
#
# Override GPU count on the command line:
#   sbatch.tinygpu --gres=gpu:a100:4 scripts/sbatch_a100.sh scenes/random20.json 50000 4
#
# Examples:
#   sbatch.tinygpu scripts/sbatch_a100.sh scenes/random20.json 50000 2
#   sbatch.tinygpu scripts/sbatch_a100.sh scenes/two_discs.json 10000
# ============================================================================

#SBATCH --gres=gpu:a100:2
#SBATCH --partition=a100
#SBATCH --time=6:00:00
#SBATCH --export=NONE
#SBATCH --job-name=md2d

unset SLURM_EXPORT_ENV

# ---- Parse arguments ----
SCENE="${1:-scenes/random20.json}"
MAX_STEPS="${2:-50000}"
NUM_GPUS="${3:-2}"

echo "Scene:      $SCENE"
echo "Max steps:  $MAX_STEPS"
echo "Num GPUs:   $NUM_GPUS"

# ---- Load modules ----
module load gcc/11.5.0
module load cuda/12.8.0

# ---- Build ----
make

# ---- Run ----
./md2d "$SCENE" "$MAX_STEPS" "$NUM_GPUS"
