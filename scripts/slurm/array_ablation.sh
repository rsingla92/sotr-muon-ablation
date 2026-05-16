#!/bin/bash
# SLURM job array template for the Phase 2 ablation grid.
# 250 jobs (10 cells × 5 seeds × 5 LRs) per PROTOCOL §9, max 24 concurrent.
# See docs/CLUSTER.md and docs/EXPERIMENTS.md.
#
# Usage:
#   1. Generate the config index:
#        python experiments/scripts/gen_phase2_configs.py
#      → produces experiments/configs/phase2/index.txt
#   2. Submit the array:
#        sbatch scripts/slurm/array_ablation.sh

#SBATCH --job-name=ablation
#SBATCH --account=rrg-timsbc
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-249%24
#SBATCH --output=results/slurm/%x-%A_%a.out
#SBATCH --error=results/slurm/%x-%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load StdEnv/2023 python/3.12 cuda/12.6 gcc/12
source ~/scratch/optimizer_experiments/venv/bin/activate

INDEX_FILE=experiments/configs/phase2/index.txt
if [[ ! -f "$INDEX_FILE" ]]; then
    echo "missing $INDEX_FILE — run experiments/scripts/gen_phase2_configs.py first" >&2
    exit 1
fi

# Map array index (0-based) to a config path (1-based line in index file).
CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$INDEX_FILE")
if [[ -z "$CONFIG" ]]; then
    echo "no config for array index $SLURM_ARRAY_TASK_ID" >&2
    exit 1
fi

# experiments/train.py is vendored from modded-nanogpt's train_gpt2.py and
# calls dist.init_process_group() at top-level, so it must be launched via
# torchrun (which sets RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT). Single-GPU
# works fine: train_gpt2.py asserts 8 % world_size == 0 and sets
# grad_accum_steps = 8 // world_size, so nproc_per_node=1 ⇒ grad_accum_steps=8.
# NB: each array task is its own process, so torchrun standalone is safe
# (no rdzv collisions between concurrent array tasks).
export PYTORCH_ALLOC_CONF="expandable_segments:True"
torchrun --standalone --nproc_per_node=1 experiments/train.py --config "$CONFIG"
