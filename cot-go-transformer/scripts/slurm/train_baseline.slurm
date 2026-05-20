#!/usr/bin/env bash
#SBATCH --job-name=gogpt-30m
#SBATCH --account=__YOUR_RAPI__
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm/%x-%j.out

set -euo pipefail
module load python/3.11 cuda/12 gcc

source ~/venv-gogpt/bin/activate

# Auto-resume: training picks up runs/<run_dir>/<run_id>/latest.pt if present.
# Preemption-safe: SIGTERM causes the dataloader to drain and the in-flight
# optimizer.step() to complete; the next save_every tick checkpoints.

# Stage data to node-local scratch.
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
    rsync -a --info=stats1 "$SCRATCH/gogpt-data/train/" "$SLURM_TMPDIR/train/" &
    rsync -a --info=stats1 "$SCRATCH/gogpt-data/val/"   "$SLURM_TMPDIR/val/" &
    wait
    sed "s|data/baseline|$SLURM_TMPDIR|g" configs/baseline_30m.yaml > "$SLURM_TMPDIR/baseline_30m.yaml"
    CONFIG="$SLURM_TMPDIR/baseline_30m.yaml"
else
    CONFIG="configs/baseline_30m.yaml"
fi

torchrun --standalone --nproc_per_node=4 -m gogpt.train --config "$CONFIG"
