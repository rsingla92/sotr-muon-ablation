#!/bin/bash
# Single-node multi-GPU SLURM template for DRAC Fir (4× H100).
# Use for Phase 3 mid-scale validation. See docs/CLUSTER.md and PROTOCOL §5.
#
# Usage:
#   sbatch scripts/slurm/multi_gpu.sh experiments/configs/phase3_sotr_500m.yaml
#
# For an 8-GPU node, change --gres to gpu:h100:8 (verify Fir node sizes via
# `sinfo --Format=Nodes,Gres -p gpu`).

#SBATCH --job-name=optexp_multi
#SBATCH --account=rrg-timsbc
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load StdEnv/2023 python/3.12 cuda/12.6 gcc/12 arrow
source ~/scratch/optimizer_experiments/venv/bin/activate

if [[ -z "${1:-}" ]]; then
    echo "usage: sbatch $0 <config.yaml>" >&2
    exit 1
fi

# nproc_per_node must match --gres count above.
NPROC=$(echo "$SLURM_JOB_GPUS" | tr ',' '\n' | wc -l)
torchrun --standalone --nproc_per_node="$NPROC" \
    experiments/train.py --config "$1"
