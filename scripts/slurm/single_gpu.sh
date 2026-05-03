#!/bin/bash
# Single-GPU SLURM template for DRAC Fir (SFU H100 cluster).
# See docs/CLUSTER.md for cluster choice and account details.
#
# Usage:
#   sbatch scripts/slurm/single_gpu.sh experiments.configs.phase1_repro_muon
#
# The arg is the dotted module path passed to train.py --config (see
# experiments/_configs.py and experiments/train.py SOTR-PATCH 1).
# Tune --time and --mem per config; defaults below are for ~124M model speedrun.

#SBATCH --job-name=optexp
#SBATCH --account=rrg-timsbc
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load StdEnv/2023 python/3.12 cuda/12.6 gcc/12
source ~/scratch/optimizer_experiments/venv/bin/activate

if [[ -z "${1:-}" ]]; then
    echo "usage: sbatch $0 <experiments.configs.dotted_module>" >&2
    exit 1
fi

python experiments/train.py --config "$1"
