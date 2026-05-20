#!/usr/bin/env bash
#SBATCH --job-name=gogpt-selfplay
#SBATCH --account=__YOUR_RAPI__
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=runs/slurm/%x-%j.out

set -euo pipefail
module load python/3.11 cuda/12 gcc

source ~/venv-gogpt/bin/activate

NUM_GAMES="${NUM_GAMES:-50000}"
OUTPUT="${OUTPUT:-$SCRATCH/gogpt-data/train}"
mkdir -p "$OUTPUT"

# One KataGo process per worker; saturate the 4 GPUs with ~8 workers each.
python scripts/generate_selfplay.py \
    --num-games "$NUM_GAMES" \
    --visits 200 \
    --analysis-visits 400 \
    --workers 32 \
    --output "$OUTPUT"
