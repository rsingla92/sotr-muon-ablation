#!/usr/bin/env bash
#SBATCH --job-name=gogpt-smoke
#SBATCH --account=__YOUR_RAPI__
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=runs/slurm/%x-%j.out

set -euo pipefail
module load python/3.11 cuda/12 gcc

source ~/venv-gogpt/bin/activate

# Stage data to fast local scratch on the compute node.
if [[ -n "${SLURM_TMPDIR:-}" && -d "$SCRATCH/gogpt-data" ]]; then
    rsync -a --info=stats1 "$SCRATCH/gogpt-data/" "$SLURM_TMPDIR/"
    export DATA_DIR="$SLURM_TMPDIR"
fi

bash scripts/smoke_test.sh
