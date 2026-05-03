#!/bin/bash
# Phase 1 reproduction — runs upstream external/modded-nanogpt/train_gpt.py
# unmodified at single-GPU on DRAC.
#
# Goal: confirm our cluster setup reproduces a published Muon number on the
# canonical FineWeb harness. PROTOCOL §6 reproduction gate is "within ±5%" of
# Keller Jordan's published single-GPU baseline.
#
# Usage (after running ./scripts/setup_drac.sh):
#     sbatch scripts/slurm/phase1_modded_nanogpt.sh
#
# Log lands at: results/slurm/phase1_modded_nanogpt-<jobid>.out
# Train log lands inside external/modded-nanogpt/logs/ (their convention).
#
# Single-GPU rationale: modded-nanogpt's train_gpt.py allows world_size ∈ {1,2,4,8}
# via the `assert 8 % world_size == 0` guard and uses grad_accum_steps = 8//N to
# keep the effective batch size constant. Newton-Muon paper (arXiv:2604.01472)
# explicitly used a single H100 against this harness — we follow the same pattern.

#SBATCH --job-name=phase1_modded_nanogpt
#SBATCH --account=rrg-timsbc
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR"

module purge
module load StdEnv/2023 python/3.12 cuda/12.6 gcc/12

VENV="${SCRATCH:-$HOME/scratch}/optimizer_experiments/venv"
if [[ ! -d "$VENV" ]]; then
    echo "ERROR: venv missing at $VENV. Run ./scripts/setup_drac.sh first." >&2
    exit 1
fi
# shellcheck source=/dev/null
source "$VENV/bin/activate"

# Match modded-nanogpt's PYTORCH_ALLOC_CONF (set in train_gpt.py at import time
# but harmless to set early too).
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# ---------------------------------------------------------------------------
# Provenance — capture what we're running on
# ---------------------------------------------------------------------------
RUN_DIR="results/phase1/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

{
    echo "=== job ==="
    echo "job_id=$SLURM_JOB_ID"
    echo "job_name=$SLURM_JOB_NAME"
    echo "submit_dir=$SLURM_SUBMIT_DIR"
    echo "host=$(hostname)"
    echo "cluster=${CC_CLUSTER:-${SLURM_CLUSTER_NAME:-unknown}}"
    echo "start_time=$(date -Iseconds)"
    echo ""
    echo "=== git ==="
    git -C "$SLURM_SUBMIT_DIR" rev-parse HEAD
    git -C "$SLURM_SUBMIT_DIR" status --short || true
    echo ""
    echo "=== external/modded-nanogpt commit ==="
    git -C external/modded-nanogpt rev-parse HEAD
    echo ""
    echo "=== gpu ==="
    nvidia-smi -L
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    echo ""
    echo "=== modules ==="
    module list 2>&1
    echo ""
    echo "=== pip freeze (key packages only) ==="
    pip freeze | grep -iE '^(torch|numpy|tiktoken|datasets|huggingface|kernels|muon|lion|optimizer-experiments)' || true
} > "$RUN_DIR/env.txt" 2>&1

echo "Provenance saved to $RUN_DIR/env.txt"
echo ""

# ---------------------------------------------------------------------------
# Run modded-nanogpt's train_gpt.py UNMODIFIED at single-GPU
# ---------------------------------------------------------------------------
# - --nproc_per_node=1: their assert 8%world_size==0 supports this
# - grad_accum_steps becomes 8 (set automatically inside train_gpt.py)
# - run from inside external/modded-nanogpt/ since their script uses relative paths
#   to data/fineweb10B/ and triton_kernels.py
echo "Starting modded-nanogpt train_gpt.py on $(hostname) at $(date -Iseconds)..."
cd external/modded-nanogpt

torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee "../../$RUN_DIR/train.log"
EXIT_CODE=${PIPESTATUS[0]}

cd "$SLURM_SUBMIT_DIR"

# ---------------------------------------------------------------------------
# Post-run summary
# ---------------------------------------------------------------------------
{
    echo ""
    echo "=== finish ==="
    echo "exit_code=$EXIT_CODE"
    echo "end_time=$(date -Iseconds)"
} >> "$RUN_DIR/env.txt"

# Capture modded-nanogpt's own log directory (it dumps to ./logs/ inside its dir).
if [[ -d external/modded-nanogpt/logs ]]; then
    cp -r external/modded-nanogpt/logs "$RUN_DIR/modded_nanogpt_logs" || true
fi

echo ""
echo "Run finished. Artifacts in $RUN_DIR/"
echo "Reproduction gate (PROTOCOL §6): final validation loss within ±5% of"
echo "Keller Jordan's published single-GPU Muon number on this harness."

exit "$EXIT_CODE"
