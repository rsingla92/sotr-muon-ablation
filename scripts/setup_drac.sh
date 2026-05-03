#!/usr/bin/env bash
# DRAC (Digital Research Alliance of Canada) login-node setup.
#
# Designed for Fir (SFU H100 cluster) but should work on any DRAC cluster
# (Narval / Trillium / Cedar / Béluga / etc.) under account rrg-timsbc.
# After running this once, you can submit SLURM jobs from this directory.
#
# Run on a *login* node (compute nodes have no internet). Idempotent — safe to
# re-run. Takes ~5 minutes the first time (mostly the FineWeb token download).
#
# Usage:
#     git clone --recurse-submodules git@github.com:rsingla92/optimizer_experiments.git \
#         ~/projects/rrg-timsbc/$USER/code/optimizer_experiments
#     cd ~/projects/rrg-timsbc/$USER/code/optimizer_experiments
#     ./scripts/setup_drac.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if [[ -z "${SLURM_CLUSTER_NAME:-}" ]] && [[ -z "${CC_CLUSTER:-}" ]]; then
    echo "WARNING: This script is intended for DRAC clusters (CC_CLUSTER unset)." >&2
    echo "         Continuing anyway — but expect failures if you're not on DRAC." >&2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

USER_DIR="${USER:-rsingla}"
SCRATCH_BASE="${SCRATCH:-$HOME/scratch}/optimizer_experiments"
PROJECT_BASE="${HOME}/projects/rrg-timsbc/${USER_DIR}/optimizer_experiments"

echo "==> Repo root:    $ROOT"
echo "==> Scratch base: $SCRATCH_BASE"
echo "==> Project base: $PROJECT_BASE"

# ---------------------------------------------------------------------------
# 1. Submodules
# ---------------------------------------------------------------------------
echo ""
echo "==> Initializing git submodules..."
git submodule update --init --recursive

# ---------------------------------------------------------------------------
# 2. DRAC modules
# ---------------------------------------------------------------------------
# IMPORTANT: arrow MUST be loaded BEFORE the venv is activated, because DRAC
# ships a "noinstall" stub pyarrow wheel that fails on purpose with a message
# pointing to the arrow module. `datasets` (used by modded-nanogpt for FineWeb
# token loading) depends transitively on pyarrow → without the arrow module,
# pip install of our package fails. See:
#   https://docs.alliancecan.ca/wiki/Arrow
echo ""
echo "==> Loading DRAC modules (StdEnv/2023, python/3.12, cuda/12.6, gcc/12, arrow)..."
module purge
module load StdEnv/2023 python/3.12 cuda/12.6 gcc/12 arrow || {
    echo "ERROR: Module load failed. Available arrow versions:" >&2
    module spider arrow 2>&1 | head -40
    echo "" >&2
    echo "If the arrow module isn't available with this StdEnv, try:" >&2
    echo "  module spider arrow            # see what's available" >&2
    echo "  module load arrow/<version>    # explicit version" >&2
    exit 1
}

# ---------------------------------------------------------------------------
# 3. Venv (in scratch — $HOME has small quota on DRAC)
# ---------------------------------------------------------------------------
mkdir -p "$SCRATCH_BASE"
VENV="$SCRATCH_BASE/venv"

if [[ ! -d "$VENV" ]]; then
    echo ""
    echo "==> Creating venv at $VENV..."
    python -m venv "$VENV"
fi

echo ""
echo "==> Activating venv..."
# shellcheck source=/dev/null
source "$VENV/bin/activate"

# NB: torch 2.11.0+computecanada has a strict `setuptools<82` constraint in its
# metadata. DRAC's wheelhouse may pull setuptools 82.x, which causes a *resolver
# warning* (not an error). We pin <82 explicitly to keep pip happy.
python -m pip install --upgrade pip wheel "setuptools<82"

# ---------------------------------------------------------------------------
# 4. PyTorch (DRAC wheelhouse)
# ---------------------------------------------------------------------------
# DRAC ships its own torch wheels compiled against their CUDA / NCCL builds.
# Use --no-index --find-links wherever possible to pull from the wheelhouse.
# If torch is already installed at >= 2.10 we skip.
echo ""
echo "==> Installing PyTorch (DRAC wheelhouse if available)..."
if python -c "import torch; assert torch.__version__ >= '2.10'" 2>/dev/null; then
    echo "    torch already installed at $(python -c 'import torch; print(torch.__version__)')"
else
    # DRAC's --no-index works because they pre-stage torch wheels.
    # Falls back to PyPI if --no-index fails (rare on Fir/Narval; possible on Cedar).
    pip install --no-index torch || pip install "torch>=2.10"
fi

# ---------------------------------------------------------------------------
# 5. Our package + dev deps
# ---------------------------------------------------------------------------
echo ""
echo "==> Installing optimizer_experiments package + dev deps..."
pip install -e ".[dev]"

# ---------------------------------------------------------------------------
# 6. External submodules as editable
# ---------------------------------------------------------------------------
echo ""
echo "==> Installing external/Muon and external/lion-pytorch as editable..."
pip install -e ./external/Muon
pip install -e ./external/lion-pytorch

# ---------------------------------------------------------------------------
# 7. modded-nanogpt's own requirements
# ---------------------------------------------------------------------------
# We don't pip install modded-nanogpt itself (it has no setup.py); we install
# what its train_gpt.py imports. We pin torch via OUR package's pyproject so
# we skip the torch line in modded-nanogpt's requirements.txt.
echo ""
echo "==> Installing modded-nanogpt's runtime requirements (skipping torch)..."
grep -v '^torch==' external/modded-nanogpt/requirements.txt | pip install -r /dev/stdin

# ---------------------------------------------------------------------------
# 8. Pre-commit hooks (best effort — local-dev concern)
# ---------------------------------------------------------------------------
if command -v pre-commit >/dev/null 2>&1; then
    echo ""
    echo "==> Installing pre-commit hooks..."
    pre-commit install || true
fi

# ---------------------------------------------------------------------------
# 9. Filesystem symlinks
# ---------------------------------------------------------------------------
# results/, checkpoints/, data/ live in scratch/project (not git).
# Symlink them from the repo so train scripts can use repo-relative paths.
echo ""
echo "==> Setting up filesystem symlinks..."
mkdir -p "$SCRATCH_BASE/results" "$SCRATCH_BASE/data" "$PROJECT_BASE/checkpoints" "$SCRATCH_BASE/results/slurm"

for link in results data; do
    if [[ ! -L "$link" ]] && [[ -d "$link" ]]; then
        # Existing real dir from the repo .gitkeep — replace with symlink.
        rm -rf "$link.bak" || true
        mv "$link" "$link.bak"
    fi
    ln -snf "$SCRATCH_BASE/$link" "$link"
done
ln -snf "$PROJECT_BASE/checkpoints" checkpoints

ls -ld results data checkpoints

# ---------------------------------------------------------------------------
# 10. FineWeb tokens (modded-nanogpt's data prep — short track: 900M tokens)
# ---------------------------------------------------------------------------
# Run on the login node — compute nodes have no internet on DRAC.
# Skips download if shards already present.
TOKEN_DIR="$SCRATCH_BASE/data/fineweb10B"
if [[ -d "$TOKEN_DIR" ]] && [[ -n "$(find "$TOKEN_DIR" -name 'fineweb_train_*.bin' -print -quit)" ]]; then
    echo ""
    echo "==> FineWeb tokens already present at $TOKEN_DIR — skipping download."
else
    echo ""
    echo "==> Downloading FineWeb tokens (first 900M, ~5 GB)..."
    pushd external/modded-nanogpt >/dev/null
    # Their script downloads to ./data/fineweb10B/. Symlink there to scratch first.
    mkdir -p "$TOKEN_DIR"
    if [[ ! -L data/fineweb10B ]] && [[ -d data ]]; then
        ln -snf "$TOKEN_DIR" data/fineweb10B
    fi
    python data/cached_fineweb10B.py 9
    popd >/dev/null
fi

# ---------------------------------------------------------------------------
# 11. Smoke check
# ---------------------------------------------------------------------------
echo ""
echo "==> Smoke check..."
python - <<'PYEOF'
import sys, torch
print(f"  python: {sys.version.split()[0]}")
print(f"  torch:  {torch.__version__}")
print(f"  cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device: {torch.cuda.get_device_name(0)}")
try:
    from muon import zeropower_via_newtonschulz5, SingleDeviceMuon, MuonWithAuxAdam  # noqa: F401
    print("  muon: OK")
except ImportError as e:
    print(f"  muon: FAIL — {e}")
try:
    from lion_pytorch import Lion  # noqa: F401
    print("  lion: OK")
except ImportError as e:
    print(f"  lion: FAIL — {e}")
try:
    from optimizers import SOTR  # noqa: F401
    print("  SOTR: OK")
except ImportError as e:
    print(f"  SOTR: FAIL — {e}")
try:
    import pyarrow  # noqa: F401
    print(f"  pyarrow: OK (from arrow module, v{pyarrow.__version__})")
except ImportError as e:
    print(f"  pyarrow: FAIL — {e}")
    print("  → did you 'module load arrow' BEFORE activating venv?")
try:
    import datasets  # noqa: F401
    print("  datasets: OK")
except ImportError as e:
    print(f"  datasets: FAIL — {e}")
PYEOF

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
cat <<EOF

==> Setup complete.

Next steps:
  1. Verify sanity gate (offline, fast):
       module load StdEnv/2023 python/3.12 cuda/12.6 gcc/12 arrow
       source $VENV/bin/activate
       make sanity

     (The arrow module must be loaded BEFORE activating the venv, every time.
     SLURM scripts already do this; this is only for interactive sessions.)

  2. Submit Phase 1 reproduction:
       sbatch scripts/slurm/phase1_modded_nanogpt.sh

  3. Watch the queue:
       squeue -u $USER

  4. Inspect results when done:
       cat results/slurm/phase1_modded_nanogpt-*.out

EOF
