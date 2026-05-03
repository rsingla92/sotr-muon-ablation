#!/usr/bin/env bash
# One-shot environment setup. Idempotent: safe to re-run.
#
# Usage:
#   ./scripts/setup.sh
#
# What this does:
#   1. Initialize submodules under external/
#   2. Create a Python venv (if none exists) and install dependencies
#   3. Install pre-commit hooks
#   4. Run a smoke check on the env

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ---------------------------------------------------------------------------
# 1. Submodules
# ---------------------------------------------------------------------------
echo "==> Initializing git submodules..."
git submodule update --init --recursive

# ---------------------------------------------------------------------------
# 2. Python environment
# ---------------------------------------------------------------------------
if command -v uv >/dev/null 2>&1; then
    echo "==> uv found — using it for env management"
    uv sync --extra dev
    PIP_CMD="uv pip"
    PYTHON_CMD="uv run python"
else
    echo "==> uv not found — falling back to venv + pip"
    if [[ ! -d .venv ]]; then
        python3 -m venv .venv
    fi
    # shellcheck source=/dev/null
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -e ".[dev]"
    PIP_CMD="pip"
    PYTHON_CMD="python"
fi

# ---------------------------------------------------------------------------
# 3. Pre-commit hooks
# ---------------------------------------------------------------------------
echo "==> Installing pre-commit hooks..."
$PYTHON_CMD -m pre_commit install || {
    echo "(pre-commit not installed; run: $PIP_CMD install pre-commit)" >&2
}

# ---------------------------------------------------------------------------
# 4. Smoke check
# ---------------------------------------------------------------------------
echo "==> Smoke check: torch + CUDA visibility"
$PYTHON_CMD - <<'PYEOF'
import sys
import torch

print(f"  python: {sys.version.split()[0]}")
print(f"  torch:  {torch.__version__}")
print(f"  cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device: {torch.cuda.get_device_name(0)}")
    print(f"  device count: {torch.cuda.device_count()}")
PYEOF

echo ""
echo "==> Done. Next:"
echo "    make sanity      # run PROTOCOL §7 sanity gate (once optimizer code lands)"
echo "    make test        # full test suite"
echo "    make lint        # ruff check"
