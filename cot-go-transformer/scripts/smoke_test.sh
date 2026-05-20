#!/usr/bin/env bash
# Smallest end-to-end pipeline: tiny model, ~100 games, 1000 training steps,
# 10 eval games. Targets < 1 hour on a single GPU; runs on CPU in minutes
# with --debug-no-data (no real games generated).
set -euo pipefail

# Resolve repo root regardless of where this is launched from.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO"

: "${KATAGO_BIN:=katago}"
: "${KATAGO_MODEL:?set KATAGO_MODEL to a 9x9-capable .bin.gz}"

DATA_DIR="${DATA_DIR:-data/smoke}"
RUN_DIR="${RUN_DIR:-runs/smoke}"

NUM_GAMES="${NUM_GAMES:-100}"
WORKERS="${WORKERS:-4}"

echo "[smoke] generating $NUM_GAMES self-play games into $DATA_DIR"
python scripts/generate_selfplay.py \
    --num-games "$NUM_GAMES" \
    --visits 50 \
    --analysis-visits 100 \
    --workers "$WORKERS" \
    --output "$DATA_DIR"

# Split: 90% train, 10% val.
mkdir -p data/smoke_split/train data/smoke_split/val
i=0
for f in "$DATA_DIR/sgf/"*.sgf; do
    if (( i % 10 == 0 )); then
        cp "$f" data/smoke_split/val/
    else
        cp "$f" data/smoke_split/train/
    fi
    i=$((i + 1))
done

echo "[smoke] training tiny model"
python -m gogpt.train --config configs/smoke.yaml --no-wandb

echo "[smoke] match games vs KataGo at 1 visit"
python -m gogpt.eval \
    --checkpoint "$RUN_DIR/$(ls "$RUN_DIR" | tail -n 1)/latest.pt" \
    --model-config configs/smoke.yaml \
    --num-games 10 \
    --visits 1 \
    --save-dir "$RUN_DIR/eval"

echo "[smoke] done. SGFs of eval games are in $RUN_DIR/eval"
