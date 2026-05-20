#!/usr/bin/env python3
"""Phase 1 deliverable: turn SGF + per-position KataGo analysis JSONL into
training-ready (state, trajectory-with-CoT) tensors on disk.

For every position in every game:
- Reconstruct the board state up to that ply.
- Extract the structured think-block via gogpt.cot_extractor.
- Concatenate [<think>] <CoT tokens> [</think>] <move-token> into the
  trajectory.
- Write one record per position to a sharded NPZ file.

Usage:
    python scripts/extract_cot_labels.py \
        --sgf-dir data/baseline/sgf \
        --jsonl-dir data/baseline/analysis \
        --output data/baseline/cot \
        --shard-size 4096

Each output shard contains:
    state_categories: (N, 81) int8
    tokens:           (N, T_max) int32 (right-padded with PASS_TOKEN)
    labels:           (N, T_max) int32 (-100 where loss-masked)
    loss_mask:        (N, T_max) int8

Reading the analysis JSONL is the hot loop; this is CPU-bound and scales
linearly with --workers.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from gogpt import BOARD_SIZE, NUM_POINTS
from gogpt.cot_extractor import extract_think_block, wrap_with_think_tags
from gogpt.data import _color_to_value, _swap_colors, parse_sgf, play_stone
from gogpt.tokenizer import (
    BOS_TOKEN,
    EOS_TOKEN,
    PASS_TOKEN,
    SEP_POS_TOKEN,
    encode_board_states,
    point_to_token,
)

log = logging.getLogger("extract_cot")


@dataclass
class CotExample:
    state_categories: np.ndarray  # (81,) int8
    tokens: np.ndarray            # (T,) int32
    labels: np.ndarray            # (T,) int32
    loss_mask: np.ndarray         # (T,) int8


def _build_example(
    board: np.ndarray,
    ko: tuple[int, int] | None,
    last_move: tuple[int, int] | None,
    cot_tokens: list[int],
    move_token: int,
    to_move: str,
    *,
    cot_supervised: bool = True,
) -> CotExample:
    """One (position, think-block + move) training example.

    Always-as-black: if to_move=='W' the caller must already have swapped
    the board AND flipped the ownership in the analysis dict.

    ``cot_supervised`` controls whether the tokens INSIDE the
    [<think>]...[</think>] region are loss-bearing. For the free-CoT (D)
    ablation this is False -- only the move-token position computes loss.
    """
    state_cats = encode_board_states(board, ko_point=ko, last_move=last_move)

    # Trajectory layout: <think> ... </think> <move>
    traj_tokens = np.asarray(
        wrap_with_think_tags(cot_tokens) + [move_token],
        dtype=np.int32,
    )

    prefix_placeholder = np.full(NUM_POINTS, PASS_TOKEN, dtype=np.int32)
    tokens = np.concatenate(
        [
            np.array([BOS_TOKEN], dtype=np.int32),
            prefix_placeholder,
            np.array([SEP_POS_TOKEN], dtype=np.int32),
            traj_tokens,
            np.array([EOS_TOKEN], dtype=np.int32),
        ]
    )

    T = tokens.shape[0]
    labels = np.full(T, -100, dtype=np.int32)
    loss_mask = np.zeros(T, dtype=np.int8)
    sep_index = 1 + NUM_POINTS
    last_traj_index = sep_index + traj_tokens.shape[0]

    if cot_supervised:
        # Standard: every trajectory position (SEP_POS .. last move) predicts
        # the next token.
        for i in range(sep_index, last_traj_index + 1):
            if i + 1 < T:
                labels[i] = int(tokens[i + 1])
        loss_mask[sep_index : last_traj_index + 1] = 1
    else:
        # Free-CoT: only the position that predicts the MOVE token is
        # loss-bearing. That position is the [</think>] token, whose
        # successor is the move token. Sequence layout starting at sep_index:
        #   sep_index    : SEP_POS
        #   +1           : THINK_OPEN
        #   +2 ..+1+N    : N content tokens
        #   +2+N         : THINK_CLOSE
        #   +3+N         : move
        think_close_pos = sep_index + len(cot_tokens) + 2
        move_pos = think_close_pos + 1
        if move_pos < T:
            labels[think_close_pos] = int(tokens[move_pos])
            loss_mask[think_close_pos] = 1

    return CotExample(
        state_categories=state_cats,
        tokens=tokens,
        labels=labels,
        loss_mask=loss_mask,
    )


def _examples_for_game(
    sgf_path: Path,
    jsonl_path: Path,
    *,
    mode: str,
) -> Iterator[CotExample]:
    game = parse_sgf(sgf_path.read_text())
    by_ply: dict[int, dict] = {}
    with jsonl_path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
                by_ply[rec["ply"]] = rec
            except Exception as e:
                log.warning("bad analysis line in %s: %s", jsonl_path, e)

    cot_supervised = mode != "free"

    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    ko: tuple[int, int] | None = None
    last_move: tuple[int, int] | None = None
    for ply, (color, rc) in enumerate(game.moves):
        analysis = by_ply.get(ply)
        if analysis is None:
            board, ko = play_stone(board, _color_to_value(color), rc)
            last_move = rc
            continue
        flip = color == "W"
        view_board = _swap_colors(board) if flip else board
        cot = extract_think_block(
            view_board,
            analysis,
            move_number=ply,
            played_move_rc=rc,
            flip_ownership=flip,
            mode=mode,
        )
        move_token = PASS_TOKEN if rc is None else point_to_token(*rc)
        yield _build_example(
            view_board, ko, last_move, cot, move_token, to_move=color,
            cot_supervised=cot_supervised,
        )
        board, ko = play_stone(board, _color_to_value(color), rc)
        last_move = rc


def _process_one(args: tuple[str, str, str]) -> list[CotExample]:
    sgf_path, jsonl_path, mode = args
    return list(_examples_for_game(Path(sgf_path), Path(jsonl_path), mode=mode))


def _write_shard(out_dir: Path, shard_idx: int, examples: list[CotExample]) -> None:
    if not examples:
        return
    T_max = max(ex.tokens.shape[0] for ex in examples)
    N = len(examples)
    tokens = np.full((N, T_max), PASS_TOKEN, dtype=np.int32)
    labels = np.full((N, T_max), -100, dtype=np.int32)
    loss_mask = np.zeros((N, T_max), dtype=np.int8)
    state_cats = np.zeros((N, NUM_POINTS), dtype=np.int8)
    for i, ex in enumerate(examples):
        T = ex.tokens.shape[0]
        tokens[i, :T] = ex.tokens
        labels[i, :T] = ex.labels
        loss_mask[i, :T] = ex.loss_mask
        state_cats[i] = ex.state_categories
    out_path = out_dir / f"shard_{shard_idx:06d}.npz"
    np.savez_compressed(
        out_path,
        state_categories=state_cats,
        tokens=tokens,
        labels=labels,
        loss_mask=loss_mask,
    )
    log.info("wrote %s (%d examples, T_max=%d)", out_path, N, T_max)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sgf-dir", required=True)
    parser.add_argument("--jsonl-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2))
    parser.add_argument(
        "--mode",
        choices=["structured", "empty", "free"],
        default="structured",
        help=(
            "CoT ablation mode. structured=B (default; full structured CoT, "
            "loss on all tokens), empty=A (no inner tokens, just <think></think>), "
            "free=D (random think-tokens, loss masked inside the region). "
            "Natural-language (C) is produced by a separate rewriter script."
        ),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    sgf_paths = sorted(Path(args.sgf_dir).glob("*.sgf"))
    pairs: list[tuple[str, str, str]] = []
    for s in sgf_paths:
        j = Path(args.jsonl_dir) / (s.stem + ".jsonl")
        if not j.exists():
            log.warning("no analysis for %s; skipping", s)
            continue
        pairs.append((str(s), str(j), args.mode))
    log.info("processing %d games (mode=%s)", len(pairs), args.mode)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    buffer: list[CotExample] = []
    shard_idx = 0
    with mp.Pool(args.workers) as pool:
        for examples in pool.imap_unordered(_process_one, pairs):
            buffer.extend(examples)
            while len(buffer) >= args.shard_size:
                _write_shard(out_dir, shard_idx, buffer[: args.shard_size])
                buffer = buffer[args.shard_size :]
                shard_idx += 1
    if buffer:
        _write_shard(out_dir, shard_idx, buffer)
    log.info("done; %d shards", shard_idx + (1 if buffer else 0))


if __name__ == "__main__":
    main()
