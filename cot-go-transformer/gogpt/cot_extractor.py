"""Structured CoT extractor.

Given a position and the KataGo analysis for it, produces the sequence of
think-block tokens that should go between [<think>] and [</think>] in the
training data. Mechanical (no LLM calls).

Inputs:
    board:        9x9 int8 array (0=empty, 1=black, 2=white) -- ALWAYS from
                  the always-as-black perspective the model trains on.
    katago:       a dict shaped like the JSONL emitted by
                  scripts/generate_selfplay.py:
                  {
                    'to_move': 'B' or 'W',
                    'root_winrate': float,           # side-to-move
                    'root_score_lead': float,        # side-to-move
                    'top_moves': [
                        {'move': 'D4', 'visits': ..., 'winrate': ...,
                         'score_lead': ..., 'prior': ..., 'order': ...}
                    ],
                    'ownership': [81 floats] or None,
                  }
    move_number:  ply count (0-indexed) for phase tagging.

Outputs:
    A list[int] of token IDs.

If KataGo flagged this position as `to_move='W'`, the caller must pre-flip
the board and ownership map before calling this function (because the
training pipeline is always-as-black).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from . import BOARD_SIZE, NUM_POINTS
from . import concepts as C
from . import cot_vocab as V
from .tokenizer import (
    THINK_OPEN_TOKEN,
    THINK_CLOSE_TOKEN,
    gtp_vertex_to_token,
    point_to_token,
)

log = logging.getLogger(__name__)


# How many weak-group and shape mentions to include per position.
MAX_WEAK_GROUPS = 3
MAX_TACTICS = 3
MAX_SHAPES = 2


def extract_think_block(
    board: np.ndarray,
    katago: dict[str, Any],
    move_number: int,
    *,
    flip_ownership: bool = False,
) -> list[int]:
    """Return the token sequence that goes between [<think>] and [</think>].

    Does NOT include the open/close tokens themselves -- the caller wraps it.
    """
    out: list[int] = []

    # --- Winrate + score
    out.append(V.winrate_bin_token(katago.get("root_winrate", 0.5)))
    out.append(V.score_lead_token(katago.get("root_score_lead", 0.0)))

    # --- Phase
    out.append(V.phase_token(move_number))

    # --- Weak groups (own + opp), up to MAX_WEAK_GROUPS by importance
    ownership_arr: np.ndarray | None = None
    raw_own = katago.get("ownership")
    if raw_own is not None:
        ownership_arr = np.asarray(raw_own, dtype=np.float32).reshape(BOARD_SIZE, BOARD_SIZE)
        if flip_ownership:
            ownership_arr = -ownership_arr  # flip the perspective

    groups = C.all_groups(board)
    weak = [g for g in groups if g.num_liberties <= 2]
    # Sort by (own first, fewer libs first, larger size first) -- own weak groups are most decision-relevant.
    weak.sort(key=lambda g: (-(g.color == C.BLACK), g.num_liberties, -g.size))
    facts_emitted = False
    for g in weak[:MAX_WEAK_GROUPS]:
        dead = (
            ownership_arr is not None
            and C.group_dead_by_ownership(g, ownership_arr)
        )
        seki = (
            ownership_arr is not None
            and C.group_in_seki(g, ownership_arr)
        )
        out.append(V.group_status_token(g.num_liberties, dead, seki))
        out.append(V.AT_VERTEX)
        rep_r, rep_c = g.representative
        out.append(point_to_token(rep_r, rep_c))
        facts_emitted = True

    # --- Local tactics around the top KataGo move
    top_moves = katago.get("top_moves") or []
    top = top_moves[0] if top_moves else None
    tactics_tokens: list[int] = []
    if top is not None and top["move"].lower() != "pass":
        try:
            top_token = gtp_vertex_to_token(top["move"])
        except Exception:
            top_token = None
        if top_token is not None and top_token < NUM_POINTS:
            rc = (top_token // BOARD_SIZE, top_token % BOARD_SIZE)
            # Atari threat
            if C.is_atari_threat(board, C.BLACK, rc):
                tactics_tokens.append(V.TAC_ATARI)
            # Capture
            caps = C.captures_if_played(board, C.BLACK, rc)
            if caps:
                tactics_tokens.append(V.TAC_CAPTURE)
            # Ko-capture
            if C.is_ko_capture(board, C.BLACK, rc):
                tactics_tokens.append(V.TAC_KO)
            # Ladder breaker
            if C.is_ladder_breaker(board, C.BLACK, rc):
                tactics_tokens.append(V.TAC_LADDER_BREAK)
            # Defense of an own weak group: top move is adjacent to a weak
            # own group's liberty.
            own_weak = [g for g in weak if g.color == C.BLACK]
            for g in own_weak:
                if rc in g.liberties:
                    tactics_tokens.append(V.TAC_DEFENSE)
                    break
            # Reduction / invasion: top move is in opponent's strong area.
            if ownership_arr is not None:
                own_val = float(ownership_arr[rc[0], rc[1]])
                if own_val <= -0.6:
                    tactics_tokens.append(V.TAC_INVASION)
                elif own_val <= -0.3:
                    tactics_tokens.append(V.TAC_REDUCTION)
            # Eye-making: top move completes a real eye for black somewhere
            # adjacent. Cheap check: after the move, is there a new eye?
            after = board.copy()
            after[rc] = C.BLACK
            new_eyes = C.count_eyes(after, C.BLACK) - C.count_eyes(board, C.BLACK)
            if new_eyes > 0:
                tactics_tokens.append(V.TAC_EYE_MAKE)
        # Ladder run: any own group currently in a captured-ladder?
        own_runners = [
            g for g in groups
            if g.color == C.BLACK and g.num_liberties <= 2 and C.is_ladder_runner(board, g)
        ]
        if own_runners:
            tactics_tokens.append(V.TAC_LADDER_RUN)
    # Dedup while preserving order, cap at MAX_TACTICS
    seen: set[int] = set()
    for t in tactics_tokens:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
        facts_emitted = True
        if len(seen) >= MAX_TACTICS:
            break

    # --- Shape observations around the top move
    if top is not None and top["move"].lower() != "pass":
        try:
            top_token = gtp_vertex_to_token(top["move"])
            rc = (top_token // BOARD_SIZE, top_token % BOARD_SIZE)
        except Exception:
            rc = None
        if rc is not None:
            shape_tokens: list[tuple[int, tuple[int, int]]] = []
            # Look for eye / tiger / bamboo near the top move.
            if C.is_tiger_mouth(board, *rc, C.BLACK):
                shape_tokens.append((V.SH_TIGER, rc))
            after = board.copy()
            after[rc] = C.BLACK
            # Check if any new eye appeared right next to the move.
            for nr, nc in C.neighbors4(*rc):
                if C.is_eye(after, nr, nc, C.BLACK) and not C.is_eye(board, nr, nc, C.BLACK):
                    shape_tokens.append((V.SH_EYE, (nr, nc)))
                    break
            if C.is_bamboo_joint(after, *rc, C.BLACK):
                shape_tokens.append((V.SH_BAMBOO, rc))
            for tok, (sr, sc) in shape_tokens[:MAX_SHAPES]:
                out.append(tok)
                out.append(V.AT_VERTEX)
                out.append(point_to_token(sr, sc))
                facts_emitted = True

    # --- Separator + predicted top move + confidence
    if not facts_emitted:
        out.append(V.NO_FACTS)
    out.append(V.SEP_FACTS)
    if top is not None:
        try:
            out.append(V.TOP_MOVE)
            out.append(gtp_vertex_to_token(top["move"]))
        except Exception:
            log.warning("could not encode top move %r", top.get("move"))
        runner_up_visits = top_moves[1]["visits"] if len(top_moves) >= 2 else 0
        out.append(V.confidence_token(top.get("visits", 0), runner_up_visits))
    return out


def wrap_with_think_tags(think_tokens: list[int]) -> list[int]:
    """Convenience: return [<think>] + tokens + [</think>]."""
    return [THINK_OPEN_TOKEN] + list(think_tokens) + [THINK_CLOSE_TOKEN]


__all__ = ["extract_think_block", "wrap_with_think_tags"]
