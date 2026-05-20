"""Structured CoT extractor.

Given a position, the KataGo analysis for it, AND the move that was actually
played (or sampled to play), produces the sequence of think-block tokens
that should go between `[<think>]` and `[</think>]` in the training data.
Mechanical (no LLM calls).

Inputs:
    board:           (9, 9) int8 (0=empty, 1=black, 2=white). Must be from
                     the always-as-black perspective the model trains on --
                     the caller should swap colors when the actual side to
                     move is white.
    katago:          a dict shaped like the JSONL emitted by
                     scripts/generate_selfplay.py (see below).
    played_move_rc:  (row, col) tuple for the move that will be emitted as
                     the outer trajectory token, or None for a pass. This
                     is what TOP_MOVE binds to: the model's CoT is an
                     explanation of THIS move, not necessarily KataGo's
                     top recommendation.
    move_number:     ply count (0-indexed) for phase tagging.
    flip_ownership:  if True, multiply the ownership map by -1 to remap
                     KataGo's black-positive convention into our flipped
                     frame.
    mode:            'structured' (default), 'empty', or 'free'.
                     - 'structured': full structured CoT.
                     - 'empty':       returns [] (caller wraps with
                                      <think></think>, total CoT cost is
                                      2 tokens).
                     - 'free':        returns N pseudo-random think-tokens
                                      to occupy the CoT region with
                                      content-free input. The training
                                      pipeline masks loss inside the
                                      region so the model isn't supervised
                                      on the content.

KataGo dict schema:
    {
      'to_move': 'B' or 'W',
      'root_winrate': float,           # from side to move
      'root_score_lead': float,        # from side to move
      'top_moves': [
          {'move': 'D4', 'visits': ..., 'winrate': ...,
           'score_lead': ..., 'prior': ..., 'order': ...},
          ...
      ],
      'ownership': [81 floats] or None,
    }

Outputs:
    list[int] of think-token IDs (no <think>/</think> wrap).
"""

from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np

from . import BOARD_SIZE, NUM_POINTS
from . import concepts as C
from . import cot_vocab as V
from .tokenizer import (
    THINK_CLOSE_TOKEN,
    THINK_OPEN_TOKEN,
    gtp_vertex_to_token,
    point_to_token,
)

log = logging.getLogger(__name__)


MAX_WEAK_GROUPS = 3
MAX_TACTICS = 3
MAX_SHAPES = 2
FREE_MODE_LENGTH = 10  # think-block length for the unsupervised 'free' ablation


def _katago_top_move_rc(top_moves: list[dict[str, Any]]) -> tuple[int, int] | None:
    if not top_moves:
        return None
    move = top_moves[0].get("move", "")
    if not move or move.lower() == "pass":
        return None
    try:
        tok = gtp_vertex_to_token(move)
    except Exception:
        return None
    if tok >= NUM_POINTS:
        return None
    return divmod(tok, BOARD_SIZE)


def _runner_up_winrate(top_moves: list[dict[str, Any]]) -> float | None:
    if len(top_moves) < 2:
        return None
    return float(top_moves[1].get("winrate", 0.0))


def _emit_facts(
    board: np.ndarray,
    katago: dict[str, Any],
    move_rc: tuple[int, int] | None,
    ownership_arr: np.ndarray | None,
    groups: list,
    weak: list,
) -> tuple[list[int], bool]:
    """Group + tactic + shape emission. Returns (tokens, any_emitted)."""
    out: list[int] = []
    emitted = False

    # Weak groups (own first, by fewest libs / largest size)
    for g in weak[:MAX_WEAK_GROUPS]:
        dead = (
            ownership_arr is not None and C.group_dead_by_ownership(g, ownership_arr)
        )
        seki = (
            ownership_arr is not None and C.group_in_seki(g, ownership_arr)
        )
        out.append(V.group_status_token(g.num_liberties, dead, seki))
        out.append(V.AT_VERTEX)
        rep_r, rep_c = g.representative
        out.append(point_to_token(rep_r, rep_c))
        emitted = True

    # Tactics, evaluated against the PLAYED MOVE (not KataGo's top).
    if move_rc is not None:
        tactics: list[int] = []
        # Atari
        if C.is_atari_threat(board, C.BLACK, move_rc):
            tactics.append(V.TAC_ATARI)
        # Capture
        caps = C.captures_if_played(board, C.BLACK, move_rc)
        if caps:
            tactics.append(V.TAC_CAPTURE)
        # Ko-capture
        if C.is_ko_capture(board, C.BLACK, move_rc):
            tactics.append(V.TAC_KO)
        # Ladder breaker
        if C.is_ladder_breaker(board, C.BLACK, move_rc):
            tactics.append(V.TAC_LADDER_BREAK)
        # Defense of an own weak group
        own_weak = [g for g in weak if g.color == C.BLACK]
        for g in own_weak:
            if move_rc in g.liberties:
                tactics.append(V.TAC_DEFENSE)
                break
        # Invasion / reduction from ownership of the move point
        if ownership_arr is not None:
            own_val = float(ownership_arr[move_rc[0], move_rc[1]])
            if own_val <= -0.6:
                tactics.append(V.TAC_INVASION)
            elif own_val <= -0.3:
                tactics.append(V.TAC_REDUCTION)
        # Eye-making (check only the 4 neighbours of the played move, not
        # the whole board)
        after = board.copy()
        after[move_rc] = C.BLACK
        for nr, nc in C.neighbors4(*move_rc):
            if C.is_eye(after, nr, nc, C.BLACK) and not C.is_eye(board, nr, nc, C.BLACK):
                tactics.append(V.TAC_EYE_MAKE)
                break
        # Ladder run (independent of the played move: do we have an own group
        # currently captured-by-ladder?)
        if any(
            g.color == C.BLACK and g.num_liberties <= 2 and C.is_ladder_runner(board, g)
            for g in groups
        ):
            tactics.append(V.TAC_LADDER_RUN)

        seen: set[int] = set()
        for t in tactics:
            if t in seen:
                continue
            seen.add(t)
            out.append(t)
            emitted = True
            if len(seen) >= MAX_TACTICS:
                break

    # Shape observations around the played move
    if move_rc is not None:
        shapes: list[tuple[int, tuple[int, int]]] = []
        if C.is_tiger_mouth(board, *move_rc, C.BLACK):
            shapes.append((V.SH_TIGER, move_rc))
        after = board.copy()
        after[move_rc] = C.BLACK
        for nr, nc in C.neighbors4(*move_rc):
            if C.is_eye(after, nr, nc, C.BLACK) and not C.is_eye(board, nr, nc, C.BLACK):
                shapes.append((V.SH_EYE, (nr, nc)))
                break
        if C.is_bamboo_joint(after, *move_rc, C.BLACK):
            shapes.append((V.SH_BAMBOO, move_rc))
        for tok, (sr, sc) in shapes[:MAX_SHAPES]:
            out.append(tok)
            out.append(V.AT_VERTEX)
            out.append(point_to_token(sr, sc))
            emitted = True

    return out, emitted


def extract_think_block(
    board: np.ndarray,
    katago: dict[str, Any],
    move_number: int,
    *,
    played_move_rc: tuple[int, int] | None = None,
    flip_ownership: bool = False,
    mode: str = "structured",
) -> list[int]:
    """Return the token sequence that goes between [<think>] and [</think>]."""
    if mode == "empty":
        return []
    if mode == "free":
        # Random tokens drawn from the structured-CoT vocab range so they
        # have the same value distribution as real CoTs (but no semantics).
        # The training pipeline masks loss inside this region; the model
        # gets a "scratchpad" with no supervision.
        rng = random.Random(
            hash((move_number, tuple(board.flatten().tolist()))) & 0xFFFFFFFF
        )
        ids = V.all_think_token_ids()
        return [rng.choice(ids) for _ in range(FREE_MODE_LENGTH)]
    if mode != "structured":
        raise ValueError(f"unknown CoT mode {mode!r}")

    # ---------------- structured mode ----------------
    out: list[int] = []

    # State-level facts (these come first because they're position-global)
    out.append(V.winrate_bin_token(katago.get("root_winrate", 0.5)))
    out.append(V.score_lead_token(katago.get("root_score_lead", 0.0)))

    # Ownership setup
    ownership_arr: np.ndarray | None = None
    raw_own = katago.get("ownership")
    if raw_own is not None:
        ownership_arr = np.asarray(raw_own, dtype=np.float32).reshape(BOARD_SIZE, BOARD_SIZE)
        if flip_ownership:
            ownership_arr = -ownership_arr

    # If played_move_rc wasn't provided, default to KataGo's top -- but the
    # caller should usually pass the actual played move so the CoT explains
    # the move that will be emitted.
    move_rc = played_move_rc
    if move_rc is None:
        move_rc = _katago_top_move_rc(katago.get("top_moves") or [])

    groups = C.all_groups(board)
    weak = [g for g in groups if g.num_liberties <= 2]
    weak.sort(key=lambda g: (-(g.color == C.BLACK), g.num_liberties, -g.size))

    fact_tokens, facts_emitted = _emit_facts(
        board, katago, move_rc, ownership_arr, groups, weak
    )
    if facts_emitted:
        out.extend(fact_tokens)
    else:
        out.append(V.NO_FACTS)

    # Phase comes AFTER grounding facts -- the model has now seen the
    # position summary and can predict phase with the right context.
    out.append(V.phase_token(move_number))

    # Conclusion: separator, played-move token, confidence.
    out.append(V.SEP_FACTS)
    out.append(V.TOP_MOVE)
    # Encode the played move as its move-vocab token. None or pass -> use
    # the PASS_TOKEN.
    if move_rc is None:
        from .tokenizer import PASS_TOKEN
        out.append(PASS_TOKEN)
    else:
        out.append(point_to_token(*move_rc))

    # Confidence: winrate gap between top KataGo move and runner-up.
    top_moves = katago.get("top_moves") or []
    top_wr = float(top_moves[0]["winrate"]) if top_moves else None
    runner_wr = _runner_up_winrate(top_moves)
    out.append(V.confidence_token(top_wr, runner_wr))

    return out


def wrap_with_think_tags(think_tokens: list[int]) -> list[int]:
    """Wrap with [<think>] / [</think>]. Empty tokens lists are OK -- they
    produce just the open/close pair, which is the no-CoT ablation."""
    return [THINK_OPEN_TOKEN] + list(think_tokens) + [THINK_CLOSE_TOKEN]


__all__ = [
    "extract_think_block",
    "wrap_with_think_tags",
    "FREE_MODE_LENGTH",
    "MAX_WEAK_GROUPS",
    "MAX_TACTICS",
    "MAX_SHAPES",
]
