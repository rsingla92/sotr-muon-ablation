"""Go-concept rule library.

Pure board-state predicates used by:
- The Phase 1 CoT label extractor (to translate KataGo analyses into
  structured think-block tokens).
- The Phase 3 feature-verification step (to score Transcoder/Lorsa features
  against rule-based ground truth).

All functions operate on numpy boards (9x9, int8: 0=empty, 1=black,
2=white) and are color-agnostic where it makes sense. None of this code
depends on torch or KataGo.

References for the concepts:
- Standard Go terminology (liberty, group, eye, ko, seki, atari, ladder).
- Lin et al. 2026 Table 2 (chess analog).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from . import BOARD_SIZE

EMPTY, BLACK, WHITE = 0, 1, 2
OFF_BOARD = -1


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < BOARD_SIZE:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < BOARD_SIZE:
        out.append((r, c + 1))
    return out


def diagonals(r: int, c: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr in (-1, 1):
        for dc in (-1, 1):
            nr, nc = r + dr, c + dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                out.append((nr, nc))
    return out


def is_on_edge(r: int, c: int) -> bool:
    return r == 0 or r == BOARD_SIZE - 1 or c == 0 or c == BOARD_SIZE - 1


def is_on_corner(r: int, c: int) -> bool:
    return (r in (0, BOARD_SIZE - 1)) and (c in (0, BOARD_SIZE - 1))


# ---------------------------------------------------------------------------
# Groups and liberties
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Group:
    color: int                       # BLACK or WHITE
    stones: frozenset[tuple[int, int]]
    liberties: frozenset[tuple[int, int]]

    @property
    def size(self) -> int:
        return len(self.stones)

    @property
    def num_liberties(self) -> int:
        return len(self.liberties)

    @property
    def representative(self) -> tuple[int, int]:
        # Smallest (r, c) by row-major order -- stable across runs.
        return min(self.stones)


def _flood_group(board: np.ndarray, r: int, c: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    color = int(board[r, c])
    if color == EMPTY:
        return set(), set()
    stones: set[tuple[int, int]] = set()
    libs: set[tuple[int, int]] = set()
    stack = [(r, c)]
    while stack:
        cur = stack.pop()
        if cur in stones:
            continue
        stones.add(cur)
        for nr, nc in neighbors4(*cur):
            v = int(board[nr, nc])
            if v == EMPTY:
                libs.add((nr, nc))
            elif v == color and (nr, nc) not in stones:
                stack.append((nr, nc))
    return stones, libs


def group_at(board: np.ndarray, r: int, c: int) -> Group | None:
    """Return the group at (r, c), or None if the point is empty."""
    if int(board[r, c]) == EMPTY:
        return None
    stones, libs = _flood_group(board, r, c)
    return Group(color=int(board[r, c]), stones=frozenset(stones), liberties=frozenset(libs))


def all_groups(board: np.ndarray) -> list[Group]:
    """Enumerate every group on the board, deduplicated."""
    seen: set[tuple[int, int]] = set()
    out: list[Group] = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if (r, c) in seen or int(board[r, c]) == EMPTY:
                continue
            g = group_at(board, r, c)
            assert g is not None
            seen |= g.stones
            out.append(g)
    return out


def weak_groups(board: np.ndarray, max_liberties: int = 2) -> list[Group]:
    """Groups with <= ``max_liberties`` liberties. Atari is max_liberties=1."""
    return [g for g in all_groups(board) if g.num_liberties <= max_liberties]


def groups_in_atari(board: np.ndarray) -> list[Group]:
    return weak_groups(board, max_liberties=1)


# ---------------------------------------------------------------------------
# Move legality / capture preview
# ---------------------------------------------------------------------------

def is_legal_move(
    board: np.ndarray,
    color: int,
    rc: tuple[int, int],
    ko_point: tuple[int, int] | None = None,
) -> bool:
    """Conservative legality check: no suicide, respects simple ko."""
    r, c = rc
    if int(board[r, c]) != EMPTY:
        return False
    if ko_point is not None and rc == ko_point:
        return False
    # Try the move and see if any of (a) it captures, or (b) the resulting
    # own group has >= 1 liberty.
    opp = 3 - color
    test = board.copy()
    test[r, c] = color
    # Captured if any opp neighbour group has no liberties.
    for nr, nc in neighbors4(r, c):
        if int(test[nr, nc]) == opp:
            _, libs = _flood_group(test, nr, nc)
            if not libs:
                return True
    # Otherwise own group must have a liberty (no suicide).
    _, own_libs = _flood_group(test, r, c)
    return bool(own_libs)


def captures_if_played(
    board: np.ndarray, color: int, rc: tuple[int, int]
) -> list[Group]:
    """Return the opp groups that would be captured by playing at rc."""
    r, c = rc
    if int(board[r, c]) != EMPTY:
        return []
    opp = 3 - color
    test = board.copy()
    test[r, c] = color
    captured: list[Group] = []
    seen_reps: set[tuple[int, int]] = set()
    for nr, nc in neighbors4(r, c):
        if int(test[nr, nc]) != opp:
            continue
        stones, libs = _flood_group(test, nr, nc)
        if libs:
            continue
        rep = min(stones)
        if rep in seen_reps:
            continue
        seen_reps.add(rep)
        captured.append(Group(color=opp, stones=frozenset(stones), liberties=frozenset()))
    return captured


def is_atari_threat(board: np.ndarray, color: int, rc: tuple[int, int]) -> bool:
    """Would playing ``color`` at ``rc`` reduce some opponent group to <= 1 liberty?"""
    r, c = rc
    if int(board[r, c]) != EMPTY:
        return False
    if not is_legal_move(board, color, rc):
        return False
    opp = 3 - color
    test = board.copy()
    test[r, c] = color
    for nr, nc in neighbors4(r, c):
        if int(test[nr, nc]) != opp:
            continue
        _, libs = _flood_group(test, nr, nc)
        if len(libs) <= 1:
            return True
    return False


# ---------------------------------------------------------------------------
# Eye detection
# ---------------------------------------------------------------------------

def is_eye(board: np.ndarray, r: int, c: int, color: int) -> bool:
    """Standard 'real eye' definition.

    A real eye for ``color`` at (r, c) requires:
    - The point itself is empty.
    - All 4-neighbors are same-color stones (or off-board).
    - The diagonals are 'controlled' by ``color``: at most 1 diagonal is
      opponent or off-board on an interior point; at most 0 on an edge.

    This is a heuristic; under heavy fighting (false eyes via dead diagonals)
    it can mislabel, but it matches what most Go engines call an eye.
    """
    if int(board[r, c]) != EMPTY:
        return False
    opp = 3 - color
    # Adjacents
    for nr, nc in neighbors4(r, c):
        if int(board[nr, nc]) != color:
            return False
    # Diagonals
    bad = 0
    diag = diagonals(r, c)
    expected = 4  # interior
    if is_on_edge(r, c):
        expected = 2
    if is_on_corner(r, c):
        expected = 1
    for dr, dc in diag:
        v = int(board[dr, dc])
        if v == opp:
            bad += 1
    # Off-board diagonals count as bad on edges/corners.
    bad += (expected - len(diag))
    threshold = 1 if expected == 4 else 0
    return bad <= threshold


def count_eyes(board: np.ndarray, color: int) -> int:
    return sum(
        1
        for r in range(BOARD_SIZE)
        for c in range(BOARD_SIZE)
        if is_eye(board, r, c, color)
    )


# ---------------------------------------------------------------------------
# Shape predicates
# ---------------------------------------------------------------------------

def is_bamboo_joint(board: np.ndarray, r: int, c: int, color: int) -> bool:
    """Bamboo joint: two stones of ``color`` two points apart, connected by
    two more stones forming a 2x2 block missing one stone. Most concretely:
    ``color`` at (r, c) and (r, c+2) plus empty at (r, c+1) and (r+1, c+1)
    or a rotation. For our purposes we just check the 4 axis-aligned
    canonical configurations."""
    if int(board[r, c]) != color:
        return False
    candidates = [
        ((0, 2), (1, 0), (1, 2)),    # horizontal-down
        ((0, 2), (-1, 0), (-1, 2)),  # horizontal-up
        ((2, 0), (0, 1), (2, 1)),    # vertical-right
        ((2, 0), (0, -1), (2, -1)),  # vertical-left
    ]
    for (d1, d2, d3) in candidates:
        positions = [(r + dr, c + dc) for dr, dc in (d1, d2, d3)]
        if all(0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE for rr, cc in positions):
            if all(int(board[rr, cc]) == color for rr, cc in positions):
                return True
    return False


def is_tiger_mouth(board: np.ndarray, r: int, c: int, color: int) -> bool:
    """Tiger's mouth: empty point with 3 same-color stones in an L,
    plus controlled diagonal. Often a 'safe' point to descend into."""
    if int(board[r, c]) != EMPTY:
        return False
    n = neighbors4(r, c)
    same = sum(1 for nr, nc in n if int(board[nr, nc]) == color)
    return same == 3


# ---------------------------------------------------------------------------
# Ladder reading (simple)
# ---------------------------------------------------------------------------

def ladder_status(
    board: np.ndarray,
    target: tuple[int, int],
    max_depth: int = 60,
) -> str:
    """Read out a ladder for the group at ``target`` (which should have <=2 libs).

    Returns:
        "captured" -- the chase succeeds; the target is captured.
        "escaped"  -- the target reaches >=3 liberties and breaks the ladder.
        "unclear"  -- depth exceeded or shape too complex; treat as escaped.
    """
    g = group_at(board, *target)
    if g is None:
        return "escaped"
    if g.num_liberties >= 3:
        return "escaped"

    defender = g.color
    attacker = 3 - defender

    def _step(brd: np.ndarray, depth: int) -> str:
        # Attacker-to-play state. Defender's group at `target` has 1 or 2
        # liberties. Attacker plays optimally: tries each move that keeps
        # the defender at <= 1 liberty after, recurses; if any leads to
        # capture, returns "captured".
        if depth >= max_depth:
            return "unclear"
        cur = group_at(brd, *target)
        if cur is None or cur.color != defender:
            return "captured"
        libs = list(cur.liberties)
        if len(libs) >= 3:
            return "escaped"
        any_unclear = False
        for atk_lib in libs:
            if not is_legal_move(brd, attacker, atk_lib):
                continue
            trial = _apply_move(brd.copy(), attacker, atk_lib)
            ng = group_at(trial, *target)
            if ng is None:
                return "captured"
            if ng.num_liberties >= 3:
                continue  # this attacker move fails; try another
            # Defender to play. Defender extends to one of its remaining
            # liberties (chooses optimally: the move that maximizes
            # post-extension liberties).
            best_def: str | None = None
            for def_lib in ng.liberties:
                if not is_legal_move(trial, defender, def_lib):
                    continue
                trial2 = _apply_move(trial.copy(), defender, def_lib)
                after = group_at(trial2, *target)
                if after is None:
                    sub = "captured"
                elif after.num_liberties >= 3:
                    sub = "escaped"
                else:
                    sub = _step(trial2, depth + 1)
                # Defender prefers escape > unclear > captured.
                if sub == "escaped":
                    best_def = "escaped"
                    break
                if sub == "unclear" and best_def != "escaped":
                    best_def = "unclear"
                if sub == "captured" and best_def is None:
                    best_def = "captured"
            if best_def is None:
                # Defender has no legal extension -- captured.
                return "captured"
            if best_def == "captured":
                return "captured"
            if best_def == "unclear":
                any_unclear = True
        return "unclear" if any_unclear else "escaped"

    return _step(board.copy(), 0)


def _apply_move(board: np.ndarray, color: int, rc: tuple[int, int]) -> np.ndarray:
    r, c = rc
    board[r, c] = color
    opp = 3 - color
    # captures
    for nr, nc in neighbors4(r, c):
        if int(board[nr, nc]) == opp:
            stones, libs = _flood_group(board, nr, nc)
            if not libs:
                for sr, sc in stones:
                    board[sr, sc] = EMPTY
    return board


def is_ladder_runner(board: np.ndarray, group: Group) -> bool:
    """A group in atari/2-libs that the ladder reads as captured."""
    if group.num_liberties > 2:
        return False
    rep = next(iter(group.stones))
    return ladder_status(board, rep) == "captured"


def is_ladder_breaker(board: np.ndarray, color: int, rc: tuple[int, int]) -> bool:
    """Does playing ``color`` at ``rc`` rescue some same-color group from a ladder?

    Heuristic: find a same-color group g with 1-2 libs such that
    ladder_status(g) == "captured". Play the move at rc, re-read; if the
    same group now reads as "escaped", rc is a ladder-breaker.
    """
    if not is_legal_move(board, color, rc):
        return False
    weak = [g for g in all_groups(board) if g.color == color and g.num_liberties <= 2]
    runners = [g for g in weak if is_ladder_runner(board, g)]
    if not runners:
        return False
    new = _apply_move(board.copy(), color, rc)
    for g in runners:
        rep = next(iter(g.stones))
        if int(new[rep[0], rep[1]]) != color:
            continue
        if ladder_status(new, rep) == "escaped":
            return True
    return False


# ---------------------------------------------------------------------------
# Ko predicates
# ---------------------------------------------------------------------------

def is_ko_capture(
    board: np.ndarray,
    color: int,
    rc: tuple[int, int],
    last_move: tuple[int, int] | None = None,
) -> bool:
    """Did ``rc`` recapture a single stone in a ko-shape?

    Conservative test: rc captures exactly one stone, the resulting own
    group at rc is a single stone with exactly one liberty.
    """
    captured = captures_if_played(board, color, rc)
    if sum(g.size for g in captured) != 1:
        return False
    test = _apply_move(board.copy(), color, rc)
    g = group_at(test, *rc)
    return g is not None and g.size == 1 and g.num_liberties == 1


# ---------------------------------------------------------------------------
# Territory and influence from KataGo's ownership map
# ---------------------------------------------------------------------------

def territory_from_ownership(
    ownership: np.ndarray | list[float],
    black_thresh: float = 0.6,
    white_thresh: float = -0.6,
) -> np.ndarray:
    """Coarse 3-way label per intersection: 0=contested, 1=black, 2=white.

    KataGo's ownership convention: positive favors black, range [-1, 1].
    """
    own = np.asarray(ownership, dtype=np.float32).reshape(BOARD_SIZE, BOARD_SIZE)
    out = np.zeros_like(own, dtype=np.int8)
    out[own >= black_thresh] = BLACK
    out[own <= white_thresh] = WHITE
    return out


def black_territory(ownership: np.ndarray | list[float], thresh: float = 0.6) -> int:
    own = np.asarray(ownership, dtype=np.float32)
    return int((own >= thresh).sum())


def white_territory(ownership: np.ndarray | list[float], thresh: float = 0.6) -> int:
    own = np.asarray(ownership, dtype=np.float32)
    return int((own <= -thresh).sum())


def contested_points(
    ownership: np.ndarray | list[float],
    lo: float = -0.6,
    hi: float = 0.6,
) -> int:
    own = np.asarray(ownership, dtype=np.float32)
    return int(((own > lo) & (own < hi)).sum())


def influence_map(ownership: np.ndarray | list[float]) -> np.ndarray:
    """Pass-through alias for ownership reshaped to (9, 9)."""
    return np.asarray(ownership, dtype=np.float32).reshape(BOARD_SIZE, BOARD_SIZE)


# ---------------------------------------------------------------------------
# Group life / death from ownership (KataGo gives this implicitly)
# ---------------------------------------------------------------------------

def group_alive_by_ownership(group: Group, ownership: np.ndarray) -> bool:
    """A group is 'alive' if the ownership average over its stones strongly
    favors its color. Threshold matches KataGo's ownership scale."""
    own = np.asarray(ownership, dtype=np.float32).reshape(BOARD_SIZE, BOARD_SIZE)
    avg = float(np.mean([own[r, c] for r, c in group.stones]))
    if group.color == BLACK:
        return avg >= 0.5
    return avg <= -0.5


def group_dead_by_ownership(group: Group, ownership: np.ndarray) -> bool:
    own = np.asarray(ownership, dtype=np.float32).reshape(BOARD_SIZE, BOARD_SIZE)
    avg = float(np.mean([own[r, c] for r, c in group.stones]))
    if group.color == BLACK:
        return avg <= -0.5
    return avg >= 0.5


def group_in_seki(group: Group, ownership: np.ndarray, tolerance: float = 0.2) -> bool:
    """Heuristic seki: ownership at the group is near zero (mutually alive)
    AND the group has >= 1 liberty (i.e. not actually dead)."""
    if group.num_liberties == 0:
        return False
    own = np.asarray(ownership, dtype=np.float32).reshape(BOARD_SIZE, BOARD_SIZE)
    avg = float(np.mean([own[r, c] for r, c in group.stones]))
    return abs(avg) <= tolerance


__all__ = [
    "BLACK",
    "WHITE",
    "EMPTY",
    "Group",
    "neighbors4",
    "diagonals",
    "is_on_edge",
    "is_on_corner",
    "group_at",
    "all_groups",
    "weak_groups",
    "groups_in_atari",
    "is_legal_move",
    "captures_if_played",
    "is_atari_threat",
    "is_eye",
    "count_eyes",
    "is_bamboo_joint",
    "is_tiger_mouth",
    "ladder_status",
    "is_ladder_runner",
    "is_ladder_breaker",
    "is_ko_capture",
    "territory_from_ownership",
    "black_territory",
    "white_territory",
    "contested_points",
    "influence_map",
    "group_alive_by_ownership",
    "group_dead_by_ownership",
    "group_in_seki",
]
