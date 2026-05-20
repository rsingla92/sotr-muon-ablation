"""Go-concept rule library tests. CPU-only, no torch / no KataGo."""

import numpy as np
import pytest

from gogpt.concepts import (
    BLACK,
    WHITE,
    all_groups,
    black_territory,
    captures_if_played,
    contested_points,
    count_eyes,
    group_alive_by_ownership,
    group_at,
    group_dead_by_ownership,
    group_in_seki,
    groups_in_atari,
    influence_map,
    is_atari_threat,
    is_bamboo_joint,
    is_eye,
    is_ko_capture,
    is_ladder_breaker,
    is_ladder_runner,
    is_legal_move,
    is_on_corner,
    is_on_edge,
    is_tiger_mouth,
    ladder_status,
    neighbors4,
    territory_from_ownership,
    weak_groups,
    white_territory,
)


def _b():
    return np.zeros((9, 9), dtype=np.int8)


# ---- Geometry ----

def test_neighbors_count():
    assert len(neighbors4(0, 0)) == 2
    assert len(neighbors4(0, 4)) == 3
    assert len(neighbors4(4, 4)) == 4
    assert len(neighbors4(8, 8)) == 2


def test_edges_and_corners():
    assert is_on_corner(0, 0)
    assert is_on_corner(8, 8)
    assert not is_on_corner(0, 4)
    assert is_on_edge(0, 4)
    assert not is_on_edge(4, 4)


# ---- Groups ----

def test_single_stone_group():
    b = _b()
    b[4, 4] = BLACK
    g = group_at(b, 4, 4)
    assert g is not None
    assert g.color == BLACK
    assert g.size == 1
    assert g.num_liberties == 4


def test_chain_group_merges():
    b = _b()
    b[4, 4] = BLACK
    b[4, 5] = BLACK
    b[4, 6] = BLACK
    g = group_at(b, 4, 4)
    assert g is not None
    assert g.size == 3
    # 4 outer + 4 outer + 4 outer - 4 internal shared = 8 unique liberties
    assert g.num_liberties == 8


def test_corner_liberty_count():
    b = _b()
    b[0, 0] = BLACK
    g = group_at(b, 0, 0)
    assert g is not None
    assert g.num_liberties == 2


def test_all_groups_dedup():
    b = _b()
    b[2, 2] = BLACK
    b[2, 3] = BLACK
    b[5, 5] = WHITE
    gs = all_groups(b)
    assert len(gs) == 2
    sizes = sorted(g.size for g in gs)
    assert sizes == [1, 2]


# ---- Atari / weak groups ----

def test_atari_detected():
    b = _b()
    b[4, 4] = WHITE
    b[3, 4] = BLACK
    b[5, 4] = BLACK
    b[4, 5] = BLACK
    # White at (4,4) has only one liberty: (4,3)
    atari = groups_in_atari(b)
    assert len(atari) == 1
    assert atari[0].color == WHITE
    assert atari[0].num_liberties == 1


def test_weak_groups_threshold():
    b = _b()
    b[4, 4] = WHITE
    b[3, 4] = BLACK
    b[5, 4] = BLACK
    # White has 2 liberties.
    weak = weak_groups(b, max_liberties=2)
    assert any(g.color == WHITE for g in weak)


# ---- Legality ----

def test_suicide_illegal():
    b = _b()
    b[3, 4] = WHITE
    b[5, 4] = WHITE
    b[4, 3] = WHITE
    b[4, 5] = WHITE
    # Black playing at (4,4) is suicide.
    assert not is_legal_move(b, BLACK, (4, 4))


def test_capture_makes_otherwise_suicide_legal():
    b = _b()
    # White single stone with one liberty at (4,4).
    b[4, 4] = WHITE
    b[3, 4] = BLACK
    b[5, 4] = BLACK
    b[4, 5] = BLACK
    # Black playing at (4,3) captures white -> legal even though without
    # capture the move would have 1 self-liberty (at (4,4) which gets cleared).
    assert is_legal_move(b, BLACK, (4, 3))


def test_ko_point_blocked():
    b = _b()
    b[4, 4] = BLACK
    assert is_legal_move(b, WHITE, (3, 4))
    assert not is_legal_move(b, WHITE, (3, 4), ko_point=(3, 4))


# ---- Captures preview ----

def test_captures_if_played_single_stone():
    b = _b()
    b[4, 4] = WHITE
    b[3, 4] = BLACK
    b[5, 4] = BLACK
    b[4, 5] = BLACK
    caps = captures_if_played(b, BLACK, (4, 3))
    assert len(caps) == 1
    assert caps[0].size == 1
    assert (4, 4) in caps[0].stones


def test_captures_if_played_none():
    b = _b()
    b[4, 4] = WHITE
    caps = captures_if_played(b, BLACK, (3, 3))
    assert caps == []


# ---- Atari threat ----

def test_atari_threat_detection():
    b = _b()
    # White at (4,4) with 2 liberties already.
    b[4, 4] = WHITE
    b[3, 4] = BLACK
    b[5, 4] = BLACK
    # Black at (4,3) drops white to 1 liberty -> atari.
    assert is_atari_threat(b, BLACK, (4, 3))


# ---- Eye detection ----

def test_real_eye_center():
    b = _b()
    # Black surrounds (4,4) on all four sides AND controls diagonals.
    for (r, c) in [(3, 4), (5, 4), (4, 3), (4, 5)]:
        b[r, c] = BLACK
    for (r, c) in [(3, 3), (3, 5), (5, 3), (5, 5)]:
        b[r, c] = BLACK
    assert is_eye(b, 4, 4, BLACK)
    assert not is_eye(b, 4, 4, WHITE)


def test_false_eye_in_center():
    b = _b()
    # 4 adjacents are black; but 2 of 4 diagonals are white => false eye.
    for (r, c) in [(3, 4), (5, 4), (4, 3), (4, 5)]:
        b[r, c] = BLACK
    b[3, 3] = WHITE
    b[5, 5] = WHITE
    assert not is_eye(b, 4, 4, BLACK)


def test_corner_eye():
    b = _b()
    # (0,0) eye: needs (0,1) and (1,0) same-color; the single off-board
    # diagonal isn't penalized; (1,1) being opposite color WOULD invalidate
    # because corner threshold is 0.
    b[0, 1] = BLACK
    b[1, 0] = BLACK
    assert is_eye(b, 0, 0, BLACK)
    b[1, 1] = WHITE  # one bad diagonal on corner -> not an eye
    assert not is_eye(b, 0, 0, BLACK)


def test_count_eyes():
    b = _b()
    # Two trivial eyes for black.
    for (r, c) in [(3, 4), (5, 4), (4, 3), (4, 5), (3, 3), (3, 5), (5, 3), (5, 5)]:
        b[r, c] = BLACK
    # And another eye on the other side.
    for (r, c) in [(0, 1), (1, 0), (1, 1)]:
        b[r, c] = BLACK
    # (0,0) is now a corner eye; (4,4) is a center eye.
    n = count_eyes(b, BLACK)
    assert n >= 2


# ---- Shape ----

def test_tiger_mouth():
    b = _b()
    b[3, 4] = BLACK
    b[5, 4] = BLACK
    b[4, 5] = BLACK
    assert is_tiger_mouth(b, 4, 4, BLACK)
    assert not is_tiger_mouth(b, 4, 4, WHITE)


def test_bamboo_joint():
    b = _b()
    # Vertical bamboo: (4,4), (4,6), (5,4), (5,6) all black; (4,5)/(5,5) empty.
    b[4, 4] = BLACK
    b[4, 6] = BLACK
    b[5, 4] = BLACK
    b[5, 6] = BLACK
    assert is_bamboo_joint(b, 4, 4, BLACK)


# ---- Ladders ----

def test_ladder_captures_in_corner():
    """A 2-liberty group running into the corner with no breaker should die."""
    b = _b()
    # Setup: black stone at (1,1) in atari/2-libs with white blockers.
    b[1, 1] = BLACK
    b[0, 1] = WHITE
    b[1, 0] = WHITE
    # Black has 2 liberties: (1,2) and (2,1). Place a white blocker on one
    # diagonal so the chase is forced into the other direction.
    b[2, 2] = WHITE
    status = ladder_status(b, (1, 1))
    assert status == "captured"


def test_is_ladder_runner_basic():
    """A 2-liberty group that reads as captured is a runner."""
    b = _b()
    b[1, 1] = BLACK
    b[0, 1] = WHITE
    b[1, 0] = WHITE
    b[2, 2] = WHITE
    g = group_at(b, 1, 1)
    assert g is not None
    assert is_ladder_runner(b, g)


def test_is_ladder_breaker_negative_cases():
    """Without a real ladder, no move counts as a breaker."""
    b = _b()
    b[4, 4] = BLACK
    # No weak black groups -> nothing to break.
    assert not is_ladder_breaker(b, BLACK, (5, 5))


@pytest.mark.xfail(
    reason=(
        "Constructing a position where the chase direction is forced "
        "(so the breaker actually matters) requires careful Go-position "
        "design. Defer: Phase 3 can use KataGo's isLadderCapture / "
        "isLadderEscape from moveInfos as ground truth."
    ),
    strict=False,
)
def test_ladder_escapes_with_breaker():
    b = _b()
    b[1, 1] = BLACK
    b[0, 1] = WHITE
    b[1, 0] = WHITE
    b[2, 2] = WHITE
    b[3, 1] = BLACK
    status = ladder_status(b, (1, 1))
    assert status == "escaped"


# ---- Ko ----

def test_ko_capture_detected():
    b = _b()
    # Ko shape (from earlier data tests).
    b[4, 4] = WHITE
    b[3, 3] = WHITE
    b[5, 3] = WHITE
    b[4, 2] = WHITE
    b[3, 4] = BLACK
    b[5, 4] = BLACK
    b[4, 5] = BLACK
    assert is_ko_capture(b, BLACK, (4, 3))


def test_ko_capture_negative():
    b = _b()
    b[4, 4] = WHITE
    b[3, 4] = BLACK
    b[5, 4] = BLACK
    b[4, 5] = BLACK
    # Capturer at (4,3) becomes a 1-stone group with 3 liberties (open space)
    # so this is NOT a ko-capture.
    assert not is_ko_capture(b, BLACK, (4, 3))


# ---- Territory / influence (ownership-based) ----

def test_territory_from_ownership():
    own = np.zeros((9, 9), dtype=np.float32)
    own[0:3, :] = 0.9   # black solid territory
    own[6:9, :] = -0.9  # white solid territory
    own[3:6, :] = 0.0   # neutral
    t = territory_from_ownership(own)
    assert (t[0:3, :] == BLACK).all()
    assert (t[6:9, :] == WHITE).all()
    assert (t[3:6, :] == 0).all()
    assert black_territory(own) == 27
    assert white_territory(own) == 27
    assert contested_points(own) == 27


def test_influence_map_shape():
    own = list(np.linspace(-1, 1, 81))
    inf = influence_map(own)
    assert inf.shape == (9, 9)
    assert np.isclose(inf.min(), -1.0)
    assert np.isclose(inf.max(), 1.0)


def test_group_life_by_ownership():
    b = _b()
    b[4, 4] = BLACK
    g = group_at(b, 4, 4)
    assert g is not None
    own_alive = np.zeros((9, 9), dtype=np.float32)
    own_alive[4, 4] = 0.9
    assert group_alive_by_ownership(g, own_alive)
    assert not group_dead_by_ownership(g, own_alive)
    own_dead = np.zeros((9, 9), dtype=np.float32)
    own_dead[4, 4] = -0.9
    assert group_dead_by_ownership(g, own_dead)
    own_seki = np.zeros((9, 9), dtype=np.float32)
    assert group_in_seki(g, own_seki)
