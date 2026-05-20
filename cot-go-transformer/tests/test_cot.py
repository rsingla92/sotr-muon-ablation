"""CoT vocab and extractor tests."""

import numpy as np

from gogpt import concepts as C
from gogpt import cot_vocab as V
from gogpt.cot_extractor import (
    FREE_MODE_LENGTH,
    extract_think_block,
    wrap_with_think_tags,
)
from gogpt.tokenizer import (
    PASS_TOKEN,
    PHASE0_VOCAB_SIZE,
    THINK_CLOSE_TOKEN,
    THINK_OPEN_TOKEN,
    VOCAB_SIZE,
    point_to_token,
)


def test_vocab_ids_in_reserved_range():
    for tid in V.all_think_token_ids():
        assert PHASE0_VOCAB_SIZE <= tid < VOCAB_SIZE, (tid,)
    assert V.THINK_TOKENS_USED <= (VOCAB_SIZE - PHASE0_VOCAB_SIZE)


def test_vocab_ids_unique():
    ids = V.all_think_token_ids()
    assert len(ids) == len(set(ids))


def test_token_ids_clustered_by_category():
    """Each category occupies a contiguous block of IDs."""
    for name, (lo, hi) in V.CATEGORY_RANGES.items():
        block = list(range(lo, hi))
        assert all(t in V._REGISTRY for t in block), (
            f"category {name} has gaps in {lo}..{hi}"
        )
    # Categories don't overlap
    seen: set[int] = set()
    for name, (lo, hi) in V.CATEGORY_RANGES.items():
        block = set(range(lo, hi))
        assert not (block & seen), f"category {name} overlaps another"
        seen |= block


def test_winrate_bin_boundaries():
    assert V.winrate_bin_token(0.5) == V.WR_BINS[10]
    assert V.winrate_bin_token(0.45) == V.WR_BINS[10]
    assert V.winrate_bin_token(0.55) == V.WR_BINS[10]
    assert V.winrate_bin_token(0.0) == V.WR_BINS[0]
    assert V.winrate_bin_token(0.99) == V.WR_BINS[9]
    assert V.winrate_bin_token(-0.1) == V.WR_BINS[0]
    assert V.winrate_bin_token(1.5) == V.WR_BINS[9]


def test_score_lead_buckets():
    assert V.score_lead_token(40) == V.SL_B_DOM
    assert V.score_lead_token(15) == V.SL_B_BIG
    assert V.score_lead_token(7) == V.SL_B_MED
    assert V.score_lead_token(3) == V.SL_B_SMALL
    assert V.score_lead_token(1) == V.SL_B_TINY
    assert V.score_lead_token(0.0) == V.SL_EVEN
    assert V.score_lead_token(-1) == V.SL_W_TINY
    assert V.score_lead_token(-7) == V.SL_W_MED
    assert V.score_lead_token(-30) == V.SL_W_DOM


def test_phase_tokens():
    assert V.phase_token(0) == V.PH_OPENING
    assert V.phase_token(20) == V.PH_MIDGAME
    assert V.phase_token(45) == V.PH_LATE_MID
    assert V.phase_token(80) == V.PH_ENDGAME


def test_group_status_token():
    assert V.group_status_token(num_liberties=5, dead=False, seki=False) == V.GRP_ALIVE
    assert V.group_status_token(num_liberties=1, dead=False, seki=False) == V.GRP_WEAK_1
    assert V.group_status_token(num_liberties=2, dead=False, seki=False) == V.GRP_WEAK_2
    assert V.group_status_token(num_liberties=5, dead=True, seki=False) == V.GRP_DEAD
    assert V.group_status_token(num_liberties=5, dead=False, seki=True) == V.GRP_SEKI
    assert V.group_status_token(num_liberties=5, dead=True, seki=True) == V.GRP_SEKI


def test_confidence_winrate_gap():
    # Big gap -> HIGH
    assert V.confidence_token(0.70, 0.50) == V.CONF_HIGH
    # Medium gap (3-10pp)
    assert V.confidence_token(0.60, 0.55) == V.CONF_MED
    # Small gap -> LOW
    assert V.confidence_token(0.52, 0.51) == V.CONF_LOW
    # Missing runner-up -> LOW (insufficient info)
    assert V.confidence_token(0.70, None) == V.CONF_LOW
    assert V.confidence_token(None, 0.50) == V.CONF_LOW


def test_extract_minimal_position_emission_order():
    """Empty board, simple analysis -> a short well-formed think block.

    Verifies the new emission order:
      WR, SL, (no facts -> NO_FACTS), PH, SEP_FACTS, TOP_MOVE, vertex, CONF
    """
    b = np.zeros((9, 9), dtype=np.int8)
    katago = {
        "to_move": "B",
        "root_winrate": 0.52,
        "root_score_lead": 0.5,
        "top_moves": [
            {"move": "E5", "visits": 400, "winrate": 0.55, "score_lead": 1.0, "prior": 0.3, "order": 0},
            {"move": "G5", "visits": 50, "winrate": 0.45, "score_lead": 0.7, "prior": 0.1, "order": 1},
        ],
        "ownership": None,
    }
    out = extract_think_block(b, katago, move_number=0, played_move_rc=(4, 4))

    # Required tokens present
    assert V.WR_BINS[10] in out          # WR_EVEN
    assert V.SL_B_TINY in out
    assert V.PH_OPENING in out
    assert V.NO_FACTS in out             # empty board -> no weak groups, no tactics
    assert V.SEP_FACTS in out
    assert V.TOP_MOVE in out
    assert point_to_token(4, 4) in out

    # Order check: phase comes AFTER NO_FACTS (was the reorder fix)
    no_facts_idx = out.index(V.NO_FACTS)
    phase_idx = out.index(V.PH_OPENING)
    sep_idx = out.index(V.SEP_FACTS)
    assert no_facts_idx < phase_idx < sep_idx

    # SEP_FACTS precedes TOP_MOVE precedes the move-vertex
    top_idx = out.index(V.TOP_MOVE)
    vertex_idx = out.index(point_to_token(4, 4))
    assert sep_idx < top_idx < vertex_idx

    # Confidence: winrate gap 0.55-0.45 = 0.10 -> HIGH
    assert V.CONF_HIGH in out


def test_top_move_is_played_move_not_katago_top():
    """The CoT explains the move we're about to play, even if it isn't KataGo's top."""
    b = np.zeros((9, 9), dtype=np.int8)
    katago = {
        "to_move": "B",
        "root_winrate": 0.6,
        "root_score_lead": 2.0,
        "top_moves": [
            {"move": "E5", "visits": 500, "winrate": 0.65, "score_lead": 3.0, "prior": 0.5, "order": 0},
        ],
        "ownership": None,
    }
    # We're playing G3 (not KataGo's E5).
    played = (6, 6)  # G3 (row 6 = "3", col 6 = "G")
    out = extract_think_block(b, katago, move_number=5, played_move_rc=played)
    # TOP_MOVE should bind to the PLAYED move (6,6), not KataGo's (4,4).
    assert point_to_token(*played) in out
    assert point_to_token(4, 4) not in out


def test_tactics_evaluated_against_played_move():
    """Played-move D5 captures a stone; CoT should emit TAC_CAPTURE even
    though KataGo's top isn't necessarily D5."""
    b = np.zeros((9, 9), dtype=np.int8)
    b[4, 4] = C.WHITE
    b[3, 4] = C.BLACK
    b[5, 4] = C.BLACK
    b[4, 5] = C.BLACK
    katago = {
        "to_move": "B",
        "root_winrate": 0.9,
        "root_score_lead": 10.0,
        # KataGo lists some unrelated move as top; we still emit TAC_CAPTURE
        # because the PLAYED move at (4,3) captures.
        "top_moves": [
            {"move": "B7", "visits": 50, "winrate": 0.4, "score_lead": -2.0, "prior": 0.2, "order": 0},
        ],
        "ownership": None,
    }
    out = extract_think_block(b, katago, move_number=8, played_move_rc=(4, 3))
    assert V.TAC_CAPTURE in out


def test_extract_with_weak_group_emit():
    b = np.zeros((9, 9), dtype=np.int8)
    b[1, 1] = C.BLACK
    b[0, 1] = C.WHITE
    b[1, 0] = C.WHITE
    katago = {
        "to_move": "B",
        "root_winrate": 0.4,
        "root_score_lead": -2.0,
        "top_moves": [
            {"move": "C2", "visits": 200, "winrate": 0.45, "score_lead": -1.0, "prior": 0.2, "order": 0},
        ],
        "ownership": None,
    }
    out = extract_think_block(b, katago, move_number=5, played_move_rc=(7, 2))
    assert V.GRP_WEAK_2 in out
    assert V.AT_VERTEX in out
    assert point_to_token(1, 1) in out


def test_extract_with_ownership_no_crash():
    b = np.zeros((9, 9), dtype=np.int8)
    b[4, 4] = C.BLACK
    b[4, 5] = C.BLACK
    own = np.zeros(81, dtype=np.float32)
    own[4 * 9 + 4] = -0.9
    own[4 * 9 + 5] = -0.9
    katago = {
        "to_move": "B",
        "root_winrate": 0.1,
        "root_score_lead": -15.0,
        "top_moves": [
            {"move": "pass", "visits": 100, "winrate": 0.1, "score_lead": -15.0, "prior": 0.5, "order": 0},
        ],
        "ownership": own.tolist(),
    }
    out = extract_think_block(b, katago, move_number=30, played_move_rc=None)
    assert V.PH_MIDGAME in out
    # pass -> PASS_TOKEN in the TOP_MOVE slot
    top_idx = out.index(V.TOP_MOVE)
    assert out[top_idx + 1] == PASS_TOKEN


def test_mode_empty():
    """Mode 'empty' returns no inner tokens."""
    b = np.zeros((9, 9), dtype=np.int8)
    katago = {
        "to_move": "B",
        "root_winrate": 0.5,
        "root_score_lead": 0.0,
        "top_moves": [],
        "ownership": None,
    }
    out = extract_think_block(b, katago, move_number=0, mode="empty")
    assert out == []
    wrapped = wrap_with_think_tags(out)
    assert wrapped == [THINK_OPEN_TOKEN, THINK_CLOSE_TOKEN]


def test_mode_free_length_and_token_range():
    """Mode 'free' returns FREE_MODE_LENGTH tokens, all within the think vocab."""
    b = np.zeros((9, 9), dtype=np.int8)
    katago = {
        "to_move": "B",
        "root_winrate": 0.5,
        "root_score_lead": 0.0,
        "top_moves": [],
        "ownership": None,
    }
    out = extract_think_block(b, katago, move_number=0, mode="free")
    assert len(out) == FREE_MODE_LENGTH
    valid_ids = set(V.all_think_token_ids())
    for t in out:
        assert t in valid_ids


def test_mode_free_deterministic_for_same_position():
    """Free-mode token sequence is deterministic given the same (board, move_number)
    so that re-running label extraction gives identical training data."""
    b = np.zeros((9, 9), dtype=np.int8)
    b[4, 4] = C.BLACK
    katago = {
        "to_move": "B",
        "root_winrate": 0.5,
        "root_score_lead": 0.0,
        "top_moves": [],
        "ownership": None,
    }
    a = extract_think_block(b, katago, move_number=7, mode="free")
    b_out = extract_think_block(b, katago, move_number=7, mode="free")
    assert a == b_out


def test_wrap_with_think_tags():
    inner = [V.WR_BINS[0], V.PH_OPENING, V.SEP_FACTS]
    wrapped = wrap_with_think_tags(inner)
    assert wrapped[0] == THINK_OPEN_TOKEN
    assert wrapped[-1] == THINK_CLOSE_TOKEN
    assert wrapped[1:-1] == inner


def test_token_name_round_trip():
    for tid in V.all_think_token_ids():
        name = V.token_name(tid)
        assert not name.startswith("<unknown")


def test_extract_unknown_mode_raises():
    import pytest
    b = np.zeros((9, 9), dtype=np.int8)
    with pytest.raises(ValueError):
        extract_think_block(b, {"top_moves": []}, move_number=0, mode="bogus")
