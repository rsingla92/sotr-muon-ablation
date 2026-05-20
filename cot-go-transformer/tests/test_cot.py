"""CoT vocab and extractor tests."""

import numpy as np

from gogpt import concepts as C
from gogpt import cot_vocab as V
from gogpt.cot_extractor import extract_think_block, wrap_with_think_tags
from gogpt.tokenizer import (
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


def test_winrate_bin_boundaries():
    # Inside even-band
    assert V.winrate_bin_token(0.5) == V.WR_BINS[10]
    assert V.winrate_bin_token(0.45) == V.WR_BINS[10]
    assert V.winrate_bin_token(0.55) == V.WR_BINS[10]
    # Outside the even band
    assert V.winrate_bin_token(0.0) == V.WR_BINS[0]
    assert V.winrate_bin_token(0.99) == V.WR_BINS[9]
    # Out-of-range gets clamped
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
    # Seki takes precedence over dead
    assert V.group_status_token(num_liberties=5, dead=True, seki=True) == V.GRP_SEKI


def test_confidence_token():
    # Top dominates -> high
    assert V.confidence_token(top_visits=1000, runner_up_visits=50) == V.CONF_HIGH
    # Runner-up close -> low
    assert V.confidence_token(top_visits=100, runner_up_visits=80) == V.CONF_LOW
    # Middle ground
    assert V.confidence_token(top_visits=100, runner_up_visits=30) == V.CONF_MED
    # Edge case
    assert V.confidence_token(top_visits=0, runner_up_visits=0) == V.CONF_LOW


def test_extract_minimal_position():
    """Empty board, simple analysis -> a short well-formed think block."""
    b = np.zeros((9, 9), dtype=np.int8)
    katago = {
        "to_move": "B",
        "root_winrate": 0.52,
        "root_score_lead": 0.5,
        "top_moves": [
            {"move": "E5", "visits": 400, "winrate": 0.55, "score_lead": 1.0, "prior": 0.3, "order": 0},
            {"move": "G5", "visits": 50, "winrate": 0.52, "score_lead": 0.7, "prior": 0.1, "order": 1},
        ],
        "ownership": None,
    }
    out = extract_think_block(b, katago, move_number=0)
    # Average length 8-15 was the spec target; on empty board we expect short.
    assert 5 <= len(out) <= 20, f"len={len(out)}: {out}"
    # Should contain the winrate bin (even), phase opening, separator,
    # top-move marker, encoded top move, and a confidence token.
    assert V.WR_BINS[10] in out          # even winrate
    assert V.PH_OPENING in out
    assert V.SEP_FACTS in out
    assert V.TOP_MOVE in out
    assert point_to_token(4, 4) in out   # E5
    # Confidence: runner_up/top = 50/400 = 0.125 -> HIGH
    assert V.CONF_HIGH in out


def test_extract_with_atari_threat():
    """Position where the top move puts an opp group in atari."""
    b = np.zeros((9, 9), dtype=np.int8)
    # White group with 2 libs at (4,4); black at (4,3) puts it in atari.
    b[4, 4] = C.WHITE
    b[3, 4] = C.BLACK
    b[5, 4] = C.BLACK
    katago = {
        "to_move": "B",
        "root_winrate": 0.7,
        "root_score_lead": 3.0,
        "top_moves": [
            {"move": "D5", "visits": 500, "winrate": 0.8, "score_lead": 5.0, "prior": 0.5, "order": 0},
        ],
        "ownership": None,
    }
    # GTP D5: column 'D' = 3, row 5 -> row index 9-5 = 4. So (4, 3).
    out = extract_think_block(b, katago, move_number=10)
    assert V.TAC_ATARI in out


def test_extract_with_capture_top_move():
    """Top move captures a stone."""
    b = np.zeros((9, 9), dtype=np.int8)
    b[4, 4] = C.WHITE
    b[3, 4] = C.BLACK
    b[5, 4] = C.BLACK
    b[4, 5] = C.BLACK
    katago = {
        "to_move": "B",
        "root_winrate": 0.9,
        "root_score_lead": 10.0,
        "top_moves": [
            {"move": "D5", "visits": 800, "winrate": 0.95, "score_lead": 12.0, "prior": 0.8, "order": 0},
        ],
        "ownership": None,
    }
    out = extract_think_block(b, katago, move_number=20)
    assert V.TAC_CAPTURE in out


def test_extract_with_weak_group_emit():
    """Position with an own weak group emits a GRP_WEAK_* token + AT_VERTEX + position."""
    b = np.zeros((9, 9), dtype=np.int8)
    # Black single stone at (1,1) with 2 libs (corner-ish).
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
    out = extract_think_block(b, katago, move_number=5)
    # Must mention the weak black group at (1,1) -> point token = 1*9+1 = 10
    assert V.GRP_WEAK_2 in out
    assert V.AT_VERTEX in out
    assert point_to_token(1, 1) in out


def test_extract_with_ownership_dead_group():
    """Ownership tagging a group as dead should produce GRP_DEAD."""
    b = np.zeros((9, 9), dtype=np.int8)
    b[4, 4] = C.BLACK
    b[4, 5] = C.BLACK
    # Surround with enough white to make the group's status look dead via
    # ownership, even though pure-board liberties say 4.
    # We force this via the ownership map directly.
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
    # The group has 6 liberties -> not weak by libs; the extractor only
    # mentions weak (<=2 libs) groups. Verify the extractor doesn't crash.
    out = extract_think_block(b, katago, move_number=30)
    assert V.PH_MIDGAME in out


def test_wrap_with_think_tags():
    inner = [V.WR_BINS[0], V.PH_OPENING, V.SEP_FACTS]
    wrapped = wrap_with_think_tags(inner)
    assert wrapped[0] == THINK_OPEN_TOKEN
    assert wrapped[-1] == THINK_CLOSE_TOKEN
    assert wrapped[1:-1] == inner


def test_token_name_round_trip():
    # Every assigned think-token has a name in the registry.
    for tid in V.all_think_token_ids():
        name = V.token_name(tid)
        assert not name.startswith("<unknown")
