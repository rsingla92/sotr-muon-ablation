"""Tokenizer + position-encoding tests. CPU-only, no KataGo required."""

import numpy as np
import pytest

from gogpt.tokenizer import (
    BOARD_SIZE,
    BOS_TOKEN,
    EOS_TOKEN,
    NUM_POINTS,
    PASS_TOKEN,
    SEP_POS_TOKEN,
    STATE_BLACK,
    STATE_EMPTY,
    STATE_LAST_MOVE,
    STATE_WHITE,
    build_input_token_stream,
    build_loss_mask,
    encode_board_states,
    gtp_vertex_to_token,
    point_to_token,
    token_to_gtp_vertex,
    token_to_point,
)


def test_point_token_roundtrip():
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            t = point_to_token(r, c)
            assert 0 <= t < NUM_POINTS
            assert token_to_point(t) == (r, c)
    assert token_to_point(PASS_TOKEN) is None


def test_pass_token_roundtrip():
    assert token_to_gtp_vertex(PASS_TOKEN) == "pass"
    assert gtp_vertex_to_token("pass") == PASS_TOKEN
    assert gtp_vertex_to_token("PASS") == PASS_TOKEN


def test_gtp_vertex_known_points():
    # A1 is the bottom-left in GTP. In our row-major board, that's row=8, col=0.
    assert gtp_vertex_to_token("A1") == point_to_token(8, 0)
    # J9 is top-right (column 'J' skips 'I'): row=0, col=8.
    assert gtp_vertex_to_token("J9") == point_to_token(0, 8)
    # E5 is the center for 9x9: row=4, col=4.
    assert gtp_vertex_to_token("E5") == point_to_token(4, 4)
    # Round-trip through token.
    for v in ["A1", "B2", "D4", "E5", "J9", "C7", "H3"]:
        assert token_to_gtp_vertex(gtp_vertex_to_token(v)) == v


def test_encode_board_states_basic():
    b = np.zeros((9, 9), dtype=np.int8)
    b[4, 4] = 1  # black at E5
    b[3, 4] = 2  # white at E6
    cats = encode_board_states(b, ko_point=None, last_move=(4, 4))
    assert cats.shape == (81,)
    # Last-move overrides stone category; (4,4) is row-major index 4*9+4=40
    assert cats[40] == STATE_LAST_MOVE
    assert cats[3 * 9 + 4] == STATE_WHITE
    # Empty point
    assert cats[0] == STATE_EMPTY


def test_input_token_stream_shape():
    traj = np.array([point_to_token(4, 4), point_to_token(3, 4), PASS_TOKEN], dtype=np.int32)
    seq = build_input_token_stream(traj)
    assert seq.shape == (1 + 81 + 1 + 3 + 1,)
    assert seq[0] == BOS_TOKEN
    assert seq[1 + 81] == SEP_POS_TOKEN
    assert seq[-1] == EOS_TOKEN
    assert seq[1 + 81 + 1] == point_to_token(4, 4)


def test_loss_mask_alignment():
    mask = build_loss_mask(trajectory_length=5)
    assert mask.shape == (1 + 81 + 1 + 5 + 1,)
    sep_index = 1 + 81
    # Positions 82..86 (SEP_POS and 4 trajectory tokens that have a next) get mask=1
    # Plus position 87 (the last trajectory token) predicts EOS at 88.
    assert int(mask[sep_index]) == 1
    assert int(mask[sep_index + 5]) == 1
    # The EOS position itself has no successor; should be 0.
    assert int(mask[-1]) == 0


@pytest.mark.parametrize("seed", range(20))
def test_position_encode_roundtrip_random(seed):
    """Encode a random board, then verify each square decodes correctly."""
    rng = np.random.default_rng(seed)
    b = rng.integers(0, 3, size=(9, 9), dtype=np.int8)
    cats = encode_board_states(b)
    for r in range(9):
        for c in range(9):
            idx = r * 9 + c
            v = b[r, c]
            expected = (STATE_EMPTY, STATE_BLACK, STATE_WHITE)[v]
            assert cats[idx] == expected
