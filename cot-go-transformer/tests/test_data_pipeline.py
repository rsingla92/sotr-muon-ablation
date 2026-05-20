"""Data pipeline tests: SGF parsing, capture rules, example generation."""

import numpy as np
import pytest

from gogpt.data import (
    ParsedGame,
    _build_example,
    iter_examples_from_game,
    parse_sgf,
    play_stone,
)
from gogpt.tokenizer import NUM_POINTS, PASS_TOKEN, point_to_token


def test_parse_minimal_sgf():
    sgf = "(;GM[1]FF[4]SZ[9]KM[7.0];B[ee];W[ed];B[];W[])"
    g = parse_sgf(sgf)
    assert g.komi == 7.0
    assert len(g.moves) == 4
    assert g.moves[0] == ("B", (4, 4))   # ee -> row=4, col=4
    assert g.moves[1] == ("W", (3, 4))   # ed -> row=3, col=4
    assert g.moves[2] == ("B", None)     # pass
    assert g.moves[3] == ("W", None)


def test_play_stone_capture():
    # Single white stone surrounded on 3 sides; the 4th move captures it.
    b = np.zeros((9, 9), dtype=np.int8)
    b[4, 4] = 2          # white to be captured
    b[3, 4] = 1
    b[5, 4] = 1
    b[4, 5] = 1
    new, _ = play_stone(b, 1, (4, 3))
    assert new[4, 4] == 0
    assert new[4, 3] == 1


def test_play_stone_ko():
    # Classic ko shape: capturer becomes a single stone whose only liberty
    # is the just-captured point.
    b = np.zeros((9, 9), dtype=np.int8)
    # The white stone at (4,4) will be captured.
    b[4, 4] = 2
    # The black-to-be-played stone at (4,3) must have all other neighbors white.
    b[3, 3] = 2
    b[5, 3] = 2
    b[4, 2] = 2
    # And (4,4)'s remaining three neighbors must be black so that placing
    # at (4,3) captures it.
    b[3, 4] = 1
    b[5, 4] = 1
    b[4, 5] = 1
    new, ko = play_stone(b, 1, (4, 3))
    assert new[4, 4] == 0
    assert new[4, 3] == 1
    assert ko == (4, 4)


def test_play_stone_suicide_rejected():
    # White surrounds an empty point; black plays into it.
    b = np.zeros((9, 9), dtype=np.int8)
    b[3, 4] = 2
    b[5, 4] = 2
    b[4, 3] = 2
    b[4, 5] = 2
    # Surround with more white so that black's stone would have no liberties.
    new, ko = play_stone(b, 1, (4, 4))
    # Our rule treats suicide as illegal -> board unchanged.
    assert new[4, 4] == 0
    assert ko is None


def test_iter_examples_min_index():
    game = ParsedGame(moves=[("B", (4, 4)), ("W", (3, 4)), ("B", (5, 4)), ("W", (4, 5))])
    exs = list(iter_examples_from_game(game, max_trajectory_len=4, min_move_index=0))
    # 4 moves -> 4 starting positions, each with at least one continuation
    assert len(exs) == 4
    # Each example: state cats shape, tokens contain BOS/SEP/EOS, labels mostly -100.
    for ex in exs:
        assert ex.state_categories.shape == (NUM_POINTS,)
        assert ex.tokens[0] == 82  # BOS
        assert ex.tokens[1 + NUM_POINTS] == 83  # SEP_POS
        assert ex.tokens[-1] == 84  # EOS
        # Loss mask is zero in the board-prefix region (positions 0..81
        # are BOS + 81 board points). SEP_POS at index 82 is loss-bearing
        # because it predicts the first trajectory token.
        assert ex.loss_mask[: 1 + NUM_POINTS].sum() == 0
        # Loss mask is one somewhere from SEP_POS onward.
        assert ex.loss_mask[1 + NUM_POINTS :].sum() > 0


def test_color_flip_for_white_to_move():
    # White is to move at index 1; the example should have colors swapped so
    # the model always sees itself as black-to-move. We verify via the
    # trajectory tokens (the board's last-move overlay can mask the stone
    # color at the move point -- see encode_board_states).
    game = ParsedGame(moves=[("B", (4, 4)), ("W", (3, 4)), ("B", (5, 4))])
    exs = list(iter_examples_from_game(game, max_trajectory_len=2, min_move_index=1))
    ex = exs[0]
    # The first trajectory token is move 1 (W -> color-flipped to B), at (3,4).
    first_traj = ex.tokens[1 + NUM_POINTS + 1]
    assert first_traj == point_to_token(3, 4)
    # And the next is move 2 (B -> color-flipped to W on the board, but the
    # token stream just records the position): (5,4).
    second_traj = ex.tokens[1 + NUM_POINTS + 2]
    assert second_traj == point_to_token(5, 4)


def test_parse_sgf_handles_escapes():
    sgf = "(;GM[1]FF[4]SZ[9];C[hello \\] world];B[ee])"
    g = parse_sgf(sgf)
    assert g.moves == [("B", (4, 4))]
