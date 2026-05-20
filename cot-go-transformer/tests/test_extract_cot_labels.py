"""End-to-end tests for scripts/extract_cot_labels.py loss-mask construction.

The script's _build_example function decides which sequence positions
compute loss for each ablation mode. Getting that index right matters --
a one-off bug points the loss at the wrong token. These tests pin the
exact positions for each mode.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO / "scripts" / "extract_cot_labels.py"


def _load_script_module():
    sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location("extract_cot_labels", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _empty_board():
    return np.zeros((9, 9), dtype=np.int8)


def test_structured_loss_mask_covers_full_trajectory():
    mod = _load_script_module()
    board = _empty_board()
    cot = [101, 102, 103]            # 3 think tokens
    move_token = 40                  # E5
    ex = mod._build_example(
        board, ko=None, last_move=None,
        cot_tokens=cot, move_token=move_token,
        to_move="B", cot_supervised=True,
    )
    sep_index = 1 + 81
    # Trajectory length = THINK_OPEN + 3 cot + THINK_CLOSE + move = 6 tokens.
    # Loss-bearing positions: sep_index .. sep_index + 6 = 12 positions.
    assert int(ex.loss_mask.sum()) == 6 + 1
    assert ex.loss_mask[sep_index] == 1                    # SEP_POS predicts THINK_OPEN
    assert ex.loss_mask[sep_index + 5] == 1                # THINK_CLOSE predicts move
    assert ex.loss_mask[sep_index + 6] == 1                # move predicts EOS
    # Beyond the trajectory: zero.
    assert ex.loss_mask[sep_index + 7] == 0


def test_free_loss_mask_only_move_position():
    """In free-CoT, only the [</think>] position is loss-bearing, and its
    label is the move token."""
    mod = _load_script_module()
    board = _empty_board()
    cot = [101] * 10  # 10 random think tokens
    move_token = 40
    ex = mod._build_example(
        board, ko=None, last_move=None,
        cot_tokens=cot, move_token=move_token,
        to_move="B", cot_supervised=False,
    )
    sep_index = 1 + 81

    # Exactly one position is loss-bearing.
    assert int(ex.loss_mask.sum()) == 1

    # That position is [</think>] at sep_index + len(cot) + 2:
    #   sep_index    : SEP_POS
    #   +1           : THINK_OPEN
    #   +2..+11      : 10 cot tokens
    #   +12          : THINK_CLOSE  <-- loss here
    #   +13          : move
    think_close_pos = sep_index + len(cot) + 2
    move_pos = think_close_pos + 1
    assert ex.loss_mask[think_close_pos] == 1
    # The token AT that position must be THINK_CLOSE, not a content token.
    from gogpt.tokenizer import THINK_CLOSE_TOKEN
    assert ex.tokens[think_close_pos] == THINK_CLOSE_TOKEN
    # And its label must be the move token.
    assert ex.labels[think_close_pos] == move_token
    # The move token itself is at the next position.
    assert ex.tokens[move_pos] == move_token
    # The position just BEFORE [</think>] (last content token) must NOT
    # be loss-bearing -- this is the bug we caught.
    assert ex.loss_mask[think_close_pos - 1] == 0


def test_empty_mode_loss_mask():
    """Empty mode: trajectory is just <think></think><move>. Loss on
    SEP_POS, THINK_OPEN, THINK_CLOSE, and move (4 positions)."""
    mod = _load_script_module()
    board = _empty_board()
    ex = mod._build_example(
        board, ko=None, last_move=None,
        cot_tokens=[], move_token=40,
        to_move="B", cot_supervised=True,
    )
    assert int(ex.loss_mask.sum()) == 4
