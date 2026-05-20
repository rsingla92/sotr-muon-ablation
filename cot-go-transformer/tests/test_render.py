"""Board renderer tests (ASCII and SVG)."""

import numpy as np

from gogpt.render import board_to_ascii, board_to_svg


def _empty_board():
    return np.zeros((9, 9), dtype=np.int8)


def test_ascii_dims_and_header():
    b = _empty_board()
    out = board_to_ascii(b)
    lines = out.splitlines()
    # One column-label header + 9 rows + one column-label footer
    assert len(lines) == 11
    assert lines[0].strip().startswith("A")
    assert lines[-1].strip().startswith("A")
    # The row labels should be 9 .. 1 going down.
    assert lines[1].startswith("9 ")
    assert lines[9].startswith("1 ")


def test_ascii_star_points():
    b = _empty_board()
    out = board_to_ascii(b)
    # Center star at (4,4) should appear as '*'
    assert "*" in out


def test_ascii_stones_and_last_move():
    b = _empty_board()
    b[4, 4] = 1  # black at E5
    b[3, 4] = 2  # white at E6
    out = board_to_ascii(b, last_move=(4, 4))
    assert "X" in out
    assert "O" in out
    # last_move should be bracketed
    assert "[X]" in out


def test_ascii_annotations():
    b = _empty_board()
    out = board_to_ascii(b, annotations={(4, 4): "A"})
    # The center cell now reads 'A'
    assert " A " in out


def test_svg_contains_grid_and_stones():
    b = _empty_board()
    b[4, 4] = 1
    b[0, 0] = 2
    svg = board_to_svg(b, last_move=(4, 4))
    assert svg.startswith("<svg ")
    assert svg.endswith("</svg>")
    # Should contain at least 9*2 grid lines.
    assert svg.count("<line") >= 18
    # Should contain at least two stone circles.
    assert svg.count("<circle") >= 2


def test_svg_highlights():
    b = _empty_board()
    svg = board_to_svg(b, highlights={(4, 4): "#ff0000"})
    assert "#ff0000" in svg
