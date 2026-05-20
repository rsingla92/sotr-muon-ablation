"""Board renderers.

ASCII for terminal-friendly inspection of positions and SGFs; SVG for the
Phase-3 pathway visualizer (HTML page). The renderers are dependency-free
(no matplotlib) so they run anywhere numpy runs.
"""

from __future__ import annotations

import numpy as np

from . import BOARD_SIZE

# GTP column labels skip 'I'.
GTP_COLS = "ABCDEFGHJ"


def board_to_ascii(
    board: np.ndarray,
    *,
    last_move: tuple[int, int] | None = None,
    ko_point: tuple[int, int] | None = None,
    annotations: dict[tuple[int, int], str] | None = None,
) -> str:
    """Multi-line ASCII rendering of a 9x9 board.

    Symbols:
        X   black stone
        O   white stone
        .   empty
        *   star points (3-3, 3-7, 5-5, 7-3, 7-7 on 9x9)
        []  highlights the last move (e.g. [X])
        #   marks the ko-banned point
    Annotations override the stone glyph for that cell, e.g. {(4, 4): 'A'}.
    """
    star_points = {(2, 2), (2, 6), (4, 4), (6, 2), (6, 6)}
    rows: list[str] = []
    rows.append("   " + " ".join(GTP_COLS))
    for r in range(BOARD_SIZE):
        cells: list[str] = []
        for c in range(BOARD_SIZE):
            stone = int(board[r, c])
            glyph = "."
            if (r, c) in star_points and stone == 0:
                glyph = "*"
            if stone == 1:
                glyph = "X"
            elif stone == 2:
                glyph = "O"
            if ko_point == (r, c):
                glyph = "#"
            if annotations and (r, c) in annotations:
                glyph = annotations[(r, c)]
            if last_move == (r, c):
                cells.append(f"[{glyph}]")
            else:
                cells.append(f" {glyph} ")
        gtp_row = BOARD_SIZE - r
        rows.append(f"{gtp_row} " + "".join(cells).rstrip())
    rows.append("   " + " ".join(GTP_COLS))
    return "\n".join(rows)


def board_to_svg(
    board: np.ndarray,
    *,
    last_move: tuple[int, int] | None = None,
    highlights: dict[tuple[int, int], str] | None = None,
    width: int = 360,
) -> str:
    """SVG rendering. ``highlights`` maps (r, c) -> CSS color string for an
    overlay circle (used by the pathway viz to mark features-of-interest).
    """
    n = BOARD_SIZE
    margin = 24
    cell = (width - 2 * margin) / (n - 1)
    height = width
    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        'style="background:#dcb35c">'
    )
    # Grid
    for i in range(n):
        x = margin + i * cell
        parts.append(f'<line x1="{x}" y1="{margin}" x2="{x}" y2="{height - margin}" stroke="#000" stroke-width="1"/>')
        parts.append(f'<line x1="{margin}" y1="{x}" x2="{width - margin}" y2="{x}" stroke="#000" stroke-width="1"/>')
    # Star points
    for (r, c) in [(2, 2), (2, 6), (4, 4), (6, 2), (6, 6)]:
        cx = margin + c * cell
        cy = margin + r * cell
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="2.5" fill="#000"/>')
    # Stones
    radius = cell * 0.45
    for r in range(n):
        for c in range(n):
            v = int(board[r, c])
            if v == 0:
                continue
            cx = margin + c * cell
            cy = margin + r * cell
            color = "#111" if v == 1 else "#fafafa"
            stroke = "#000" if v == 2 else "none"
            parts.append(
                f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="{color}" '
                f'stroke="{stroke}" stroke-width="1"/>'
            )
            if last_move == (r, c):
                # small dot on the last move
                dot_color = "#fafafa" if v == 1 else "#111"
                parts.append(f'<circle cx="{cx}" cy="{cy}" r="{radius/3}" fill="{dot_color}"/>')
    # Highlight overlays
    if highlights:
        for (r, c), color in highlights.items():
            cx = margin + c * cell
            cy = margin + r * cell
            parts.append(
                f'<circle cx="{cx}" cy="{cy}" r="{radius * 0.7}" fill="none" '
                f'stroke="{color}" stroke-width="3"/>'
            )
    # Column / row labels
    for i in range(n):
        x = margin + i * cell
        parts.append(
            f'<text x="{x}" y="{margin - 8}" text-anchor="middle" font-size="11" font-family="monospace">{GTP_COLS[i]}</text>'
        )
        parts.append(
            f'<text x="{margin - 12}" y="{margin + i * cell + 4}" text-anchor="end" font-size="11" font-family="monospace">{n - i}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


__all__ = ["board_to_ascii", "board_to_svg"]
