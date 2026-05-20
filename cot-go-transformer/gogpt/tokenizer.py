"""Hybrid prefix-LM tokenizer for 9x9 Go.

The model sees two regions in its input sequence:

1. **Board prefix** (positions 1..81). One input embedding per intersection.
   Each embedding is a learned vector indexed by the per-point *state category*
   (empty / black / white / ko-banned / last-move). These are NOT entries in
   the move vocabulary -- they're a separate embedding table, looked up by the
   board-prefix path. The prefix is preceded by [BOS] and followed by
   [SEP_POS].

2. **Trajectory** (positions 82..). Move tokens (one per ply played from this
   position), optionally interleaved with [<think>] ... [</think>] reasoning
   blocks (filled in Phase 1).

Loss is computed only on trajectory tokens. The tokenizer exposes:
- ``encode_position(board) -> (state_categories, move_prefix_tokens)`` where
  ``state_categories`` is the length-81 array of board-square categories and
  ``move_prefix_tokens`` is [BOS, ...SEP_POS] used as the literal token-ID
  stream wrapping the prefix (see ``Tokenizer.build_input_sequence``).
- ``encode_game(sgf_bytes) -> List[EncodedPosition]`` one entry per move.
- A round-trip ``decode_move(token_id) -> vertex`` for legality / SGF output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import BOARD_SIZE, NUM_POINTS

# ---------------------------------------------------------------------------
# Board-square state categories (input embedding indices, NOT vocab tokens)
# ---------------------------------------------------------------------------

STATE_EMPTY = 0
STATE_BLACK = 1
STATE_WHITE = 2
STATE_KO_BANNED = 3
STATE_LAST_MOVE = 4
NUM_STATE_CATEGORIES = 5

# ---------------------------------------------------------------------------
# Move + special token vocab
# ---------------------------------------------------------------------------
# Layout (Phase 0):
#   0..80     : board points, row-major  (point_token(r, c) = r * 9 + c)
#   81        : PASS
#   82        : BOS
#   83        : SEP_POS
#   84        : EOS
#   85        : THINK_OPEN     [<think>]
#   86        : THINK_CLOSE    [</think>]
#   87..      : reserved for Phase 1 structured think-block vocab
#
# Total Phase-0 vocab size = 87 (Phase 1 will extend with ~200 think-tokens).

PASS_TOKEN = NUM_POINTS         # 81
BOS_TOKEN = NUM_POINTS + 1      # 82
SEP_POS_TOKEN = NUM_POINTS + 2  # 83
EOS_TOKEN = NUM_POINTS + 3      # 84
THINK_OPEN_TOKEN = NUM_POINTS + 4   # 85
THINK_CLOSE_TOKEN = NUM_POINTS + 5  # 86

PHASE0_VOCAB_SIZE = NUM_POINTS + 6  # 87
# Reserve room for Phase 1 think-block tokens.
RESERVED_THINK_VOCAB = 200
VOCAB_SIZE = PHASE0_VOCAB_SIZE + RESERVED_THINK_VOCAB

MOVE_TOKEN_IDS: frozenset[int] = frozenset(range(NUM_POINTS + 1))  # 0..81 incl. PASS


def point_to_token(row: int, col: int) -> int:
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        raise ValueError(f"point ({row},{col}) out of range for 9x9 board")
    return row * BOARD_SIZE + col


def token_to_point(token: int) -> tuple[int, int] | None:
    """Decode a move token to (row, col), or None for PASS."""
    if token == PASS_TOKEN:
        return None
    if not (0 <= token < NUM_POINTS):
        raise ValueError(f"token {token} is not a board-point move")
    return divmod(token, BOARD_SIZE)


def token_to_gtp_vertex(token: int) -> str:
    """Convert a move token to a GTP vertex string ('pass' or e.g. 'D4').

    GTP columns skip 'I'. For a 9x9 board the columns are A..J skipping I,
    i.e. 'A','B','C','D','E','F','G','H','J'. Rows are 1..9 with 1 at the
    bottom (GTP convention).
    """
    if token == PASS_TOKEN:
        return "pass"
    row, col = token_to_point(token)  # type: ignore[misc]
    letters = "ABCDEFGHJ"
    return f"{letters[col]}{BOARD_SIZE - row}"


def gtp_vertex_to_token(vertex: str) -> int:
    v = vertex.strip().lower()
    if v == "pass":
        return PASS_TOKEN
    letters = "abcdefghj"
    col = letters.index(v[0])
    row = BOARD_SIZE - int(v[1:])
    return point_to_token(row, col)


# ---------------------------------------------------------------------------
# Position / game encoding
# ---------------------------------------------------------------------------

@dataclass
class EncodedPosition:
    """One training example: board state + trajectory continuation."""

    state_categories: np.ndarray  # int8, shape (81,), values in 0..4
    # The literal token stream for the trajectory portion of the sequence,
    # NOT including the board prefix. The full sequence built by the model
    # is: [BOS] + board-prefix-embeddings(81) + [SEP_POS] + trajectory.
    trajectory_tokens: np.ndarray  # int32, shape (T,)
    # Mask: 1 where loss is computed (non-prefix tokens), 0 elsewhere. Same
    # length as trajectory_tokens.
    loss_mask: np.ndarray  # int8, shape (T,)
    # To-move color at this position: 'B' or 'W'. Used by the dataloader
    # to flip colors for white-to-move positions, since the model is trained
    # always-as-black for symmetry.
    to_move: str = "B"


def encode_board_states(
    board: np.ndarray,
    ko_point: tuple[int, int] | None = None,
    last_move: tuple[int, int] | None = None,
) -> np.ndarray:
    """Map a 9x9 board to length-81 state-category array.

    ``board`` is shape (9, 9) with values 0=empty, 1=black, 2=white. The
    encoded array uses the STATE_* constants above; ko-banned and last-move
    overlays *override* the stone/empty value (last_move is shown on a
    coloured stone in real games, but here we use a single category for
    compactness; Phase 1 may split this into a multi-hot channel set if
    needed).
    """
    if board.shape != (BOARD_SIZE, BOARD_SIZE):
        raise ValueError(f"board must be {BOARD_SIZE}x{BOARD_SIZE}, got {board.shape}")
    out = np.empty(NUM_POINTS, dtype=np.int8)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            v = board[r, c]
            if v == 0:
                out[r * BOARD_SIZE + c] = STATE_EMPTY
            elif v == 1:
                out[r * BOARD_SIZE + c] = STATE_BLACK
            elif v == 2:
                out[r * BOARD_SIZE + c] = STATE_WHITE
            else:
                raise ValueError(f"invalid board value {v} at ({r},{c})")
    if ko_point is not None:
        r, c = ko_point
        out[r * BOARD_SIZE + c] = STATE_KO_BANNED
    if last_move is not None:
        r, c = last_move
        out[r * BOARD_SIZE + c] = STATE_LAST_MOVE
    return out


def build_input_token_stream(trajectory_tokens: np.ndarray) -> np.ndarray:
    """Wrap a trajectory in [BOS] ... [SEP_POS] ... [EOS].

    The board-prefix embeddings are inserted by the model between [BOS] and
    [SEP_POS] using the ``state_categories`` lookup; from the *token stream*'s
    point of view, the prefix occupies positions 1..81 with placeholder IDs
    that the embedding layer overrides. We use PASS_TOKEN as a harmless
    placeholder there (it never receives loss).
    """
    prefix_placeholder = np.full(NUM_POINTS, PASS_TOKEN, dtype=np.int32)
    return np.concatenate(
        [
            np.array([BOS_TOKEN], dtype=np.int32),
            prefix_placeholder,
            np.array([SEP_POS_TOKEN], dtype=np.int32),
            trajectory_tokens.astype(np.int32),
            np.array([EOS_TOKEN], dtype=np.int32),
        ]
    )


def build_loss_mask(trajectory_length: int) -> np.ndarray:
    """Loss mask aligned to ``build_input_token_stream`` output.

    The mask is 1 on positions where the model should predict the *next*
    token and that next token is a real trajectory token. Concretely we put
    the 1s on the SEP_POS position (predicts the first trajectory token)
    and on each interior trajectory position (predicts the next one), and a
    final 1 on the last trajectory token (predicts EOS). Phase 1 may want
    EOS prediction skipped; for Phase 0 this is fine.
    """
    total_len = 1 + NUM_POINTS + 1 + trajectory_length + 1
    mask = np.zeros(total_len, dtype=np.int8)
    # The SEP_POS token is at index 1 + NUM_POINTS. From there through the
    # last trajectory token (inclusive), we want to predict the next token.
    sep_index = 1 + NUM_POINTS
    last_traj_index = sep_index + trajectory_length
    mask[sep_index : last_traj_index + 1] = 1
    return mask


__all__ = [
    "BOARD_SIZE",
    "NUM_POINTS",
    "NUM_STATE_CATEGORIES",
    "STATE_EMPTY",
    "STATE_BLACK",
    "STATE_WHITE",
    "STATE_KO_BANNED",
    "STATE_LAST_MOVE",
    "PASS_TOKEN",
    "BOS_TOKEN",
    "SEP_POS_TOKEN",
    "EOS_TOKEN",
    "THINK_OPEN_TOKEN",
    "THINK_CLOSE_TOKEN",
    "PHASE0_VOCAB_SIZE",
    "RESERVED_THINK_VOCAB",
    "VOCAB_SIZE",
    "MOVE_TOKEN_IDS",
    "EncodedPosition",
    "point_to_token",
    "token_to_point",
    "token_to_gtp_vertex",
    "gtp_vertex_to_token",
    "encode_board_states",
    "build_input_token_stream",
    "build_loss_mask",
]
