"""SGF parsing and batched dataloader for prefix-LM training.

Each training example is a *position-conditioned continuation*: from one
position in a self-play game, we feed the board prefix and ask the model to
predict the remaining moves of the game (up to ``max_trajectory_len`` tokens).

For Phase 0 this is sufficient; Phase 1 will insert structured think-block
tokens before each move token in the trajectory.

Color symmetry: we train always-as-black. For white-to-move positions we
swap stone colors so the side to move is always black. KataGo analysis
labels (used in Phase 1) should be flipped to match.
"""

from __future__ import annotations

import gzip
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

# torch is imported lazily so the SGF parser / rules can be exercised on
# environments without torch (CI, local quick-checks).
try:
    import torch
    from torch.utils.data import IterableDataset
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    IterableDataset = object  # type: ignore[assignment,misc]
    _HAS_TORCH = False

from .tokenizer import (
    BOS_TOKEN,
    EOS_TOKEN,
    NUM_POINTS,
    PASS_TOKEN,
    SEP_POS_TOKEN,
    encode_board_states,
    point_to_token,
)

log = logging.getLogger(__name__)
BOARD_SIZE = 9


# ---------------------------------------------------------------------------
# Minimal SGF parsing -- just enough for KataGo-emitted 9x9 games
# ---------------------------------------------------------------------------

@dataclass
class ParsedGame:
    moves: list[tuple[str, tuple[int, int] | None]]  # (color, (r, c)) or (color, None) for pass
    komi: float = 7.0
    result: str = ""


def _sgf_coord_to_rc(s: str) -> tuple[int, int] | None:
    if s == "" or s == "tt":  # pass conventions
        return None
    if len(s) != 2:
        raise ValueError(f"invalid SGF coord {s!r}")
    col = ord(s[0]) - ord("a")
    row = ord(s[1]) - ord("a")
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        raise ValueError(f"SGF coord {s!r} outside 9x9")
    return row, col


def parse_sgf(text: str) -> ParsedGame:
    """Parse a KataGo 9x9 SGF (single game, no branches) into moves.

    This is a deliberately tiny parser; for anything more complex, use sgfmill.
    """
    moves: list[tuple[str, tuple[int, int] | None]] = []
    komi = 7.0
    result = ""
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c in (";", "(", ")", "\n", " ", "\t", "\r"):
            i += 1
            continue
        # Property: 1-2 uppercase letters followed by [...]
        j = i
        while j < n and text[j].isalpha() and text[j].isupper():
            j += 1
        if j == i:
            i += 1
            continue
        key = text[i:j]
        # one or more bracketed values
        values: list[str] = []
        while j < n and text[j] == "[":
            k = j + 1
            buf = []
            while k < n:
                if text[k] == "\\" and k + 1 < n:
                    buf.append(text[k + 1])
                    k += 2
                    continue
                if text[k] == "]":
                    break
                buf.append(text[k])
                k += 1
            values.append("".join(buf))
            j = k + 1
        if key == "B":
            moves.append(("B", _sgf_coord_to_rc(values[0]) if values else None))
        elif key == "W":
            moves.append(("W", _sgf_coord_to_rc(values[0]) if values else None))
        elif key == "KM":
            try:
                komi = float(values[0])
            except Exception:
                pass
        elif key == "RE":
            result = values[0] if values else ""
        i = j
    return ParsedGame(moves=moves, komi=komi, result=result)


# ---------------------------------------------------------------------------
# Lightweight rules for stone placement / capture
# ---------------------------------------------------------------------------

def _neighbors(r: int, c: int) -> Iterator[tuple[int, int]]:
    if r > 0:
        yield r - 1, c
    if r + 1 < BOARD_SIZE:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < BOARD_SIZE:
        yield r, c + 1


def _flood_group(board: np.ndarray, r: int, c: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """Return (group cells, liberty cells) for the stone at (r, c)."""
    color = board[r, c]
    if color == 0:
        return set(), set()
    group: set[tuple[int, int]] = set()
    libs: set[tuple[int, int]] = set()
    stack = [(r, c)]
    while stack:
        cur = stack.pop()
        if cur in group:
            continue
        group.add(cur)
        for nr, nc in _neighbors(*cur):
            v = board[nr, nc]
            if v == 0:
                libs.add((nr, nc))
            elif v == color and (nr, nc) not in group:
                stack.append((nr, nc))
    return group, libs


def play_stone(
    board: np.ndarray,
    color: int,
    rc: tuple[int, int] | None,
) -> tuple[np.ndarray, tuple[int, int] | None]:
    """Play a stone, doing captures. Returns (new_board, ko_point_or_none).

    Ko detection here is the simple ko rule: if exactly one stone was just
    captured by a move that itself becomes a single-stone group with one
    liberty (the captured point), mark that liberty as ko-banned.
    """
    if rc is None:
        return board.copy(), None
    r, c = rc
    new = board.copy()
    new[r, c] = color
    opp = 3 - color
    captured: list[tuple[int, int]] = []
    for nr, nc in _neighbors(r, c):
        if new[nr, nc] == opp:
            group, libs = _flood_group(new, nr, nc)
            if not libs:
                for gr, gc in group:
                    new[gr, gc] = 0
                    captured.append((gr, gc))
    # Self-capture suicide rule: if our own group now has no liberties, the
    # move is illegal under most rulesets. For Tromp-Taylor (KataGo's default)
    # suicide is technically allowed, but we conservatively reject it here.
    own_group, own_libs = _flood_group(new, r, c)
    if not own_libs:
        # Illegal under non-Tromp; revert.
        return board.copy(), None
    ko: tuple[int, int] | None = None
    if len(captured) == 1 and len(own_group) == 1 and len(own_libs) == 1:
        ko = captured[0]
    return new, ko


# ---------------------------------------------------------------------------
# Position iterator -- yields one example per (game, move-index)
# ---------------------------------------------------------------------------

@dataclass
class TrainingExample:
    state_categories: np.ndarray   # (81,) int8
    tokens: np.ndarray             # (T,) int64 -- full input stream
    labels: np.ndarray             # (T,) int64 -- shifted targets (-100 where ignored)
    loss_mask: np.ndarray          # (T,) int8


def _color_to_value(color: str) -> int:
    return 1 if color.upper() == "B" else 2


def _swap_colors(board: np.ndarray) -> np.ndarray:
    out = board.copy()
    out[board == 1] = 2
    out[board == 2] = 1
    return out


def iter_examples_from_game(
    game: ParsedGame,
    max_trajectory_len: int,
    min_move_index: int = 0,
    always_as_black: bool = True,
    rng: random.Random | None = None,
) -> Iterator[TrainingExample]:
    """Generate training examples by walking the game.

    For each move index ``i`` (>= min_move_index), the prefix is the board
    after the first ``i`` moves and the trajectory is the next moves
    truncated to ``max_trajectory_len``.
    """
    rng = rng or random.Random()
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    ko: tuple[int, int] | None = None
    last_move: tuple[int, int] | None = None
    history: list[tuple[str, tuple[int, int] | None]] = []
    for i, (color, rc) in enumerate(game.moves):
        if i >= min_move_index:
            ex = _build_example(
                board, ko, last_move, history, game.moves[i:], max_trajectory_len,
                to_move=color, always_as_black=always_as_black,
            )
            if ex is not None:
                yield ex
        board, ko = play_stone(board, _color_to_value(color), rc)
        last_move = rc
        history.append((color, rc))


def _build_example(
    board: np.ndarray,
    ko: tuple[int, int] | None,
    last_move: tuple[int, int] | None,
    history: list[tuple[str, tuple[int, int] | None]],
    future_moves: list[tuple[str, tuple[int, int] | None]],
    max_trajectory_len: int,
    to_move: str,
    always_as_black: bool,
) -> TrainingExample | None:
    if not future_moves:
        return None

    if always_as_black and to_move.upper() == "W":
        board = _swap_colors(board)
        # Flip the future-move colors too so the next move is by black.
        future_moves = [("B" if c == "W" else "W", rc) for c, rc in future_moves]

    state_cats = encode_board_states(board, ko_point=ko, last_move=last_move)

    # Trajectory tokens: for Phase 0, just the sequence of move tokens.
    traj_tokens: list[int] = []
    for _, rc in future_moves[:max_trajectory_len]:
        traj_tokens.append(PASS_TOKEN if rc is None else point_to_token(*rc))
    traj_arr = np.asarray(traj_tokens, dtype=np.int64)

    # Build the full input stream: [BOS] [prefix x 81] [SEP_POS] [traj] [EOS]
    prefix_placeholder = np.full(NUM_POINTS, PASS_TOKEN, dtype=np.int64)
    tokens = np.concatenate(
        [
            np.array([BOS_TOKEN], dtype=np.int64),
            prefix_placeholder,
            np.array([SEP_POS_TOKEN], dtype=np.int64),
            traj_arr,
            np.array([EOS_TOKEN], dtype=np.int64),
        ]
    )

    # Labels: at position t, predict tokens[t+1]. We loss-mask everything
    # except SEP_POS .. last-trajectory-token (predicts EOS); the prefix
    # positions get -100 labels.
    T = tokens.shape[0]
    labels = np.full(T, -100, dtype=np.int64)
    sep_index = 1 + NUM_POINTS
    last_traj_index = sep_index + traj_arr.shape[0]
    # Position i predicts token i+1.
    for i in range(sep_index, last_traj_index + 1):
        if i + 1 < T:
            labels[i] = int(tokens[i + 1])

    loss_mask = np.zeros(T, dtype=np.int8)
    loss_mask[sep_index : last_traj_index + 1] = 1

    return TrainingExample(
        state_categories=state_cats,
        tokens=tokens,
        labels=labels,
        loss_mask=loss_mask,
    )


# ---------------------------------------------------------------------------
# IterableDataset over a shard list
# ---------------------------------------------------------------------------

class SgfShardDataset(IterableDataset):
    """Streams positions from a list of (possibly gzipped) SGF files.

    Each shard contains one or more SGFs separated by parentheses. For
    KataGo self-play output, one file = one game.
    """

    def __init__(
        self,
        shard_paths: list[str | Path],
        max_trajectory_len: int = 64,
        min_move_index: int = 0,
        shuffle_shards: bool = True,
        seed: int = 0,
        always_as_black: bool = True,
        max_examples_per_game: int | None = 4,
    ) -> None:
        super().__init__()
        self.shards = [Path(p) for p in shard_paths]
        self.max_trajectory_len = max_trajectory_len
        self.min_move_index = min_move_index
        self.shuffle_shards = shuffle_shards
        self.seed = seed
        self.always_as_black = always_as_black
        self.max_examples_per_game = max_examples_per_game

    def __iter__(self) -> Iterator[TrainingExample]:
        if not _HAS_TORCH:
            raise RuntimeError("SgfShardDataset requires torch; install via pip")
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        rng = random.Random(self.seed + worker_id * 1009)
        shards = list(self.shards)
        if self.shuffle_shards:
            rng.shuffle(shards)
        # Sharding across workers: take every nth.
        shards = shards[worker_id::num_workers]
        for p in shards:
            try:
                yield from self._iter_shard(p, rng)
            except Exception as e:
                log.warning("failed to read %s: %s", p, e)
                continue

    def _iter_shard(self, p: Path, rng: random.Random) -> Iterator[TrainingExample]:
        text = _read_text(p)
        game = parse_sgf(text)
        if not game.moves:
            return
        examples = list(
            iter_examples_from_game(
                game,
                self.max_trajectory_len,
                min_move_index=self.min_move_index,
                always_as_black=self.always_as_black,
                rng=rng,
            )
        )
        if not examples:
            return
        if self.max_examples_per_game is not None and len(examples) > self.max_examples_per_game:
            examples = rng.sample(examples, self.max_examples_per_game)
        rng.shuffle(examples)
        for ex in examples:
            yield ex


def _read_text(p: Path) -> str:
    if str(p).endswith(".gz"):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            return f.read()
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def collate(batch: list[TrainingExample]) -> "dict[str, torch.Tensor]":
    """Right-pad to the max sequence in the batch."""
    if not _HAS_TORCH:
        raise RuntimeError("collate requires torch; install via pip")
    T_max = max(ex.tokens.shape[0] for ex in batch)
    B = len(batch)
    tokens = torch.full((B, T_max), PASS_TOKEN, dtype=torch.int64)
    labels = torch.full((B, T_max), -100, dtype=torch.int64)
    loss_mask = torch.zeros((B, T_max), dtype=torch.int8)
    state_cats = torch.zeros((B, NUM_POINTS), dtype=torch.int64)
    for i, ex in enumerate(batch):
        T = ex.tokens.shape[0]
        tokens[i, :T] = torch.from_numpy(ex.tokens)
        labels[i, :T] = torch.from_numpy(ex.labels)
        loss_mask[i, :T] = torch.from_numpy(ex.loss_mask)
        state_cats[i] = torch.from_numpy(ex.state_categories.astype(np.int64))
    return {
        "tokens": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "state_categories": state_cats,
    }


__all__ = [
    "SgfShardDataset",
    "TrainingExample",
    "ParsedGame",
    "parse_sgf",
    "iter_examples_from_game",
    "collate",
    "play_stone",
]
