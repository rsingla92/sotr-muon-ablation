"""Position-conditioned move generation.

Phase 0 scope: greedy / temperature-sampling next-move from a model. Phase 1
will extend this with [<think>] block generation, KV caching for the prefix,
best-of-N self-consistency, and value-head ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .model import GoGPT
from .tokenizer import (
    BOS_TOKEN,
    EOS_TOKEN,
    MOVE_TOKEN_IDS,
    NUM_POINTS,
    PASS_TOKEN,
    SEP_POS_TOKEN,
    encode_board_states,
)


@dataclass
class GeneratedMove:
    token: int
    logits: torch.Tensor
    prob: float
    is_legal_move: bool


def _legality_mask(num_outputs: int) -> torch.Tensor:
    """Mask restricting sampling to the move vocab (board points + PASS).

    Returns a (num_outputs,) bool tensor. The model's head only outputs move
    tokens (num_outputs == 82), so this is mostly a no-op safety check.
    """
    mask = torch.zeros(num_outputs, dtype=torch.bool)
    for t in MOVE_TOKEN_IDS:
        if t < num_outputs:
            mask[t] = True
    return mask


@torch.no_grad()
def generate_move(
    model: GoGPT,
    board: np.ndarray,
    *,
    ko_point: tuple[int, int] | None = None,
    last_move: tuple[int, int] | None = None,
    history_tokens: list[int] | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
    device: Optional[torch.device] = None,
    legal_mask: Optional[np.ndarray] = None,
) -> GeneratedMove:
    """Sample the next move given a board state and optional move history.

    ``legal_mask`` is a (82,) bool array; if provided, illegal moves get
    -inf logits. Phase 0's data path is colorflipped to always-as-black, so
    callers should similarly flip stones before invoking this.
    """
    device = device or next(model.parameters()).device
    history_tokens = history_tokens or []

    state_cats = encode_board_states(board, ko_point=ko_point, last_move=last_move)
    state_cats_t = torch.from_numpy(state_cats.astype(np.int64)).unsqueeze(0).to(device)

    # Build input sequence: [BOS] + prefix-placeholder(81) + [SEP_POS] + history.
    prefix_placeholder = np.full(NUM_POINTS, PASS_TOKEN, dtype=np.int64)
    seq = np.concatenate(
        [
            np.array([BOS_TOKEN], dtype=np.int64),
            prefix_placeholder,
            np.array([SEP_POS_TOKEN], dtype=np.int64),
            np.asarray(history_tokens, dtype=np.int64),
        ]
    )
    tokens = torch.from_numpy(seq).unsqueeze(0).to(device)
    out = model(tokens=tokens, state_categories=state_cats_t)
    logits = out["logits"][0, -1, :].float()  # (num_move_outputs,)

    if legal_mask is not None:
        lm = torch.from_numpy(legal_mask).to(device)
        logits = logits.masked_fill(~lm, float("-inf"))

    if temperature <= 0:
        token = int(torch.argmax(logits).item())
        probs = F.softmax(logits, dim=-1)
    else:
        scaled = logits / temperature
        if top_k is not None:
            v, _ = torch.topk(scaled, k=min(top_k, scaled.shape[-1]))
            scaled = scaled.masked_fill(scaled < v[-1], float("-inf"))
        probs = F.softmax(scaled, dim=-1)
        token = int(torch.multinomial(probs, num_samples=1).item())

    is_legal = legal_mask[token] if legal_mask is not None else True
    return GeneratedMove(
        token=token,
        logits=logits.detach().cpu(),
        prob=float(probs[token].item()),
        is_legal_move=bool(is_legal),
    )


__all__ = ["generate_move", "GeneratedMove"]
