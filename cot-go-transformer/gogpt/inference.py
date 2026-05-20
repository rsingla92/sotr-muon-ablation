"""Position-conditioned move generation.

Phase 0 scope: greedy / temperature-sampling next-move from a model.
Phase 1+ extensions:
- ``best_of_n_sample`` runs N parallel single-move samples and returns the
  most-frequent move (a stub for the eventual self-consistency procedure
  that will fold over (CoT, move) pairs).

Note on autoregressive CoT generation: the current model emits only
82-way move logits, so step-by-step generation of think-block tokens
isn't supported in the head as-is. Expanding the head to full VOCAB_SIZE
is a Phase-1 task (see docs/phase1_plan.md).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .model import GoGPT
from .tokenizer import (
    BOS_TOKEN,
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


@dataclass
class BestOfNResult:
    chosen_token: int
    counts: dict[int, int]                # how many times each move was sampled
    samples: list[GeneratedMove]          # all N samples in draw order
    method: str = "majority"              # 'majority' | 'value' (Phase 1+)


@torch.no_grad()
def best_of_n_sample(
    model: GoGPT,
    board: np.ndarray,
    n: int,
    *,
    ko_point: tuple[int, int] | None = None,
    last_move: tuple[int, int] | None = None,
    history_tokens: list[int] | None = None,
    temperature: float = 0.8,
    top_k: int | None = None,
    device: Optional[torch.device] = None,
    legal_mask: Optional[np.ndarray] = None,
) -> BestOfNResult:
    """Draw ``n`` samples and return the majority-vote move.

    Implementation: batched forward across all N samples. For 9x9 + the
    short trajectories we expect, this is fast enough to skip the KV
    cache. If we later find batched generation isn't enough (e.g. at
    very large N or with long think-blocks once Phase-1 lands), the
    natural follow-up is a per-layer K/V cache for the board prefix that
    we reuse across the N parallel samples.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    device = device or next(model.parameters()).device
    history_tokens = history_tokens or []

    state_cats = encode_board_states(board, ko_point=ko_point, last_move=last_move)
    state_cats_t = (
        torch.from_numpy(state_cats.astype(np.int64))
        .unsqueeze(0)
        .expand(n, -1)
        .contiguous()
        .to(device)
    )
    prefix_placeholder = np.full(NUM_POINTS, PASS_TOKEN, dtype=np.int64)
    seq = np.concatenate(
        [
            np.array([BOS_TOKEN], dtype=np.int64),
            prefix_placeholder,
            np.array([SEP_POS_TOKEN], dtype=np.int64),
            np.asarray(history_tokens, dtype=np.int64),
        ]
    )
    tokens = (
        torch.from_numpy(seq)
        .unsqueeze(0)
        .expand(n, -1)
        .contiguous()
        .to(device)
    )

    out = model(tokens=tokens, state_categories=state_cats_t)
    logits = out["logits"][:, -1, :].float()  # (n, num_move_outputs)

    if legal_mask is not None:
        lm = torch.from_numpy(legal_mask).to(device)
        logits = logits.masked_fill(~lm.unsqueeze(0), float("-inf"))

    if temperature <= 0:
        sampled = torch.argmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)
    else:
        scaled = logits / temperature
        if top_k is not None:
            v, _ = torch.topk(scaled, k=min(top_k, scaled.shape[-1]), dim=-1)
            cutoff = v[:, -1:].expand_as(scaled)
            scaled = scaled.masked_fill(scaled < cutoff, float("-inf"))
        probs = F.softmax(scaled, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)

    sample_list: list[GeneratedMove] = []
    counts: Counter[int] = Counter()
    for i in range(n):
        tok = int(sampled[i].item())
        is_legal = bool(legal_mask[tok]) if legal_mask is not None else True
        sample_list.append(
            GeneratedMove(
                token=tok,
                logits=logits[i].detach().cpu(),
                prob=float(probs[i, tok].item()),
                is_legal_move=is_legal,
            )
        )
        counts[tok] += 1
    chosen, _ = counts.most_common(1)[0]
    return BestOfNResult(
        chosen_token=chosen,
        counts=dict(counts),
        samples=sample_list,
        method="majority",
    )


__all__ = ["generate_move", "GeneratedMove", "best_of_n_sample", "BestOfNResult"]
