"""Prefix-LM transformer for 9x9 Go.

Sequence layout (token positions, 0-indexed):
    0           : [BOS]
    1 .. 81     : board prefix (81 intersections, row-major)
    82          : [SEP_POS]
    83 ..       : trajectory tokens (moves and Phase-1 think-block tokens)

Attention mask:
    - Within the board prefix (positions 1..81): bidirectional.
    - Everything else: causal.
    - [BOS] (position 0) attends only to itself (mask consistent with causal).
    - [SEP_POS] (position 82) and trajectory positions attend causally to all
      earlier positions (including the bidirectionally-mixed board prefix).

Embedding strategy:
    - Token embedding table (vocab_size, d_model) for the move/special vocab.
    - State-category embedding table (5, d_model) added to board-prefix
      positions, indexed by per-point state.
    - 2D learned positional embedding (81, d_model) added to the board prefix.
    - RoPE applied to trajectory positions inside attention (queries+keys).

Output:
    Move-logit head over the full move vocab (82 entries: 81 points + PASS).
    The head is applied at every position but loss is masked to non-prefix
    positions only.

This is a Phase-0-ready scaffold. FlashAttention-2 wiring is gated behind
the ``use_flash_attn`` flag; when off, we fall back to ``F.scaled_dot_product_attention``
with an explicit additive mask (which can use the fused SDPA kernels when
the mask shape is supported).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import (
    NUM_POINTS,
    NUM_STATE_CATEGORIES,
    VOCAB_SIZE,
)

# Number of move-logit outputs (board points + PASS).
NUM_MOVE_OUTPUTS = NUM_POINTS + 1  # 82


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GoGPTConfig:
    n_layers: int = 12
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    vocab_size: int = VOCAB_SIZE
    max_trajectory_len: int = 256
    dropout: float = 0.0
    rope_base: float = 10_000.0
    tie_word_embeddings: bool = False
    use_flash_attn: bool = False  # set True on CUDA with flash-attn installed

    # Derived/static
    num_state_categories: int = NUM_STATE_CATEGORIES
    num_board_points: int = NUM_POINTS
    num_move_outputs: int = NUM_MOVE_OUTPUTS

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def max_seq_len(self) -> int:
        # [BOS] + 81 prefix + [SEP_POS] + trajectory
        return 1 + self.num_board_points + 1 + self.max_trajectory_len


# ---------------------------------------------------------------------------
# Norm and FFN
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d)
        norm = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm.to(x.dtype)) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        # SwiGLU uses gate + value pair, see Shazeer 2020.
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_value = nn.Linear(d_model, d_ff, bias=False)
        self.w_out = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_out(F.silu(self.w_gate(x)) * self.w_value(x))


# ---------------------------------------------------------------------------
# RoPE for trajectory positions
# ---------------------------------------------------------------------------

def build_rope_cache(
    max_seq_len: int, head_dim: int, base: float, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin) tables of shape (max_seq_len, head_dim)."""
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (T, head_dim/2)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).to(dtype)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).to(dtype)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    apply_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE only at positions where ``apply_mask`` is True.

    x:        (B, n_heads, T, head_dim)
    cos, sin: (T, head_dim)
    apply_mask: (T,) bool -- True where RoPE should be applied (trajectory + SEP).
    """
    cos_b = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
    sin_b = sin.unsqueeze(0).unsqueeze(0)
    rotated = x * cos_b + _rotate_half(x) * sin_b
    mask = apply_mask.view(1, 1, -1, 1).to(x.dtype)
    return rotated * mask + x * (1 - mask)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class PrefixLMAttention(nn.Module):
    def __init__(self, cfg: GoGPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,                  # (B, T, d_model)
        attn_mask: torch.Tensor,          # (T, T) bool, True = allowed
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        rope_apply_mask: torch.Tensor,    # (T,) bool
    ) -> torch.Tensor:
        B, T, _ = x.shape
        cfg = self.cfg
        q = self.q_proj(x).view(B, T, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, cfg.n_heads, cfg.head_dim).transpose(1, 2)

        q = apply_rope(q, rope_cos[:T], rope_sin[:T], rope_apply_mask[:T])
        k = apply_rope(k, rope_cos[:T], rope_sin[:T], rope_apply_mask[:T])

        # The mask combines bidirectional-in-prefix and causal-elsewhere. We
        # pass it as a boolean mask to SDPA where True = allowed.
        # SDPA expects an additive mask or a boolean mask (True keeps).
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask.unsqueeze(0).unsqueeze(0),  # (1,1,T,T)
            dropout_p=cfg.dropout if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, cfg.d_model)
        return self.dropout(self.o_proj(out))


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, cfg: GoGPTConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = PrefixLMAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        rope_apply_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), attn_mask, rope_cos, rope_sin, rope_apply_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ---------------------------------------------------------------------------
# Mask construction
# ---------------------------------------------------------------------------

def build_prefix_lm_mask(seq_len: int, prefix_start: int, prefix_end: int) -> torch.Tensor:
    """Bool mask of shape (seq_len, seq_len). True = attention allowed.

    Positions ``[prefix_start, prefix_end)`` are bidirectional; everything
    else is causal. The prefix range is INCLUSIVE on the lower end and
    EXCLUSIVE on the upper end -- for our layout, prefix_start=1 and
    prefix_end=82 (positions 1..81 inclusive are the 81 board points).

    Rules for ``mask[i, j]`` (whether query at i may attend to key at j):
        - If both i and j are in the prefix range: allowed.
        - Otherwise: causal (allowed iff j <= i).
    """
    idx = torch.arange(seq_len)
    in_prefix = (idx >= prefix_start) & (idx < prefix_end)
    in_prefix_i = in_prefix.unsqueeze(1)  # (T,1)
    in_prefix_j = in_prefix.unsqueeze(0)  # (1,T)
    bidir = in_prefix_i & in_prefix_j
    causal = idx.unsqueeze(0) <= idx.unsqueeze(1)
    return bidir | causal


def build_rope_apply_mask(seq_len: int, sep_pos_index: int) -> torch.Tensor:
    """RoPE applies to SEP_POS and all trajectory positions; not to BOS/prefix.

    For a sequence of layout [BOS][prefix x 81][SEP_POS][traj...], we apply
    RoPE from ``sep_pos_index`` onward. The position used by RoPE for SEP_POS
    is 0, the next trajectory token is 1, etc.
    """
    mask = torch.zeros(seq_len, dtype=torch.bool)
    mask[sep_pos_index:] = True
    return mask


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GoGPT(nn.Module):
    def __init__(self, cfg: GoGPTConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.state_emb = nn.Embedding(cfg.num_state_categories, cfg.d_model)
        self.board_pos_emb = nn.Embedding(cfg.num_board_points, cfg.d_model)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.move_head = nn.Linear(cfg.d_model, cfg.num_move_outputs, bias=False)

        self.apply(self._init_weights)

        # RoPE cache is created lazily (depends on device/dtype).
        self._rope_cos: torch.Tensor | None = None
        self._rope_sin: torch.Tensor | None = None

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _maybe_build_rope(self, device: torch.device, dtype: torch.dtype) -> None:
        need = (
            self._rope_cos is None
            or self._rope_cos.device != device
            or self._rope_cos.dtype != dtype
        )
        if need:
            cos, sin = build_rope_cache(
                self.cfg.max_seq_len, self.cfg.head_dim, self.cfg.rope_base, device, dtype
            )
            self._rope_cos = cos
            self._rope_sin = sin

    def forward(
        self,
        tokens: torch.Tensor,         # (B, T) int64
        state_categories: torch.Tensor,  # (B, 81) int64
        labels: Optional[torch.Tensor] = None,    # (B, T) int64 or -100 to ignore
        loss_mask: Optional[torch.Tensor] = None,  # (B, T) bool/0-1, optional
    ) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        B, T = tokens.shape
        if T > cfg.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds max_seq_len {cfg.max_seq_len}")
        device = tokens.device

        # Token embeddings everywhere.
        x = self.tok_emb(tokens)  # (B, T, d_model)

        # Overlay board-prefix embeddings at positions 1..81.
        # state_categories: (B, 81); board_pos: 0..80
        prefix_start = 1
        prefix_end = prefix_start + cfg.num_board_points  # 82
        state_e = self.state_emb(state_categories)  # (B, 81, d)
        pos_idx = torch.arange(cfg.num_board_points, device=device)
        pos_e = self.board_pos_emb(pos_idx)  # (81, d)
        prefix_e = state_e + pos_e.unsqueeze(0)
        # Replace token embeddings in the prefix region with the prefix
        # composite. The token IDs at those positions are placeholders.
        x = x.clone()
        x[:, prefix_start:prefix_end, :] = prefix_e

        # Masks (cached on first use of this seq_len).
        attn_mask = build_prefix_lm_mask(T, prefix_start, prefix_end).to(device)
        sep_pos_index = prefix_end  # the SEP_POS token sits at index 82
        rope_apply_mask = build_rope_apply_mask(T, sep_pos_index).to(device)

        self._maybe_build_rope(device, x.dtype)
        assert self._rope_cos is not None and self._rope_sin is not None
        rope_cos = self._rope_cos
        rope_sin = self._rope_sin

        for block in self.blocks:
            x = block(x, attn_mask, rope_cos, rope_sin, rope_apply_mask)

        x = self.final_norm(x)
        logits = self.move_head(x)  # (B, T, num_move_outputs)

        out: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            # CE on positions where loss_mask is True. Labels at masked-out
            # positions can be anything; we set them to -100 via the mask.
            tgt = labels.clone()
            if loss_mask is not None:
                tgt = torch.where(loss_mask.bool(), tgt, torch.full_like(tgt, -100))
            loss = F.cross_entropy(
                logits.view(-1, cfg.num_move_outputs),
                tgt.view(-1),
                ignore_index=-100,
            )
            out["loss"] = loss
        return out

    @torch.no_grad()
    def num_parameters(self, only_trainable: bool = True) -> int:
        params = [p for p in self.parameters() if (p.requires_grad or not only_trainable)]
        return sum(p.numel() for p in params)


__all__ = [
    "GoGPT",
    "GoGPTConfig",
    "build_prefix_lm_mask",
    "build_rope_apply_mask",
    "NUM_MOVE_OUTPUTS",
]
