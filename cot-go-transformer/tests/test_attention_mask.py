"""Prefix-LM attention mask correctness.

This is the verification-gate test from Phase 0: gradients at board-prefix
positions depend on all other prefix positions but NOT on trajectory tokens;
gradients at trajectory position t depend only on positions <= t.

We verify two things:
 1. The pure boolean mask matches the spec exactly.
 2. A 2-layer model's input-gradient sparsity matches the mask (i.e. zero
    grad through positions that should be unreachable).
"""

import pytest

torch = pytest.importorskip("torch")

from gogpt.model import GoGPT, GoGPTConfig, build_prefix_lm_mask
from gogpt.tokenizer import NUM_POINTS


def test_mask_bidirectional_in_prefix_causal_outside():
    T = 1 + NUM_POINTS + 1 + 5
    mask = build_prefix_lm_mask(T, prefix_start=1, prefix_end=1 + NUM_POINTS)
    # Position 0 (BOS): causal -- only attends to itself.
    assert mask[0, 0] is torch.tensor(True).item() or bool(mask[0, 0])
    for j in range(1, T):
        assert not bool(mask[0, j]), f"BOS should not see position {j}"
    # Inside prefix: bidirectional.
    for i in range(1, 1 + NUM_POINTS):
        for j in range(1, 1 + NUM_POINTS):
            assert bool(mask[i, j]), f"prefix {i} should see prefix {j}"
        # But shouldn't see SEP_POS or any later token.
        for j in range(1 + NUM_POINTS, T):
            assert not bool(mask[i, j]), f"prefix {i} should not see {j}"
    # SEP_POS (position 1 + NUM_POINTS) sees prefix + itself, not later.
    sep = 1 + NUM_POINTS
    for j in range(0, sep + 1):
        assert bool(mask[sep, j])
    for j in range(sep + 1, T):
        assert not bool(mask[sep, j])
    # Trajectory positions are strictly causal.
    for i in range(sep + 1, T):
        for j in range(T):
            assert bool(mask[i, j]) == (j <= i), f"({i},{j}) causal violation"


def test_gradient_sparsity_through_attention():
    """Gradients at a trajectory position must not flow to positions > t."""
    cfg = GoGPTConfig(
        n_layers=2, d_model=32, n_heads=4, d_ff=64,
        max_trajectory_len=4, vocab_size=128,
        use_flash_attn=False,
    )
    model = GoGPT(cfg).double()  # use fp64 for sensitivity
    model.eval()

    B = 1
    T = 1 + NUM_POINTS + 1 + 3 + 1
    tokens = torch.randint(0, 82, (B, T))
    tokens[0, 0] = 82          # BOS_TOKEN
    tokens[0, 1 + NUM_POINTS] = 83  # SEP_POS_TOKEN
    tokens[0, -1] = 84              # EOS_TOKEN
    state = torch.randint(0, 5, (B, NUM_POINTS))

    # Embed tokens through a leaf parameter we can autograd against.
    # Trick: we run forward, then call backward from the logit at a
    # trajectory position and check that grads w.r.t. positions > t are zero.
    tokens = tokens.long()
    state = state.long()

    # Make the token embedding output a leaf via a hook so we can grab grads.
    # Simpler: differentiate w.r.t. tok_emb output via register_hook.
    captured: dict[str, torch.Tensor] = {}

    def _hook(grad: torch.Tensor) -> None:
        captured["grad"] = grad.clone()

    tok_emb = model.tok_emb(tokens)
    tok_emb.requires_grad_(True)
    tok_emb.retain_grad()

    # Mirror the forward path but using our leaf tok_emb.
    x = tok_emb.clone()
    state_e = model.state_emb(state)
    pos_idx = torch.arange(NUM_POINTS)
    pos_e = model.board_pos_emb(pos_idx)
    prefix_e = state_e + pos_e.unsqueeze(0)
    x = x.clone()
    x[:, 1 : 1 + NUM_POINTS, :] = prefix_e

    from gogpt.model import build_prefix_lm_mask, build_rope_apply_mask, build_rope_cache
    attn_mask = build_prefix_lm_mask(T, 1, 1 + NUM_POINTS)
    rope_apply = build_rope_apply_mask(T, 1 + NUM_POINTS)
    cos, sin = build_rope_cache(cfg.max_seq_len, cfg.head_dim, cfg.rope_base, x.device, x.dtype)

    for block in model.blocks:
        x = block(x, attn_mask, cos, sin, rope_apply)
    x = model.final_norm(x)
    logits = model.move_head(x)

    # Backward from the logit at trajectory position t = sep + 1 (first move slot).
    t = 1 + NUM_POINTS + 1  # = 83 for 9x9
    target_logit = logits[0, t, 0]
    target_logit.backward()
    g = tok_emb.grad
    assert g is not None
    # Grad at any position j > t must be (near) zero.
    g_abs = g.abs().sum(dim=-1)[0]  # (T,)
    for j in range(t + 1, T):
        assert g_abs[j].item() < 1e-8, f"grad at j={j} should be zero, got {g_abs[j].item()}"
    # And there should be some nonzero grad somewhere at or before t.
    assert g_abs[: t + 1].sum().item() > 0
