"""Tiny model forward + backward smoke test. Verifies shapes, finite loss,
and that gradients flow. CPU-runnable (skips if torch not installed)."""

import pytest

torch = pytest.importorskip("torch")

from gogpt.model import GoGPT, GoGPTConfig
from gogpt.tokenizer import (
    BOS_TOKEN,
    EOS_TOKEN,
    NUM_POINTS,
    SEP_POS_TOKEN,
)


def _tiny_cfg():
    return GoGPTConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        d_ff=128,
        max_trajectory_len=8,
        vocab_size=128,
        use_flash_attn=False,
    )


def test_forward_shapes_and_finite_loss():
    cfg = _tiny_cfg()
    model = GoGPT(cfg)
    B, traj = 2, 5
    T = 1 + NUM_POINTS + 1 + traj + 1
    tokens = torch.randint(0, 82, (B, T))
    tokens[:, 0] = BOS_TOKEN
    tokens[:, 1 + NUM_POINTS] = SEP_POS_TOKEN
    tokens[:, -1] = EOS_TOKEN
    state = torch.randint(0, 5, (B, NUM_POINTS))
    labels = torch.full_like(tokens, -100)
    sep_idx = 1 + NUM_POINTS
    labels[:, sep_idx:-1] = tokens[:, sep_idx + 1:]
    # Clamp labels to the model's output vocab (0..81).
    labels[labels >= 0] = labels[labels >= 0] % 82
    loss_mask = torch.zeros_like(tokens, dtype=torch.int8)
    loss_mask[:, sep_idx:-1] = 1

    out = model(tokens=tokens, state_categories=state, labels=labels, loss_mask=loss_mask)
    assert out["logits"].shape == (B, T, 82)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients flowed"
    total = sum(g.abs().sum().item() for g in grads)
    assert total > 0
