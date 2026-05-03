"""PROTOCOL §7 sanity check #8 — SOTR enforces 2D-only parameters.

SOTR is designed for hidden 2D weight matrices (transformer.h.* in NanoGPT).
Embeddings, head, biases, and LayerNorm gains/biases must be optimized with
AdamW. This test verifies SOTR fails loudly on 1D params rather than silently
producing nonsense.

The actual param-group split lives in the training script (PROTOCOL §6 follows
modded-nanogpt's convention of using ``transformer.h.parameters()``); this test
verifies the optimizer's defensive check.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from optimizers import SOTR


@pytest.mark.sanity
def test_sotr_rejects_1d_params() -> None:
    """A 1D parameter (e.g. a bias) should raise on step()."""
    p = nn.Parameter(torch.randn(64))
    opt = SOTR([p], lr=0.01)
    p.grad = torch.randn(64)

    with pytest.raises(RuntimeError, match=r"matrix parameters \(ndim >= 2\)"):
        opt.step()


@pytest.mark.sanity
def test_sotr_rejects_scalar_params() -> None:
    """0-dim (scalar) parameter should also raise."""
    p = nn.Parameter(torch.randn(()))
    opt = SOTR([p], lr=0.01)
    p.grad = torch.randn(())

    with pytest.raises(RuntimeError, match=r"matrix parameters \(ndim >= 2\)"):
        opt.step()


@pytest.mark.sanity
def test_sotr_accepts_2d_params() -> None:
    """2D parameter is the canonical case; should run without error."""
    p = nn.Parameter(torch.randn(64, 32))
    opt = SOTR([p], lr=0.01)
    p.grad = torch.randn(64, 32)
    opt.step()  # no exception
    assert opt.clip_total == 1


@pytest.mark.sanity
def test_sotr_accepts_4d_conv_params() -> None:
    """4D conv filters are reshaped to 2D internally (matches Muon's behavior)."""
    p = nn.Parameter(torch.randn(32, 16, 3, 3))  # out_channels, in_channels, h, w
    opt = SOTR([p], lr=0.01)
    p.grad = torch.randn(32, 16, 3, 3)
    opt.step()  # should reshape and not raise
    assert opt.clip_total == 1


@pytest.mark.sanity
def test_sotr_validates_alpha_range() -> None:
    """α must be in [0, 1]."""
    p = nn.Parameter(torch.randn(8, 8))
    with pytest.raises(ValueError, match=r"alpha must be in"):
        SOTR([p], alpha=-0.1)
    with pytest.raises(ValueError, match=r"alpha must be in"):
        SOTR([p], alpha=1.5)


@pytest.mark.sanity
def test_sotr_validates_delta() -> None:
    """Δ must be > 0 (use float('inf') to disable). Negative or zero rejected."""
    p = nn.Parameter(torch.randn(8, 8))
    with pytest.raises(ValueError, match=r"delta must be > 0"):
        SOTR([p], delta=0.0)
    with pytest.raises(ValueError, match=r"delta must be > 0"):
        SOTR([p], delta=-1.0)
    # inf is allowed:
    SOTR([p], delta=float("inf"))


@pytest.mark.sanity
def test_sotr_validates_ns_steps() -> None:
    """ns_steps must be >= 0 (0 reduces to Frobenius normalization)."""
    p = nn.Parameter(torch.randn(8, 8))
    with pytest.raises(ValueError, match=r"ns_steps must be >= 0"):
        SOTR([p], ns_steps=-1)
    SOTR([p], ns_steps=0)  # allowed
