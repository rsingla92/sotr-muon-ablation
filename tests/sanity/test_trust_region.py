"""PROTOCOL §7 sanity check #6 — Frobenius trust region triggers correctly.

Verifies that:
- A small Δ activates the cap on most steps (cap path is reachable).
- A large Δ never activates (the conditional is gated correctly).
- The post-cap update has Frobenius norm exactly Δ when capped.

The trust region is the cleanly novel piece of SOTR (PROTOCOL §1, knowledge/03,
knowledge/07). Verifying it works is critical.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from optimizers import SOTR


@pytest.mark.sanity
def test_small_delta_clips_almost_every_step() -> None:
    """With Δ ≪ typical update norm, the cap should fire on essentially every step."""
    shape = (64, 64)
    torch.manual_seed(0)
    p = nn.Parameter(torch.randn(*shape))

    # NS produces O with ||O||_F ≈ sqrt(min(m,n)) ≈ 8 for 64x64.
    # Δ = 0.01 is two orders of magnitude smaller → cap fires on every step.
    opt = SOTR([p], lr=0.001, momentum=0.95, alpha=1.0, delta=0.01, ns_steps=5)

    n_steps = 20
    for step in range(n_steps):
        p.grad = torch.randn(*shape, generator=torch.Generator().manual_seed(step))
        opt.step()

    assert opt.clip_total == n_steps
    assert opt.clip_rate >= 0.95, (
        f"Δ=0.01 should trigger the cap on nearly every step on a 64x64 problem; "
        f"got clip_rate = {opt.clip_rate:.2%}. The cap path may be unreachable."
    )


@pytest.mark.sanity
def test_large_delta_never_clips() -> None:
    """With Δ ≫ typical update norm, the cap should never fire."""
    shape = (64, 64)
    torch.manual_seed(0)
    p = nn.Parameter(torch.randn(*shape))

    # ||O||_F ≈ 8; Δ = 100 is much larger → cap never triggers.
    opt = SOTR([p], lr=0.001, momentum=0.95, alpha=1.0, delta=100.0, ns_steps=5)

    n_steps = 20
    for step in range(n_steps):
        p.grad = torch.randn(*shape, generator=torch.Generator().manual_seed(step))
        opt.step()

    assert opt.clip_hits == 0, (
        f"Δ=100 should never trigger the cap on a 64x64 problem; got "
        f"clip_hits = {opt.clip_hits}. The conditional may be incorrectly gated."
    )


@pytest.mark.sanity
def test_inf_delta_disables_clipping() -> None:
    """Δ = inf must never trigger; this is the corner-case for SOTR ≡ Muon at α=1."""
    shape = (64, 64)
    torch.manual_seed(0)
    p = nn.Parameter(torch.randn(*shape))
    opt = SOTR([p], lr=0.001, alpha=1.0, delta=float("inf"), ns_steps=5)

    for step in range(10):
        p.grad = torch.randn(*shape)
        opt.step()

    assert opt.clip_hits == 0


@pytest.mark.sanity
def test_capped_update_has_norm_at_most_delta_times_per_shape_scale() -> None:
    """When capped, ``||u_pre_scale||_F ≤ delta`` exactly, so post-scale ≤ delta·scale."""
    from optimizers.sotr import _per_shape_scale, sotr_update

    shape = (64, 64)
    torch.manual_seed(0)
    grad = torch.randn(*shape)
    momentum_buffer = torch.zeros_like(grad)

    delta = 0.5
    update, clipped = sotr_update(
        grad.clone(),
        momentum_buffer,
        alpha=1.0,
        delta=delta,
        beta=0.95,
        ns_steps=5,
        nesterov=True,
        eps=1e-12,
    )
    assert clipped, "Δ=0.5 should clip an O(8)-norm orthogonal update."
    # The cap is applied pre-per-shape-scale; post-scale norm ≤ delta * scale.
    scale = _per_shape_scale(update)
    actual_norm = update.norm().item()
    expected_max = delta * scale + 1e-3  # bf16 slack
    assert actual_norm <= expected_max, (
        f"Capped update norm {actual_norm:.4f} exceeds delta * scale = {expected_max:.4f}"
    )
    # Also check it's not way smaller (the cap should saturate).
    assert actual_norm >= delta * scale * 0.95, (
        f"Capped update norm {actual_norm:.4f} is far below delta * scale = {delta * scale:.4f}; "
        "the cap may be over-clipping."
    )
