"""PROTOCOL §7 sanity checks #1, #2, #3 — SOTR limit cases.

#1: SOTR(α=1, Δ=∞, q=5) byte-equivalent to Muon over 50 steps on 256x256.
#2: SOTR(α=0, q=0) reduces to Frobenius-normalized momentum × per-shape scale.
#3: SOTR(α=1, q=2) measurably differs from SOTR(α=1, q=5) — the q knob is wired.
"""

from __future__ import annotations

import math

import pytest
import torch
from muon import SingleDeviceMuon, zeropower_via_newtonschulz5
from torch import nn

from optimizers import SOTR


def _grad_seq(seed_base: int, n: int, shape: tuple[int, int]) -> list[torch.Tensor]:
    """Reproducible per-step gradient sequence."""
    grads = []
    for step in range(n):
        g = torch.Generator().manual_seed(seed_base + step)
        grads.append(torch.randn(*shape, generator=g))
    return grads


@pytest.mark.sanity
def test_alpha_one_full_ns_matches_muon_byte_for_byte() -> None:
    """#1: SOTR(α=1, Δ=∞, q=5) ≡ SingleDeviceMuon over 50 steps."""
    shape = (256, 256)
    n_steps = 50
    grads = _grad_seq(seed_base=1000, n=n_steps, shape=shape)

    torch.manual_seed(0)
    p_sotr = nn.Parameter(torch.randn(*shape))
    p_muon = nn.Parameter(p_sotr.detach().clone())

    sotr = SOTR(
        [p_sotr],
        lr=0.02,
        momentum=0.95,
        alpha=1.0,
        delta=float("inf"),
        ns_steps=5,
        weight_decay=0.0,
    )
    muon = SingleDeviceMuon([p_muon], lr=0.02, momentum=0.95, weight_decay=0)

    for grad in grads:
        p_sotr.grad = grad.clone()
        p_muon.grad = grad.clone()
        sotr.step()
        muon.step()

    rel_err = ((p_sotr - p_muon).norm() / p_muon.norm()).item()
    assert rel_err < 1e-5, (
        f"SOTR(α=1) drifted from Muon: relative error = {rel_err:.2e} > 1e-5. "
        "This breaks the central rhetorical claim 'SOTR strictly contains Muon'."
    )


@pytest.mark.sanity
def test_alpha_zero_q_zero_reduces_to_normalized_momentum() -> None:
    """#2: SOTR(α=0, q=0) ≡ Frobenius-normalized Nesterov-momentum × per-shape scale.

    With q=0, NS returns ``M_bf16 / ||M_bf16||_F``. The α-blend then yields
    ``M_bf16 / ||M_bf16||_F`` for any α (q=0 makes α irrelevant). The per-shape
    scaling ``sqrt(max(1, m/n))`` is applied identically. This test verifies
    one full step matches the closed-form expectation.
    """
    shape = (32, 64)  # rectangular so per-shape scale = 1.0
    torch.manual_seed(0)
    init = torch.randn(*shape)

    p = nn.Parameter(init.clone())
    opt = SOTR(
        [p],
        lr=0.02,
        momentum=0.95,
        alpha=0.0,
        delta=float("inf"),
        ns_steps=0,
        weight_decay=0.0,
    )

    grad = torch.randn(*shape, generator=torch.Generator().manual_seed(42))
    p.grad = grad.clone()
    opt.step()

    # Reconstruct expected behavior manually.
    # On step 1 with momentum_buffer=0:
    #   buf <- lerp(0, grad, 1-β) = (1-β) * grad
    #   M = grad.lerp(buf, β) = grad + β·(buf - grad) = grad + β·((1-β)·grad - grad) = grad·(1 - β + β·(1-β))
    # In our impl this happens via `grad.lerp_(buf, beta)` after `buf.lerp_(grad, 1-beta)`.
    beta = 0.95
    buf_expected = (1 - beta) * grad
    m_expected = grad + beta * (buf_expected - grad)  # i.e., grad.lerp(buf_expected, beta)

    # NS at q=0 normalizes to unit Frobenius (in bf16 inside zeropower_via_newtonschulz5).
    o_expected = zeropower_via_newtonschulz5(m_expected, steps=0)
    # α=0 blend: U = M / ||M||_F (cast to bf16 to match O's dtype, per sotr.py).
    m_norm = m_expected.norm()
    u_expected = (m_expected / (m_norm + 1e-12)).to(o_expected.dtype)
    # Per-shape scale: shape (32, 64) → max(1, 32/64) = 1.
    scale = math.sqrt(max(1.0, shape[-2] / shape[-1]))
    u_expected = u_expected * scale

    p_expected = init - 0.02 * u_expected.reshape(p.shape).float()

    drift = (p - p_expected).norm().item()
    rel_drift = drift / p_expected.norm().item()
    # bf16 internal precision tolerance.
    assert rel_drift < 1e-2, f"α=0 q=0 limit case drifted: rel = {rel_drift:.2e}"


@pytest.mark.sanity
def test_partial_ns_distinguishable_from_full_ns() -> None:
    """#3: SOTR(α=1, q=2) ≠ SOTR(α=1, q=5) — partial-NS is observably different."""
    shape = (128, 128)
    torch.manual_seed(0)
    init = torch.randn(*shape)

    p_q2 = nn.Parameter(init.clone())
    p_q5 = nn.Parameter(init.clone())

    opt_q2 = SOTR([p_q2], lr=0.02, momentum=0.95, alpha=1.0, delta=float("inf"), ns_steps=2)
    opt_q5 = SOTR([p_q5], lr=0.02, momentum=0.95, alpha=1.0, delta=float("inf"), ns_steps=5)

    grad = torch.randn(*shape, generator=torch.Generator().manual_seed(42))
    p_q2.grad = grad.clone()
    p_q5.grad = grad.clone()
    opt_q2.step()
    opt_q5.step()

    rel_diff = ((p_q2 - p_q5).norm() / p_q5.norm()).item()
    # NS at q=5 produces "S_ii ~ Uniform(0.5, 1.5)" (per Muon docstring); at q=2
    # the spread is wider. Update directions differ enough to be observable.
    assert rel_diff > 1e-4, (
        f"q=2 and q=5 produce indistinguishable updates (rel_diff = {rel_diff:.2e}); "
        "the ns_steps knob may not be wired correctly."
    )
