"""Soft-Orthogonal Trust Region (SOTR) optimizer.

The single novel optimizer in this repo. See:
- ``PROTOCOL.md`` §7 for the locked algorithm and sanity checks.
- ``knowledge/03_sotr_design.md`` for the design rationale and the corrected
  Muon-compatible step ordering.
- ``knowledge/07_spectral_interpretation.md`` for the singular-value-space
  identity that anchors the paper's theoretical claim.

SOTR composes three soft-orthogonalization mechanisms applied per 2D weight matrix:

1. **Partial Newton-Schulz** (``ns_steps``: typically 1 or 2 instead of Muon's 5).
2. **α-blend** between the NS-orthogonalized direction ``O = NS(M)`` and the
   Frobenius-normalized momentum ``M / ||M||_F``. At ``α=1`` we recover ``O``;
   at ``α=0`` we recover ``M / ||M||_F``.
3. **Per-matrix Frobenius trust region** (``delta``): the post-blend update is
   capped to Frobenius norm ``≤ delta``. Δ=∞ disables the cap.

The Newton-Schulz iteration is imported from ``external/Muon``
(``zeropower_via_newtonschulz5``); we do not reimplement it. At the corner case
``α=1, Δ=∞, q=5`` the SOTR step is byte-equivalent to Muon's ``muon_update``,
which is verified by ``tests/sanity/test_sotr_limits.py``.
"""

from __future__ import annotations

import math

import torch
from muon import zeropower_via_newtonschulz5
from torch.optim.optimizer import Optimizer


def _per_shape_scale(update: torch.Tensor) -> float:
    """Match Muon's per-shape RMS calibration: ``sqrt(max(1, m / n))``.

    For a 2D matrix of shape ``(m, n)``, this scales updates so the typical
    per-element RMS is comparable across layer shapes (Muon's recipe). For
    near-square matrices this is ≈ 1; for very rectangular matrices it
    grows like ``sqrt(aspect_ratio)``.
    """
    return math.sqrt(max(1.0, update.size(-2) / update.size(-1)))


def sotr_update(
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    *,
    alpha: float,
    delta: float,
    beta: float,
    ns_steps: int,
    nesterov: bool,
    eps: float,
) -> tuple[torch.Tensor, bool]:
    """Compute one SOTR update direction.

    Mirrors ``muon.muon_update``'s in-place momentum + Nesterov-mix logic so
    that at ``alpha=1, delta=inf, ns_steps=5, nesterov=True`` the result is
    identical to Muon. The novel SOTR pieces (α-blend and Frobenius cap) are
    inserted between NS and the per-shape scaling.

    Returns ``(update, clipped)`` where ``clipped`` is True iff the Frobenius
    cap was hit on this step.

    The argument names mirror Muon's ``muon_update``. ``grad`` and
    ``momentum_buffer`` are mutated in place (matching Muon's convention);
    callers should not rely on ``grad`` being unchanged after this call.
    """
    assert grad.ndim >= 2, "SOTR is for matrix parameters only (use AdamW for 1D params)."

    # Steps 1–2: momentum buffer update + Nesterov mix. Identical to muon_update.
    # Note: with nesterov=True, grad is mutated in place (matches Muon's convention).
    momentum_buffer.lerp_(grad, 1 - beta)
    m_mixed = grad.lerp_(momentum_buffer, beta) if nesterov else momentum_buffer

    # Conv filters: collapse trailing dims to 2D, matching Muon.
    if m_mixed.ndim == 4:
        m_mixed = m_mixed.view(len(m_mixed), -1)

    # Step 3: NS the (Nesterov-mixed) value. Returns bf16 (Muon's convention).
    o = zeropower_via_newtonschulz5(m_mixed, steps=ns_steps)

    # Step 4: α-blend in singular-value space (see knowledge/07_spectral_interpretation.md).
    # We branch on α to guarantee byte-equivalence with Muon at α=1.
    if alpha == 1.0:
        u = o
    else:
        # M / ||M||_F computed in fp32 then cast to match O's dtype (bf16).
        # The eps avoids division by zero on a pathological all-zero gradient.
        m_norm = m_mixed.norm()
        m_normalized = (m_mixed / (m_norm + eps)).to(o.dtype)
        u = m_normalized if alpha == 0.0 else o.mul(alpha).add_(m_normalized, alpha=1 - alpha)

    # Step 5: Frobenius trust region. Cap is per-matrix (the novel piece).
    clipped = False
    if math.isfinite(delta):
        u_norm = u.norm()
        if u_norm > delta:
            u = u.mul(delta / u_norm)
            clipped = True

    # Step 6: per-shape RMS scaling (matches Muon's muon_update).
    u = u * _per_shape_scale(u)

    return u, clipped


class SOTR(Optimizer):
    """Soft-Orthogonal Trust Region optimizer.

    Apply only to 2D (or 4D conv) hidden weight parameters. Use AdamW for
    embeddings, head, biases, LayerNorm gains/biases. See
    ``muon.MuonWithAuxAdam`` for the canonical param-group split convention.

    Args:
        params: iterable of 2D-or-higher parameters.
        lr: learning rate. Muon-style; typical 0.02 for hidden weights.
        weight_decay: AdamW-style decoupled weight decay.
        momentum: SGD momentum coefficient (Muon calls this ``beta``).
            Typical 0.95.
        alpha: SOTR α-blend in [0, 1]. ``1`` recovers Muon (NS only); ``0``
            recovers Frobenius-normalized momentum SGD; intermediate values
            are linearly interpolated singular-value rescalings.
        delta: per-matrix Frobenius trust region radius. Use ``float("inf")``
            to disable the cap.
        ns_steps: Newton-Schulz iteration count. Muon uses 5; SOTR's
            recommended 1–2 for the partial-NS regime.
        nesterov: whether to use Nesterov momentum mix (matches Muon's default).
        eps: numerical stability for ``M / ||M||_F``.

    Statistics:
        ``self.clip_hits`` and ``self.clip_total`` track Frobenius-cap activations
        across all (parameter × step) events since last :meth:`reset_stats`.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        alpha: float = 0.5,
        delta: float = 1.0,
        ns_steps: int = 2,
        nesterov: bool = True,
        eps: float = 1e-12,
    ):
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if delta <= 0:
            raise ValueError(f"delta must be > 0 (use float('inf') to disable), got {delta}")
        if ns_steps < 0:
            raise ValueError(f"ns_steps must be >= 0, got {ns_steps}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            alpha=alpha,
            delta=delta,
            ns_steps=ns_steps,
            nesterov=nesterov,
            eps=eps,
        )
        super().__init__(params, defaults)

        self.clip_hits = 0
        self.clip_total = 0

    def reset_stats(self) -> None:
        self.clip_hits = 0
        self.clip_total = 0

    @property
    def clip_rate(self) -> float:
        return self.clip_hits / self.clip_total if self.clip_total else 0.0

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim < 2:
                    raise RuntimeError(
                        f"SOTR requires matrix parameters (ndim >= 2); got ndim={p.ndim}. "
                        "Use AdamW for biases / LayerNorm / 1D params."
                    )

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)

                update, clipped = sotr_update(
                    p.grad,
                    state["momentum_buffer"],
                    alpha=group["alpha"],
                    delta=group["delta"],
                    beta=group["momentum"],
                    ns_steps=group["ns_steps"],
                    nesterov=group["nesterov"],
                    eps=group["eps"],
                )
                self.clip_total += 1
                if clipped:
                    self.clip_hits += 1

                # Decoupled (AdamW-style) weight decay, then update.
                if group["weight_decay"] != 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss
