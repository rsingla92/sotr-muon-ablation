"""PROTOCOL §7 sanity check #9 — numerical verification of the SOTR blend identity.

The α-blend ``U = α·O + (1−α)·M/||M||_F`` is a *linear* combination in matrix
space. Its behavior in singular-value space depends on whether NS preserves
M's singular-vector basis. In exact (fp32) arithmetic, NS does preserve the
basis (it operates only on singular values), so the blend reduces to a
singular-value rescaling. **In Muon's bf16 NS implementation, the basis is
preserved only approximately** — empirically the off-diagonal "leak" in M's
basis is ~30% of the Frobenius norm at q=5. See
``knowledge/07_spectral_interpretation.md`` for the derivation and the bf16
caveat.

These tests verify the implementation is correct without overstating the
spectral identity in bf16:

1. **Linearity** (tautological but catches gross bugs): U = α·O + (1−α)·M_norm.
2. **q=0 collapse**: at q=0, NS returns M_norm, so the α knob has no effect
   and U = M_norm regardless of α.
3. **Diagonal-in-M-basis identity**: the diagonal of U expressed in M's
   singular basis matches α·f_q(σ_i) + (1−α)·σ_i/||M||_F, where f_q(σ_i) is
   the (numerically extracted) action of NS on each σ_i. This is the strongest
   thing that holds in bf16 — the diagonal portion is exactly the predicted
   linear combination, while the off-diagonal carries NS's bf16 imprecision.
4. **Off-diagonal scales linearly**: the off-diagonal portion of U in M's
   basis is exactly α times that of O (M_normalized has zero off-diagonal).
   Catches any leak of off-diagonal energy from M_normalized.
"""

from __future__ import annotations

import pytest
import torch
from muon import zeropower_via_newtonschulz5


def _make_matrix(
    m: int = 64,
    n: int = 64,
    *,
    sigma_decay: float = 2.0,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build M = U·Σ·V^T with controlled spectrum. Returns (M, U, sigma, V)."""
    g = torch.Generator().manual_seed(seed)
    U, _ = torch.linalg.qr(torch.randn(m, m, generator=g))
    V, _ = torch.linalg.qr(torch.randn(n, n, generator=g))
    rank = min(m, n)
    sigma = torch.tensor([sigma_decay**-i for i in range(rank)], dtype=torch.float32)
    Sigma = torch.zeros(m, n)
    Sigma[:rank, :rank] = torch.diag(sigma)
    M = U @ Sigma @ V.T
    return M, U, sigma, V


def _blend(O: torch.Tensor, M: torch.Tensor, alpha: float) -> torch.Tensor:
    """The exact blend computation SOTR performs (cast to fp32 for analysis)."""
    M_norm = M.norm()
    M_normalized = (M / (M_norm + 1e-12)).to(O.dtype).float()
    O_fp32 = O.float()
    if alpha == 1.0:
        return O_fp32
    if alpha == 0.0:
        return M_normalized
    return alpha * O_fp32 + (1 - alpha) * M_normalized


# ---------------------------------------------------------------------------


@pytest.mark.sanity
@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_linearity_blend_is_correct(alpha: float) -> None:
    """The blend output equals α·O + (1−α)·M_norm element-wise (catches sign/tensor bugs)."""
    M, _, _, _ = _make_matrix(sigma_decay=1.5)
    O = zeropower_via_newtonschulz5(M.unsqueeze(0), steps=5).squeeze(0)
    actual = _blend(O, M, alpha)

    # Manual recomputation, no shortcuts.
    O_fp32 = O.float()
    M_normalized = (M / (M.norm() + 1e-12)).to(O.dtype).float()
    expected = alpha * O_fp32 + (1 - alpha) * M_normalized

    assert torch.allclose(actual, expected, atol=1e-7), (
        f"Blend output deviates from α·O + (1-α)·M_norm at α={alpha}"
    )


@pytest.mark.sanity
@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_q0_collapses_to_normalized_M_regardless_of_alpha(alpha: float) -> None:
    """At q=0, NS returns the bf16-normalized M, so U = M_norm for any α."""
    M, _, _, _ = _make_matrix(sigma_decay=1.5)
    O = zeropower_via_newtonschulz5(M.unsqueeze(0), steps=0).squeeze(0)

    M_norm_bf16 = (M / (M.norm() + 1e-12)).to(O.dtype)

    # Both NS output and M_normalized should be byte-identical (both = bf16 of M/||M||).
    # NS at q=0 has X = G/||G||_F + the eps inside, which differs slightly from our 1e-12 eps.
    # Tolerance reflects that.
    assert torch.allclose(O.float(), M_norm_bf16.float(), atol=1e-3), (
        "NS at q=0 should equal M / ||M||_F; got significant deviation. "
        "Either the NS normalization changed or our M_norm computation is wrong."
    )

    U = _blend(O, M, alpha)
    drift = (U - M_norm_bf16.float()).norm() / M_norm_bf16.float().norm()
    assert drift < 5e-3, (
        f"At q=0, blend should equal M_normalized for any α; got rel drift = {drift:.2e} "
        f"at α={alpha}. This means the α knob is not properly degenerate at q=0."
    )


@pytest.mark.sanity
def test_diagonal_in_M_basis_matches_blend_formula() -> None:
    """The diagonal of U_blend (in M's basis) is α·f_q(σ_i) + (1−α)·σ_i/||M||_F.

    This is the "spectral identity" of the SOTR blend. We don't need analytical
    f_q — we extract it numerically from O's diagonal in M's basis.
    """
    M, U_M, sigma_M, V_M = _make_matrix(sigma_decay=2.0)
    M_norm = M.norm()

    O = zeropower_via_newtonschulz5(M.unsqueeze(0), steps=5).squeeze(0).float()

    # Numerically extract f_q(σ_i): diagonal of O in M's basis
    O_in_M_basis = U_M.T @ O @ V_M
    f_sigma = torch.diagonal(O_in_M_basis)  # (rank,)

    sigma_normalized = sigma_M / M_norm

    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        U_blend = _blend(O, M, alpha)
        U_blend_in_M_basis = U_M.T @ U_blend @ V_M
        diag_actual = torch.diagonal(U_blend_in_M_basis)
        diag_predicted = alpha * f_sigma + (1 - alpha) * sigma_normalized

        # bf16 internal precision sets the floor.
        max_err = (diag_actual - diag_predicted).abs().max().item()
        assert max_err < 5e-3, (
            f"Diagonal-in-M-basis identity failed at α={alpha}: max diagonal error = "
            f"{max_err:.2e}. The blend's projection onto M's basis does not match "
            f"α·f_q(σ_i) + (1-α)·σ_i/||M||_F. Implementation bug likely."
        )


@pytest.mark.sanity
def test_offdiagonal_in_M_basis_scales_linearly_with_alpha() -> None:
    """The off-diagonal portion of U_blend in M's basis is α times that of O.

    This is a direct consequence of linearity: M_normalized has zero off-diagonal
    in M's basis (since it's diag(σ/||M||) ), so the blend's off-diagonal is
    entirely from α·O. Catches if M_normalized is somehow leaking off-diagonal.
    """
    M, U_M, _, V_M = _make_matrix(sigma_decay=2.0)
    O = zeropower_via_newtonschulz5(M.unsqueeze(0), steps=5).squeeze(0).float()

    O_in_M_basis = U_M.T @ O @ V_M
    O_off_diag = O_in_M_basis - torch.diag(torch.diagonal(O_in_M_basis))

    for alpha in (0.25, 0.5, 0.75):
        U_blend = _blend(O, M, alpha)
        U_in_M_basis = U_M.T @ U_blend @ V_M
        U_off_diag = U_in_M_basis - torch.diag(torch.diagonal(U_in_M_basis))

        expected_off_diag = alpha * O_off_diag
        rel_err = (U_off_diag - expected_off_diag).norm() / O_off_diag.norm().clamp(min=1e-12)
        assert rel_err < 1e-3, (
            f"At α={alpha}, off-diagonal of U is not α·(off-diagonal of O): "
            f"rel error = {rel_err:.2e}. M_normalized may be leaking off-diagonal "
            "components in M's basis (indicating M's SVD computation is wrong)."
        )
