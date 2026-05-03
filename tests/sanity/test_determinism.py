"""PROTOCOL §7 sanity check #7 — same seed produces same trajectory.

Bit-identical on CPU (1e-7 tolerance for fp accumulation); within 1e-4 on GPU
(non-deterministic CUDA kernels in NS may add small drift). Two SOTR runs
with the same seed and same gradient sequence must produce the same final
parameters.

This catches:
- Hidden non-determinism in the optimizer (uninitialized buffers, etc.).
- Accidentally seeded-then-unseeded random number generation.
- Use of non-deterministic ops in the SOTR step.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from optimizers import SOTR


def _run_one_trajectory(seed: int, n_steps: int, shape: tuple[int, int]) -> torch.Tensor:
    """Initialize parameter from seed, run SOTR for n_steps, return final p."""
    torch.manual_seed(seed)
    p = nn.Parameter(torch.randn(*shape))
    opt = SOTR([p], lr=0.01, momentum=0.95, alpha=0.5, delta=1.0, ns_steps=2)
    for step in range(n_steps):
        g = torch.Generator().manual_seed(seed + step + 1)
        p.grad = torch.randn(*shape, generator=g)
        opt.step()
    return p.detach().clone()


@pytest.mark.sanity
def test_same_seed_same_trajectory_cpu() -> None:
    """CPU: bit-identical or within fp accumulation noise."""
    p1 = _run_one_trajectory(seed=0, n_steps=20, shape=(128, 64))
    p2 = _run_one_trajectory(seed=0, n_steps=20, shape=(128, 64))

    drift = (p1 - p2).norm().item()
    assert drift < 1e-7, (
        f"Same-seed CPU runs drifted by {drift:.2e} > 1e-7. "
        "There's hidden non-determinism in SOTR's step."
    )


@pytest.mark.sanity
def test_different_seeds_produce_different_trajectories() -> None:
    """Sanity-of-sanity: if different seeds give the same result, the seed isn't being used."""
    p1 = _run_one_trajectory(seed=0, n_steps=10, shape=(64, 64))
    p2 = _run_one_trajectory(seed=1, n_steps=10, shape=(64, 64))

    diff = (p1 - p2).norm().item()
    assert diff > 1e-3, (
        f"Different seeds produced suspiciously similar trajectories (diff = {diff:.2e}). "
        "The seed may not be threading through correctly."
    )


@pytest.mark.sanity
@pytest.mark.gpu
def test_same_seed_same_trajectory_gpu() -> None:
    """GPU: within 1e-4 (non-deterministic CUDA kernels in NS bf16 ops)."""
    device = torch.device("cuda")

    def run(seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        p = nn.Parameter(torch.randn(128, 64, device=device))
        opt = SOTR([p], lr=0.01, momentum=0.95, alpha=0.5, delta=1.0, ns_steps=2)
        for step in range(10):
            g = torch.Generator(device="cpu").manual_seed(seed + step + 1)
            p.grad = torch.randn(128, 64, generator=g).to(device)
            opt.step()
        return p.detach().clone()

    p1 = run(seed=0)
    p2 = run(seed=0)
    drift = (p1 - p2).norm().item()
    assert drift < 1e-4, (
        f"Same-seed GPU runs drifted by {drift:.2e} > 1e-4. "
        "Excessive non-determinism even allowing for CUDA tolerance."
    )
