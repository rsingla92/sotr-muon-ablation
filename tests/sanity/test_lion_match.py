"""PROTOCOL §7 sanity check #4 — imported Lion matches frozen reference trajectory.

Verifies that ``lion_pytorch.Lion`` (from ``external/lion-pytorch``) produces
the trajectory recorded in ``tests/fixtures/lion_reference.pt``. Catches:

- Accidental drift if the submodule is bumped to a new commit.
- Environment differences (different torch version, hardware) producing
  numerically different updates.
- Unintended changes to the optimizer's hyperparameter defaults.

If this test fails after a deliberate submodule bump, regenerate the fixture
via ``python -m tests.fixtures.generate_references`` and record the bump as
an amendment in PROTOCOL.md §15.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from lion_pytorch import Lion
from torch import nn

FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "lion_reference.pt"


@pytest.mark.sanity
def test_lion_matches_frozen_reference() -> None:
    if not FIXTURE_PATH.exists():
        pytest.skip(
            f"Fixture {FIXTURE_PATH} missing — run "
            "`python -m tests.fixtures.generate_references` to create it."
        )

    fix = torch.load(FIXTURE_PATH, weights_only=False)

    torch.manual_seed(fix["initial_seed"])
    p = nn.Parameter(torch.randn(*fix["shape"]))
    assert torch.allclose(p, fix["initial"], atol=0), (
        "Initial parameter mismatch — torch.randn behavior may have changed across versions."
    )

    opt = Lion(
        [p],
        lr=fix["lr"],
        betas=fix["betas"],
        weight_decay=fix["weight_decay"],
    )

    for step in range(fix["n_steps"]):
        g = torch.Generator().manual_seed(fix["grad_seed_base"] + step)
        p.grad = torch.randn(*fix["shape"], generator=g)
        opt.step()

        if step in fix["checkpoints"]:
            expected = fix["checkpoints"][step]
            drift = (p - expected).norm().item()
            assert drift < 1e-5, (
                f"Lion drifted from reference at step {step}: ||drift|| = {drift:.2e}. "
                "Either external/lion-pytorch was bumped (regenerate fixture) or your "
                "environment differs from the one that generated the fixture."
            )
