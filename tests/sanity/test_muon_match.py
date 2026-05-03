"""PROTOCOL §7 sanity check #5 — imported Muon matches frozen reference trajectory.

Same purpose as ``test_lion_match.py`` but for ``muon.SingleDeviceMuon``. Catches
drift in ``external/Muon`` due to submodule bumps or environment changes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from muon import SingleDeviceMuon
from torch import nn

FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "muon_reference.pt"


@pytest.mark.sanity
def test_muon_matches_frozen_reference() -> None:
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

    opt = SingleDeviceMuon(
        [p],
        lr=fix["lr"],
        momentum=fix["momentum"],
        weight_decay=fix["weight_decay"],
    )

    for step in range(fix["n_steps"]):
        g = torch.Generator().manual_seed(fix["grad_seed_base"] + step)
        p.grad = torch.randn(*fix["shape"], generator=g)
        opt.step()

        if step in fix["checkpoints"]:
            expected = fix["checkpoints"][step]
            drift = (p - expected).norm().item()
            # Slightly looser tolerance than Lion: NS runs in bf16 internally
            # and bf16 ops may differ across CUDA versions / CPUs, but on the
            # same machine they should be identical.
            assert drift < 1e-4, (
                f"Muon drifted from reference at step {step}: ||drift|| = {drift:.2e}. "
                "Either external/Muon was bumped (regenerate fixture) or your "
                "environment differs from the one that generated the fixture."
            )
