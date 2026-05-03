"""Generate frozen reference trajectories for sanity tests #4 and #5.

Output ``.pt`` files are committed to the repo so tests run offline.

**Fixtures are tied to the torch build** that generates them. ``torch.randn`` is
not bit-identical across torch builds (e.g., ``2.11.0`` vs ``2.11.0+computecanada``)
because they link different math libraries (MKL, AOCL, Accelerate, etc.). The
``test_lion_match`` / ``test_muon_match`` sanity tests skip themselves if the
current torch build doesn't match the build recorded in the fixture, with a
clear pointer to regenerate.

Regenerate when:
- ``external/Muon`` or ``external/lion-pytorch`` is intentionally bumped to a
  new pinned commit (record in PROTOCOL.md §15 amendment), OR
- The active torch build changes (e.g., switching between local and DRAC).

Usage:
    python -m tests.fixtures.generate_references
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from lion_pytorch import Lion
from muon import SingleDeviceMuon
from torch import nn

OUT_DIR = Path(__file__).parent
SHAPE = (64, 64)
N_STEPS = 100


def _grad_for_step(step: int) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(2026_05_02 + step)
    return torch.randn(*SHAPE, generator=g)


def gen_lion_reference() -> None:
    """Frozen Lion trajectory with documented hyperparameters."""
    torch.manual_seed(0)
    p = nn.Parameter(torch.randn(*SHAPE))
    initial = p.detach().clone()

    opt = Lion([p], lr=1e-4, betas=(0.9, 0.99), weight_decay=0.01)

    # Only checkpoint at evenly spaced steps to keep fixture small (~30 KB).
    # 11 checkpoints over 100 steps catches drift early without bloating the repo.
    checkpoint_steps = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
    checkpoints = {}
    for step in range(N_STEPS):
        p.grad = _grad_for_step(step)
        opt.step()
        if step in checkpoint_steps:
            checkpoints[step] = p.detach().clone()

    out_path = OUT_DIR / "lion_reference.pt"
    torch.save(
        {
            "version": 2,  # bumped: now records torch_version
            "torch_version": torch.__version__,
            "shape": SHAPE,
            "n_steps": N_STEPS,
            "initial_seed": 0,
            "grad_seed_base": 2026_05_02,
            "lr": 1e-4,
            "betas": (0.9, 0.99),
            "weight_decay": 0.01,
            "initial": initial,
            "checkpoint_steps": checkpoint_steps,
            "checkpoints": checkpoints,
        },
        out_path,
    )
    print(f"wrote {out_path}  ({out_path.stat().st_size / 1024:.1f} KB; torch={torch.__version__})")


def gen_muon_reference() -> None:
    """Frozen SingleDeviceMuon trajectory with documented hyperparameters."""
    torch.manual_seed(0)
    p = nn.Parameter(torch.randn(*SHAPE))
    initial = p.detach().clone()

    opt = SingleDeviceMuon([p], lr=0.02, momentum=0.95, weight_decay=0)

    # Only checkpoint at evenly spaced steps to keep fixture small (~30 KB).
    # 11 checkpoints over 100 steps catches drift early without bloating the repo.
    checkpoint_steps = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
    checkpoints = {}
    for step in range(N_STEPS):
        p.grad = _grad_for_step(step)
        opt.step()
        if step in checkpoint_steps:
            checkpoints[step] = p.detach().clone()

    out_path = OUT_DIR / "muon_reference.pt"
    torch.save(
        {
            "version": 2,  # bumped: now records torch_version
            "torch_version": torch.__version__,
            "shape": SHAPE,
            "n_steps": N_STEPS,
            "initial_seed": 0,
            "grad_seed_base": 2026_05_02,
            "lr": 0.02,
            "momentum": 0.95,
            "weight_decay": 0,
            "initial": initial,
            "checkpoint_steps": checkpoint_steps,
            "checkpoints": checkpoints,
        },
        out_path,
    )
    print(f"wrote {out_path}  ({out_path.stat().st_size / 1024:.1f} KB; torch={torch.__version__})")


def main() -> int:
    gen_lion_reference()
    gen_muon_reference()
    return 0


if __name__ == "__main__":
    sys.exit(main())
