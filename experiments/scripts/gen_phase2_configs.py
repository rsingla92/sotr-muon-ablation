"""Generate the Phase 2 ablation grid (PROTOCOL §9).

10 cells × 5 seeds × 5 LRs = 250 runs. Each (cell × seed × LR) tuple becomes
one Python config module under ``experiments/configs/phase2/``. The flat
``index.txt`` is what ``scripts/slurm/array_ablation.sh`` reads.

Generated configs are gitignored — see ``.gitignore``. The generator is the
source of truth: regenerate after any cell-grid change in PROTOCOL §9 by
running ``python -m experiments.scripts.gen_phase2_configs``.

Cell definitions are mirrored from PROTOCOL §9. Keep this in sync if §9
changes (and the meta-test in ``tests/sanity/test_sanity_coverage.py`` will
catch drift between PROTOCOL.md and our test suite).

Each emitted config module exposes a single ``config = RunConfig(...)``
attribute so ``train.py --config experiments.configs.phase2.<cell>.<seed>_<lr>``
loads it.
"""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

from experiments._configs import OptimizerKind, RunConfig
from experiments.configs._phase2_base import base_config

# ---------------------------------------------------------------------------
# Cell definitions (PROTOCOL §9 ablation grid)
# ---------------------------------------------------------------------------


def _cells() -> list[dict]:
    """Each dict is a complete Phase 2 cell, keyed for filename + RunConfig fields."""
    return [
        # A. Full SOTR
        dict(
            name="A_sotr_full",
            optimizer_name=OptimizerKind.SOTR,
            sotr_alpha=0.5,
            sotr_delta=1.0,
            sotr_ns_steps=2,
        ),
        # B. Drop α-blend (α=1, partial NS, with Frobenius cap)
        dict(
            name="B_drop_alpha",
            optimizer_name=OptimizerKind.SOTR,
            sotr_alpha=1.0,
            sotr_delta=1.0,
            sotr_ns_steps=2,
        ),
        # C. Drop Δ cap (α-blend, partial NS, no cap)
        dict(
            name="C_drop_delta",
            optimizer_name=OptimizerKind.SOTR,
            sotr_alpha=0.5,
            sotr_delta=math.inf,
            sotr_ns_steps=2,
        ),
        # D. Drop both (Muon-like with q=2, no SOTR knobs)
        dict(
            name="D_drop_both",
            optimizer_name=OptimizerKind.SOTR,
            sotr_alpha=1.0,
            sotr_delta=math.inf,
            sotr_ns_steps=2,
        ),
        # E. Drop NS (α-blend + Fro cap, no orth)
        dict(
            name="E_drop_ns",
            optimizer_name=OptimizerKind.SOTR,
            sotr_alpha=0.5,
            sotr_delta=1.0,
            sotr_ns_steps=0,
        ),
        # F. Full NS (α-blend, q=5)
        dict(
            name="F_full_ns",
            optimizer_name=OptimizerKind.SOTR,
            sotr_alpha=0.5,
            sotr_delta=1.0,
            sotr_ns_steps=5,
        ),
        # G. α schedule  — flagged TODO; α scheduling is not yet wired through
        #    train.py. We emit the static-α=0.5 config here for now and revisit
        #    once the scheduler is implemented (see PROTOCOL §9 cell G note).
        dict(
            name="G_alpha_schedule",
            optimizer_name=OptimizerKind.SOTR,
            sotr_alpha=0.5,
            sotr_delta=1.0,
            sotr_ns_steps=2,
        ),
        # H. Δ scheduled — same caveat as G; static Δ=1.0 for now.
        dict(
            name="H_delta_schedule",
            optimizer_name=OptimizerKind.SOTR,
            sotr_alpha=0.5,
            sotr_delta=1.0,
            sotr_ns_steps=2,
        ),
        # I. Muon + Frobenius cap only
        # (Cell I uses optimizer_name=SOTR with α=1 + Δ=1.0 + q=5: the
        # SOTR(α=1, q=5) limit is byte-equivalent to Muon, then we re-apply
        # the cap. This isolates Δ as the sole novel mechanism.)
        dict(
            name="I_muon_plus_cap",
            optimizer_name=OptimizerKind.SOTR,
            sotr_alpha=1.0,
            sotr_delta=1.0,
            sotr_ns_steps=5,
        ),
        # J. Partial-NS Muon (α=1, Δ=∞, q=2 — Muon-like with partial NS only)
        dict(
            name="J_partial_ns_muon",
            optimizer_name=OptimizerKind.SOTR,
            sotr_alpha=1.0,
            sotr_delta=math.inf,
            sotr_ns_steps=2,
        ),
        # K. Canonical Muon (α=1, Δ=∞, q=5 — SOTR's full-Muon limit).
        # Added by 2026-05-26 amendment to anchor PROTOCOL §3 H1 (SOTR > Muon).
        # Without this cell, we have no Phase-2 Muon baseline to compare F (the
        # best q=5 SOTR variant) against. The SOTR(α=1, Δ=∞, q=5) form is
        # byte-equivalent to canonical Muon — same NS polynomial, same scale.
        dict(
            name="K_muon_canonical",
            optimizer_name=OptimizerKind.SOTR,
            sotr_alpha=1.0,
            sotr_delta=math.inf,
            sotr_ns_steps=5,
        ),
    ]


# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------


SEEDS = (0, 1, 2, 3, 4)

# LR sweep: 5 log-spaced points around the upstream Muon default of 0.02.
# Range is half a decade above and below: 0.005 to 0.08.
LRS = (0.005, 0.01, 0.02, 0.04, 0.08)

# Extended LRs above 0.08 — added by the 2026-05-26 amendment. Phase 2 results
# showed both F and I peaking at the upper edge (LR=0.08) of the original sweep,
# suggesting their true optima may be higher. We extend symmetrically for F, I,
# and the newly-added K so the H1 comparison covers the same LR grid.
LRS_EXTENSION = (0.12, 0.16)
EXTENSION_CELLS = ("F_full_ns", "I_muon_plus_cap", "K_muon_canonical")


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------


_HEADER = '''"""AUTO-GENERATED by experiments/scripts/gen_phase2_configs.py.

Do not edit by hand. Regenerate with:
    python -m experiments.scripts.gen_phase2_configs

Cell {cell_name}, seed={seed}, lr={lr}.
"""

import math

from experiments._configs import OptimizerKind, RunConfig
from experiments.configs._phase2_base import base_config

config = '''


def _config_for(cell: dict, seed: int, lr: float) -> RunConfig:
    """Build a single RunConfig from base_config + cell + seed + lr."""
    new_hparams = replace(base_config.hparams, muon_learning_rate=lr)
    new_name = f"phase2_{cell['name']}_seed{seed}_lr{lr:g}"
    return replace(
        base_config,
        name=new_name,
        hparams=new_hparams,
        seed=seed,
        **{k: v for k, v in cell.items() if k != "name"},
    )


def _emit(cell: dict, seed: int, lr: float, out_dir: Path) -> tuple[str, str]:
    """Write one config file. Return (module_path, file_path)."""
    cell_dir = out_dir / cell["name"]
    cell_dir.mkdir(parents=True, exist_ok=True)
    # Init files for the package path.
    (cell_dir / "__init__.py").touch(exist_ok=True)

    file_name = f"seed{seed}_lr{lr:g}.py".replace(".", "p", 1).replace("p", ".", 1)
    # Filename: replace any '.' inside the lr part with 'p' for valid module names.
    safe_lr = f"{lr:g}".replace(".", "p")
    file_name = f"seed{seed}_lr{safe_lr}.py"
    file_path = cell_dir / file_name

    # Build the literal `RunConfig(...)` form we want to write — easier to
    # round-trip than `repr(config)` because RunConfig has nested defaults.
    cell_kwargs = {k: v for k, v in cell.items() if k != "name"}
    kwargs_str_parts = []
    for k, v in cell_kwargs.items():
        if isinstance(v, OptimizerKind):
            kwargs_str_parts.append(f"        {k}=OptimizerKind.{v.name},")
        elif v == math.inf:
            kwargs_str_parts.append(f"        {k}=math.inf,")
        else:
            kwargs_str_parts.append(f"        {k}={v!r},")
    kwargs_str = "\n".join(kwargs_str_parts)

    body = _HEADER.format(cell_name=cell["name"], seed=seed, lr=lr) + (
        "RunConfig(\n"
        f'        name="{_config_for(cell, seed, lr).name}",\n'
        f'        purpose="phase2",\n'
        f'        description="Phase 2 cell {cell["name"]}, seed={seed}, lr={lr:g}.",\n'
        "        hparams=base_config.hparams.__class__(\n"
        "            **{**base_config.hparams.__dict__,\n"
        f'              "muon_learning_rate": {lr!r}}}),\n'
        f"        seed={seed},\n"
        f"{kwargs_str}\n"
        "    )\n"
    )
    file_path.write_text(body)
    module_path = f"experiments.configs.phase2.{cell['name']}.seed{seed}_lr{safe_lr}"
    return module_path, str(file_path)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "experiments" / "configs" / "phase2"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "__init__.py").touch(exist_ok=True)

    cells = _cells()
    index_lines: list[str] = []
    n_emitted = 0

    # Pass 1: original §9 grid (cells A-J + K) at standard LRs.
    # Order matters — indices 0..249 must continue to match the cells/seeds/LRs
    # we already ran, so existing run dirs (results/phase2/<run_id>/) keep their
    # array-index mapping. New cell K appends at indices 250..274.
    for cell in cells:
        for seed in SEEDS:
            for lr in LRS:
                module_path, _file_path = _emit(cell, seed, lr, out_dir)
                index_lines.append(module_path)
                n_emitted += 1

    # Pass 2: LR extension for F, I, K — appended at indices 275..304.
    by_name = {c["name"]: c for c in cells}
    for cell_name in EXTENSION_CELLS:
        cell = by_name[cell_name]
        for seed in SEEDS:
            for lr in LRS_EXTENSION:
                module_path, _file_path = _emit(cell, seed, lr, out_dir)
                index_lines.append(module_path)
                n_emitted += 1

    index_path = out_dir / "index.txt"
    index_path.write_text("\n".join(index_lines) + "\n")

    expected = len(cells) * len(SEEDS) * len(LRS) + len(EXTENSION_CELLS) * len(SEEDS) * len(LRS_EXTENSION)
    print(f"emitted {n_emitted} configs to {out_dir}")
    print(f"index → {index_path} ({len(index_lines)} entries; expected {expected})")
    print(
        f"  main grid:   indices 0..{len(cells) * len(SEEDS) * len(LRS) - 1} "
        f"({len(cells)} cells × {len(SEEDS)} seeds × {len(LRS)} LRs)"
    )
    print(
        f"  LR extension: indices {len(cells) * len(SEEDS) * len(LRS)}..{expected - 1} "
        f"({len(EXTENSION_CELLS)} cells × {len(SEEDS)} seeds × {len(LRS_EXTENSION)} extra LRs)"
    )
    if n_emitted != expected:
        print(
            f"WARNING: emitted {n_emitted} != expected {expected}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
