# Architecture

How the repo is organized and why. Read alongside [`CONTRIBUTING.md`](../CONTRIBUTING.md) (rules) and [`PROTOCOL.md`](../PROTOCOL.md) (what we're proving).

## Directory layout

```
sotr-muon-ablation/
├── PROTOCOL.md             Pre-registered methodology (the contract)
├── README.md               Project entry point
├── CONTRIBUTING.md         Coding/testing standards
├── LICENSE                 MIT
├── pyproject.toml          Deps, ruff, pytest config
├── Makefile                Common entry points (setup, sanity, test, lint)
├── .pre-commit-config.yaml Auto-formatting/linting hooks
├── conftest.py             Pytest root config
│
├── docs/
│   ├── ARCHITECTURE.md     This file
│   ├── CLUSTER.md          DRAC (Compute Canada) cluster specifics
│   ├── EXPERIMENTS.md      How to define and run an experiment
│   ├── PHASE1.md           Phase 1 reproduction procedure
│   └── PHASE2.md           Phase 2 ablation procedure
│
├── knowledge/              Literature synthesis (00–09)
│   ├── 00_index.md
│   ├── 01_muon_landscape.md
│   ├── 02_muon_scalability.md
│   ├── 03_sotr_design.md
│   ├── 04_proposals_existing.md
│   ├── 05_open_directions.md
│   ├── 06_lit_update_2026_05.md
│   ├── 07_spectral_interpretation.md    Reframing after realizing α-blend isn't cleanly novel
│   ├── 08_muonbp_block_periodic.md      Subagent synthesis of arxiv 2510.16981
│   └── 09_demo_decoupled_momentum.md    Subagent synthesis of arxiv 2411.19870
│
├── external/               Pinned reference repos as git submodules (read-only)
│   ├── README.md           Submodule policy + pinned commits
│   ├── Muon/               KellerJordan/Muon — reference Muon impl
│   ├── modded-nanogpt/     KellerJordan/modded-nanogpt — pinned to dd2224b
│   ├── lion-pytorch/       lucidrains/lion-pytorch — Lion reference
│   └── dion/               microsoft/dion — Dion reference
│
├── optimizers/             Our optimizer implementations
│   ├── __init__.py         Public API
│   └── sotr.py             SOTR — the ONE novel file (225 lines)
│
├── experiments/            Run scripts + configs
│   ├── _configs.py         Typed RunConfig dataclass + OptimizerKind enum
│   ├── _logging.py         JSONL logger + PROTOCOL §8 stability incident detection
│   ├── _run_id.py          Deterministic run-ID generator
│   ├── train.py            Vendored modded-nanogpt + optimizer dispatcher
│   ├── configs/            Python config modules (@dataclass instances, no YAML)
│   │   ├── _phase2_base.py           Shared Phase 2 base config
│   │   ├── phase1_repro_muon.py      Phase 1 reproduction config
│   │   └── phase2/                   Cell-level dirs; per-run configs are gitignored (regen from gen_phase2_configs.py)
│   ├── scripts/
│   │   └── gen_phase2_configs.py     Code-as-source-of-truth for the 305-config grid
│   └── analysis/
│       └── phase2_summary.py         Paired bootstrap + Holm–Bonferroni + decision tree
│
├── scripts/                Repo-level utilities
│   ├── setup.sh            Local-dev environment setup
│   ├── setup_drac.sh       DRAC login-node setup (idempotent)
│   ├── ablation_status.sh  One-shot progress report for a Phase 2 array
│   └── slurm/
│       ├── single_gpu.sh              Generic 1× GPU template
│       ├── multi_gpu.sh               4× H100 single-node (Phase 3)
│       ├── array_ablation.sh          Phase 2 SLURM array
│       └── phase1_modded_nanogpt.sh   Phase 1 reproduction
│
├── tests/
│   ├── conftest.py         Shared fixtures
│   ├── sanity/             PROTOCOL §7 limit-case gates (make sanity)
│   │   ├── test_sotr_limits.py         Kill switch Hkill2: SOTR ≡ Muon at (α=1, Δ=∞, q=5)
│   │   ├── test_muon_match.py          Frozen-trajectory match against external/Muon
│   │   ├── test_lion_match.py          Frozen-trajectory match against lion_pytorch
│   │   ├── test_trust_region.py        Δ actually caps the update Frobenius norm
│   │   ├── test_spectral_identity.py   Numerical verification of σ_i ↦ α + (1−α)·σ_i/‖M‖_F
│   │   ├── test_determinism.py         Same seed → same loss (CPU bit-identical)
│   │   ├── test_param_groups.py        SOTR only on 2D transformer.h.*
│   │   └── test_sanity_coverage.py     Meta-test: PROTOCOL §7 ↔ tests/sanity/ don't drift
│   ├── unit/
│   │   ├── test_phase2_analysis.py     17 tests for the analysis pipeline
│   │   └── test_logging.py             JSONL logger + incident detection
│   └── fixtures/                       Frozen tensors for the *_match tests
│
├── results/                Run outputs (gitignored except .gitkeep)
├── checkpoints/            Model checkpoints (gitignored)
└── data/                   Tokenized corpora (gitignored)
```

## Key abstractions

### `optimizers/`

We write exactly one optimizer: **`sotr.py`** (225 lines). Everything else is imported from a canonical reference:

```python
from optimizers import SOTR                        # ours
from muon import Muon, zeropower_via_newtonschulz5 # from external/Muon
from lion_pytorch import Lion                      # from external/lion-pytorch
from torch.optim import AdamW                      # stdlib
```

**Why we don't write our own Newton-Schulz / Lion / Muon.**

- **Newton-Schulz iteration** is in `external/Muon/muon.py` as `zeropower_via_newtonschulz5(G, steps)`. It's the multi-author-tuned routine that all the speedrun records use. SOTR imports it directly. Reimplementing risks subtle bugs in the polynomial coefficients and out-tunes nothing.
- **Lion** is in `external/lion-pytorch` (lucidrains, MIT-licensed). Reimplementing risks the well-known `betas` / sign-update / decoupled-WD bugs.
- **Muon** doesn't need a separate file. It's the SOTR configuration `(α=1, Δ=∞, q=5)`, which by construction produces identical updates. Sanity test `test_sotr_limits.py` verifies this byte-equivalence on a synthetic problem and is the enforcement of kill switch `Hkill2` in `PROTOCOL.md`.

This keeps the surface area for bugs to a single ~225-line `sotr.py`.

### `experiments/`

Training pipeline:

- **`_configs.py`** — `RunConfig` dataclass + `OptimizerKind` enum. One dataclass field per hyperparameter; validation happens in `__post_init__`.
- **`_run_id.py`** — deterministic run-ID generator from `(timestamp, config hash, salt)`.
- **`_logging.py`** — JSONL logger with stable schema + PROTOCOL §8 stability-incident detection.
- **`train.py`** — vendored `modded-nanogpt` train loop (pinned to `dd2224b`) with a minimal optimizer-dispatch patch. Loads a config module via `--config <module.path>`.
- **`configs/`** — Python modules, not YAML. Each is a small file exposing `config = RunConfig(...)`.
- **`configs/phase2/`** — 305 generated config modules for the ablation grid, all gitignored. The generator ([`scripts/gen_phase2_configs.py`](../experiments/scripts/gen_phase2_configs.py)) is single-source-of-truth.
- **`analysis/phase2_summary.py`** — offline aggregation: discover runs → load final val_loss → best-LR-per-cell → paired bootstrap → Holm–Bonferroni → decision-tree narrative.

The train loop is built on top of `external/modded-nanogpt/train_gpt2.py` — we adapt rather than rewrite, since matching that harness is the whole point of comparability (PROTOCOL §5).

### `tests/`

Three tiers, separated by directory:

| Directory | What | When run |
|---|---|---|
| `tests/sanity/` | PROTOCOL §7 limit-case gates | `make sanity`. Required before any Phase 2 result is reported. |
| `tests/unit/` | Pure-function correctness (analysis pipeline, logger) | `make test`. Every commit via pre-commit. |
| `tests/fixtures/` | Frozen reference tensors for `*_match` sanity tests | Loaded by tests, not run. |

`tests/conftest.py` provides fixtures: synthetic tensors, a `tiny_transformer` stub, deterministic seed setup, CI-aware GPU-test skips.

### `external/`

Pinned git submodules. **We never modify them.** When we need to use code from them:

1. Copy the file into `optimizers/` or `experiments/` with a vendoring header (`CONTRIBUTING.md` §"Comments").
2. Run a sanity test (`tests/sanity/test_<thing>_match.py`) verifying step-by-step equivalence with the upstream.

This is exactly what `experiments/train.py` does: vendored from `external/modded-nanogpt/train_gpt2.py` at commit `dd2224b`, with a header citing the source and a ~30-line optimizer-dispatch patch.

## Rationale for choices

### Why is `optimizers/` so small?

Because we write exactly one optimizer (SOTR). Every baseline — Muon, Lion, AdamW — is imported from a canonical reference. The "MuonLike" sanity baseline is a SOTR configuration, not a separate file. This is deliberate: each line of code we write is a line we have to maintain, test, and defend, and bugs in baselines invalidate every comparison.

### Why Python config modules and not YAML?

- **Type checking.** A `RunConfig` dataclass gives us mypy/ruff coverage. YAML gives us runtime errors.
- **Composition.** `replace(base_config, muon_learning_rate=lr)` beats YAML anchors + overrides for the ablation grid.
- **Generator symmetry.** Grid expansion in `gen_phase2_configs.py` writes Python that imports the same base config, so the generated files parse and type-check with no separate schema.
- **No framework.** Hydra / OmegaConf add composition complexity we don't need — we have <10 base configs.

### Why no notebooks?

- Notebooks hide order-of-execution bugs.
- Diffs are unreviewable.
- Notebooks tempt "let me just paste this here" copy-paste.
- Interactive exploration: use `ipython` against the codebase or a scratch dir outside the repo.

### Why two separate `sanity` and `unit` test tiers?

Sanity tests are *gating* (PROTOCOL §7 — must pass before reporting anything). Unit tests are *fast feedback during development*. Conflating them makes the sanity gate slow (so it gets skipped) or makes development feedback laggy (so unit tests get neglected).

### Why submodules for external repos and not pip dependencies?

- Pinned commits guarantee bit-reproducibility.
- We can read the source easily (`grep -r` in `external/`).
- We don't depend on PyPI for research-prototype code that might never be packaged.
- Updating is explicit (see `external/README.md`).

## What lives where (decision flowchart)

| Type of code | Location |
|---|---|
| Newton-Schulz iteration | **Don't write — import** `from muon import zeropower_via_newtonschulz5` |
| Lion baseline | **Don't write — import** `from lion_pytorch import Lion` |
| Muon baseline | **Don't write — import** `from muon import Muon, MuonWithAuxAdam` |
| MuonLike sanity baseline | **Don't write — it's `SOTR(α=1, Δ=∞, q=5)`**; equivalence proven by `test_sotr_limits.py` |
| Frobenius norm | `tensor.norm()` or `torch.linalg.norm()` — built-in |
| Momentum buffer / weight decay machinery | `torch.optim.Optimizer` base class — built-in |
| **SOTR step logic** | `optimizers/sotr.py` |
| Config dataclass | `experiments/_configs.py` |
| Per-run Python config module | `experiments/configs/<purpose>.py` |
| Ablation-grid generator | `experiments/scripts/gen_phase2_configs.py` |
| Training loop entry point | `experiments/train.py` |
| Analysis pipeline | `experiments/analysis/phase2_summary.py` |
| SLURM submission script | `scripts/slurm/<purpose>.sh` |
| Repo setup, environment | `scripts/setup.sh` (local), `scripts/setup_drac.sh` (DRAC) |
| Sanity gate test | `tests/sanity/test_<thing>.py` |
| Pure-function unit test | `tests/unit/test_<thing>.py` |
| Reusable across experiments | If used 3+ times → factor; else inline |

## What this repo does *not* contain

- Pre-trained model weights (use `checkpoints/` locally; gitignored)
- Training data (use `data/`; gitignored; FineWeb downloaded via modded-nanogpt's script)
- Notebooks
- A web UI
- A custom config-merge framework
- Code for optimizers we're not going to use (don't keep "for later")
