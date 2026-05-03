# Architecture

How the repo is organized and why. Read alongside `CONTRIBUTING.md` (rules) and `PROTOCOL.md` (what we're proving).

## Directory layout

```
optimizer_experiments/
├── PROTOCOL.md             Pre-registered methodology (the contract)
├── README.md               Project entry point
├── CONTRIBUTING.md         Coding/testing standards
├── CLAUDE.md               Skill routing for Claude Code (don't edit casually)
├── pyproject.toml          Deps, ruff, pytest config
├── Makefile                Common entry points (setup, sanity, test, lint)
├── .pre-commit-config.yaml Auto-formatting/linting hooks
├── conftest.py             Pytest root config
│
├── docs/
│   ├── ARCHITECTURE.md     This file
│   ├── CLUSTER.md          UBC cluster (SLURM) specifics
│   └── EXPERIMENTS.md      How to define and run an experiment
│
├── knowledge/              Literature summaries from source PDFs
│   ├── 00_index.md
│   ├── 01_muon_landscape.md
│   ├── 02_muon_scalability.md
│   ├── 03_sotr_design.md
│   ├── 04_proposals_existing.md
│   ├── 05_open_directions.md
│   └── 06_lit_update_2026_05.md
│
├── external/               Pinned reference repos as git submodules (read-only)
│   ├── README.md           Submodule policy
│   ├── Muon/               KellerJordan/Muon — reference Muon impl
│   ├── modded-nanogpt/     KellerJordan/modded-nanogpt — speedrun harness
│   ├── lion-pytorch/       lucidrains/lion-pytorch — Lion reference
│   └── dion/               microsoft/dion — Dion reference
│
├── optimizers/             Our optimizer implementations
│   ├── __init__.py         Public API (SOTR, Lion, MuonLike, ...)
│   ├── _newton_schulz.py   Shared NS polynomial routine
│   ├── _utils.py           Frobenius norm, trust-region helpers
│   ├── sotr.py             SOTR optimizer
│   ├── muon_like.py        Sanity baseline (= SOTR with α=1)
│   └── lion.py             Lion (vendored from external/lion-pytorch)
│
├── experiments/            Run scripts + configs
│   ├── _configs.py         Typed config dataclasses
│   ├── _run_id.py          Run-ID generator
│   ├── train.py            Training entry point
│   ├── configs/            YAML configs (one per run)
│   │   ├── sanity_shakespeare.yaml
│   │   ├── phase1_repro_muon.yaml
│   │   └── ...
│   └── scripts/            Shell wrappers (rare; prefer Python)
│
├── scripts/                Repo-level utilities
│   ├── slurm/              SLURM job templates (UBC-specific)
│   │   ├── single_gpu.sh
│   │   └── multi_gpu.sh
│   └── setup.sh            One-shot environment setup
│
├── tests/
│   ├── conftest.py         Shared fixtures
│   ├── sanity/             PROTOCOL §7 limit-case checks (gating)
│   │   ├── test_sotr_limits.py
│   │   ├── test_muon_match.py
│   │   ├── test_lion_match.py
│   │   ├── test_trust_region.py
│   │   ├── test_determinism.py
│   │   └── test_param_groups.py
│   ├── unit/               Pure-function tests
│   │   ├── test_newton_schulz.py
│   │   └── test_utils.py
│   └── fixtures/           Test data (saved tensors, etc.)
│
├── results/                Run outputs (gitignored except .gitkeep)
├── checkpoints/            Model checkpoints (gitignored)
└── data/                   Tokenized corpora (gitignored)
```

## Key abstractions

### `optimizers/`

Each optimizer is a single file. Public API exported from `__init__.py`:

```python
from optimizers import SOTR, Lion, MuonLike
```

Shared helpers (`_newton_schulz`, `_utils`) are underscore-prefixed → not part of public API. They live in this directory only because they're optimizer-specific.

The Newton-Schulz polynomial iteration is implemented **once**, in `_newton_schulz.py`. Both SOTR and MuonLike call it with different iteration counts. We do not maintain separate NS implementations.

### `experiments/`

Training pipeline split into:

- `_configs.py`: typed config dataclasses (one per phase/scenario, all inheriting from a common `BaseConfig`)
- `_run_id.py`: one-line generator for unique, sortable run IDs
- `train.py`: the actual training loop. Takes `--config path/to/config.yaml`, validates against the dataclass, runs.
- `configs/`: YAML files. One file per actual run. Filename matches its purpose.

The train loop is built on top of `external/modded-nanogpt/train_gpt.py` — we adapt rather than rewrite, since matching that harness is the whole point of comparability (see PROTOCOL §5).

### `tests/`

Three tiers, separated by directory:

| Directory | What | When run |
|---|---|---|
| `tests/sanity/` | PROTOCOL §7 limit-case gates | Before any Phase 2 result is reported. `make sanity`. |
| `tests/unit/` | Pure-function correctness (helpers, math routines) | Every commit via pre-commit `pytest` (fast subset). |
| `tests/integration/` (later) | One-step training to verify pieces compose | Pre-Phase-1 manual; not on every commit. |

`tests/conftest.py` provides fixtures: small synthetic tensors, a `tiny_transformer` stub model, deterministic seed setup.

### `external/`

Pinned git submodules. **We never modify them.** When we need to use code from them:

1. Copy the file into `optimizers/` or `experiments/` with a vendoring header (`CONTRIBUTING.md` §"Comments")
2. Run a sanity test (`tests/sanity/test_<thing>_match.py`) verifying step-by-step equivalence with the upstream

This keeps our git history clean while preserving exact reproducibility.

## Rationale for choices

### Why a flat `optimizers/` instead of `optimizers/sotr/`, `optimizers/lion/` package-per-optimizer?

We have ~5 optimizers planned. Each is < 300 lines. Per-optimizer packages would add directory bloat for no benefit. If we ever ship 20 optimizers, we revisit.

### Why YAML configs and not Hydra / OmegaConf / argparse?

- YAML is human-readable and diffable.
- Validation via `@dataclass.__post_init__` gives us type safety without a framework.
- Hydra adds composition complexity we don't need — we have a few dozen configs at most.

### Why no notebooks?

- Notebooks hide order-of-execution bugs.
- Diffs are unreviewable.
- Notebooks tempt "let me just paste this here" copy-paste.
- If interactive exploration is needed: use `ipython` against the codebase or `marimo`/`jupytext` outside the repo.

### Why two separate "sanity" and "unit" test tiers?

Sanity tests are *gating* (PROTOCOL §7 — must pass before reporting anything). Unit tests are *fast checks during development*. Conflating them makes the sanity gate slow (so it gets skipped) or makes development feedback laggy (so unit tests get neglected).

### Why submodules for external repos and not pip dependencies?

- Pinned commits guarantee bit-reproducibility.
- We can read the source easily (`grep -r` in `external/`).
- We don't depend on PyPI for research-prototype code that might never be packaged.
- Updating is explicit (see `external/README.md`).

## What lives where (decision flowchart)

| Type of code | Location |
|---|---|
| Optimizer step logic | `optimizers/<name>.py` |
| Math helper used by multiple optimizers | `optimizers/_<helper>.py` |
| Config dataclass | `experiments/_configs.py` |
| YAML config | `experiments/configs/<purpose>.yaml` |
| Training loop entry point | `experiments/train.py` |
| Eval entry point | `experiments/eval.py` (when needed) |
| SLURM submission script | `scripts/slurm/<purpose>.sh` |
| Repo setup, environment | `scripts/setup.sh` |
| Sanity gate test | `tests/sanity/test_<thing>.py` |
| Pure-function unit test | `tests/unit/test_<thing>.py` |
| Adapted from external repo | Vendor with header; sanity-test against original |
| Reusable across experiments | If used 3+ times → factor; else inline |

## What this repo does *not* contain

- Pre-trained model weights (use `checkpoints/` locally; gitignored)
- Training data (use `data/`; gitignored; FineWeb downloaded via modded-nanogpt's script)
- Notebooks
- A web UI
- A custom config-merge framework
- Code for optimizers we're not going to use (don't keep "for later")
