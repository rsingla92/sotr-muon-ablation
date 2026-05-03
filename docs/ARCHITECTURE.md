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
├── optimizers/             Our optimizer implementations (only what's novel)
│   ├── __init__.py         Public API: re-exports SOTR + Lion (from lion_pytorch)
│   └── sotr.py             SOTR optimizer — the ONLY novel file we write
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

We write only what's novel: **`sotr.py`**. Everything else is imported from canonical references.

```python
from optimizers import SOTR, Lion          # SOTR is ours; Lion is re-exported from lion_pytorch
from muon import Muon, MuonWithAuxAdam     # Muon imported directly from external/Muon
```

**Why we don't write our own NS / Lion / MuonLike.**

- **Newton-Schulz iteration** is in `external/Muon/muon.py` as `zeropower_via_newtonschulz5(G, steps)`. It's the precisely-tuned, multi-author–optimized routine that all the speedrun records use. SOTR imports it directly. Reimplementing risks subtle bugs in the polynomial coefficients and out-tunes nothing.
- **Lion** is in `external/lion-pytorch` as `lion_pytorch.Lion`. Faithful Chen 2023 reference (lucidrains, MIT-licensed, pip-installable from the submodule). Reimplementing risks the well-known `betas` / sign-update / decoupled-WD bugs.
- **MuonLike** doesn't exist as a separate optimizer. It's the configuration `SOTR(α=1, Δ=∞, q=5)`, which by construction produces the same updates as Muon. Sanity test #1 (PROTOCOL §7) verifies this byte-equivalence on a synthetic problem.

This keeps the surface area for bugs to a single ~100-line `sotr.py`.

### `experiments/`

Training pipeline split into:

- `_configs.py`: typed config dataclasses (one per phase/scenario, all inheriting from a common `BaseConfig`)
- `_run_id.py`: one-line generator for unique, sortable run IDs
- `train.py`: the actual training loop. Takes `--config path/to/config.yaml`, validates against the dataclass, runs.
- `configs/`: YAML files. One file per actual run. Filename matches its purpose.

The train loop is built on top of `external/modded-nanogpt/train_gpt2.py` — we adapt rather than rewrite, since matching that harness is the whole point of comparability (see PROTOCOL §5).

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

### Why is `optimizers/` so empty?

Because we write exactly one optimizer (SOTR). Every baseline — Muon, Lion, AdamW — is imported from a canonical reference (`external/Muon`, `external/lion-pytorch`, `torch.optim.AdamW`). The "MuonLike" sanity baseline is a SOTR configuration, not a separate file. This is deliberate: each line of code we write is a line we have to maintain and test, and bugs in baselines invalidate the comparison.

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
| Newton-Schulz iteration | **Don't write — import from `external/Muon`** (`from muon import zeropower_via_newtonschulz5`) |
| Lion baseline | **Don't write — import from `external/lion-pytorch`** (`from lion_pytorch import Lion`) |
| Muon baseline | **Don't write — import from `external/Muon`** (`from muon import Muon, MuonWithAuxAdam`) |
| MuonLike sanity baseline | **Don't write — it's the configuration `SOTR(α=1, Δ=∞, q=5)`**; equivalence proven by sanity test #1 |
| Frobenius norm | `tensor.norm()` or `torch.linalg.norm()` — built-in |
| Momentum buffer / weight decay machinery | `torch.optim.Optimizer` base class — built-in |
| **SOTR step logic** | `optimizers/sotr.py` (the one novel file) |
| Config dataclass | `experiments/_configs.py` |
| YAML config | `experiments/configs/<purpose>.yaml` |
| Training loop entry point | `experiments/train.py` |
| Eval entry point | `experiments/eval.py` (when needed) |
| SLURM submission script | `scripts/slurm/<purpose>.sh` |
| Repo setup, environment | `scripts/setup.sh` |
| Sanity gate test | `tests/sanity/test_<thing>.py` |
| Pure-function unit test | `tests/unit/test_<thing>.py` |
| Reusable across experiments | If used 3+ times → factor; else inline |
| Helper used in 2+ files | Inline copy until the third use case actually arrives |

## What this repo does *not* contain

- Pre-trained model weights (use `checkpoints/` locally; gitignored)
- Training data (use `data/`; gitignored; FineWeb downloaded via modded-nanogpt's script)
- Notebooks
- A web UI
- A custom config-merge framework
- Code for optimizers we're not going to use (don't keep "for later")
