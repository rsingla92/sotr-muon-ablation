# Contributing — coding and testing standards

This is the **single source of truth** for how code in this repo is written. If a rule below conflicts with something elsewhere, this document wins. PROTOCOL.md governs *what* gets done; this governs *how*.

## Design philosophy

This is research code, not a production library. But research code that's hard to read, full of dead options, or untested produces results no one trusts. The bar:

- **Boring is good.** Prefer clear over clever. A reader new to the file understands it within 5 minutes.
- **No premature abstraction.** Three similar lines beats a wrong abstraction. Don't build a config framework before you have three configs.
- **Reproducibility is non-negotiable.** Every run reconstructible from a single (commit hash + config file + seed).
- **Delete aggressively.** Dead code, commented-out blocks, unused parameters, defensive `try/except` for things that can't happen — all deleted, not retained "just in case."

## Repository layout

See `docs/ARCHITECTURE.md` for the full structure. The short version:

```
optimizers/      SOTR (our one novel optimizer); Lion re-exported from lion_pytorch
experiments/     Run scripts, Python config modules, analysis pipeline
scripts/         Repo-level utilities (setup, SLURM templates, ablation status)
tests/           Sanity (PROTOCOL §7 gates) + unit tests
external/        Pinned reference repos (read-only submodules)
results/         Run outputs (gitignored except .gitkeep)
knowledge/       Literature synthesis
docs/            Architecture, cluster, per-phase procedures
```

## Code style

### Formatting and linting

Enforced by `ruff` (config in `pyproject.toml`). Run locally before pushing:

```bash
make lint        # check
ruff format .    # auto-format
```

**CI runs the same checks on every push and PR** to `main` (`.github/workflows/lint.yml`). A push that fails ruff lint or format check will be visible in the GitHub Actions tab. Local fixing is faster than waiting for CI feedback, but CI is the source of truth.

**Pre-commit hooks are optional and not auto-installed.** The hook configuration is in `.pre-commit-config.yaml` if you want immediate-at-commit feedback locally; install with:

```bash
pip install pre-commit && pre-commit install
```

On DRAC, this is **discouraged** — pre-commit's cache lives in `$HOME/.cache/pre-commit/` by default, which is on a slow network filesystem; first commits take minutes. If you want pre-commit on DRAC anyway, redirect the cache to scratch:

```bash
export PRE_COMMIT_HOME="$SCRATCH/pre-commit-cache"
pre-commit install
```

### Python

- **Python 3.10+ syntax.** Use `X | None`, not `Optional[X]`. Use built-in generics: `list[int]`, not `List[int]`.
- **Type hints on every public function and class.** Private helpers (underscore-prefixed) may omit them when context is obvious.
- **f-strings**, not `%` or `.format()`.
- **Imports**: stdlib → third-party → first-party, separated by blank lines. Sorted within each group (ruff handles this).
- **One concept per file.** A file named `sotr.py` contains the SOTR optimizer and nothing else. Helpers go in `optimizers/_utils.py` or similar.

### Naming

- `snake_case` for functions, variables, modules
- `PascalCase` for classes
- `UPPER_SNAKE` for module-level constants
- Names describe *what*, not *how*. `compute_partial_polar` not `do_step_1`.
- Avoid abbreviations except established ones (`lr`, `wd`, `bs`, `cfg`, `idx`).

### Comments

Per the project default (`CLAUDE.md`): **default to writing no comments.** Add one only when the *why* is non-obvious. Don't restate what the code does. Don't reference current task or PR.

Exception: every file vendored or adapted from an external repo gets a header comment:

```python
# Adapted from KellerJordan/Muon @ commit bd1758a
# Source: external/Muon/muon.py (vendored 2026-05-02)
# Changes: added trust_region argument; renamed _orth → newton_schulz
```

This is a hard requirement, not optional.

### Functions and classes

- Functions: do one thing. If you can't summarize in one sentence, split.
- Classes: prefer `@dataclass(frozen=True)` for configs and result containers. Only inherit from `torch.optim.Optimizer` for actual optimizers.
- No mutable global state. Pass state explicitly.
- No `**kwargs` passthrough unless it's a thin wrapper around a documented interface. Be explicit.

### Tensors and shapes

- **Document tensor shapes** in docstrings using a consistent notation: `(B, T, D)` for batch/time/dim, `(M, N)` for matrices.
- Use `einops.rearrange` over manual `.view()` / `.permute()` chains when the shape transform is non-trivial.
- `torch.no_grad()` on optimizer step bodies. Always.
- `dtype` and `device` always explicit at tensor creation when not derived from input.

### Configs

All experiment hyperparameters live in Python config modules under `experiments/configs/`. Each is a small file that constructs a single `RunConfig` dataclass instance and assigns it to the module-level `config` name. No hardcoded numbers in training scripts. The training script imports the config module by dotted path (`--config experiments.configs.<name>`), validates on load via `__post_init__`, and passes typed values down. Example dataclass:

```python
@dataclass(frozen=True)
class TrainConfig:
    optimizer: str
    lr: float
    seed: int
    n_layer: int
    # ...
```

Config dataclasses live in `experiments/_configs.py`. Validation (LR > 0, etc.) goes in `__post_init__`.

### Logging

- `logging.getLogger(__name__)`, not `print` — except in scripts where stdout is the user-facing output.
- Each run produces a `results/<phase>/<run_id>/` directory containing: `train.log`, `train.jsonl` (per-step metrics), `eval.jsonl` (validation losses), `final_metrics.json`, `stability_incidents.jsonl` (PROTOCOL §8), and `env.txt` (`pip freeze` + `nvidia-smi` + git SHA).
- Run IDs are timestamps + 6-char hash: `2026-05-02_143022_a3f2c1`. Implemented once in `experiments/_run_id.py`, used everywhere.

### Error handling

- Validate at boundaries (config load, file I/O, user input). After validation, trust the data — no defensive checks scattered through internals.
- Raise specific exceptions: `ValueError` for bad config, `RuntimeError` for unexpected state, `NotImplementedError` for stubs.
- Never `except Exception: pass`. Never bare `except:`.

## Testing

See `tests/README.md` for the layout. Philosophy:

### What we test

1. **Sanity tests** (`tests/sanity/`): PROTOCOL §7 limit-case checks. These are gating — must pass before any Phase 2 result is reported. They are not optional or "nice to have."
2. **Unit tests** (`tests/unit/`): pure functions in `optimizers/`. Newton-Schulz output shape, Frobenius norm helper, trust-region clip behavior, etc.
3. **Integration tests** (`tests/integration/`, when needed): one-step training on a tiny model to ensure pieces compose. Numerical regression NOT included — too flaky across hardware.

### What we don't test

- **No regression tests on numerical loss values across hardware.** Fundamentally flaky.
- **No tests of `torch` itself.** We trust upstream.
- **No tests of external/ submodules.** Their job, not ours.

### Test conventions

- File: `test_<thing>.py` mirrors the source file path.
- Function: `test_<behavior>` — describe what is verified, not "test_1".
- Each test should fail for one specific reason. If the test message would be "something is broken," the test does too much.
- Tolerance for floating-point: explicit `atol`/`rtol` in `torch.allclose`. Never `==` on floats.
- Determinism: every test that uses RNG sets the seed. Tests using GPU also set deterministic algorithm flags where feasible.
- Fixtures (small tensors, model stubs) live in `tests/conftest.py` or `tests/fixtures/`.

### Sanity gate

`make sanity` runs `tests/sanity/`. The PROTOCOL §7 list maps 1:1 to test files there. PR description must include the output of `make sanity` if it touches `optimizers/`.

## Git workflow

- **Commit frequently with focused messages.** One concept per commit.
- Commit message format: `<area>: <imperative summary>` first line, blank, body explaining *why*. Examples: `optimizers: add SOTR with α-blend and Frobenius cap`, `protocol: lock hardware to UBC cluster`.
- **Never** force-push to `main`. Branches OK to rewrite.
- **Never** bypass hooks (`--no-verify`). If a hook fails, fix the underlying issue.
- Co-authoring: include the trailer `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` on commits where Claude wrote substantial code or docs.
- PROTOCOL.md changes get their own commit, prefixed `protocol:`, and require an Amendment section in §15.

## Dependencies

- New top-level dependency: discuss before adding. The bar is "we'd write this from scratch otherwise and it'd take >1 day."
- Pin major versions in `pyproject.toml`. Patch versions float.
- **No** dependency on a service requiring credentials at runtime (W&B optional only, never required).

## What this codebase will *not* have

- Plugin architectures. We write one optimizer at a time.
- Custom config DSLs (we use plain Python dataclasses in `experiments/configs/`).
- Web dashboards (we use `tensorboard --logdir results/` if visualization needed).
- A test that has to be skipped on CI ("flaky"). Either the test is right and the code is wrong, or the test is wrong.
- "TODO" comments in committed code. Use a tracked task instead.

## When in doubt

The reader of this code in six months will be a stranger (often, you). Optimize for them. If you're unsure whether to refactor, write the comment, or add the abstraction — *don't*. We can always add it later when the third use case arrives.
