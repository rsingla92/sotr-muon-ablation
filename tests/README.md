# tests/

See `CONTRIBUTING.md` §"Testing" for the philosophy. This file documents the layout.

## Tiers

| Directory | Purpose | When run |
|---|---|---|
| `sanity/` | PROTOCOL §7 limit-case gates. Must pass before any Phase 2 result is reported. | `make sanity`. Required before paper-bound runs. |
| `unit/` | Pure-function correctness for helpers in `optimizers/_*.py`. | `make test`. Every commit (pre-commit). |
| `integration/` (added later) | One-step end-to-end training to verify pieces compose. | Pre-Phase-1 manual. Not on every commit. |
| `fixtures/` | Saved tensors, model stubs, reference trajectories. | Loaded by tests; not run themselves. |

## Sanity tests (mapped to PROTOCOL §7)

| Test file | PROTOCOL §7 check |
|---|---|
| `test_sotr_limits.py` | #1 SOTR(α=1, Δ=∞, q=5) ≡ Muon; #2 SOTR(α=0, q=0) ≡ normalized G; #3 SOTR(α=0, q=2) ≠ Muon |
| `test_lion_match.py` | #4 Our Lion impl matches Chen 2023 reference |
| `test_muon_match.py` | #5 Our integration of `external/Muon` agrees with running it directly |
| `test_trust_region.py` | #6 Trust region triggers correctly at small Δ |
| `test_determinism.py` | #7 Same seed → same loss curve (within tolerance) |
| `test_param_groups.py` | #8 Muon/SOTR apply only to `transformer.h.*` 2D weights |

The mapping is verified in `test_sanity_coverage.py` — that test fails if any §7 item lacks a corresponding test.

## Conventions

- Each test file mirrors a source file path. `tests/unit/test_newton_schulz.py` tests `optimizers/_newton_schulz.py`.
- Test functions: `test_<behavior>` — what's verified, not "test_1".
- Floating-point comparisons use `torch.allclose` with explicit `atol`/`rtol`. Never `==`.
- Determinism: seeds are auto-set per test (see `conftest.py`).
- GPU tests: marked `@pytest.mark.gpu`. Auto-skipped on CPU-only systems.

## Running

```bash
make sanity              # just sanity tier (gating)
make test                # full suite
pytest tests/unit -k newton_schulz   # targeted
pytest tests/ -m "not slow"          # skip slow tests
pytest tests/ -m "gpu" -v            # only GPU tests
pytest tests/ -x                     # stop on first failure
```

## What we don't test

- **Loss values across hardware.** Flaky; not informative.
- **`torch` itself.** Trust upstream.
- **External submodules.** Their job, not ours.
- **Anything that requires downloading data or model checkpoints.** Tests must run offline, fast, and on minimal hardware.

## Adding a new test

1. Decide tier: limit case → `sanity/`; pure function → `unit/`; multi-component → `integration/`.
2. Create `tests/<tier>/test_<thing>.py`. Mirror the source file name.
3. One assertion or scenario per `test_*` function. If a test would fail for >1 reason, split it.
4. If you need a new fixture, add it to `tests/conftest.py`. Don't duplicate fixtures across files.
5. Run `make lint && make test` before committing.
