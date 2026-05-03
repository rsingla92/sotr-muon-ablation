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
| `test_sotr_limits.py` | #1 SOTR(α=1, Δ=∞, q=5) ≡ Muon (byte-equivalent on synthetic); #2 SOTR(α=0, q=0) ≡ Frobenius-normalized momentum with per-shape RMS scale; #3 SOTR(α=1, q=2) ≠ SOTR(α=1, q=5) — partial-NS knob is wired |
| `test_lion_match.py` | #4 `lion_pytorch.Lion` (imported) reproduces a frozen 100-step trajectory in `tests/fixtures/lion_reference.pt` (catches upstream drift) |
| `test_muon_match.py` | #5 `muon.Muon` (imported from external/Muon) reproduces a frozen 100-step trajectory in `tests/fixtures/muon_reference.pt` |
| `test_trust_region.py` | #6 SOTR(α=1, Δ=0.01, q=5) hits the Frobenius cap on >50% of steps when typical update is O(1) |
| `test_determinism.py` | #7 Same seed → same loss curve (CPU bit-identical; GPU within 1e-4) |
| `test_param_groups.py` | #8 SOTR applied only to 2D `transformer.h.*` parameters; embed/head/biases/LayerNorm get AdamW |
| `test_spectral_identity.py` | #9 Numerical verification of the SVD-space identity `σ_i ↦ α + (1−α)·σ_i/||M||_F` (derivation in `knowledge/07_spectral_interpretation.md`); catches bugs that limit-case tests miss |

The mapping is enforced in `test_sanity_coverage.py` — that test fails if any §7 item lacks a corresponding test file. Prevents drift between PROTOCOL §7 and the test suite.

**Reference trajectories** (`tests/fixtures/*.pt`) are generated once by running `tests/fixtures/generate_references.py` against pinned external/ commits. They are committed so the repo is self-checking. Regenerate only when an external/ submodule is intentionally bumped (record in PROTOCOL.md §15 if so).

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
