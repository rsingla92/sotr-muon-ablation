# Phase 0 plan and verification gate

## Goal

A working end-to-end pipeline at toy scale: KataGo produces games and
analysis, a small transformer trains on them and plays legal Go moves, we
can evaluate strength against KataGo at varied visit counts.

## Verification gate (must all be checked before Phase 1)

- [ ] **Smoke test runs end-to-end in under 1 hour on a single GPU.**
  Generate 100 games -> train a 4-layer model for 1000 steps -> play 10
  games vs KataGo at 1 visit. The model produces legal moves >99% of the
  time.
  - Driver: `scripts/smoke_test.sh`
  - Config: `configs/smoke.yaml`
- [ ] **KataGo wrapper unit tests pass.** 100 calls to `analyze()` return
  well-formed results with no subprocess leaks (verify `pgrep katago`
  returns nothing after teardown).
  - Test file: `tests/test_katago_smoke.py` (TODO; cluster-only)
- [ ] **Tokenizer round-trip test passes** on 100 randomly sampled
  positions and a full game.
  - Test file: `tests/test_tokenizer.py` (passing locally on CPU as of
    initial scaffold commit)
- [ ] **Bidirectional/causal attention mask verified.** In a 4-layer
  model, gradients at board-prefix positions depend on all other prefix
  positions but not on any trajectory positions; gradients at trajectory
  position t depend on positions <= t.
  - Test file: `tests/test_attention_mask.py` (requires torch)
- [ ] **30M baseline trains successfully** on 50k-100k games for >=24h on
  4x H100. Val loss decreases monotonically over the last 25% of
  training. Final val cross-entropy is reported.
  - Config: `configs/baseline_30m.yaml`
  - SLURM script: `scripts/slurm/train_baseline.slurm`
- [ ] **Baseline strength benchmark.** 30M model wins >=40% vs KataGo at
  1 visit, >=10% vs KataGo at 10 visits.
- [ ] **Reproducibility check.** Re-running training from the same seed
  and config produces val loss within 0.5% of the original.
- [ ] **Tag and document.** `git tag phase-0-complete`, README section
  with strength curve.

## Open questions / parking lot

- KataGo's "ladder" detection in `moveInfos` doesn't directly expose ladder
  status; Phase 1 may need a custom rule-based ladder predicate.
- The current LAST_MOVE overlay collapses with stone color; if we observe
  systematic prediction errors near the last move, split LAST_MOVE into a
  separate channel rather than overriding the stone category.
- Suicide rule: we conservatively reject; KataGo defaults to Tromp-Taylor
  which allows suicide. If KataGo emits a suicide move in self-play, our
  data pipeline will silently skip it. Audit the first 1000 games for
  this case.
