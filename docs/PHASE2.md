# Phase 2 — Small-scale ablation

The PROTOCOL §3 evaluation phase. **This is where the pre-registered hypotheses (H1, H2) are actually tested.** We sweep an 11-cell × 5-seed × 5-LR grid at reduced scale (1,500 iters, batch 128) on the vendored modded-nanogpt harness (`experiments/train.py`), then compare cells with paired bootstrap and Holm–Bonferroni correction.

Phase 2 differs from Phase 1 in three ways:

1. **Our optimizer, not upstream.** We're running `experiments/train.py` (vendored + optimizer dispatcher), not `external/modded-nanogpt/train_gpt2.py`.
2. **Reduced scale.** 1,500 iters and batch 128 (vs 5,100 iters and batch 512 in Phase 1) so the full grid fits in ~50 GPU-days on the DRAC queue.
3. **Statistical testing, not gate-matching.** Per-cell median across seeds → best LR per cell → paired bootstrap between cell pairs. See [`experiments/analysis/phase2_summary.py`](../experiments/analysis/phase2_summary.py).

## The 11-cell ablation grid

Cell definitions from `PROTOCOL.md` §9. All eleven cells use `optimizer_name=SOTR`; the differences are in the `(α, Δ, q)` triple.

| Cell | α | Δ | q | Purpose |
|---|---|---|---|---|
| **A** `sotr_full` | 0.5 | 1.0 | 2 | Full SOTR (all three knobs on) |
| **B** `drop_alpha` | 1.0 | 1.0 | 2 | Drop α-blend — H2 sub-test 1 |
| **C** `drop_delta` | 0.5 | ∞ | 2 | Drop Δ cap — H2 sub-test 2 |
| **D** `drop_both` | 1.0 | ∞ | 2 | Drop α + Δ, keep partial NS |
| **E** `drop_ns` | 0.5 | 1.0 | 0 | Drop NS entirely — H2 sub-test 3 |
| **F** `full_ns` | 0.5 | 1.0 | 5 | Full NS with α-blend + Δ cap |
| G, H | 0.5 | 1.0 | 2 | α / Δ schedule cells — **deferred** (static in Phase 2; see §9 amendment 2026-05-03) |
| **I** `muon_plus_cap` | 1.0 | 1.0 | 5 | Muon + Frobenius cap (isolates Δ) |
| **J** `partial_ns_muon` | 1.0 | ∞ | 2 | Muon-like with only partial NS |
| **K** `muon_canonical` | 1.0 | ∞ | 5 | Canonical Muon (SOTR's full-Muon limit) — **added 2026-05-26** |

The LR sweep is `{0.005, 0.01, 0.02, 0.04, 0.08}` — log-spaced around Muon's upstream default of 0.02. A 2026-05-26 amendment added `{0.12, 0.16}` for cells F, I, K after both F and I peaked at the upper edge of the original sweep.

Seeds: `{0, 1, 2, 3, 4}`.

Total run counts:

- Main grid: 11 cells × 5 seeds × 5 LRs = **275 runs**
- LR extension: 3 cells (F, I, K) × 5 seeds × 2 extra LRs = **30 runs**
- **Grand total: 305 runs** (indices 0–304 in `experiments/configs/phase2/index.txt`)

## Procedure

### Generate configs (idempotent)

The 305 config modules are gitignored (regenerable from the generator, which is the source of truth):

```bash
python -m experiments.scripts.gen_phase2_configs
# emitted 305 configs to experiments/configs/phase2
# index → experiments/configs/phase2/index.txt (305 entries; expected 305)
```

The generator is single-source-of-truth for the ablation grid. Editing `_cells()`, `SEEDS`, `LRS`, or `LRS_EXTENSION` in [`experiments/scripts/gen_phase2_configs.py`](../experiments/scripts/gen_phase2_configs.py) → rerun → the config tree is rewritten. This is deliberate; no hand-edited per-run configs.

### Submit the array (Fir)

```bash
sbatch scripts/slurm/array_ablation.sh
```

The array job reads `experiments/configs/phase2/index.txt` and launches one task per line. It's parameterized by `SLURM_ARRAY_TASK_ID` and defaults to 24-way concurrency to stay a good citizen on `gpubase_bygpu`.

Each task launches via `torchrun --standalone --nproc_per_node=1` because `experiments/train.py` calls `dist.init_process_group` at import time (inherited from upstream).

### Monitor

```bash
./scripts/ablation_status.sh              # auto-detects most recent 'ablation' array
./scripts/ablation_status.sh <jobid>      # explicit
```

Reports: state counts (completed / running / pending / failed), the last five completed val-losses with cell identification, the failure table with node names, and an ETA estimate based on median elapsed per completed task. Handles both the main array and explicit-index redo arrays (auto-detects array size from `scontrol`).

Bad-node handling: two nodes have been observed to crash all tasks with a CUDA-driver init failure (`fc10506`, `fc10417`). Resubmit failed indices with `sbatch --exclude=fc10506,fc10417 --array=<failed-indices>%24 scripts/slurm/array_ablation.sh`.

### Analyze

```bash
python -m experiments.analysis.phase2_summary
```

The analysis script (483 lines, [`experiments/analysis/phase2_summary.py`](../experiments/analysis/phase2_summary.py)) does:

1. Walk `results/slurm/ablation-*.out` and parse the `[SOTR] run_id=… cfg=…` header emitted by `train.py` to recover `(run_id, cell, seed, lr)` for each completed task.
2. Load the corresponding `results/phase2/<run_id>/eval.jsonl` and take the final `val_loss_nats`.
3. Per cell, pick the LR whose across-seed median is lowest.
4. Run paired bootstrap (10k resamples) on per-seed pairs for each pre-registered H2 sub-comparison.
5. Apply Holm–Bonferroni step-down correction across the H2 family.
6. Emit `_summary_<jobid>.md` (narrative decision tree) and `_per_run_<jobid>.csv` (raw per-run data).

The analysis code has unit tests: [`tests/unit/test_phase2_analysis.py`](../tests/unit/test_phase2_analysis.py) covers 17 cases including the Holm–Bonferroni step-down logic, bootstrap null-case, length-mismatch handling, and end-to-end `main()` on synthetic data.

## Pass/fail criteria

### H1 (primary — SOTR vs Muon)

Best SOTR cell (post-tuning) vs cell K (canonical Muon) at K's best LR, paired bootstrap over 5 seeds. Pre-registered: SOTR passes H1 if it achieves either

- (a) lower val_loss with 95% paired-bootstrap CI excluding 0 in SOTR's favor, or
- (b) equal val_loss (paired CI within ±0.01 nats) with fewer stability incidents (Fisher's exact p < 0.05).

### H2 (secondary — component necessity)

For each of {α-blend (A vs B), Δ (A vs C), partial NS (A vs E)}, paired bootstrap on best-LR-per-cell across 5 seeds, Holm–Bonferroni corrected at family-wise α=0.05. A component is necessary iff its drop causes a Holm-significant degradation.

### Cell K amendment (P-K1)

F's best LR vs K's best LR, paired bootstrap over 5 seeds. Predicted: `F_best < K_best` by ≥ 0.02 nats Holm-significant. Falsification = canonical Muon already dominates the best SOTR variant tested.

## Current state (2026-07-03)

- **250 original runs complete** (cells A–J × 5 seeds × 5 LRs, across job arrays 40082656, 40103605, 41186293).
- **Multiple H2 sub-predictions falsified**: A vs B (α-blend), A vs C (Δ cap), and F vs A (q=2 vs q=5 sufficiency) all rejected in ways the protocol didn't predict — see the results table in `README.md`.
- **Cell K + LR extension in progress**: `job 41671913`, 48 tasks after two bad-node reruns. Preliminary seed-0 result on K puts canonical Muon **~0.05 nats below** F's best (3.7730 vs 3.819), which — if it holds across seeds — falsifies P-K1 and closes out the SOTR positive-result path at Phase 2 scale.

## After Phase 2

Two branches, depending on the K verdict:

**If F > K by ≥ 0.02 nats Holm-sig** → Phase 3 (mid-scale, 300–500M, 4× H100, ~5 GPU-days). Config-promotion script (~80 LOC) writes `phase3_F_promoted.py` and `phase3_K_baseline.py` at full upstream scale. Multi-GPU SLURM template is already in `scripts/slurm/multi_gpu.sh`.

**If F ≈ K or F < K** → publish as a pre-registered negative result. Skip Phase 3/4 compute. Optional pivot to Paper 2 (Muon-family in RLHF/DPO/GRPO; see `PROTOCOL.md` §15 for the amendment sketch).

## What Phase 2 does *not* prove

- It does not test H3 (cross-scale consistency) — that's Phase 3.
- It does not test H4 (generalization to a held-out task) — that's Phase 4.
- It does not settle whether the Frobenius trust region is useful at *scale*. Reduced-scale results can under- or over-estimate the value of magnitude control; hence the mid-scale (Phase 3) gate before any headline claim.
