# sotr-muon-ablation

**Pre-registered ablation of a soft-orthogonal Muon-family optimizer.**

**Author.** Rohit Singla, MD PhD — postdoc + resident, UBC. [LinkedIn](https://www.linkedin.com/in/rsingla92/) · rsingla@ece.ubc.ca

**Status (2026-07-03).** Phase 1 reproduced ✓ · Phase 2 250-run ablation complete · Cell K + LR extension in progress · Phase 3/4 pending K verdict.

---

## What

**SOTR (Soft-Orthogonal Trust Region)** is a Muon-family optimizer with three knobs:

- `α ∈ [0, 1]` — blend between Newton-Schulz-orthogonalized momentum (α=1) and Frobenius-normalized momentum (α=0)
- `Δ ∈ (0, ∞]` — per-matrix Frobenius trust region cap on the post-blend update
- `q ∈ {0, 1, 2, …}` — number of Newton-Schulz iterations

The corner case `(α=1, Δ=∞, q=5)` is byte-equivalent to canonical Muon, and this equivalence is a kill-switch test in `tests/sanity/test_sotr_limits.py`. SOTR generalizes Muon along these three axes to ask whether adding per-matrix magnitude control (`Δ`) and a partial-orthogonalization schedule (`α`, `q`) meaningfully improves training.

This repo is a pre-registered evaluation of that question. The **methodology is the artifact**: hypotheses locked before code was written, gates and kill switches defined up front, statistical tests specified in advance, decision rules dated in the commit log.

## Why

Muon is state-of-the-art on the modded-nanogpt speedrun and has been adopted for Kimi K2 pretraining, but it has no per-matrix magnitude control — Newton-Schulz projects the momentum onto (approximately) the closest orthogonal matrix and the whole update is scaled by a scalar learning rate. Trust-region methods are standard in policy-gradient RL (TRPO, PPO) and second-order optimization (Levenberg–Marquardt) precisely because a step direction can be right while the magnitude is wrong.

**The per-matrix Frobenius trust region (`Δ`) is the cleanly novel piece.** The α-blend and partial-NS ideas exist in the literature — see [`knowledge/07_spectral_interpretation.md`](knowledge/07_spectral_interpretation.md) for the honest scope note (α-blend is a specific parameterization within the singular-value-rescaling family studied by PolarGrad, Lau 2025). The empirical contribution being tested is the *interaction map* over `(α, Δ, q)`, which has not been published.

## How

Everything downstream of §1 of `PROTOCOL.md` was decided in advance:

- **Four hypotheses (H1–H4)** with numeric success criteria, paired-bootstrap CIs, Holm–Bonferroni correction across the H2 family, and named kill conditions.
- **Four phases with explicit gates:**
  1. Reproduction of published Muon val-loss on modded-nanogpt (within ±5% band)
  2. 11-cell × 5-seed × 5-LR ablation at reduced scale (250 → 305 runs after LR extension)
  3. Mid-scale (300–500M) validation of the best Phase 2 config vs canonical Muon
  4. 8× H100 headline reproduction at the modded-nanogpt speedrun protocol
- **Kill switches** for implementation bugs (`Hkill2`: SOTR must byte-match Muon at the corner case), reproduction failure (`Hkill1`), and rank reversal across scales (H3).
- **Analysis pipeline** with paired bootstrap over per-seed pairs and Holm–Bonferroni family-wise error control across the three H2 sub-hypotheses. Full spec in [`PROTOCOL.md`](PROTOCOL.md) §3, §7, §9.

Cluster: DRAC Fir (SFU H100). Harness: modded-nanogpt pinned to `dd2224b`, vendored into `experiments/train.py` with a minimal optimizer-dispatch patch.

## Results so far

### Phase 1 — reproduction gate (passed)

Muon on modded-nanogpt: val_loss **3.2911**, inside the published-reference band `[3.12, 3.44]`. Gate passed on first successful run after installing the DRAC Triton wheel (`3.6.0+computecanada`).

### Phase 2 — 250-run ablation (multiple pre-registered predictions falsified)

Reduced scale (1,500 iters, batch 128, 5 seeds × 5 LRs per cell). Pre-registered predictions and outcomes:

| Prediction | Cells compared | Outcome |
|---|---|---|
| Dropping α-blend degrades performance (H2) | A vs B | **Falsified** — B ≈ A within noise |
| Dropping Δ trust region degrades performance at q=2 (H2) | A vs C | **Falsified in the opposite direction** — C beats A |
| Partial NS (q=2) matches full NS (q=5) at α=0.5, Δ=1 | A vs F | **Falsified** — F beats A by ~0.22 nats |
| α-blend adds no value when Δ is capped at q=5 | F vs I | **Falsified** — F (α=0.5) beats I (α=1) |

Best SOTR variant seen so far: cell F (α=0.5, Δ=1, q=5) at LR=0.08 → val_loss **3.819**. Both F and I peak at the upper edge of the original LR sweep, which motivated the 2026-05-26 amendment adding LRs `{0.12, 0.16}` and a canonical-Muon cell K.

### Cell K (canonical Muon) — in progress

The original grid lacked a Phase-2 canonical-Muon baseline, so cell K (α=1, Δ=∞, q=5) was added to anchor H1 (the primary hypothesis: SOTR beats Muon). Preliminary seed-0 results:

```
K seed0 LR=0.005 → 3.7879
K seed0 LR=0.01  → 3.7743
K seed0 LR=0.02  → 3.7730   ← seed-0 optimum
K seed0 LR=0.04  → 3.7917
K seed0 LR=0.08  → 3.8195
```

K's seed-0 optimum lands **~0.05 nats below F's best of 3.819**. If this margin holds across all five seeds, prediction P-K1 (F beats K by ≥0.02 nats Holm-significant) is falsified, and canonical Muon already outperforms the best SOTR variant tested at Phase 2 scale. Full-seed verification is running on the DRAC array (`job 41671913`).

### What this looks like as science

Multiple pre-registered predictions being falsified is not "the experiment didn't work" — it's what pre-registration is for. The methodological artifact (protocol + analysis + decision tree) survives regardless of which optimizer wins. Whether this becomes a positive-result paper or a pre-registered negative result depends on the K-cell verdict.

## What's in progress

- Cell K + LR extension resubmit (`job 41671913`, 48 tasks after bad-node reruns) draining on DRAC Fir.
- Decision fork on completion:
  - **F > K by ≥ 0.02 nats Holm-sig** → proceed to Phase 3 mid-scale (~5 GPU-days).
  - **F ≈ K or worse** → publish as a pre-registered negative result and pivot to Paper 2 (Muon-family optimizers in RLHF/DPO/GRPO), whose protocol amendment lives in `PROTOCOL.md` §15.

Phase 3/4 SLURM scripts and analysis code (~230 LOC total) will land after the K verdict.

## Start here

Five files that let a skim-reader verify this isn't LLM-generated boilerplate:

1. **[`PROTOCOL.md`](PROTOCOL.md)** (504 lines) — the pre-registration. Hypotheses, kill switches, phase gates, statistical tests, dated amendments.
2. **[`optimizers/sotr.py`](optimizers/sotr.py)** (225 lines) — the optimizer. Muon-byte-compatible at the `(α=1, Δ=∞, q=5)` corner; the NS polynomial is imported from `external/Muon`, not reimplemented.
3. **[`experiments/analysis/phase2_summary.py`](experiments/analysis/phase2_summary.py)** (483 lines) — analysis pipeline with paired bootstrap (10k resamples), Holm–Bonferroni across the H2 family, and decision-tree rendering.
4. **[`tests/unit/test_phase2_analysis.py`](tests/unit/test_phase2_analysis.py)** + **[`tests/sanity/test_sotr_limits.py`](tests/sanity/test_sotr_limits.py)** — both the analysis code and the optimizer have real unit tests. `test_sotr_limits` is the enforcement of kill switch `Hkill2`.
5. **[`knowledge/07_spectral_interpretation.md`](knowledge/07_spectral_interpretation.md)** — the honest scope note. Where α-blend fits in the singular-value-rescaling family and why the paper's contribution claim was reframed after that realization.

## Repo layout

```
PROTOCOL.md                Pre-registered methodology (§1–§16)
LICENSE                    MIT
optimizers/sotr.py         SOTR implementation
experiments/
  train.py                 modded-nanogpt vendored + optimizer dispatch
  configs/                 Config dataclasses (Phase 1 + Phase 2 grid)
  scripts/gen_phase2_configs.py    Code-as-source-of-truth for the ablation
  analysis/phase2_summary.py       Statistics + decision tree
tests/
  sanity/                  PROTOCOL §7 gates (Muon byte-match, spectral identity, …)
  unit/                    Analysis and logging tests
scripts/
  setup.sh, setup_drac.sh  Local + DRAC bootstrap
  slurm/                   SLURM templates (single/multi/8-GPU + array)
  ablation_status.sh       One-shot progress report for a Phase 2 array
knowledge/                 Literature synthesis (00-index + 9 notes)
external/                  Pinned reference repos as submodules
docs/                      Architecture, cluster specifics, per-phase procedures
```

Full breakdown: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Reproduce

Requires Python ≥ 3.10 and CUDA-capable PyTorch ≥ 2.10.

```bash
git clone --recurse-submodules git@github.com:rsingla92/sotr-muon-ablation.git
cd sotr-muon-ablation
./scripts/setup.sh

make sanity     # PROTOCOL §7 gates (Muon byte-match, spectral identity, …)
make test       # full test suite
make lint       # ruff
```

Per-phase procedures: [`docs/PHASE1.md`](docs/PHASE1.md), [`docs/PHASE2.md`](docs/PHASE2.md), [`docs/CLUSTER.md`](docs/CLUSTER.md).

## How this was built

This project was executed with Claude Code as a research assistant across many sessions. The division of labor:

**I directed:** hypothesis choice (H1–H4 + kill conditions), success criteria and effect-size thresholds, choice of statistical tests (paired bootstrap + Holm–Bonferroni over the H2 family), harness (modded-nanogpt) and cluster (DRAC Fir) selection, phase gates and kill switches, PROTOCOL amendments after each surprising result, decision rules for each Phase-2 finding, and the reframed contribution claim after realizing the α-blend was not cleanly novel.

**AI accelerated:** implementation of the optimizer (guided by the pre-registered spec, verified against `tests/sanity/test_sotr_limits.py`), SLURM plumbing, config-generation script, analysis-pipeline scaffolding and its unit tests, and lit-review synthesis of two Muon-family papers ([`knowledge/08_muonbp_block_periodic.md`](knowledge/08_muonbp_block_periodic.md), [`knowledge/09_demo_decoupled_momentum.md`](knowledge/09_demo_decoupled_momentum.md)) — each drafted by a subagent from the source arxiv PDF and reviewed before commit.

The pre-registration and my willingness to record my own falsified predictions in the commit log are what distinguish this from a positive-result-only vibe-coded artifact. When the analysis says "P-K1 falsified," the protocol says exactly what happens next.

## License

MIT. See [`LICENSE`](LICENSE).
