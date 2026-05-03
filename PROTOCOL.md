# Experimental Protocol — optimizer_experiments

**Pre-registered.** This document locks the methodology before experiments are run. It is the analog of a clinical trial protocol: hypotheses, endpoints, baselines, statistical tests, and decision rules are committed in advance. Any change after the first experimental run requires a git commit with rationale, and substantive changes (success criteria, hypotheses, baselines) must be flagged in any final paper.

**Date of pre-registration:** 2026-05-02
**Repository:** https://github.com/rsingla92/optimizer_experiments
**Author:** Rohit Singla (UBC)

---

## 1. Overview

Two papers are planned (see `knowledge/05_open_directions.md` and `knowledge/06_lit_update_2026_05.md`):

- **Paper 1 — SOTR** (Soft-Orthogonal Trust Region): a soft-orthogonalization optimizer with a tunable α blend between Newton-Schulz polar and normalized gradient, plus a per-matrix Frobenius trust region.
- **Paper 2 — PSORL** (Muon-family optimizers in RLHF/DPO/GRPO): an empirical study of orthogonalized optimizers in alignment training.

This protocol covers Paper 1 in full. Paper 2 will get its own protocol amendment once Paper 1 reaches Phase 2.

---

## 2. Hypotheses

### Primary

**H1 (SOTR vs Muon, validation loss):** There exists a configuration `(α*, Δ*, q*)` with `α* ∈ (0, 1)` and `Δ* < ∞` such that, at small scale (10–50M parameters), SOTR achieves either:

- (a) lower validation loss than tuned Muon at fixed wallclock seconds, with paired bootstrap 95% CI excluding 0 in SOTR's favor, **OR**
- (b) equal validation loss (paired CI within ±0.01 nats) with strictly fewer stability incidents (Fisher's exact p < 0.05),

across ≥5 seeds, with the comparison conducted on the AdamW-tuned version of every baseline (see §6).

### Secondary

**H2 (Component necessity):** Each of {α-blend, Δ trust region, partial NS} contributes positively. Tested by full-grid ablation (drop one at a time, see §9). A component "fails" the necessity test if dropping it does not statistically degrade primary endpoint at α = 0.05 with Holm-Bonferroni correction across the three drops.

**H3 (Cross-scale consistency):** If H1 holds at small scale, the best `(α*, Δ*, q*)` from Phase 2 retains its qualitative ranking (top-1 or top-2 of the SOTR variants tested) at mid-scale (300–500M), without re-tuning. Quantitative changes in margin are allowed; **rank reversal is a kill condition** (see §11).

**H4 (Generalization):** Best small-scale config beats AdamW on at least one task held out from Phase 2 tuning (e.g., FineWeb subset if Phase 2 was on Pile-subset, or vice versa).

### Null and kill hypotheses

**H0 (Null):** SOTR's improvements vanish when baselines are properly tuned and stability incidents are intention-to-treat. Failure to reject H0 means we write a negative-result report, not a positive-claim paper.

**Hkill1:** Phase 1 reproduction fails (we cannot reproduce a published Muon or AdamW number within 5% on identical infrastructure). Implementation is suspect; we halt before any Phase 2 claim.

**Hkill2:** SOTR(α=1, Δ=∞, q=5) does not match Muon step-by-step within 1e-5 in update magnitude on a 50-step toy problem. Implementation has a bug; we halt.

---

## 3. Phase plan with explicit decision gates

| Phase | Goal | Compute budget | Gate to advance |
|---|---|---|---|
| **0. Infrastructure** | Repo, baselines wired, logging, sanity checks | <1 GPU-day | All sanity checks pass (see §7) |
| **1. Reproduction** | Match published Muon and AdamW numbers on Shakespeare-char | ~3 GPU-days | Within ±5% of published reference numbers |
| **2. Small-scale ablation** | Multi-seed sweep over (α, Δ, q), tuned baselines | ~50 GPU-days | H1 met (or explicitly rejected) |
| **3. Mid-scale validation** | 300–500M, best config vs tuned baselines | ~200 GPU-days | H3 met (or rank reversal recorded) |
| **4. Open release** | Public repo, configs, checkpoints, third-party reproduction attempt | ~5 GPU-days | Independent run reproduces key Table 2 cells |

Phase 4 is mandatory before any preprint submission.

---

## 4. Datasets and tasks

| Phase | Dataset | Model | Tokens | Source |
|---|---|---|---|---|
| 1 | Shakespeare-char | NanoGPT 10M | ~1M (overfit-friendly) | karpathy/nanoGPT default |
| 2 | OpenWebText subset (FineWeb-edu-10B subset acceptable substitute) | NanoGPT 10–50M | 1B–3B | modded-nanogpt default |
| 3 | FineWeb-edu (full or subset) | GPT-2-style 300–500M | 10B–30B | published reference dataset |
| 4 | Held-out: opposite of Phase 2/3 choice | Same as Phase 3 | matched | — |

Tokenizer: GPT-2 BPE for everything ≥ NanoGPT scale; char-level for Shakespeare-char.

**No test-set tuning.** Validation set used for hyperparameter selection. Test set used for final reporting only and accessed at most once per phase per method.

---

## 5. Hardware and software lock

We follow the **modded-nanogpt speedrun protocol** as our canonical comparison harness. This is the framework where Muon, AdamW, DistributedShampoo, SOAP (and now Muon+, NorMuon, AdaMuon) have all been benchmarked apples-to-apples; tracking it gives us free comparability with the published literature.

**Software lock (from `external/modded-nanogpt/Dockerfile`):**

| Item | Locked value |
|---|---|
| Framework | PyTorch ≥ 2.10 (matching modded-nanogpt's pinned `torch==2.10` requirement) |
| Python | 3.12.7 (Docker image) |
| CUDA | 12.6 (Docker image) |
| Precision | BF16 mixed; **FP32 NS body**; FP8 matmul where modded-nanogpt enables it (head only) |
| Determinism | `torch.manual_seed(seed)`, deterministic CUDA where feasible. NS may use non-deterministic kernels — flagged when so. |
| Multi-GPU | `torchrun --standalone --nproc_per_node=8` |
| Compile | `torch.compile` opt-in; modded-nanogpt convention: `coordinate_descent_tuning` is **banned** for speedrun comparisons (>30 min compile). We follow this. |
| Data shuffle | Fixed seed per `(method, seed)` pair → same data ordering across methods. |
| Param-group split | `raw_model.transformer.h.parameters()` → matrix optimizer; embedding + head → Adam (matches the canonical optimizer comparison in `records/track_1_short/2024-10-29_Optimizers/`). |
| LR schedule | Trapezoidal (warmup-stable-decay) — empirically optimal in the speedrun protocol. |
| Weight decay | **0** for matrix optimizer (per speedrun convention); decoupled-AdamW WD optional for aux Adam. |

**What's actually canonical across papers:**

A literature survey (Apr–May 2026) of recent Muon-family papers (Dion, AdaMuon, NorMuon, Muon+, Mousse, Newton-Muon, Mano) shows hardware varies widely (single H100 → 8× A100 → 8× H200 → 4× H800 → 16× A100). What is consistent across *every* recent paper:

- **FineWeb (or FineWeb-Edu) dataset** via modded-nanogpt's data pipeline
- **modded-nanogpt's `train_gpt.py` training harness**
- **Trapezoidal (warmup-stable-decay) LR schedule**
- **`transformer.h.*` split for matrix optimizer; embed/head for AdamW**
- **~20 hyperparameter attempts per baseline** for fair comparison
- **`torch.compile` + BF16 mixed precision; FP8 matmul for head**

These are the locked items. Hardware is **recorded per run** but not constrained beyond the requirement that any direct comparison (e.g., SOTR vs Muon) must run on the same hardware.

**Hardware tier protocol (where each phase runs):**

| Phase | Hardware | Why | Cost estimate |
|---|---|---|---|
| **Phase 0 (sanity, dev)** | Any single GPU — UBC cluster A100/L4/H100 OK | Limit-case unit tests don't need scale | ~free |
| **Phase 1 (reproduction)** | Single H100 or A100 (UBC or rented). Match a *published modded-nanogpt single-GPU baseline number* (e.g., the Newton-Muon paper's Record #4 on single H100). **Not** the official 8× H100 5-min record. | Reproducing the canonical optimizer-comparison protocol on whatever hardware we have. | ~$5–20 per run if rented |
| **Phase 2 (ablation, 200 runs)** | 1–2× H100 or A100 with **reduced-scale config** (smaller `n_layer`/`n_embd`, matched token budget) | 200 runs × full-scale would be infeasible; reduced scale preserves directional signal | ~$5/run × 200 = ~$1k |
| **Phase 3 (mid-scale validation, ~20 runs)** | 4–8× H100 (or whatever multi-GPU we have) at full modded-nanogpt config | Primary claims at full canonical setup; matches Mousse/NorMuon-class papers | ~$100–300 × 20 = ~$2–6k |
| **Phase 4 (release replication)** | Whatever Phase 3 used | External-replication evidence | ~$50 |

**Estimated total compute:** ~$3–8k for Paper 1 if we rent. Significantly less if UBC cluster covers Phases 0–2.

**Hardware constraints for *valid* claims:**
- Any direct A-vs-B comparison (SOTR vs Muon, etc.) runs on **identical hardware** — same GPU type, same count, same node configuration, same data sharding.
- Hardware is recorded with every result table.
- We do **not** claim the official 8× H100 speedrun record. We claim "matches/improves Muon under hardware-matched conditions on FineWeb with modded-nanogpt's harness." This is exactly what Newton-Muon (April 2026) and similar papers claim.
- Cross-hardware comparison (e.g., our Phase 2 reduced-scale on A100 vs Phase 3 full-scale on H100) is **never** the basis for a primary claim. Phase 2 is a *filter* for Phase 3.

**Reduced-scale config for Phase 2 (proposed; final values during Phase 0 sanity):**
- Same model architecture as modded-nanogpt's `train_gpt.py`
- Reduce `n_layer` 12→6, `n_embd` 768→384, `n_head` 6→3 (≈ 25M params)
- Reduce target tokens proportionally to maintain Chinchilla-like ratio
- Same validation loss target scaled appropriately (TBD via Phase 1 calibration)
- Single H100/A100/L4 acceptable; record GPU type per result

---

## 6. Baselines (locked specifications)

All baselines get an independent learning-rate sweep at each scale. **No default-LR comparisons.**

| Baseline | Source | Commit/version | LR sweep |
|---|---|---|---|
| **AdamW** | `torch.optim.AdamW` (`fused=True`), betas=(0.9, 0.95), eps=1e-8, decoupled WD ∈ {0, 0.01, 0.1} | PyTorch ≥ 2.4 | 5 LRs log-spaced around 3e-4 (1e-4, 3e-4, 1e-3, 3e-3, 1e-2) |
| **Lion** | Chen et al. 2023 (arXiv:2302.06675) reference; vendored to `optimizers/lion_official.py` | as released | 5 LRs around 1e-4 |
| **Muon** | KellerJordan/Muon repo, `MuonWithAuxAdam`, NS poly (3.4445, -4.7750, 2.0315), 5 NS steps, BF16 NS | Pinned to specific commit; recorded in lockfile | 5 LRs around 0.02 (Muon's published) for hidden weights; AdamW LR for aux params |
| **AdaMuon** | Si et al. 2507.11005 reference impl | Pinned commit | LR sweep matching their paper's range |
| **MuonLike (sanity baseline)** | SOTR with α=1, Δ=∞, q=5; same NS poly as Muon | This repo | Same LR range as Muon |

**Tuning protocol per baseline:** 5 LRs × 3 seeds = 15 runs per baseline per scale. Best LR (lowest mean validation loss across seeds, ties broken by stability) becomes that baseline's headline number for the primary comparison.

**Reference numbers we must hit (Phase 1 reproduction):**
- AdamW on Shakespeare-char NanoGPT: published reference loss within 5%
- Muon on modded-nanogpt speedrun: within 5% of Keller Jordan's published number on the same hardware class

If we can't hit these, we don't have a working baseline, and the comparison is meaningless.

---

## 7. Sanity checks (Phase 0 → Phase 1 gate)

All must pass before proceeding to Phase 2. Each is a unit test in `tests/`.

1. **Limit case I:** `SOTR(α=1, Δ=∞, q=5)` produces parameter updates within `||ΔW_SOTR - ΔW_Muon||_F / ||ΔW_Muon||_F < 1e-5` step-by-step over 50 steps on a synthetic 256×256 problem. Same NS polynomial.
2. **Limit case II:** `SOTR(α=0, q=0)` produces updates within 1e-7 of `G / (||G||_F + ε)`.
3. **Limit case III:** `SOTR(α=0, q=2)` does *not* match Muon (we want the partial-NS to be *visible*).
4. **Lion match:** Our `Lion` impl matches Chen 2023 reference impl on a saved 100-step trajectory within 1e-5.
5. **Muon match:** Our integration of `KellerJordan/Muon` agrees with running their repo directly on the same seed for 100 steps within 1e-5.
6. **Trust region triggers correctly:** with `α=1, Δ=0.01, q=5`, hit-rate >50% on a problem where typical update is O(1).
7. **Determinism:** two runs with same seed and code produce bit-identical loss curves on CPU; on GPU, within 1e-4 (tolerance for non-deterministic CUDA ops).
8. **Param-group split correctness:** Muon and SOTR apply only to `transformer.h.*` 2D weights; embeddings/head/biases/LayerNorm get AdamW. Verified by inspecting `param_groups` printout.

Failure of any sanity check halts progress until fixed. No exceptions.

---

## 8. Stability incident definitions (intention-to-treat)

Pre-defined so they can't be retro-fitted. **All incidents reported across all seeds; no exclusions.**

| Incident | Definition | Severity |
|---|---|---|
| **Spike** | Validation loss > 2× rolling-100-step mean | Recoverable |
| **Crash** | Any NaN or Inf in loss, gradient, or parameter | Terminal |
| **Blowup** | Update Frobenius norm > 10× rolling-100-step mean of update norm | Recoverable |
| **Grad spike** | Per-step gradient norm > 100× rolling-1000-step median | Recoverable |
| **Rank collapse** | Stable rank of any tracked weight matrix drops > 50% from initialization | Concerning, not terminal |

For each (method × LR × seed) run, we report:
- Total incident count by type
- Time to first incident (in steps)
- Whether the run completed the budget
- Final loss, regardless of incidents (intention-to-treat)

A method with mean loss 0.01 lower than baseline but 30% terminal-crash rate is **not** considered better.

---

## 9. Ablation grid (locked)

Run at small scale only initially; promote winners to mid-scale.

| Config | α | Δ | q (NS iters) | Reduces to / tests |
|---|---|---|---|---|
| **A. SOTR full** | 0.5 | 1.0 | 2 | The proposed method |
| **B. Drop α-blend** | 1.0 | 1.0 | 2 | Muon + Fro cap (no soft blend) |
| **C. Drop Δ cap** | 0.5 | ∞ | 2 | α-blend only (no trust region) |
| **D. Drop both** | 1.0 | ∞ | 2 | "MuonLike q=2" — partial NS only |
| **E. Drop NS** | 0.5 | 1.0 | 0 | Renorm-SGD with Fro cap + blend (skips orth) |
| **F. Full NS** | 0.5 | 1.0 | 5 | Does *partial* NS matter? |
| **G. α schedule** | 0→0.5 over 10k steps | 1.0 | 2 | Annealing matters? |
| **H. Δ scheduled** | 0.5 | start∞ → 1.0 over 10k | 2 | Late-onset trust region? |

Each cell: 5 seeds × 1 LR sweep (5 LRs) = 25 runs. Total ablation: 8 × 25 = 200 small-scale runs.

Each cell uses the *same* per-config LR sweep — no shared LR across cells unless justified.

**Pre-registered prediction:** A > {B, C, D} significantly; F ≈ A or A slightly better (partial NS suffices); E < A (NS contributes); G ≥ A (annealing helps).

---

## 10. Statistical methodology

| Test | Use |
|---|---|
| **Paired bootstrap (10,000 resamples)** on validation loss | Primary endpoint comparison: SOTR vs each baseline at fixed wallclock |
| **Fisher's exact** | Stability incident rate comparison (binomial) |
| **Mann-Whitney U** | Time-to-target-loss when distributions are non-normal |
| **Holm-Bonferroni** | Multiple-comparison correction across baselines |
| **Cohen's d** | Effect size, reported alongside every p-value |

**Significance level:** α = 0.05 throughout. Effect sizes reported regardless of significance.

**Sample size:** 5 seeds is the floor; 10 seeds for any number that goes in a final paper figure or table.

**No p-hacking:**
- Pre-registered tests only. New tests added later are flagged as exploratory and cannot support primary claims.
- All seeds reported. None dropped.
- All configs reported. None dropped.

---

## 11. Decision rules and kill switches

| Trigger | Action |
|---|---|
| Phase 1 reproduction off by >5% | **HALT.** Debug infrastructure. Do not run Phase 2. |
| Sanity check fails | **HALT.** Fix bug. Do not report any number. |
| Phase 2: H1 not met after full ablation | Drop SOTR claim. Either (a) write up as negative result, or (b) refine SOTR design and re-pre-register. |
| Phase 2: best config has >50% stability-incident rate | That config is dropped from claims regardless of mean loss. |
| Phase 3: rank reversal vs Phase 2 | Report explicitly. Paper restricted to small-scale claim with caveat, or killed. |
| Component necessity test fails for any of {α, Δ, q<5} | That component dropped from the proposed method; protocol amended. |
| Cross-hardware comparison required | Flag explicitly; cannot be primary evidence. |
| Statistical test result depends on which seeds we exclude | Implementation is broken or sample size too small. Add seeds, do not exclude. |

---

## 12. Reporting requirements (every result table)

For every reported number:

- Mean and standard deviation across seeds
- Worst-seed value (intention-to-treat)
- Number of seeds (n)
- Number of stability incidents and type breakdown
- Compute used (GPU-hours, hardware type)
- Effect size (Cohen's d) versus the relevant comparator
- Confidence interval (95%) where applicable
- All raw seed-level numbers in supplementary material
- Code and config commit hash

**Tables that omit any of the above are not publishable.**

---

## 13. What we commit to *not* do

- ❌ "Best of N" without showing all N
- ❌ Removing failed seeds from averages
- ❌ Introducing new metrics after seeing results
- ❌ Using baseline numbers from other papers (we rerun on our infra)
- ❌ Tuning on the test set
- ❌ Default-LR baseline comparisons
- ❌ Re-running with different seeds until significance is achieved
- ❌ Quietly dropping ablation cells whose results are inconvenient
- ❌ Comparing across hardware classes for primary claims
- ❌ Reporting wallclock without reporting steps/FLOPs alongside

---

## 14. Amendment process

- Any change to this PROTOCOL.md after Phase 0 begins requires:
  - A git commit dedicated to the change
  - A `## Amendment YYYY-MM-DD` section appended below documenting what changed and why
  - A note in the final paper if the change affected hypothesis or success criteria

- Phase amendments before that phase begins are free (e.g., Phase 3 details can be refined after Phase 2 results inform them).

- **Unamendable** items: H0, H1, H2, the sanity checks (§7), the stability definitions (§8). These cannot be relaxed mid-study.

---

## 15. Paper 2 (PSORL) — placeholder

Pre-registration for Paper 2 (Muon-family in RLHF/DPO/GRPO) will be drafted as a separate amendment to this document once Paper 1 reaches Phase 2. Outline:

- Hypotheses about optimizer-mismatch effects across SFT → RLHF transitions
- Algorithms covered: GRPO, DPO, PPO, IPO
- Optimizers covered: AdamW, Lion, Muon, AdaMuon, Muon+, ROOT, SOTR
- Models: Pythia/Qwen-small base, 125M–1B
- Stability protocol specific to RL (reward hacking, KL-divergence blowups)

---

## Amendments

### Amendment 2026-05-02 (revised) — Lock the *protocol*, not the hardware
*(Pre-Phase-0; no experimental data yet → free amendment.)*

**Change:** §5 (Hardware and software lock) revised. The earlier draft over-locked on 8× H100. After surveying recent Muon-family papers (Dion, AdaMuon, NorMuon, Muon+, Mousse, Newton-Muon, Mano) for their actual hardware:

- Newton-Muon: single H100 or single L40S
- Mano: 4× H800 PCIe (and 4× RTX-4090 for smallest models)
- Mousse: 8× H200
- NorMuon: 8× A100 (single node) for ≤350M; 16× A100 for 1.1B+
- Muon+: H100 or A100, count unspecified
- Dion: H100, count unspecified
- AdaMuon: not specified at all

**Hardware varies wildly across labs.** What's locked across every paper is the *protocol*: FineWeb dataset, modded-nanogpt's `train_gpt.py` harness, trapezoidal LR schedule, `transformer.h.*` split, ~20 HP attempts per baseline, BF16 + FP8-head precision. We adopt the protocol and treat hardware as recorded-but-not-constrained, with the rule that any direct A-vs-B comparison must run on identical hardware.

**Practical implication:** We can run our Phase 1 reproduction on whatever single GPU we have (Newton-Muon precedent: single H100 against modded-nanogpt records). Phase 2 ablations on 1–2× H100/A100 with reduced model config. Phase 3 on 4–8× H100 if available, or single-node multi-GPU equivalent. Estimated cost drops from ~$5–15k → ~$3–8k.

**No claim of the official 8× H100 5-min speedrun record.** Our claims are of the form "SOTR matches/improves Muon under hardware-matched conditions on FineWeb with modded-nanogpt's harness" — exactly the framing recent papers (Newton-Muon, Mano) use.

**Constraint preserved:** No primary claim may be supported by Phase 2 reduced-scale numbers alone. Phase 2 is a *filter* for Phase 3.

### Amendment 2026-05-02 — Lock §5 hardware to modded-nanogpt speedrun protocol *(SUPERSEDED by revision above)*
Original wording over-locked on 8× H100. Replaced because hardware is not standardized across papers. See revised amendment immediately above.
