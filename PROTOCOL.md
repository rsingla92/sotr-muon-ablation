# Experimental Protocol — optimizer_experiments

**Pre-registered.** This document locks the methodology before experiments are run. It is the analog of a clinical trial protocol: hypotheses, endpoints, baselines, statistical tests, and decision rules are committed in advance. Any change after the first experimental run requires a git commit with rationale, and substantive changes (success criteria, hypotheses, baselines) must be flagged in any final paper.

**Date of pre-registration:** 2026-05-02
**Repository:** https://github.com/rsingla92/optimizer_experiments
**Author:** Rohit Singla (UBC)

---

## 1. Overview

Two papers are planned (see `knowledge/05_open_directions.md` and `knowledge/06_lit_update_2026_05.md`):

- **Paper 1 — SOTR** (Soft-Orthogonal Trust Region): a study of three soft-orthogonalization mechanisms applied per-matrix to weight updates, with one cleanly novel piece (a per-matrix Frobenius trust region) and two pre-existing knobs (Newton–Schulz iteration count `q`, and an additive-linear singular-value blend parameterized by `α`). Paper 1 maps the interaction of `(α, Δ, q)` and identifies which combinations outperform Muon under hardware-matched conditions on the modded-nanogpt FineWeb harness.
- **Paper 2 — PSORL** (Muon-family optimizers in RLHF/DPO/GRPO): an empirical study of orthogonalized optimizers in alignment training.

This protocol covers Paper 1 in full. Paper 2 will get its own protocol amendment once Paper 1 reaches Phase 2.

**Honest scope note.** The α-blend is *not* a new family of singular-value rescalings — it's a particular parameterization within a family already explored by PolarGrad (Lau 2025) and "Delving into Muon and Beyond" (2602.04669); see `knowledge/07_spectral_interpretation.md` for the derivation. The cleanly novel piece is the per-matrix Frobenius trust region. The empirical contribution is the *interaction map* of all three knobs, which has not been published. This is reflected in the H1–H4 hypotheses below.

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
| Precision | BF16 mixed; **BF16 NS body** (matches Muon's canonical impl in `external/Muon/muon.py` — `zeropower_via_newtonschulz5` runs the polynomial in bf16); FP8 matmul where modded-nanogpt enables it (head only). FP32 NS body available as an optional ablation if numerical issues are observed. |
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

We target **UBC research computing infrastructure** (Sockeye and/or Compute Canada / DRAC). No paid cloud rentals planned. Cluster specifics, SLURM templates, and quotas in [`docs/CLUSTER.md`](docs/CLUSTER.md).

| Phase | Hardware (UBC) | Why |
|---|---|---|
| **Phase 0 (sanity, dev)** | Any single GPU — Sockeye A100/V100, DRAC A100, or local | Limit-case unit tests don't need scale |
| **Phase 1 (reproduction)** | Single A100 or H100 on Sockeye / DRAC. Match a *published modded-nanogpt single-GPU baseline number* (e.g., Newton-Muon's Record #4 on single H100). | Reproducing the canonical optimizer-comparison protocol on the hardware we'll actually use |
| **Phase 2 (ablation, 200 runs)** | 1× A100 per job, submitted as a SLURM job array (`scripts/slurm/array_ablation.sh`). Reduced-scale model config so each run completes in ~1 hour. | 200 runs × full-scale infeasible; reduced scale preserves directional signal. Job arrays are how SLURM expects this kind of sweep. |
| **Phase 3 (mid-scale validation, ~20 runs)** | 4–8× A100 or H100 single-node (Sockeye GPU partition or DRAC large GPU). | Primary claims at full canonical setup; matches Mousse / NorMuon class. |
| **Phase 4 (release replication)** | Same partition as Phase 3, single confirmation run after open release. | External-replication evidence. |

**Estimated compute:** zero dollars (UBC allocation-based). Bound is GPU-hour quota and queue time, not budget. Full Paper 1 scope (~250 runs) fits comfortably in a typical postdoc allocation.

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

All must pass before proceeding to Phase 2. Each is a unit test in `tests/sanity/`.

We use the **Muon-compatible step ordering** for SOTR (see Amendment in §15 for the design fix versus the original PDF pseudocode). Concretely SOTR's per-step update is:

```
1. momentum.lerp_(grad, 1-β)                                  # update buffer
2. M = grad.lerp(momentum, β) if nesterov else momentum       # Nesterov-mixed value
3. O = zeropower_via_newtonschulz5(M, steps=q)                # NS the mixed value
4. U = α·O + (1-α)·M / (||M||_F + ε)                          # SOTR α-blend
5. if ||U||_F > Δ: U *= Δ / ||U||_F                            # SOTR Frobenius cap
6. U *= max(1, m/n)**0.5                                       # match Muon's per-shape RMS scale
7. p -= lr · U                                                  # decoupled WD applied separately
```

At `α=1, Δ=∞`, line 4 collapses to `U = O` and line 5 is a no-op, so steps 1–3, 6–7 reproduce Muon's `muon_update` exactly.

**Sanity checks:**

1. **Limit case I (Muon equivalence):** `SOTR(α=1, Δ=∞, q=5)` produces parameter updates within `||ΔW_SOTR - ΔW_Muon||_F / ||ΔW_Muon||_F < 1e-5` step-by-step over 50 steps on a synthetic 256×256 problem, when SOTR shares the NS routine and per-shape scaling with `external/Muon/muon.py`. Verifies the corner-case design and rules out ordering bugs.
2. **Limit case II (no-orth limit):** `SOTR(α=0, q=0)` produces updates equal to Frobenius-normalized (Nesterov-)momentum with per-shape RMS scaling: `U = (M / ||M||_F) · max(1, m/n)**0.5`. Tolerance 1e-6.
3. **Limit case III (partial NS visible):** `SOTR(α=1, q=2)` is *measurably different* from `SOTR(α=1, q=5)` (Muon) — `||ΔW_SOTR_q2 - ΔW_Muon||_F / ||ΔW_Muon||_F > 1e-3` after the first NS step. Ensures the `q` knob is wired correctly.
4. **Lion match:** The `Lion` baseline imported from `external/lion-pytorch` (lucidrains' reference impl) matches a frozen 100-step reference trajectory in `tests/fixtures/lion_reference.pt` within 1e-5. Catches accidental version drift in the upstream submodule.
5. **Muon match:** The `Muon` baseline imported from `external/Muon` matches a frozen 100-step reference trajectory in `tests/fixtures/muon_reference.pt` within 1e-5. Same purpose as #4.
6. **Trust region triggers:** with `SOTR(α=1, Δ=0.01, q=5)`, the per-matrix Frobenius cap fires on >50% of steps for a problem where typical update Frobenius norm is O(1). Verifies the cap path is reachable.
7. **Determinism:** two runs with same seed and code produce bit-identical loss curves on CPU; on GPU, within 1e-4 (tolerance for non-deterministic CUDA kernels in NS).
8. **Param-group split correctness:** SOTR is applied only to 2D parameters in `transformer.h.*`; embeddings/head/biases/LayerNorm receive AdamW. Verified by inspecting `param_groups` after construction.
9. **Spectral identity (numerical):** for a synthetic 2D `M` with controlled SVD `M = U·Σ·Vᵀ`, run a SOTR step with `q = 5`, `Δ = ∞`, and various `α ∈ {0.25, 0.5, 0.75}`. Verify that the singular values of `U_blend` (post-blend, pre-cap, pre-scale) match the closed form `σ'_i = α + (1−α)·σ_i / ||M||_F` within `1e-3` (loose because `f_5(σ) ≈ 1` not exactly 1). For `q = 0`, verify `σ'_i = σ_i / ||M||_F` exactly within `1e-6`. Catches implementation bugs that limit-case tests #1, #2 can miss (sign errors, wrong normalization, NS applied to wrong tensor). Derivation in `knowledge/07_spectral_interpretation.md`.

**Coverage meta-check:** `tests/sanity/test_sanity_coverage.py` fails if any of #1–#9 lacks a corresponding test file. Prevents drift between this list and the test suite.

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
| **B. Drop α-blend** | 1.0 | 1.0 | 2 | Partial-NS Muon + Fro cap (no soft blend) |
| **C. Drop Δ cap** | 0.5 | ∞ | 2 | Partial-NS + α-blend, no trust region |
| **D. Drop both** | 1.0 | ∞ | 2 | "MuonLike q=2" — partial NS only |
| **E. Drop NS** | 0.5 | 1.0 | 0 | Renorm-SGD with Fro cap + blend (skips orth) |
| **F. Full NS + SOTR** | 0.5 | 1.0 | 5 | Does *partial* NS matter? |
| **G. α schedule** | 0→0.5 over 10k steps | 1.0 | 2 | Annealing matters? *(deferred — see Amendment 2026-05-03 (G/H deferral); first ablation pass emits static α=0.5, identical to A)* |
| **H. Δ scheduled** | 0.5 | start∞ → 1.0 over 10k | 2 | Late-onset trust region? *(deferred — see Amendment 2026-05-03 (G/H deferral); first ablation pass emits static Δ=1.0, identical to A)* |
| **I. Muon + Fro cap only** | 1.0 | 1.0 | 5 | Isolates the Frobenius trust region as the *sole* novel mechanism. Cleanest test of "is Δ alone enough?" |
| **J. Partial-NS Muon** | 1.0 | ∞ | 2 | Isolates partial NS's effect with no blend and no cap. Decouples q from the other knobs. |

Each cell: 5 seeds × 1 LR sweep (5 LRs) = 25 runs. Total ablation: 10 × 25 = 250 small-scale runs.

Each cell uses the *same* per-config LR sweep — no shared LR across cells unless justified.

**Decomposition:** by including cells D, I, J alongside the full SOTR (A) and Muon baseline (q=5, no SOTR), we can attribute SOTR's effect to its components:

- D = Muon + partial NS only
- I = Muon + Frobenius cap only (the genuinely novel piece)
- J = same as D — Muon with partial NS, sanity duplicate
- B = D + Frobenius cap
- C = D + α-blend
- A = D + Frobenius cap + α-blend

Comparing (Muon, D, I, J, B, C, A) tells us: does partial NS help (D vs Muon)? Does Frobenius cap help on top of Muon (I vs Muon)? On top of partial-NS Muon (B vs D)? Does α-blend help on top of partial-NS Muon (C vs D)? On top of partial-NS Muon + Frobenius cap (A vs B)?

**Pre-registered predictions:** A > B (α-blend contributes some); A > C (trust region contributes some); A > D (combined > partial-NS alone); E < D (NS contributes); F ≈ A or A slightly better (partial NS suffices over full NS); I > Muon (trust region helps even without SOTR's other knobs); J ≈ D (sanity); G ≥ A (annealing helps).

**Decision rules tied to ablation outcomes:**
- If A ≈ I (Frobenius cap alone explains the win): paper is "we show Frobenius trust region per matrix is sufficient; α and partial-NS knobs add nothing." Still publishable, leaner contribution.
- If I ≈ Muon (Frobenius cap alone doesn't help): SOTR's contribution rests on the *combination*, paper claim narrows to "the combination of partial NS + α-blend + Frobenius cap is necessary."
- If A ≈ Muon (no combination beats Muon): negative result, write up honestly. PROTOCOL §11 kill switch.

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

### Amendment 2026-05-02 (spectral interpretation) — Reframe contribution; expand ablation; add sanity #9
*(Pre-Phase-0; no experimental data yet → free amendment.)*

**Trigger.** Working through the SVD algebra of SOTR's α-blend reveals that the blend `α·O + (1−α)·M/||M||_F` reduces to a singular-value rescaling `σ_i ↦ α + (1−α)·σ_i/||M||_F` (full derivation in `knowledge/07_spectral_interpretation.md`). This places SOTR's α-knob in the same family as PolarGrad's `σ^v` and 2602.04669's `σ^p`. The α-blend is a parameterization choice within an existing family, not a new family.

**Changes:**

1. **§1 Overview reframed.** Paper 1's contribution is no longer "introduces a tunable soft-orthogonalization with trust region." It is now: "a study of three soft-orthogonalization mechanisms (per-matrix Frobenius trust region [novel], partial NS [pre-existing knob], additive-linear singular-value blend [parameterization within known family]) and their interaction." Honest scope note added.

2. **§7 sanity check #9 added.** Numerically verify the spectral identity: for synthetic `M`, check that `U_blend`'s singular values match the closed form `α + (1−α)·σ_i / ||M||_F` (q=5 case, tol 1e-3) and `σ_i / ||M||_F` (q=0 case, tol 1e-6). Catches sign/normalization/wrong-tensor bugs that limit-case tests #1, #2 can miss.

3. **§9 ablation grid expanded.** Two new cells:
   - **Cell I:** `α=1, Δ=1.0, q=5` — Muon + Frobenius cap only. Isolates Δ as sole novel mechanism.
   - **Cell J:** `α=1, Δ=∞, q=2` — Muon with partial NS. Isolates q's effect alone.
   
   Total ablation runs: 10 cells × 5 seeds × 5 LRs = 250 (was 200). Compute increase ~25%, comfortably within UBC allocation. New decision rules (§9) tied to specific ablation outcomes.

4. **Empirical focus.** The spectral derivation shows the α-blend is most informative at small `q` (partial NS) or for non-uniform spectra. We will report results stratified by these conditions in the paper.

**No effect on hypotheses H1–H4 or success criteria.** H1 was always "for some `(α*, Δ*, q*)`, SOTR beats Muon" — still well-defined. The reframing is rhetorical (how we describe the contribution) rather than substantive (what we measure).

**No effect on PROTOCOL §13 ("what we don't do").** All discipline rules unchanged.

### Amendment 2026-05-02 (design fix) — Lock SOTR step ordering and NS precision
*(Pre-Phase-0; no experimental data yet → free amendment.)*

**Two related changes**, both informed by reading `external/Muon/muon.py` and reconciling against the SOTR PDF's draft pseudocode:

**(a) SOTR step ordering — corrected to be Muon-compatible.**

The SOTR design PDF specified the per-step order as `NS(grad) → α-blend → Frobenius cap → momentum → update`. With this order, at `α=1, Δ=∞, q=5` the resulting trajectory is `momentum(NS(grad))`, while Muon computes `NS(Nesterov(grad, momentum))`. These are not equivalent, so the PDF's claim "α=1 reduces to Muon" is wrong as written.

Since "strictly contains Muon at α=1" is the central rhetorical claim of the SOTR design, we adopt the **Muon-compatible ordering** documented in §7: momentum-update first, then form the Nesterov-mixed value `M`, then NS, then α-blend (between `O = NS(M)` and `M / ||M||_F`), then Frobenius cap, then per-shape RMS scaling, then weight update. At `α=1, Δ=∞, q=5` this is byte-equivalent to `external/Muon`'s `muon_update`.

The α-blend interpolates `M`'s orthogonalized form against `M`'s Frobenius-normalized form. At `α=0, q=0` SOTR reduces to Frobenius-normalized (Nesterov-)momentum SGD with per-shape RMS scaling — a clean limit, similar in spirit to AuON's "unit-norm momentum."

**(b) NS body precision — bf16, not fp32.**

Earlier wording in §5 specified "FP32 NS body." Inspection of `external/Muon/muon.py` shows the canonical `zeropower_via_newtonschulz5` runs the polynomial in **bfloat16**, and every modded-nanogpt speedrun record uses bf16 NS. Forcing fp32 would mean SOTR(α=1) no longer matches Muon at the bit level. We amend §5 to default to bf16 NS body (matching Muon) and reserve fp32 NS as an optional ablation if numerical issues are observed.

**Implementation consequence.** `optimizers/sotr.py` is the *only* novel file. It imports `zeropower_via_newtonschulz5` from `external/Muon` (no reimplementation). Lion is imported from `external/lion-pytorch`. There is no separate `MuonLike` optimizer — the equivalence is established by sanity test #1 over the parameterized SOTR.

**No effect on hypotheses or success criteria** (H1–H4, kill switches). The fix moves an internal design detail; the headline claims unchanged.

### Amendment 2026-05-03 — Pin `external/modded-nanogpt` to `dd2224b` (Oct 2024 Muon era)
*(Pre-Phase-2; no Phase-2 experimental data yet → free amendment.)*

**Trigger:** While planning the Phase 2 vendoring of `train_gpt.py`, the research subagent found that upstream HEAD (`6399c65`, May 2026) had moved far from the simple Muon baseline. Current HEAD uses `NorMuonAndAdam` — a single bespoke optimizer with sharded comms, FP8 lm_head, sparse bigram comms, banked weights, MTP, YaRN. Vendoring this with a clean 4-optimizer dispatcher (`{adamw, lion, muon, sotr}`) would force ~100 lines of patches around two execution paths and would not give an apples-to-apples Muon comparison anyway (NorMuon has many features SOTR lacks).

**Change:** `external/modded-nanogpt` pinned to commit **`dd2224b`** (2024-10-29). At this commit:

- `train_gpt2.py` is **537 lines** (vs 2022 at HEAD)
- Optimizer construction is the simple two-optimizer pattern: `optimizers = [AdamW(lm_head), Muon(transformer.h)]`
- `requirements.txt` is just `numpy / tqdm / torch / huggingface-hub` (no `tiktoken`, `datasets`, `kernels`, or `pyarrow`)
- This is the canonical "Optimizers" comparison harness used in `records/track_1_short/2024-10-29_Optimizers/` (AdamW vs DistributedShampoo vs SOAP vs Muon)
- File rename: HEAD calls it `train_gpt.py`; this older commit calls it `train_gpt2.py`. We preserve upstream's filename when vendoring.

**Consequences:**

- The currently-pending Phase 1 reproduction job (`38333414`, submitted with the old HEAD train_gpt.py) becomes a "scratch" run — it will produce a NorMuon number, not the apples-to-apples baseline we want. We submit a fresh Phase 1 job against `train_gpt2.py` once the vendoring + repinned scripts are merged. Both runs are useful (the NorMuon one as a "what does today's record look like" data point), but only the dd2224b run satisfies PROTOCOL §6.
- Cleaner Phase 2 work: vendoring is a ~30-line patch (vs ~100), and the 4 baselines (`adamw, lion, muon, sotr`) all run through the same `optimizers = [aux_optim, hidden_optim]` interface.
- Drop `arrow` module load from `setup_drac.sh` and SLURM scripts (no `datasets` dep at this pin).
- Trim `pyproject.toml` deps: removed `tiktoken`, `datasets`, `einops`, `tqdm`, `pyyaml`, `rich` (none used by our code; modded-nanogpt's runtime deps installed separately by `setup_drac.sh`).

**Reproduction target unchanged:** PROTOCOL §6 still says final FineWeb val loss within ±5% of the published number. At dd2224b, the canonical short-track Muon record is **12.0 minutes on 8× H100** to 3.28 val loss (Record #7, "Upgraded PyTorch 2.5.0", 2024-10-18). On a single H100 with `grad_accum=8` we expect ~1.5–2 hours — faster than the earlier ~3 h estimate that was based on the heavier HEAD script.

### Amendment 2026-05-26 — Add cell K (canonical Muon) + LR extension for F/I/K
*(Pre-Phase-3; no Phase-3 data yet → free amendment.)*

**Trigger:** Phase 2 results (250 runs across original 10 cells) showed:
1. Cell F (α=0.5, Δ=1, q=5) crushes cell A by 0.22 nats — *partial* NS (q=2) is harmful, full NS dominates.
2. Pre-registered "A > B" (α-blend contributes) was **falsified** (Δ ≈ 0, p=0.53).
3. Pre-registered "A > C" (Δ contributes at q=2) was **falsified** in the opposite direction — C is *better* than A by 0.084 at q=2.
4. **F's and I's best LRs are both at LR=0.08 — the upper edge of our sweep.** True optima may be higher.
5. The original §9 grid has no canonical Muon cell. We cannot test H1 (SOTR > Muon) directly on Phase 2 data without one.

**Change:**

1. New cell **K_muon_canonical** (α=1, Δ=∞, q=5): byte-equivalent to canonical Muon (SOTR's α=1 + Δ=∞ + q=5 limit). Provides the H1 baseline that was missing. 5 seeds × 5 LRs = 25 runs.
2. **LR extension** for cells {F, I, K} at LR ∈ {0.12, 0.16}: 3 cells × 5 seeds × 2 LRs = 30 runs.

Total new compute: 55 runs ≈ 9 hours wall-clock on Fir at Phase-2 scale.

**Rationale:** Without K we cannot honestly evaluate H1 — F's 0.22 win over A may be entirely a function of partial-NS being broken, with F itself only matching or slightly exceeding canonical Muon. The symmetric LR extension across {F, I, K} avoids favoring any cell — if F's true optimum is higher than 0.08, K's might be too.

**New pre-registered prediction (P-K1):** F_best_LR val_loss < K_best_LR val_loss by ≥ 0.02 nats with Holm-significant paired bootstrap. If F ≈ K (CI brackets 0 or |Δ| < 0.02), **H1 is falsified at Phase 2 scale** and Paper 1's SOTR-vs-Muon claim collapses — we either accept the falsification (publish negative result at a smaller venue) or pivot to Paper 2.

**Consequences:**
- §9 grid: 10 → 11 cells. Original 10 cells unchanged.
- LR sweep: 5 → 7 LRs for cells {F, I, K}; still 5 LRs for everyone else.
- §3 H1 becomes directly testable on Phase 2 data.
- §3 H2 (component necessity) is essentially falsified at Phase 2 scale by the existing data — α-drop and partial-NS-vs-full-NS predictions both failed. We will note this in the paper.
- Existing 250-run array indices (0..249) are preserved by appending new cells at the end of `index.txt`. New runs are array indices 250..304 (55 tasks).
- No changes to kill switches, statistical tests, or other pre-registered methods.

### Amendment 2026-05-03 (G/H deferral) — Static α/Δ for first Phase 2 ablation pass
*(Pre-Phase-2; no Phase-2 experimental data yet → free amendment.)*

**Trigger:** During Phase 2 vendoring of `train.py`, α/Δ schedules (cells G and H in §9) were not wired through the training loop. The optimizer step accepts a single static α and Δ; per-step scheduling would require either (a) a callback re-binding the SOTR optimizer's hyperparameters mid-run, or (b) a small scheduler hook adjacent to the LR schedule. Neither is built.

**Change:** For the first Phase 2 ablation pass, cells G and H emit *static* configs (α=0.5, Δ=1.0, q=2 — byte-identical to cell A). The §9 grid still lists 10 cells × 5 seeds × 5 LRs = 250 runs; G and H produce duplicate-of-A data points for the first pass.

**Consequences:**
- **What we still learn from the first pass:** the 8 non-G/H cells answer the core ablation questions (α-blend contribution, Δ-cap contribution, partial-NS contribution, isolated mechanisms via I and J). G/H are about *scheduling*, which is orthogonal to the necessity-of-components question (H2).
- **What we don't learn:** whether α-annealing or late-onset trust regions improve over their static counterparts. Pre-registered prediction "G ≥ A" (§9) is **withdrawn** for the first pass; it will be reinstated in a follow-up amendment if/when the scheduler ships.
- **Statistical bookkeeping:** the Holm-Bonferroni correction in §3 (H2) is applied across the *three* component drops {α, Δ, NS} only — G and H are not part of that family in the first pass, so they don't change the correction count.
- **Cells G/H rerun:** after the scheduler is wired, G and H will be regenerated and rerun (5 seeds × 5 LRs × 2 cells = 50 additional runs). These count as a follow-up sweep, not as part of the original 250.

**Rationale:** Shipping the wired scheduler before any Phase 2 data exists would block the much higher-value 8-cell ablation on a feature that's only relevant if the static-knob results recommend exploring scheduling. Better to run the first pass, see whether α/Δ even matter (cells B vs A, C vs A), and only then invest in the scheduler.

**The generator (`experiments/scripts/gen_phase2_configs.py`) and §9 table both flag G/H with this caveat so future readers see the deferral inline.**

### Amendment 2026-05-02 (UBC cluster) — Switch to UBC compute, drop dollar estimates
*(Pre-Phase-0; no experimental data yet → free amendment.)*

**Change:** §5 hardware tier table updated to assume UBC research computing (Sockeye / DRAC) instead of paid PrimeIntellect rentals. Cost estimates removed (UBC is allocation-based, not dollar-based). SLURM-specific guidance (`docs/CLUSTER.md`) referenced for cluster account, partitions, and job templates.

**Rationale:** User confirmed UBC cluster availability; cloud rentals not needed for the planned scope. Allocation-based bound is tighter on queue time and GPU-hours than on dollars, and the SLURM array pattern (one config per array index) is the cleanest fit for the Phase 2 200-run ablation grid.

**No effect on hypotheses or success criteria.** Hardware tier is "what we run on," not "what we measure" — primary claims are still hardware-matched A-vs-B, not absolute wallclock targets.

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
