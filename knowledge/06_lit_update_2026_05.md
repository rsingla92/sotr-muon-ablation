# Literature update — Feb → May 2026

Done 2026-05-02 via web search. Filling the 3-month gap from the user's PDFs.

## New papers since Feb 2026

| Paper | arXiv | Date | Relevance |
|---|---|---|---|
| **Newton-Muon** (Du & Su) | 2604.01472 | Apr 2026 | Muon as implicit Newton step missing the right preconditioner; ~6% fewer iters. **Overlaps Curvature-Aware Muon.** |
| **Mousse** (Curvature-Aware Muon) | 2603.09697 | Mar 2026 | Kronecker-factored Shampoo-style whitening + Muon. ~12% step reduction at 160M–800M. **Subsumes Second-Order Muon.** |
| **MUD** (MomentUm Decorrelation) | 2603.17970 | Mar 2026 | Cholesky-style triangular whitening replacing polar; 10–50% wallclock vs Muon. New non-orthogonal direction. |
| **MuonRec** | 2603.00416 | Mar 2026 | First serious deployment of Muon for generative recommendation. Domain extension. |
| **MuonEq** | 2603.28254 | Mar 2026 | Diagonal row/col equilibration *before* NS — pre-conditioning. |
| **IFNSO** (Iteration-Free NS) | 2602.02500 (v3 Mar 2026) | Mar 2026 | Single learned polynomial replacing iterative NS. **Relevant to FPPP — provides learned-coefficient alternative.** |
| **Muon+** | 2602.21545 | Late Feb 2026 | One additional post-NS normalization. **Closest published "Muon++" — likely what user is conflating with Lions & Muons paper.** |
| **NAMO / NAMO-D** ("Adam Improves Muon") | 2602.17080 | Feb 2026 | Adaptive Adam-style scalar stepsize on orthogonalized momentum. AdaMuon-adjacent. |
| **Delving into Muon and Beyond** | 2602.04669 | Feb 2026 | UΣ^p V^T family for p∈{0, 1/4, 1/2, 1}. **Singular-value-exponent family — *not* Schatten-p, but closest published competitor.** |
| **Gram Newton-Schulz** (Tri Dao blog + code) | tridao.me/blog/2026/gram-newton-schulz/ | 2026 | Hardware-aware NS via symmetric GEMMs. 40–50% NS runtime cut. CuTeDSL kernels for Hopper + Blackwell consumer. **Strongly relevant to FPPP framing.** |
| **Mano** (Oblique manifold) | 2601.23000 | Jan 30 2026 | Beats Muon and AdamW with *less* memory, no spectral preconditioner. Manifold-side competitor. |
| **Manifold Muon** | thinkingmachines.ai/blog/modular-manifolds/ | 2026 | Stiefel/spectral steepest descent angle. |
| **SPEL** | 2601.21487 | Jan 2026 | Spectral steepest descent. |
| **NorMuon** | 2510.05491 | Oct 2025 (missed earlier) | Per-row/neuron normalization after NS. |
| **AdaMuon** (Si et al.) | 2507.11005 | Jul 2025 (missed earlier) | Element-wise second-moment + sign-stab + RMS rescaling on orthogonalized updates. **THIS is the AdaMuon paper.** |
| **Lions and Muons** (Pethick et al.) | 2506.04192 | Jun 2025 (missed earlier) | Heavy-tail-robust Lion/Muon via Stochastic Frank-Wolfe. **Does NOT introduce "Muon++" — user is conflating.** |
| **ROOT** (Robust Orthogonalized Optimizer) | 2511.20626 | Nov 2025 (missed earlier) | Decompose momentum into robust + outlier; soft-threshold outlier; orthogonalize robust part. Per-shape adaptive NS coefficients. **Closest to Robust Muon idea.** |
| **POME** (Post-hoc Muon-style projection) | 2510.06627 | Oct 2025 | Post-training SVD edit on RLHF-finetuned model deltas, +2.5% GSM8K. Not a training optimizer but used in RLHF context. |
| **Effective Quantization of Muon Optimizer States** | 2509.23106 | Sep 2025 | 8-bit quantization of optimizer state (likely INT8, not FP8). Adjacent to FPPP but not a substitute. |
| **What Really Matters in Matrix-Whitening Optimizers** | 2510.25000 | Oct 2025 | Survey/comparison; calls AdaMuon strongest balance of wallclock + final loss. |
| **Trion** (Dion + DCT column selection + NS) | (cited) | 2026 | Extends Dion. |
| **Improved Convergence Rates of Muon** | 2601.19400 | Jan 2026 | Different paper from Kim & Oh's 2601.19156 — verify which one user actually cites. |

## Reaction — how each of his eight ideas looks now

| Idea | Status after lit update | Rec |
|---|---|---|
| **SOTR** | The 3-ingredient combo (partial NS + α-blend + per-matrix Fro cap) is genuinely **open**. Closest neighbors: TrasMuon (global energy clip), NorMuon (post-NS row norm), ROOT (soft outlier split), Mano/Manifold-Muon (manifold). None do α-blend + per-matrix Fro. | **PROCEED** — but reviewers will demand baselines vs Muon+, AdaMuon, ROOT, Mano, NorMuon, not just AdamW/Muon/Lion. Differentiator framing is "softness in the *NS body*" vs all these "softness *outside* NS". |
| **Schatten-p Muon** | Partially overlapped by "Delving into Muon and Beyond" 2602.04669 (UΣ^p family). The strict Schatten-p-norm steepest-descent variant is distinct, but motivation must differ from the σ-exponent family. PolarGrad covers nuclear-scaling. | **DROP unless you can articulate a sharp differentiator from 2602.04669.** Otherwise it'll get desk-rejected as "same family with different parameterization." |
| **Curvature-Aware / Second-Order Muon** | **SUBSUMED.** Mousse (2603.09697) is exactly this with Kronecker factors. Newton-Muon (2604.01472) is the input-Gram version. SOAP+Muon covers it from the SOAP side. | **DROP.** No reasonable differentiator left. |
| **Robust Muon / Muon++** | Largely covered. ROOT (2511.20626) does outlier-thresholding before NS. Muon+ (2602.21545) is the actual published "Muon+". Lions-and-Muons (2506.04192) gives the FW-theoretic robust framing. | **DROP unless framed as a specific extension of ROOT** (e.g., element-wise pre-clip + ROOT's robust split + post-NS Fro normalization, with theory). |
| **AdaMuon** | **PUBLISHED** — Si et al. 2507.11005, ICLR 2026 submission. | **DROP** — already exists. |
| **PSORL (Muon in RL/RLHF)** | Barely populated. OpenRLHF added Muon as engineering, no paper. POME is post-hoc, not training. Kimi K2's MuonClip is for *pretraining*. No published systematic Muon-family-in-RLHF/DPO/GRPO. | **STRONG OPPORTUNITY.** Largest open whitespace. Empirical study showing Muon (or AdaMuon, or SOTR) stabilizing GRPO/DPO would land cleanly. |
| **FPPP (FP8 polar)** | Open. No primary-source FP8 NS paper found. Gram Newton-Schulz (Tri Dao 2026) is the speed-of-light bf16 implementation but not FP8. INT8-quantized optimizer state work exists (2509.23106) but is on state, not the NS iteration. | **OPEN — systems contribution.** Frame as "symmetric-GEMM + FP8" combined story (Gram NS + FP8 stability analysis + fused Triton). Different reviewer pool than algo papers. |
| **SOTR-on-Dion** | Open. Trion and others extend Dion, but no per-matrix Frobenius cap on the low-rank update. | **CHEAP FOLLOW-UP** to SOTR — single ablation once SOTR works. |

## New angles he didn't consider

1. **Manifold-side competitors are now serious** (Mano, Manifold-Muon, SPEL). Any new SOTR paper *must* benchmark against Mano, not just AdamW/Muon. Mano beats Muon with less memory.
2. **Pre-NS / in-NS / post-NS is now the explicit design axis.** Muon+ / NorMuon / AdaMuon are post-NS. MuonEq is pre-NS. MUD replaces NS. SOTR sits *inside* NS (partial) + α-blend — a third axis. Strong rhetorical positioning if framed correctly.
3. **Learned NS coefficients** (IFNSO, ROOT's per-shape adaptive coeffs) emerging as alternative to hand-tuned Chebyshev (CANS). Combine with FP8 → "learned FP8-stable NS polynomial."
4. **Symmetric-GEMM hardware story** (Gram NS) is the new speed-of-light NS. FPPP framed as symmetric-GEMM + FP8 is much stronger than just "FP8 NS."
5. **Domain extension** (MuonRec, long-tail) — Muon-family is generalizing beyond LM pretraining. RL/RLHF is the obvious next vertical, supporting PSORL prioritization.

## Cite fixes

- **"Lions & Muons paper" introducing Muon++:** does not exist. The actual paper is arXiv:2506.04192 (Pethick et al. "Lions and Muons", Stochastic Frank-Wolfe). The published "Muon+" is arXiv:2602.21545. User is conflating.
- **AdaMuon cite:** arXiv:2507.11005 (Si et al., Jul 2025).
- **Lions-and-Muons cite:** arXiv:2506.04192 (Jun 2025).
- **Muon convergence:** verify whether user's cite is arXiv:2601.19156 (Kim & Oh) or arXiv:2601.19400 (different "Improved Convergence Rates of Muon").
- **Polar Express primary source:** arXiv:2505.16932 (Amsel et al., May 2025). Tri Dao's blog references it but isn't the primary source. Gram Newton-Schulz is a separate distinct contribution.
- **AGGC arXiv:2601.11864:** not surfaced; user should re-verify.
- **Turbo-Muon arXiv:2512.04632:** suspicious indexing; might be misindexed or a late-2025 preprint.

## Summary verdict

Of the 8 ideas the user has on the board:
- **2 are genuinely open and worth proceeding:** SOTR (his flagship), PSORL (Muon in RL/RLHF).
- **2 are open and cheap follow-ups:** SOTR-on-Dion, FPPP (as a systems story).
- **4 are subsumed or near-subsumed and should be dropped or sharply refocused:** Schatten-p Muon, Curvature-Aware Muon, Robust Muon, AdaMuon.

The biggest signal from this update: **the optimizer-in-RLHF gap is conspicuous.** Multiple recent papers (POME, Kimi K2, OpenRLHF integration) brush up against it without doing the systematic study. That's where I'd put the second bet after SOTR.
