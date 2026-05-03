# Existing proposals — what ChatGPT already wrote up

From the 100pp Muon scaling PDF. Four proposals are detailed; five are placeholder names.

## 1. Schatten-p Muon (Beyond Spectral Norm)

**Idea:** Muon enforces spectral norm = 1 (Schatten-∞). Generalize to any Schatten-p:
- p = ∞ → Muon (all singular values = 1)
- p = 2 → Frobenius-renormalized SGD (uniform scaling)
- p = 1 → nuclear-norm constraint → low-rank update (sparsifies singular spectrum)
- p ∈ {2, 4, ...} → intermediate "partial flattening"

**Method:** SVD `M = UΣVᵀ`, project `Σ` onto Schatten-p sphere of fixed budget `C`:
```
O_p = U · diag(t_i) · Vᵀ   where (t_1,...,t_r) = proj_to_lp_ball(Σ, p, C)
```
Or: tunable exponent `α ∈ [0,1]` interpolating: `t_i = σ_iᵅ` then renorm.

For p = ∞ closed form is clipping; p = 1 is soft-thresholding; intermediate p uses binary search on Lagrange multiplier or partial-NS approximation.

**Cost:** SVD per step (expensive). Mitigations: partial SVD (top-k), Polar Express polynomial iterations targeting non-uniform spectra, infrequent updates (every 10 steps).

**Validation:** modded-NanoGPT speedrun (126M on FineWeb), 1.3B on 100B tokens, ablate p ∈ {1, 1.5, 2, 4, 8, ∞}.

**Status:** novel; user said "yes" to fleshing out further.

## 2. Adaptive & Hybrid Preconditioned Muon — DECLARED ALREADY-DONE

This was the second proposal in v1, but **the user noted "Adaptive Muon was its own paper"** and treated it as covered (likely AdaMuon or similar). The PDF content describes:

- Per-matrix or row/col Adafactor-style RMS scaling layered on Muon
- Pre- vs post-scaling (preserve orthogonality vs add stability)
- Match global update RMS to AdamW's ~0.3 → reuse same LR

**Status:** treat as published; do not re-propose.

## 3. Second-Order / Curvature-Aware Muon

**Idea:** Make Muon's update curvature-aware by injecting Hessian/Fisher estimates.

**Method (Kronecker-factored, Shampoo-style):**
1. Maintain `A_t = β₂A_{t-1} + (1-β₂)·MᵀM` (n×n) and `B_t = β₂B_{t-1} + (1-β₂)·MMᵀ` (m×m).
2. **Whiten momentum**: `M̃_t = B_t^{-1/4} · M_t · A_t^{-1/4}`.
3. **Polar in whitened space**: `Õ_t = polar(M̃_t)` via NS.
4. **Reproject**: `O_t = B_t^{1/4} · Õ_t · A_t^{1/4}`.
5. Update: `W ← W − η·O_t`.

The `−1/4` powers compose with polar's implicit `(MᵀM)^{-1/2}` to give effective `−1/2` Mahalanobis preconditioning — it's a polar step in the metric defined by `A, B`.

**Alternative:** diagonal Hessian (Sophia-style), scale `O_t` elementwise by `1/√(h_ij + ε)`.

**Cost:** Heavy but tractable — mirror Shampoo's amortization (update inverse roots every k steps). Polar Express polynomials can compute matrix inverse roots cheaply.

**Validation:** ResNet-50 on ImageNet vs AdamW/Shampoo/Muon; GPT-2 medium vs AdamW/AdaMuon/Muon; synthetic quadratic where ideal Newton converges in 1 step.

**Status:** novel as a Muon extension; closely related to existing Shampoo+momentum but the polar-in-whitened-space ordering is the new piece.

## 4. Low-Rank Muon (Communication-Efficient)

**Idea:** Restrict update to rank `r << min(m,n)` using PowerSGD-style power iteration with momentum error feedback. Closely related to **Dion**, but framed as a research extension.

**Method per layer per step:**
1. `B_t = M_{t-1} + G_t` (gradient + momentum buffer).
2. **Power iteration:** `P'_t = B_t · Q_{t-1}` (n×r) → orthonormalize via QR → `P_t` (m×r); compute `R_t = B_tᵀ · P_t` (n×r); `Q_t = ColumnNormalize(R_t)`.
3. **Error feedback:** `M_t = B_t − (1−μ)·P_t·R_tᵀ` (residual stays in momentum).
4. Update: `W ← W − η·(P_t · Q_tᵀ + λ·W)`.

**Communication:** all-reduce `P'_t` (m×r) and `R_t` (n×r) instead of full `B_t` (m×n). For 4096×4096 weights with r=4: 16M floats → ~32k floats, **500× reduction**.

**Theoretical guarantee** (per Dion's Thm A.1): equivalent to centralized full-sync update over time, even when local momentum diverges.

**Validation:** throughput on 8/16/32 GPUs, GPT-2 perplexity vs full Muon at r ∈ {1, 2, 4, 8}, geographically distributed simulation with throttled bandwidth.

**Status:** **already done by Dion at this level of detail.** SOTR-on-top-of-Dion or something more is what the user would need to add.

## 5. Robust Muon (Heavy-Tailed Resilience)

**Idea:** Muon's polar step normalizes spectral norm but doesn't kill outlier elements that dominate top singular vectors. Add element-wise clipping pre-NS + global update normalization post-NS.

**Method:**
1. **Pre-clip:** `Clip_τ(G)` element-wise with adaptive `τ = c · median(|G|)` or running quantile.
2. Momentum: `M = β·M_{t-1} + (1-β)·Clip_τ(G_t)`.
3. Polar: `O = polar(M)`.
4. **Post-normalize:** `D = O + λW`; if `||D||_F > ζ`, scale `D ← D · ζ/||D||_F`.

**Theoretical backing:** Cutkosky & Mehta (2021) — clipping + momentum + normalization gives high-prob convergence under heavy-tailed noise. "Lion++" / "Muon++" (Lions & Muons paper) advocate clipping for orthogonalized optimizers.

**Validation:**
- Synthetic heavy-tail: α-stable noise injection
- CIFAR with 20% label noise
- Small-batch language modeling
- ImageNet/Wikitext-103 to verify no degradation on clean data

**Status:** plausible novel, but Lions & Muons' Muon++ may already cover the core idea; needs lit check.

## 5 Placeholder Landmark Idea Names (no algorithms yet)

The deep-research prompt at the end of the PDF *seeded* these names but did not develop algorithms:

1. **SOTR** — Soft-Orthogonal Trust Region (this is the user's flagship; algorithm exists)
2. **CGPU** — Curvature-Gated Polar Updates: polar direction + cheap curvature gate
3. **SCOO** — State-Compressed Orthogonal Optimizer: compress moments + preserve update geometry
4. **PSORL** — Preference-Stable Orthogonal RL Optimizer: optimizer designed for RLHF/DPO stability
5. **FPPP** — FP8-Friendly Polynomial Polar: precision-aware polar iterations, kernel-efficient

These 5 are blank placeholders; the deep-research prompt explicitly says "the researcher must refine or replace them based on actual literature."

## Tangential proposal in the same PDF

The user also asked ChatGPT for a separate **data valuation for medical imaging** student project proposal (Shapley values, influence functions, OT, persistent homology / TDA). This is a **separate research thread** (PI Tim Salcudean), not part of the optimizer line. References include CheXpert, ISIC, BraTS, MIMIC. Save mentally as "the other thing the user is working on."
