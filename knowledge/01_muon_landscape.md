# Muon-family optimizer landscape

The "matrix-aware" / orthonormalized-update family from late 2024 onward. All operate on **2D weight matrices** (linear/conv reshaped) and do something to the singular spectrum of the update.

## Core method: Muon (Jordan et al., late 2024)

- For each 2D weight, replace SGD-momentum update `M` with `polar(M) ≈ NS(M)` — the closest semi-orthogonal matrix (singular values snapped to 1).
- Polar factor approximated by **Newton–Schulz polynomial iteration** (Muon writeup uses a tuned quintic with coefficients `(3.4445, -4.7750, 2.0315)` and 5 iterations) — no SVD needed, runs in bf16.
- Applied only to hidden 2D weights (`transformer.h.*`); embeddings, head, biases, LayerNorm use AdamW.
- ~2× speedup over AdamW on GPT-style pretraining; ~52% of the FLOPs to match AdamW quality at 3B scale (Moonlight MoE).
- Per-step overhead is tiny (~0.5–0.7% extra FLOPs at 405B scale because hidden width grows slower than total params).
- Memory: only first-moment buffer, no Adam second-moment → ~half of AdamW state.

**Why it works (consensus hypotheses):**
- Conditioning: equalizes the singular values, amplifying "rare directions" the gradient would otherwise underuse.
- Implicit spectral-norm trust region: `||ΔW||₂ ≤ 1`, similar to a hard constraint.
- Removes √rank penalty from SGD's matrix-parameter convergence (Kim & Oh, ICLR 2026).

## Variants

| Method | What's constrained / changed | Hard or soft | Compute | Distinguishing claim |
|---|---|---|---|---|
| **Muon** (Jordan 2024) | Update matrix → semi-orthogonal | Hard polar (NS, q=5) | O(m·n²) for NS | ~2× LLM speedup vs AdamW |
| **Dion** (Ahn 2025, ICLR'26) | Low-rank orthonormal update via amortized power iteration + error feedback | Hard, but rank-r truncated | O(r·m·n) per step + (m+n)·r comm | Matches Muon up to 3B with comm reduced (m+n)r/(mn) |
| **PolarGrad** (Lau 2025) | Polar `Q·norm` family; nuclear-scaling reproduces Muon | Hard exact polar | SVD or NS | Outperforms Adam/Muon; unifies the class |
| **AuON** (Maity 2025) | Spectral norm ≤ 1 via cosh-style elementwise scaling, optional NS | Soft + L2 renorm | O(n²) or linear | Linear-time alternative; matches Muon on speedruns |
| **AdaMuon** (Adaptive Muon, separate paper) | Muon direction × Adam-style RMS scale | Soft, additive | + 1 EMA buffer | Per-parameter scale matching → reuse AdamW LR; treated as already published in user's research |
| **MSign** (Ren 2026) | **Weight matrix** singular values reset to 1 periodically | Hard (matrix sign) | O(m·n²), infrequent | Prevents stable-rank collapse, <7% overhead; trains 5M–3B stably |
| **MuonClip** (EmergentMind 2025) | Muon + weight clipping + QK-logit clip | Hard | Cheap | Specialized fix for attention logit blow-ups |
| **TrasMuon** (Cheng 2026) | Muon + global RMS calibration + energy-based column trust region | Semi-soft (global) | + O(mn) for stats | Faster than Muon, no warmup needed |
| **CANS** (Grishina 2025) | Better NS polynomial coefficients via Chebyshev / Remez | Hard | Same NS structure | Faster orthonormality at fixed mat-mul count |
| **Schatten-p Muon** ("Beyond Spectral") | p-norm ball on update spectrum: p=∞ → Muon, p=1 → low-rank, p=2 → renorm SGD | Soft via p choice | SVD or partial SVD | Tunable from rank-selective to full-orth |
| **Polar Express** | Optimized polynomial iteration for polar; works in bf16 | Hard | NS-style | Faster polar in low precision |
| **Gram NS** (Tri Dao blog) | Hardware-aware NS via `X·Xᵀ` first | Hard | Faster on H100/B200 | Mixed-precision stable |

## Adjacent / supporting work

- **Lion** (Chen 2023): elementwise sign(momentum) update. State-of-the-art LLM baseline. Ben Newhouse "Old Optimizer New Norm" anthology framed Lion as L∞-ball steepest descent and Muon as spectral-norm steepest descent — same Frank-Wolfe lens, different norm.
- **AdaGC** (Wang ICML 2025) / **AGGC** (Chen 2026): adaptive per-tensor or per-module gradient clipping using EMA thresholds — eliminates LLM loss spikes.
- **SWAN** (Ma 2024): stateless preprocessing of gradients via PCA-whitening + normalization — matches Adam with 50% memory reduction.
- **SinkGD** (Scetbon 2025): multi-norm normalization framework; subsumes LAMB/Adafactor.
- **Pethick "Clipped Scion"** (2025): generalizes gradient clipping to non-Euclidean norms; spectral clipping is "Clipped Muon."
- **CondLR** (Vicencio NeurIPS 2023): factorize weights as U·S·Vᵀ, project to bound condition number.
- **Sophia / ADAHessian** (2024): diagonal Hessian preconditioner; ~2× over Adam in GPT training.
- **Shampoo / KFAC**: Kronecker-factored second-order; expensive but powerful.
- **PowerSGD** (Vogels 2019): low-rank gradient compression with error feedback — direct ancestor of Dion's distributed update.
- **Stiefel-manifold optimizers** (Cayley/landing/QR retractions): keep weights orthogonal; exact but heavy.

## Theoretical results

- **Kim & Oh, ICLR 2026** — MuOn convergence with q NS steps: matches exact polar SVD up to a factor decaying *doubly-exponentially* in q. This is why 1–2 NS iterations work in practice. Removes √rank penalty for matrix params.
- **Thrampoulidis et al., 2025** — non-asymptotic rates for MuOn-EMA and MuOn-VR; standard nonconvex rates without dimension penalty.
- Sfyraki et al. (2025) — MuOn with weight decay finds KKT points of a spectral-norm-constrained problem.

## Key practical knobs

- **NS step count `q`**: 1–2 enough in theory, Muon repo uses 5 for headroom.
- **Newton-Schulz polynomial coefficients**: `(3.4445, -4.7750, 2.0315)` is Muon's quintic. CANS shows Chebyshev-optimized variants are better.
- **Update RMS calibration**: scale Muon's update by ~`√max(m,n)·0.2` to match AdamW's typical update RMS (so a single LR works for both groups).
- **bf16 vs fp32 in NS**: bf16 fine for NS body (Polar Express, Gram NS). Watch for blow-ups in low-precision per Tri Dao GramNS analysis.
- **Param groups**: Muon for `transformer.h.*` 2D weights, AdamW for embeddings/head/biases/LayerNorm.
