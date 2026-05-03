# Open directions — what's actually wide open

Synthesizing the lit reviews and proposals: where is the genuine whitespace?

## Already crowded / done

- **Spectral-norm orthogonalization** (Muon, Dion, PolarGrad, AuON): mature, clear winners exist.
- **Per-tensor / per-group adaptive clipping** (AdaGC, AGGC): mature.
- **Sign-momentum / heavy-tail-robust optimizers** (Lion, Lion++, Muon++): increasingly crowded.
- **Distributed orthogonalized updates** (Dion, Distributed Muon ZeRO-1): solved at the level of "matches centralized."
- **Diagonal Hessian / Sophia**: mature.
- **Stable-rank restoration on weights** (MSign): single-paper but well-justified.
- **Schatten-p generalization concept**: Bernstein & Newhouse and Pethick framed it; user's "Beyond Spectral Norm Muon" would compete with their framings — risky novelty.

## Actually open

### A. SOTR's claimed niche

**Per-matrix Frobenius-norm trust region + tunable α blend** is genuinely under-explored:
- Pethick's "Clipped Spectral" is the closest theoretical neighbor but uses spectral cap.
- TrasMuon does global energy-based clipping, not per-matrix Fro.
- AdaGC/AGGC are adaptive but don't include the orthonormalization blending.

If SOTR + α schedule + per-layer Δ ∝ √(d_in + d_out) demonstrably stabilizes training without sacrificing Muon's speed, that's a real publishable contribution.

### B. Optimizer ↔ fine-tuning / RL gap

**Nobody has shown Muon-family wins in RL** at all. PSORL placeholder is real. Open questions:
- Does Muon orthonormalization amplify policy-gradient noise rather than rare-but-useful directions?
- Does pre-clipping (Robust Muon style) + Muon work for PPO / DPO?
- Optimizer mismatch on transition pretraining → SFT → RLHF: can a Muon-aware fine-tune optimizer (e.g. SOTR with α annealed to 0 over fine-tune) close the gap?

### C. Hardware/precision-aware polar (FPPP placeholder)

- Tri Dao's GramNS already shows hardware-aware NS for H100/B200 in bf16 with `Y·Yᵀ` first.
- **FP8** polar iterations: largely untouched. Polynomial coefficients robust to FP8 quantization noise — open.
- Fused Triton kernels for the full SOTR step (NS + Fro norm + clip + momentum) — open.

### D. Combining ideas

The PDFs treat each proposal independently. Combinations are open:
- **SOTR + Schatten-p**: blend factor α + spectral exponent ν as orthogonal knobs. Two-axis ablation.
- **SOTR + Curvature-Aware**: trust region in Mahalanobis (Hessian-defined) norm, not Frobenius.
- **SOTR + low-rank** (i.e., SOTR-on-Dion): does per-matrix Fro cap help when updates are already rank-r?
- **Adaptive α**: drive α from per-layer gradient statistics (e.g., condition number of `MᵀM`). Auto-tune softness instead of fixed schedule.

### E. Theory gaps

- **Convergence proof for SOTR specifically**: is there a non-asymptotic rate matching Muon's (Kim & Oh, Thrampoulidis) but with the trust region active?
- **Why doubly-exponential NS error decay holds for *blended* orthogonalization** (when α < 1 the iterates aren't pure NS).
- **Generalization implications**: Muon's spectral cap is also implicit regularization. Does Fro cap + α blend trade off worse generalization for better optimization?

### F. Diagnostic infrastructure

- Standardized **trust-region hit-rate logging** per layer per step: nobody has shown the per-layer dynamics across training.
- **Singular-value spectrum tracking** during training: cheap if done sparsely (every 1000 steps on one tracked layer).
- **Update-direction stability** across optimizer step: a metric that captures "how rotating the update is" — would distinguish softness regimes.

## Risks the lit warns about

- **Mixed-precision NS blow-up** in low precision — Tri Dao's GramNS analysis. SOTR likely needs FP32 NS body even in bf16 training.
- **Excessive orthonormalization "washing out" learning rate signals** — overfit α, Δ to small-scale and break at scale.
- **Optimizer mismatch** when chaining pretraining → fine-tuning with different optimizers.
- **Embedding/LayerNorm/head treatment** — Muon explicitly excludes; SOTR should too.
- **Momentum ordering** matters: orth-then-momentum (Muon, AuON, PolarGrad consensus) preferred over momentum-then-orth.
