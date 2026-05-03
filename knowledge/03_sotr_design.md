# SOTR — Soft-Orthogonal Trust Region

The user's lead design idea. From `Research Ideas - SOTR Optimizer Design.pdf`.

> **Note (2026-05-02):** This file summarizes the SOTR design as written in the source PDF. The PDF's pseudocode has a step-ordering bug that breaks the "α=1 reduces to Muon" claim. Our implementation in `optimizers/sotr.py` uses the **corrected Muon-compatible ordering** documented in [PROTOCOL.md §7](../PROTOCOL.md) and the §15 amendment. See the "Design correction" section below.

## Motivation

Muon's full orthonormalization is powerful but rigid:
- Discards magnitude information in the gradient
- Computationally costly (5 NS steps in the canonical impl)
- Sensitive to hyperparameters early in training

Goals:
- Enforce orthogonality **softly** (tunable, not exact)
- Preserve **some** gradient magnitude info
- Bound update size via a **per-matrix Frobenius trust region** (different from global grad clipping or Muon's spectral cap)
- Smoothly interpolate between Adam-like normalized-grad behavior and full Muon

## Algorithm (4 steps per 2D weight)

For each 2D weight `W` with gradient `G`:

**Step 1 — Partial orthogonalization**

```
O = NewtonSchulz_p(G)   # p = 1 or 2 iterations (vs Muon's 5)
                        # OR svd_orth(G) = U @ Vᵀ for ablations only
```

**Step 2 — Soft orthogonal blending**

```
U = α · O + (1 − α) · G / (||G||_F + ε)
```

- α ∈ [0, 1] — tunable
- α = 1 → reduces to Muon
- α = 0 → reduces to spectrally-normalized gradient (no orth)
- α = 0.5 starting point; can schedule (e.g., ramp 0 → 0.5 over first 10k steps)

**Step 3 — Frobenius trust region**

```
if ||U||_F > Δ:
    U ← U · Δ / ||U||_F
```

Whole-matrix cap, **not elementwise clipping**. `Δ ≈ 1.0` initially; could scale per-layer.

**Step 4 — Momentum + weight update**

```
m ← β₁ · m + (1 − β₁) · U
W ← W − lr · m   (with optional decoupled weight decay AdamW-style)
```

## Why each piece

- **Partial NS (q=1–2):** Kim & Oh prove convergence error decays doubly-exponentially in q. q=1–2 captures nearly all of full polar's benefit at fraction of cost. Compute: 1–2 mat-muls (vs 5 in Muon) — negligible vs fwd/bwd.
- **Blend with normalized gradient:** retains *some* magnitude info from G. PolarGrad already showed soft spectral interpolation (their ν exponent in `diag(σⁱ^ν)`) helps. SOTR's α blend in matrix space is a different but related softening.
- **Per-matrix Frobenius cap:** novel. Existing trust regions are global (gradient clipping, AdaGC) or per-tensor (AGGC) — but on *Frobenius* norm specifically, per matrix, in update space. Prevents any single weight matrix from making an outsized move.
- **Momentum on the blended update:** PolarGrad analysis suggests orth-then-momentum preserves accumulated direction better than momentum-then-orth. AuON corroborates. SOTR follows this.

## Hyperparameters

| Name | Default | Notes |
|---|---|---|
| `lr` | 1e-3 (or 0.02 for Muon-group) | Same range as AdamW if RMS-matched |
| `α` | 0.5 | Schedule possible: 0→0.5 over first N steps |
| `Δ` | 1.0 | Per-layer scaling: `Δ ∝ √(d_in + d_out)` candidate |
| `ns_iters` | 2 | 1 or 2; 0 disables (pure normalized-grad path) |
| `β₁` | 0.9 (AdamW), 0.95 (Muon-style) | Momentum |
| `weight_decay` | 0 to 0.01 | Decoupled, AdamW-style |
| `block_size` | None | Optional MuonBP-style block-wise NS |

## Limit behavior

- **α = 1, Δ = ∞**: SOTR ≡ Muon (with `ns_iters` instead of 5 by default).
- **α = 0, Δ = ∞, ns_iters = 0**: pure normalized-gradient (closer to L2-renormalized SGD).
- **α = 0, Δ = ∞, ns_iters > 0**: still computes O but discards it — wasted compute, only useful as ablation.
- **α = 1, Δ small**: Muon + global per-matrix step cap. Closest to TrasMuon's idea but per-matrix.

## Position vs related work (from PDF tables)

| Method | Constraint | Hard/Soft | vs SOTR |
|---|---|---|---|
| Muon | spectral norm = 1 | Hard | SOTR adds blend + Fro cap; Muon recovered at α=1, Δ=∞ |
| AuON | spectral norm ≤ 1 via cosh | Soft | Different geometry (spectral vs Frobenius); both partial-orth |
| PolarGrad | nuclear-scaled polar | Hard | SOTR's α blend is matrix-space, PolarGrad's ν is spectral-space |
| Lion | sign(momentum) | Hard, elementwise | SOTR keeps direction with partial orth, not sign |
| AdaGC/AGGC | per-tensor / per-group grad clip | Hard | SOTR's per-matrix Fro cap is in same family but adds orth blending |
| Pethick "Clipped Spectral" (2025) | spectral clip on update | Hard | Closest theoretically; SOTR uses Fro cap + α blend instead |
| MSign | sign on **weights** (singular = 1) | Hard, infrequent | Different target (weights vs updates) |

## Implementation notes (from ChatGPT's drafted code in PDF)

The user's deep-research conversation produced a NanoGPT-ready implementation:

```
optimizers/
  __init__.py
  lion_official.py       # Lion baseline (betas=(0.9, 0.99), decoupled WD)
  sotr.py                # SOTR with muon_newton_schulz polynomial
  muon_like.py           # SOTR with α=1 — apples-to-apples isolation
```

Key code choices baked in:
- `muon_newton_schulz`: uses Muon's tuned coefficients `(3.4445, -4.7750, 2.0315)`, runs in bf16, normalizes by `||X||_F` first, transposes if `m > n` for efficiency.
- `_apply_trust_region`: Fro-norm cap with hit-rate logging.
- Param-group split: Muon-style hidden weights (`transformer.h.*`, ndim≥2) → SOTR; embeddings/head/LayerNorm/biases → AdamW. The optimizer is exposed as a tuple `(opt_hidden, opt_other)` and `train.py` calls `step()`/`zero_grad()` on both.
- Distributed: NanoGPT default DDP via `torchrun`, BF16, optional `torch.compile`.

User's stated baseline preference: **AdamW + Lion + Muon (Keller Jordan repo) + MuonLike (own NS, no aux-Adam) + SOTR**. *Ignore* MuonBP. Compare on NanoGPT (Shakespeare or OpenWebText).

## Validation plan from PDF

**Small-scale speedrun (NanoGPT 10–50M, Shakespeare or OWT subset):**
- Baselines: AdamW, Lion, Muon, MuonLike
- SOTR with α ∈ {0.3, 0.5, 0.7, 1.0}, Δ = 1.0, ns_iters ∈ {1, 2}
- Metrics: steps/time-to-target-loss, trust-region hit rate, occasional singular-value spread on a tracked layer

**Mid-scale (300M–500M, Pile subset):**
- AdamW, Muon, best SOTR config
- Final perplexity, training stability, gradient conditioning stats

**Mandatory ablations:**
- Δ = ∞ (no trust region)
- Fixed α, vary Δ
- α schedule: linear 0 → 0.5 over first N steps
- ns_iters: 1 vs 2
- Distributed run on 8 GPUs via FSDP/DDP if feasible (block-wise NS on local shards)

**Success criteria:** for some α ∈ (0,1), SOTR (a) matches Muon's conditioning benefits, (b) is more stable / faster than AdamW, (c) needs less tuning than Muon.

---

## Design correction (2026-05-02): Muon-compatible step ordering

The PDF's draft pseudocode lists the SOTR step as:

```
G = grad(W)
O = orthogonalize(G)              # NS on raw G
U = α·O + (1-α) · G / ||G||_F     # blend
if ||U||_F > Δ: U *= Δ / ||U||_F  # trust region
m ← β·m + (1-β)·U                 # momentum
W ← W - lr·m
```

**The bug:** at `α=1, Δ=∞, q=5` this produces `m ← β·m + (1-β)·NS(grad)`, then `W -= lr·m`. So the parameter delta is `lr · momentum(NS(grad))`.

**Muon's actual update** (from `external/Muon/muon.py`'s `muon_update`) is:

```
m ← lerp(m, grad, 1-β)                  # buffer update
M = lerp(grad, m, β) if nesterov else m # Nesterov mix
O = zeropower_via_newtonschulz5(M, q)   # NS the mix
O *= max(1, m_dim/n_dim)**0.5           # per-shape RMS scaling
W -= lr · O
```

So Muon's parameter delta is `lr · NS(Nesterov(grad, momentum)) · √(m/n)`. The two are not equivalent — they apply NS at different points in the pipeline (before vs after momentum).

Since "strictly contains Muon at α=1" is the central rhetorical claim of SOTR, we adopt the **Muon-compatible ordering** in our implementation:

```
1. m.lerp_(grad, 1-β)                              # update momentum
2. M = grad.lerp(m, β) if nesterov else m          # Nesterov-mixed value
3. O = zeropower_via_newtonschulz5(M, steps=q)     # NS the mixed value
4. U = α·O + (1-α) · M / (||M||_F + ε)             # SOTR α-blend (against M, not raw G)
5. if ||U||_F > Δ: U *= Δ / ||U||_F                 # SOTR Frobenius cap
6. U *= max(1, m_dim/n_dim)**0.5                    # match Muon's per-shape RMS scale
7. W -= lr·U                                         # weight update; decoupled WD applied separately
```

At `α=1, Δ=∞, q=5`: line 4 collapses to `U = O`; line 5 is a no-op; lines 1–3, 6–7 are byte-equivalent to Muon's `muon_update`. PROTOCOL §7 sanity check #1 verifies this.

**The α-blend interpolates between two well-defined optimizers:**

- `α = 1`: `U = NS(M)` → Muon (with the per-shape scaling)
- `α = 0`: `U = M / ||M||_F` → Frobenius-normalized (Nesterov-)momentum SGD with per-shape scaling. A clean limit, similar in spirit to AuON's "unit-norm momentum" but with Nesterov instead of plain momentum and Frobenius normalization instead of cosh scaling.

**Why blend with `M` rather than raw `G`?** Two reasons:

1. **Continuity at α=0.** Blending with `G` would make the momentum buffer unused at α=0 (the buffer-update happens but its value is never read), creating a discontinuity in the optimizer's behavior as α→0. Blending with `M` keeps the momentum machinery active across all α.
2. **Geometric interpretation.** With `U = α·NS(M) + (1-α)·M/||M||_F`, the blend is between two normalizations of the *same* tensor `M` — its orthogonalized form and its Frobenius-normalized form. This is a clean geometric soft-projection.

This correction has no effect on PROTOCOL §2 hypotheses or §11 success criteria — it's an internal implementation detail. PROTOCOL.md §15 records the amendment for the paper trail.
