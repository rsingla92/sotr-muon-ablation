# SOTR — spectral interpretation

The theoretical anchor of the SOTR paper. Shows that SOTR's α-blend acts entirely in *singular-value space*, places it in the well-studied family of singular-value rescaling optimizers (PolarGrad, "Delving into Muon and Beyond"), and clarifies which regime the α-knob is actually interesting in.

## Setup

Let `M` be the (Nesterov-mixed) momentum value at a given training step. Take its SVD:

```
M = U · Σ · Vᵀ            with  Σ = diag(σ_1, …, σ_r),  σ_1 ≥ … ≥ σ_r ≥ 0
```

Newton-Schulz iteration `f_q` (the polynomial Muon uses with `q` steps) operates *element-wise on the singular values* and **preserves U and V** in exact arithmetic:

```
NS(M, q) = U · diag(f_q(σ_1), …, f_q(σ_r)) · Vᵀ        (in exact / fp32 arithmetic)
```

This is a textbook property of NS: the iteration is `X ← aX + b·X·XᵀX + c·X·(XᵀX)²` (Muon's quintic), which only acts on the singular values when written in the SVD basis. As `q → ∞`, `f_q(σ) → 1` for all `σ ∈ (0, ∞)` (the orthogonal polar factor). For the Muon-tuned coefficients `(3.4445, -4.7750, 2.0315)` and `q = 5`, `f_5(σ) ≈ 1` to good approximation across the typical singular-value range.

> **bf16 caveat — empirical, important.** Muon's `zeropower_via_newtonschulz5`
> casts to bfloat16 internally for speed. In bf16, basis preservation holds
> only approximately: writing `O = NS(M, q)` in M's singular basis (i.e.,
> `Uᵀ O V`), the diagonal carries f_q(σ_i) but the off-diagonal portion is
> *not* zero — empirically ~30% of the Frobenius norm at q=5 on a 64×64
> heavy-tailed input. Re-running the same NS in fp32 gives essentially zero
> off-diagonal. The implication: the σ-rescaling identity below is exact in
> fp32 and approximate in bf16. **The blend's linearity in matrix space is
> exact regardless of precision** — what bf16 perturbs is the assertion
> "the blend lives entirely in the diagonal of M's basis." See sanity test #9
> in `tests/sanity/test_spectral_identity.py` for what is actually
> verified numerically.

## The α-blend in singular-value space

SOTR computes:

```
O = NS(M, q)
U_blend = α·O + (1−α)·M / (||M||_F + ε)
```

Substituting the SVD:

```
U_blend = α · U · diag(f_q(σ_i)) · Vᵀ  +  (1−α) · U · diag(σ_i / ||M||_F) · Vᵀ
        = U · diag( α·f_q(σ_i) + (1−α)·σ_i / ||M||_F ) · Vᵀ
```

So **the α-blend leaves the singular vectors `U`, `Vᵀ` untouched and rescales the singular values**. Each singular value is mapped:

```
σ_i  ↦  σ'_i  =  α · f_q(σ_i)  +  (1 − α) · σ_i / ||M||_F             (*)
```

This is the central identity. Everything else in this document follows from it.

## Limit cases

Equation (*) immediately gives:

| Setting | σ'_i | What it is |
|---|---|---|
| `α = 1, q = 5` | `f_5(σ_i) ≈ 1` | **Muon.** All singular values flattened to ≈ 1. |
| `α = 0, any q` | `σ_i / ||M||_F` | **Frobenius-normalized momentum.** Spectrum of `M` preserved, scaled to unit Frobenius norm. The α knob has no effect. |
| `α = 1, q = 0` | `σ_i / ||M||_F` (NS at q=0 returns the input pre-normalized) | Same as above. The q knob has no effect when α=0. |
| `α ∈ (0, 1), q = 5` | `α + (1−α)·σ_i/||M||_F` | **Linear blend in σ-space:** flat spectrum (Muon) shifted toward the normalized M spectrum. |
| `α ∈ (0, 1), q = 1 or 2` | `α·f_q(σ_i) + (1−α)·σ_i/||M||_F` | **Partial-NS regime.** `f_q` is non-trivial polynomial, blend has richer behavior. |

## Where is the α-knob actually interesting?

The α-blend reshapes the singular-value spectrum **only insofar as the spectrum is non-uniform**. We can see this from (*):

- Suppose all `σ_i ≈ s` (uniform spectrum). Then `σ_i / ||M||_F = 1/√r`, a constant across `i`. Equation (*) becomes `σ'_i ≈ α + (1−α)/√r`, also a constant. So `U_blend ≈ c · UVᵀ` for a scalar `c`. **This is just Muon's update scaled by `c` — i.e., a learning-rate change in disguise.**
- Suppose `σ_i` are heavy-tailed (one large σ, many small). Then `σ_i / ||M||_F` varies a lot across `i`, and (*) yields a non-trivial spectrum. The α-blend genuinely changes the *shape* of the update, not just its magnitude.

**Conclusion:** the α-blend is most informative *empirically* when:

1. `q` is small (partial NS), so `f_q(σ_i)` is non-trivial and adds to the heterogeneity, and/or
2. `M` has heavy-tailed singular spectrum, which is common in deep nets (especially in attention layers) but not always pronounced

This sharpens the experimental design: the small-scale ablation (PROTOCOL §9) should pay particular attention to `q = 1` and `q = 2` cells for the α-sweep, and to layers known to have heavy-tailed gradient spectra.

## Family membership: SOTR's place in the literature

Several existing optimizers can be viewed as "singular-value rescaling families" — they all rewrite `M = UΣVᵀ` as `U · g(Σ) · Vᵀ` for some function `g` parameterized by a single scalar:

| Optimizer | Rescaling function `g(σ_i; θ)` | Parameter |
|---|---|---|
| **AdamW** (matrix view) | identity | — (no rescaling) |
| **Muon** (Jordan 2024) | `1` (constant) | — |
| **PolarGrad** (Lau 2025, σ^v) | `σ_i^v / Z` for normalization Z | `v ∈ [0, 1]` |
| **Delving into Muon and Beyond** (2602.04669) | `σ_i^p` with `p ∈ {0, 1/4, 1/2, 1}` | `p` |
| **AuON** (Maity 2025) | `cosh`-based scaling toward unit spectral norm | overhead schedule |
| **MSign** (Ren 2026) | `sign(σ_i) ≡ 1` (applied to weights, not updates) | — |
| **SOTR** (this work) | `α · f_q(σ_i) + (1−α) · σ_i / ||M||_F` | `(α, q) ∈ [0,1] × ℤ_{≥0}` |

PolarGrad and 2602.04669 use **multiplicative-power** rescaling `σ ↦ σ^v`; SOTR uses **additive-linear** rescaling `σ ↦ α·f(σ) + (1−α)·σ̂`. Both interpolate between Muon (all σ → 1) and a normalized version of `M`, but along different curves through σ-space.

This is informative for the paper's positioning. **The α-blend is a parameterization choice within an existing family**, not a new family. Reviewers will (correctly) ask why we chose additive-linear over multiplicative-power. Defensible answers:

1. **No SVD required.** SOTR's blend reuses the NS output that we already compute; PolarGrad's `σ^v` requires an actual SVD. SOTR is therefore cheaper at scale.
2. **Strictly contains Muon as a corner case** at `α = 1` (with `q = 5`). PolarGrad's `v = 0` also gives Muon, so this is parity, not advantage.
3. **Simpler single-parameter knob.** `α` is bounded in [0, 1] and has clear endpoint semantics; `v` is also bounded but the σ^v curve is less intuitive between endpoints.
4. **Composable with the per-matrix Frobenius trust region.** PolarGrad does not propose a trust region; the *combination* of the two soft-projection mechanisms is novel.

## What's actually novel about SOTR

After this analysis, the cleanly novel contributions are:

1. **Per-matrix Frobenius trust region on the post-orthogonalization update.** Not in PolarGrad, not in TrasMuon (global energy clip), not in AdaGC/AGGC (per-tensor on raw gradients), not in Pethick's "Clipped Spectral" (per-matrix but spectral norm not Frobenius). **This is genuinely new.**

2. **The combined-knob study: Frobenius cap × α-blend × partial NS.** The interaction of these three knobs has not been mapped. Even if any one knob individually proves uninteresting, characterizing where they help and where they don't is a contribution.

3. **The spectral derivation above** as a tool for understanding what α actually does. We can draw `σ'(σ)` curves for all members of the family side-by-side and show empirically which curves help on which tasks.

## Implications for the paper's framing

- Lead with the **trust region** as the clean novelty, not the α-blend.
- Frame the α-blend as a *parameterization choice* within a known family, with the design rationale (cheap, no-SVD, contains Muon, composable with trust region).
- The empirical question is: **does the combination of partial-NS + α-blend + Frobenius cap outperform any single member of the family?** This is a meaningful question even if the answer turns out to be "no — pure Muon + Frobenius cap is enough."

## Sanity-check consequence

Equation (*) is verifiable numerically. Add to PROTOCOL §7 sanity check #9: construct a 2D `M` with known SVD, run a SOTR step manually, and check that the singular values of `U_blend` match `α + (1−α)·σ_i/||M||_F` for `q = 5` (where `f_5 ≈ 1`). If this fails, our implementation has a bug that the limit-case tests (#1, #2) can miss — for example, a sign error, wrong normalization, or accidental application of NS to the wrong tensor.

## What this changes for the experiment

The PROTOCOL §9 ablation grid is amended to include:

- **Cell I:** SOTR with `α=1, Δ=1.0, q=5` — "Muon + Frobenius cap only." Isolates the trust region as the sole novel contribution.
- **Cell J:** SOTR with `α=1, Δ=∞, q=2` — "Muon with partial NS." Isolates the partial-NS contribution.

With cells I and J alongside the existing cells A–H, we can decompose SOTR's effect into its three components (partial NS, Frobenius cap, α-blend) and report which actually contribute.
