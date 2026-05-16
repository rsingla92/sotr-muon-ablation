# DeMo — Decoupled Momentum Optimization

**Paper:** *DeMo: Decoupled Momentum Optimization*
**Authors:** Bowen Peng, Lizhang Chen, Baiyu Su, Jeffrey Quesnelle, Diederik P. Kingma, Qiang Liu
**Venue / date:** arXiv:2411.19870 (cs.LG), Nov 29 2024 (v1); revised Feb 2026 (v2).

## TL;DR

DeMo is a **communication-efficient distributed optimizer** that decouples each worker's local momentum state from the global synchronized update. It is positioned as a drop-in replacement for AdamW-DDP / momentum SGD with up to ~85x lower per-step inter-worker bandwidth and convergence on par with — or better than — AdamW-DDP for LLM pretraining at 300M and 1B parameters. The core mechanism is: each worker keeps its **own local momentum buffer**, projects the gradient (and momentum) into a frequency basis via **DCT applied per-tile**, picks the **top-k** highest-energy DCT coefficients to synchronize across workers, and **subtracts** the synced component out of the local momentum (so it is not re-broadcast next step — this is the "decoupling" and acts as built-in error feedback). The unsynced residual stays in the local momentum forever, giving each worker a persistent, divergent local state on top of a globally-agreed low-frequency update. **DeMo is not in the Muon family.** It does not orthogonalize, does not Newton-Schulz, does not touch singular values, and shares only the very loose property "an orthonormal transform appears somewhere" (DCT for compression — not polar/SVD-orthogonalization of the update). The relevance to our work is essentially zero on the algorithm axis; the only contact point is "another optimizer that landed in late-2024 LLM-pretraining benchmarks alongside Muon," and even there the framing (distributed comm) is orthogonal to ours.

## Key method

- **Per-worker local momentum buffer `m_i`** maintained without all-reduce. Standard EMA-style update with decay β.
- **DCT compression of gradient + momentum:** weight matrices are tiled into small blocks (e.g. 32×32-ish); a 2D DCT is applied per tile. This is a *fixed orthonormal basis* — not data-dependent like SVD — so all workers transform into the same frame without coordination.
- **Top-k coefficient selection:** retain only the top-k highest-magnitude DCT coefficients per tile (k a small fraction, e.g. 1–10%). These are the "fast components" that everyone agrees to synchronize.
- **All-reduce on the sparse top-k mask + values only**, in the DCT frame. Sparse pattern + small payload → up to ~85x bandwidth reduction vs full-tensor all-reduce in AdamW-DDP.
- **Decoupling step:** after synchronization, **subtract the synced component out of the local momentum** (in DCT space). The local buffer keeps only the unsynced residual. This is the key trick — without it, the same low-frequency component is re-broadcast every step (wasted bandwidth and unstable behavior). With it, DCT-top-k acts as an *error-feedback compressor*: what is not communicated this step accumulates locally and gets a chance to enter the top-k next step.
- **Parameter update** uses the synchronized global signal (everyone applies the same step in DCT space, inverse-DCT'd back).
- **No orthogonalization of the update matrix.** No NS, no polar factor, no SVD, no spectral cap. No Frobenius cap. No singular-value rescaling of any kind.

## Key results

- **Models:** 300M and 1B-parameter decoder-only LMs (OLMo-style).
- **Baselines:** AdamW-DDP (the standard distributed pretraining recipe).
- **Headline:** up to **85x reduction** in inter-worker bandwidth per step, with **on-par or marginally better** final validation loss vs AdamW-DDP at matched tokens.
- **Topology-agnostic:** because each worker has its own local momentum and only the top-k DCT slice is synced, DeMo tolerates heterogeneous / multi-datacenter / loose-coupling settings that vanilla DDP cannot.
- **Memory:** DeMo's optimizer state is roughly the same as SGD-momentum (one buffer); AdamW carries two (m, v). So DeMo also wins on per-worker optimizer-state memory vs AdamW, though this is a secondary point.
- **Brief comparison to Muon** is mentioned in passing (v2 revision), but Muon is not the primary baseline — AdamW-DDP is. The framing is comm-efficiency, not orthogonalization.

## Ablations

Reported (not exhaustively extracted from PDF):

- **k (top-k fraction):** sensitivity sweep — smaller k → more bandwidth savings, eventually degrades loss. The sweet spot is reportedly in the 1–10% range.
- **Tile size for DCT:** small (32×32 ish) tiles work best; too-large tiles lose the spatial locality benefit, too-small tiles add overhead.
- **Decay β:** essentially standard momentum decay tuning.
- **Decoupling on/off:** ablating the "subtract synced component out of local momentum" step is reported as significantly worse — confirms that this is the load-bearing piece, not just DCT-top-k compression alone.

## Relation to existing Muon-family work

DeMo is **not in the Muon family** by any reasonable definition. The taxonomy:

- **Muon, AdaMuon, Muon+, NorMuon, Mousse, Newton-Muon, Mano, SOTR:** orthogonalize / rescale singular values of the per-matrix update. Compute axis. Single-node or all-gather-then-NS distributed.
- **MuonBP (08_muonbp_block_periodic.md):** parallelism wrapper around Muon — block-periodic all-gather of `M` before NS. Comm-cost axis, but *for an orthogonalization-based optimizer*.
- **Dion:** orthogonalization-based, comm-reduced by going low-rank — sends `(m+n)·r` instead of `mn`. Comm-cost axis *within* the Muon family.
- **DeMo (this paper):** compresses the raw gradient/momentum signal with DCT-top-k + error feedback. Underlying optimizer is closer to SGD-momentum than to Muon. Comm-cost axis, *outside* the Muon family.

The only superficial point of contact: DeMo uses an orthonormal transform (DCT) inside its compression pipeline, and Muon-family optimizers use an orthonormal transform (the polar factor of `M`) inside their update rule. But these are **structurally different uses of "orthonormal"** — DCT is a fixed, data-independent basis used for sparse coding; the polar factor is a data-dependent transform of the singular vectors used to equalize singular values. There is no algorithmic overlap.

DeMo is also distinct from **DeMo's namesake-collisions in the literature** (there are at least two unrelated "DeMo" papers in vision and RL — this one is the Peng/Kingma/Liu 2024 LM pretraining DeMo).

## Implications for SOTR work

**None on the algorithm axis. Possibly relevant to Paper 2 framing (distributed-training context). Not prior art for any SOTR component.** Detail by axis:

- **α-blend, Δ Frobenius trust region, partial-NS q:** DeMo touches none of these. It does not orthogonalize, so an α-blend between an orthogonalized form and a Frobenius-normalized form has no analog in DeMo. It applies error-feedback compression to the gradient/momentum signal, but error-feedback compression is unrelated to per-matrix update soft-projection. **Not prior art for any of SOTR's three knobs.**
- **Pre-registered hypotheses (PROTOCOL §2):** no effect. H1/H2/H3 are stated against tuned Muon and tuned AdamW. DeMo is neither — it is a comm-efficient distributed version of momentum SGD that lands closer to AdamW than to Muon in the algorithm family tree. Adding DeMo as a baseline would be a *separate research question* ("does SOTR beat DeMo on a comm-constrained training setup?"), not part of the Paper 1 ablation.
- **Kill switches (PROTOCOL §11):** none triggered. DeMo does not invalidate Hkill1 (Muon reproduction) or Hkill2 (SOTR(α=1, Δ=∞, q=5) == Muon). The `dd2224b` modded-nanogpt commit pin and the Muon/AdamW baseline list (§6) are unaffected.
- **Ablation grid (PROTOCOL §9):** **no changes recommended.** Adding a DeMo cell would conflate the algorithm axis (does soft-orthogonalization help?) with the systems axis (does comm-efficient distributed training help?). Phase 1/2 is single-node or single-GPU on `dd2224b`; comm efficiency is not the bottleneck we are measuring.
- **Baseline choice (`dd2224b`):** **unchanged.** modded-nanogpt at `dd2224b` uses AdamW + Muon as its baseline pair. DeMo would only become a relevant baseline if/when we explicitly run a multi-node experiment that stresses cross-worker bandwidth — and even then, it would be a *different* paper (or a Paper 2 extension), not Paper 1.
- **Prior art for SOTR?** **No.** The "α-blend in singular-value space" identity in `07_spectral_interpretation.md` requires an SVD-aligned update; DeMo's DCT compression is in a fixed data-independent basis, not the SVD basis of `M`. The "per-matrix Frobenius trust region" is per-matrix and applied to the orthogonalized update; DeMo has no analog. The "partial NS" knob is meaningless without NS, which DeMo does not use. **None of the three SOTR knobs has a DeMo precedent.** No citation precedence concern.

**One small flag for Paper 2 (PSORL).** If RLHF/DPO/GRPO rollouts ever get pushed to a comm-constrained multi-datacenter setup (a realistic 2026+ trend in production RLHF), DeMo becomes a relevant **non-Muon** distributed baseline, alongside AdamW-DDP. Worth noting in §15 of PROTOCOL.md as a future-work pointer, but not actionable for Paper 1.

**One small flag for the literature-review section.** Recent Muon-family papers (NorMuon, Mousse, Mano, MuonBP) all emphasize comm-cost as the next-frontier concern at large scale. DeMo is the canonical **non-orthogonalization** entry in that conversation — useful as a citation when we (in Paper 1's related-work section) explain that comm-cost interventions are orthogonal to our algorithmic contribution. One sentence in related work, not a baseline.

**Net:** DeMo is interesting LLM-pretraining infrastructure and irrelevant to SOTR's algorithmic contribution. No protocol amendment needed. No prior-art concern.
