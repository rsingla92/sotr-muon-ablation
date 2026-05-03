# Muon scalability — what's known

From the 100-page Muon scaling research PDF.

## Will Muon scale to 20B+ params, 1T+ tokens?

**Yes — strong evidence.**

- 1.5B-param transformer: GPT-2 XL quality in 10h vs 13.3h with AdamW.
- 3B/16B-param **MoE** (Moonlight) trained on **5.7T tokens** with Muon replacing AdamW.
- Same model quality at **~52% of training FLOPs** vs AdamW.
- Per-step overhead for NS: 0.5–0.7% extra FLOPs even at 405B (Llama-style); under 1% in typical LLM settings. NS only operates on each weight matrix's `max(m,n)` dimension, which grows much slower than total params.
- Memory: Muon stores only first-moment buffer → **half the optimizer state of AdamW**. Big win at 20B+.

**Stability fixes needed at scale:**
1. **Weight decay (decoupled, AdamW-style)** — vanilla Muon allows weight norms to drift in long runs.
2. **Per-shape update rescaling** — Muon's theoretical update RMS is `1/√max(m,n)`, so very wide layers get tiny per-element updates and thin layers get huge ones. Multiply by ~`√max(m,n)·0.2` to match AdamW's update RMS, then a single LR works across layer shapes.

## Distributing Newton-Schulz across a GPU cluster

**Yes — feasible, demonstrated.**

The challenge: NS needs the *full* matrix `G`, but in sharded training each GPU only holds a partition.

**ZeRO-1-style Distributed Muon:**
1. Each DP worker computes its local gradient shard, updates its momentum shard.
2. **All-gather** the momentum shards across the DP group → full `M`.
3. Run NS in **bf16** (smaller comm payload).
4. Each worker keeps only its slice of the orthonormalized `O` and applies it locally.

Communication: **~1.25× of distributed AdamW** (worst case). With overlap, close to 1.0–1.1× in practice. NS itself adds ~1–3% to iteration time, hidden by fwd/bwd.

Already integrated in an open-source Megatron-LM PR.

## Pretraining vs. fine-tuning vs. RL

**Pretraining: Muon shines clearly.**

**Fine-tuning: roughly parity with AdamW.** When fine-tuning a 7B on instruction data, Muon gets parity not a clear win.

- Likely reason: Muon's "rare directions" benefit needs a long, rich training process to amortize. Fine-tuning is short and adjusts already-learned features, where Adam's per-coord adaptivity is enough.
- **Optimizer-mismatch effect:** if pretraining was AdamW, switching to Muon for fine-tuning gives little. Vice versa: Muon-pretrained model fine-tuned with AdamW is suboptimal. Weight statistics each optimizer encourages differ.
- Practical advice: use the same optimizer end-to-end if possible.

**RL: untested, plausibly tricky.**

- RL gradients are noisy, sparse, non-i.i.d., often without the low-rank structure NS exploits. Orthonormalizing might amplify noise directions.
- Muon's heavy momentum may struggle when the reward landscape shifts (Adam's adaptivity adjusts faster).
- No published Muon-in-RL work at the time of the user's research.

## Reference quotes

> "for optimal performance, it is more effective to apply Muon during the pretraining phase rather than during supervised fine-tuning"

> "Muon may not provide clear gains for RL or could even be counterproductive without modifications"
