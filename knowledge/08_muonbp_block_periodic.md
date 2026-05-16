# MuonBP — Block-Periodic Orthogonalization for Muon

**Paper:** *MuonBP: Faster Muon via Block-Periodic Orthogonalization*
**Authors:** Ahmed Khaled, Kaan Ozkara, Tao Yu, Mingyi Hong, Youngsuk Park
**Venue / date:** arXiv:2510.16981 (cs.LG), Oct 19 2025.

## TL;DR

MuonBP is a **systems / distributed-training** variant of Muon. In tensor-parallel + ZeRO-sharded training, Muon's Newton-Schulz step requires gathering each weight matrix in full before orthogonalization, which adds non-trivial gather/scatter communication. MuonBP replaces this with a hybrid scheme: on most steps, each device runs NS *locally on its own matrix shard* (block-wise orthogonalization, no comm); periodically — every `K` steps — it runs the full, exact NS on the gathered matrix. The authors provide convergence guarantees showing the two regimes need **two distinct step sizes** (block step and full step), and report **+8% throughput** vs Muon on an 8B-parameter training run with TP+ZeRO, **with no degradation** in final loss. It is orthogonal to algorithmic Muon variants — purely a comm-cost intervention.

## Key method

- **Two phases interleaved during training:**
  1. *Block step (cheap, frequent):* each device runs NS on the local shard of `M` only, no all-gather. Update applied locally with step size `η_block`.
  2. *Full step (expensive, infrequent — every `K` steps):* all-gather `M`, run exact NS on the full matrix, scatter, apply update with step size `η_full`.
- **Two step-size design** is theoretically motivated: block-NS is a *biased* approximation of polar (because the local shard's singular values are not the full matrix's singular values), so `η_block` must be tuned smaller / differently. Convergence theorem derived for the resulting algorithm.
- **Block structure:** matches the model-parallel sharding boundary (TP groups + ZeRO partition), so block-NS exactly equals running NS on a contiguous block of rows or columns. No additional partitioning.
- Vanilla NS polynomial (the Muon quintic) is reused inside each block; MuonBP does **not** change the NS coefficients, the number of NS iterations, or the orthogonalization target.

## Key results

- **8B-parameter LM trained with TP + ZeRO-sharded optimizer state:**
  - **+8% end-to-end throughput** vs Muon.
  - **No degradation** in final validation loss / downstream metrics (reported as on-par with Muon baseline).
- Theoretical convergence guarantee for the two-step-size scheme (paper claim — not independently verified here).
- Authors position MuonBP as a **drop-in replacement** for Distributed Muon when comm bandwidth is the bottleneck (e.g., cross-node TP, ZeRO-3, smaller bf16 NS payload).

## Ablations

Reported in abstract / contributions (full grid not extracted, PDF too large for inline fetch):

- Periodicity `K` (how often the full sync happens) — appears as a sensitivity sweep.
- Two-step-size requirement — paper explicitly argues a single step size for both phases is *not* sufficient; this is a load-bearing design choice rather than an optional knob.

## Relation to existing Muon-family work

- **Distributed Muon / Moonlight ZeRO-1 scheme** (see `02_muon_scalability.md`): all-gathers momentum every step before NS. MuonBP **amortizes** this: most steps do not all-gather, only every `K`-th step does. Same destination (orthogonalized update), cheaper path.
- **MuonClip** (Kimi K2 pretraining): orthogonal axis — stability fix for attention logits, not a comm-cost intervention. Composable with MuonBP.
- **Dion**: reduces communication by going **low-rank** (sends `(m+n)·r` instead of `mn`). MuonBP reduces communication by going **temporal-sparse** (full comm only every `K` steps, full rank). These are complementary: one could in principle do block-periodic + low-rank.
- **Gram NS / Polar Express**: reduce the *compute* cost of NS (symmetric GEMMs, better polynomial). MuonBP reduces the *communication* cost. Strictly orthogonal — both speedups should stack.
- **MuonEq / Muon+ / NorMuon / AdaMuon**: all change the algorithm (pre-NS / post-NS scaling). MuonBP changes only when/where NS runs. Should compose cleanly with any of them — block-NS could itself be Muon+ or AdaMuon-flavored.

## Implications for SOTR work

**Almost none — MuonBP is a parallelism story, SOTR is an algorithmic story.** Detail by axis:

- **α-blend, Δ trust region, partial NS (q):** MuonBP does not touch any of these. It uses the standard Muon NS (q=5, standard quintic, full polar). Our three knobs are orthogonal to its block-periodic mechanism. SOTR could in principle run inside a MuonBP framework (block-SOTR + periodic full-SOTR with two step sizes) — but that's a Phase 4+ scaling concern, not a Phase 2 ablation question.
- **Pre-registered hypotheses (PROTOCOL §2):** no effect. H1/H2/H3/H4 are stated in terms of validation loss and stability at fixed wallclock against a Muon baseline. MuonBP would shift the *Muon baseline's wallclock*, but our Phase 1/2 is on 10–50M models on a single node, where the gather/scatter cost MuonBP eliminates is essentially zero. Only relevant at Phase 3 (300–500M, multi-GPU) and only if comm becomes the bottleneck — and even then, both SOTR and Muon would gain symmetrically from a MuonBP wrapper, so the comparison stays fair.
- **Kill switches (PROTOCOL §11):** none triggered. MuonBP does not invalidate Hkill1 (reproduction) or Hkill2 (SOTR(α=1, Δ=∞, q=5) == Muon). The `dd2224b` modded-nanogpt commit baseline is single-node speedrun-style; MuonBP is a non-issue there.
- **Ablation grid (PROTOCOL §9):** **no changes recommended.** Adding a MuonBP cell would conflate algorithmic and systems variables. If we want to claim SOTR scales, we instead validate Phase 3 head-to-head against tuned Muon on the same hardware — and if comm-cost matters at that scale, we can apply MuonBP to both sides as a wrapper (or report it as a Phase 4 systems extension).
- **Baseline choice (dd2224b):** **unchanged.** modded-nanogpt's reference Muon implementation is single-node and does not use MuonBP; matching the published Muon literature requires matching their setup. MuonBP would only become a relevant baseline if/when we scale beyond what the speedrun commit covers, and even then it is *added*, not *substituted*.

**One small flag for the Paper 2 (PSORL) track:** if Muon-in-RLHF ever runs on multi-node TP+ZeRO setups (e.g., 7B+ RLHF rollouts), MuonBP becomes the natural production baseline rather than vanilla Distributed Muon. Worth noting for §15 but not actionable now.

**Net:** MuonBP is good news for Muon's production viability and irrelevant to SOTR's small-scale ablation phase. No protocol amendment needed.
