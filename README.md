# sotr-muon-ablation

A pre-registered evaluation of a soft-orthogonal Muon-family optimizer.

**Author.** Rohit Singla, rsingla@ece.ubc.ca, [LinkedIn](https://www.linkedin.com/in/rsingla92/)

## Table of contents

- [Current status](#current-status)
- [How this project came about](#how-this-project-came-about)
- [What this project is](#what-this-project-is)
- [Why explore soft-orthogonal Muon variants](#why-explore-soft-orthogonal-muon-variants)
- [How the evaluation is designed](#how-the-evaluation-is-designed)
- [Results so far](#results-so-far)
- [What is still open](#what-is-still-open)
- [If you are skimming, look at these](#if-you-are-skimming-look-at-these)
- [Repository layout](#repository-layout)
- [How to reproduce](#how-to-reproduce)
- [How this project was built, and how I used AI tools](#how-this-project-was-built-and-how-i-used-ai-tools)
- [License](#license)

## Current status

- Phase 1 reproduction: passed (validation loss 3.29, inside the published Muon band).
- Phase 2 ablation: all 250 runs of the original 10-cell grid are complete.
- Canonical-Muon cell (K) plus an extended learning-rate sweep: currently running on the Fir H100 cluster.
- Phase 3 (mid-scale validation) and Phase 4 (headline reproduction): not started. Both are gated on the canonical-Muon verdict.
- Snapshot date: 3 July 2026.

## How this project came about

I wanted to spend serious time in an area I did not know well. My training is in medicine and biomedical engineering, not optimization theory. The [Muon optimizer](https://github.com/KellerJordan/Muon) and the surrounding literature struck me as a hard, currently active problem where the habits I already use every day (careful trial design, pre-registration, statistical inference) might have something to add. This repository is that exploration, treated as a research exercise rather than a side project, and run under a pre-registered protocol so the eventual answer is credible in whichever direction it lands.

## What this project is

SOTR, short for Soft-Orthogonal Trust Region, is a Muon-family optimizer with three tunable knobs:

- An α parameter in the interval [0, 1] that blends between a Newton-Schulz-orthogonalized momentum (at α = 1) and a Frobenius-normalized momentum (at α = 0).
- A Δ parameter that caps the Frobenius norm of the post-blend update on a per-matrix basis, giving a per-matrix trust region.
- A q parameter that sets the number of Newton-Schulz iterations.

The corner case (α = 1, Δ = ∞, q = 5) reproduces canonical Muon byte-for-byte. This equivalence is enforced as a kill-switch test in `tests/sanity/test_sotr_limits.py`. SOTR is a strict generalization of Muon along these three axes, which lets us ask a concrete question: does adding per-matrix magnitude control (via Δ) and a partial-orthogonalization schedule (via α and q) produce a measurable improvement in training?

Hypotheses were committed before code was written, phase gates and kill switches were defined up front, statistical tests were specified in advance, and every decision rule is dated in the commit log.

## Why explore soft-orthogonal Muon variants

[Muon](https://github.com/KellerJordan/Muon) is the current state-of-the-art result on the [modded-nanogpt speedrun benchmark](https://github.com/KellerJordan/modded-nanogpt), and it was used for pretraining [Kimi K2 at Moonshot AI](https://huggingface.co/moonshotai/Kimi-K2-Instruct). What makes it interesting is the update rule: it orthogonalizes the momentum matrix before applying it, via a Newton-Schulz iteration that projects the momentum onto (approximately) the closest orthogonal matrix. The whole update is then scaled by a single scalar learning rate. There is no direct control over the magnitude of the update matrix by matrix.

Trust-region methods are standard elsewhere in optimization. They anchor policy-gradient reinforcement-learning methods (TRPO, PPO), classical second-order methods (Levenberg-Marquardt, trust-region Newton), and constrained convex optimization. The reason is straightforward. A step direction can be correct while the magnitude is wrong, and controlling magnitude explicitly is often what separates a stable optimizer from an unstable one. Adding a per-matrix Frobenius cap to Muon is a small, mechanistic change that lets us ask whether that principle helps in first-order deep-learning optimizers as well.

The per-matrix Frobenius trust region is the piece of SOTR that is actually new. The α-blend and partial-Newton-Schulz ideas already exist in the literature. See [`knowledge/07_spectral_interpretation.md`](knowledge/07_spectral_interpretation.md) for the honest scope note: the α-blend is a specific parameterization within a singular-value-rescaling family already studied by PolarGrad (Lau, 2025). The empirical contribution being tested here is the interaction map over the three knobs, which has not been published.

## How the evaluation is designed

Everything downstream of section 1 of `PROTOCOL.md` was decided in advance.

Four hypotheses (H1 through H4) were pre-registered, each with numeric success criteria, paired-bootstrap confidence intervals, Holm-Bonferroni family-wise correction over the component-necessity family, and named kill conditions. Four phases were laid out with explicit gates:

1. Reproduce the published Muon validation loss on the modded-nanogpt harness, within a ±5% band.
2. Run an 11-cell by 5-seed by 5-learning-rate ablation at reduced scale (originally 250 runs, expanded to 305 after a mid-project amendment).
3. Validate the best Phase 2 configuration at mid-scale (300 to 500 million parameters) against canonical Muon.
4. Reproduce the modded-nanogpt speedrun protocol at the canonical 8-GPU scale as the headline result.

Two kill switches were named for the failure modes that would invalidate everything downstream. `Hkill1` requires that Phase 1 reproduce the published number (an environment / hardware catch). `Hkill2` requires that SOTR match Muon byte-for-byte at the (α = 1, Δ = ∞, q = 5) corner (an implementation-bug catch). The full analysis pipeline is specified in sections 3, 7, and 9 of `PROTOCOL.md`.

Compute: the Fir partition of the [Digital Research Alliance of Canada](https://alliancecan.ca/en) at Simon Fraser University, which provides the H100 nodes used for every reported number. Harness: modded-nanogpt pinned to commit `dd2224b`, vendored into `experiments/train.py` with a minimal optimizer-dispatch patch.

## Results so far

### Phase 1: reproduction gate passed

Muon on modded-nanogpt reached a validation loss of **3.2911**, comfortably inside the published reference band of [3.12, 3.44]. The gate cleared on the first successful run, after installing the cluster-specific Triton wheel (`3.6.0+computecanada`).

### Phase 2: the 250-run ablation

At reduced scale (1,500 training iterations, batch size 128, five seeds across five learning rates per cell), several pre-registered predictions were not supported by the data:

| Pre-registered prediction | Cells compared | What we observed |
|---|---|---|
| Dropping the α-blend degrades performance (H2). | A vs. B | Not supported. B and A were indistinguishable within seed noise. |
| Dropping the Δ trust region degrades performance at q = 2 (H2). | A vs. C | Not supported, and reversed: C beat A. |
| Partial Newton-Schulz (q = 2) matches full Newton-Schulz (q = 5) at α = 0.5, Δ = 1. | A vs. F | Not supported: F beat A by about 0.22 nats. |
| The α-blend adds no value when Δ is capped and q = 5. | F vs. I | Not supported: F (α = 0.5) beat I (α = 1). |

The best SOTR variant seen so far is cell F (α = 0.5, Δ = 1, q = 5) at learning rate 0.08, reaching validation loss **3.819**. Both F and I peaked at the upper edge of the original learning-rate sweep, which prompted the 2026-05-26 amendment adding learning rates {0.12, 0.16} and a canonical-Muon cell K.

### The canonical-Muon baseline, still running

The original grid lacked a Phase 2 canonical-Muon baseline, so cell K (α = 1, Δ = ∞, q = 5) was added to anchor the primary hypothesis. Preliminary seed-0 results on K:

```
K seed 0, lr 0.005 → 3.7879
K seed 0, lr 0.01  → 3.7743
K seed 0, lr 0.02  → 3.7730   (seed-0 optimum)
K seed 0, lr 0.04  → 3.7917
K seed 0, lr 0.08  → 3.8195
```

K's seed-0 optimum lands about 0.05 nats below F's best of 3.819. If that margin holds after averaging across all five seeds, prediction P-K1 (F beats K by at least 0.02 nats after Holm-Bonferroni correction) will not be supported, and canonical Muon already matches or beats the best SOTR variant tested at this scale. Full-seed verification is running now (job 41671913).

### What I learned so far

I did not see a clean benefit over canonical Muon from the SOTR knobs at the reduced scale we tested, and if the seed-0 pattern on cell K holds across all five seeds I probably will not. A few takeaways regardless:

- Pre-registration paid off. Without it I would have been tempted to reframe each surprising result as a discovery. The commit log locks the original bets in.
- The α-blend and the per-matrix Frobenius cap did not do as much as I expected. Whatever room there is over Muon at this scale probably sits elsewhere, most likely in higher q or in scheduling of the knobs.
- Building a real pre-registered optimizer evaluation is doable in a few weeks of part-time work if you lean on existing infrastructure: a good training harness, an existing Newton-Schulz routine, an academic cluster.
- Working alongside an AI coding tool for research code was a shift in how I worked. I got much faster at going from "what am I actually trying to measure" to runnable, tested code.

## What is still open

- The cell K plus learning-rate-extension resubmit (48 tasks after two bad-node reruns) is currently draining on Fir.
- The decision fork on completion is already spelled out in the protocol:
  - If F beats K by at least 0.02 nats with Holm-Bonferroni significance, proceed to Phase 3 mid-scale (roughly 5 GPU-days).
  - If F and K are indistinguishable, or K beats F, publish this as a pre-registered negative result and pivot to the Paper 2 sketch on Muon-family optimizers in reinforcement learning from human feedback and related alignment training, whose protocol amendment lives in section 15 of `PROTOCOL.md`.
- Phase 3 and Phase 4 SLURM scripts and analysis code (approximately 230 lines total) will land after the canonical-Muon verdict comes in.

## If you are skimming, look at these

Five files worth opening if you want to see the actual work.

1. **[`PROTOCOL.md`](PROTOCOL.md)** (504 lines). The pre-registration itself. Hypotheses, kill switches, phase gates, statistical tests, dated amendments.
2. **[`optimizers/sotr.py`](optimizers/sotr.py)** (225 lines). The optimizer implementation. Muon-byte-compatible at the (α = 1, Δ = ∞, q = 5) corner. The Newton-Schulz polynomial is imported from `external/Muon`, not reimplemented.
3. **[`experiments/analysis/phase2_summary.py`](experiments/analysis/phase2_summary.py)** (483 lines). The analysis pipeline: paired bootstrap with 10,000 resamples, Holm-Bonferroni correction across the component-necessity family, and a decision-tree renderer.
4. **[`tests/unit/test_phase2_analysis.py`](tests/unit/test_phase2_analysis.py)** together with **[`tests/sanity/test_sotr_limits.py`](tests/sanity/test_sotr_limits.py)**. The analysis pipeline and the optimizer both have real unit tests. `test_sotr_limits` is the enforcement of kill switch Hkill2.
5. **[`knowledge/07_spectral_interpretation.md`](knowledge/07_spectral_interpretation.md)**. The honest scope note. Where α-blend sits inside the singular-value-rescaling family, and why the paper's contribution claim was reframed after I noticed that part of what I thought was new had already been published.

## Repository layout

```
PROTOCOL.md                Pre-registered methodology (sections 1 through 16)
LICENSE                    MIT
optimizers/sotr.py         SOTR implementation
experiments/
  train.py                 modded-nanogpt vendored + optimizer dispatch
  configs/                 Config dataclasses (Phase 1 base, Phase 2 grid)
  scripts/gen_phase2_configs.py    Grid generator (single source of truth)
  analysis/phase2_summary.py       Statistics and decision tree
tests/
  sanity/                  PROTOCOL section 7 gates (Muon byte-match, spectral identity, etc.)
  unit/                    Analysis and logging tests
scripts/
  setup.sh, setup_drac.sh  Local and cluster bootstrap
  slurm/                   SLURM templates (single, multi, 8-GPU, array)
  ablation_status.sh       One-shot progress report for a Phase 2 array
knowledge/                 Literature synthesis (index plus 9 notes)
external/                  Pinned reference repos as submodules
docs/                      Architecture, cluster specifics, per-phase procedures
```

A fuller breakdown lives in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## How to reproduce

Requires Python 3.10 or newer and a CUDA-capable install of PyTorch 2.10 or newer.

```bash
git clone --recurse-submodules git@github.com:rsingla92/sotr-muon-ablation.git
cd sotr-muon-ablation
./scripts/setup.sh

make sanity     # PROTOCOL section 7 gates (Muon byte-match, spectral identity, etc.)
make test       # full test suite
make lint       # ruff
```

Per-phase procedures live in [`docs/PHASE1.md`](docs/PHASE1.md), [`docs/PHASE2.md`](docs/PHASE2.md), and [`docs/CLUSTER.md`](docs/CLUSTER.md). Every reported number in this repository was produced on the [Digital Research Alliance of Canada](https://alliancecan.ca/en) Fir cluster at Simon Fraser University; the same protocol should run on any single-node H100 setup with minor changes to the SLURM templates.

## How this project was built, and how I used AI tools

This is a solo research-engineering exercise, and I owned the loop end to end. That meant translating an unfamiliar problem into a pre-registered protocol with numeric gates, choosing the statistical tests, picking the training harness and compute, amending the protocol as results came in, and reading each phase against pre-committed decision rules. Claude Code was my execution collaborator across many sessions, and every artifact went through my review before it landed on `main`.

**I owned:** hypothesis choice (H1 through H4 plus the two kill conditions), success criteria and effect-size thresholds, statistical tests (paired bootstrap plus Holm-Bonferroni over the component-necessity family), the training harness ([modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)), the compute environment (Fir cluster on the Digital Research Alliance of Canada), phase gates and kill switches, every protocol amendment after a surprising result, and the reframed contribution claim after I noticed the α-blend was not cleanly novel.

**Claude Code accelerated:** the SOTR implementation (written from the pre-registered spec, then verified against `tests/sanity/test_sotr_limits.py`), the SLURM plumbing, the config-generation script, the analysis-pipeline scaffolding together with its unit tests, and the literature-review synthesis of two Muon-family papers ([`knowledge/08_muonbp_block_periodic.md`](knowledge/08_muonbp_block_periodic.md) and [`knowledge/09_demo_decoupled_momentum.md`](knowledge/09_demo_decoupled_momentum.md)). Each of those two notes was drafted by a subagent from the source arxiv PDF and reviewed before commit.

The shape of the work is close to what forward-deployed and applied research engineers do inside frontier AI labs: own scope and methodology, drive AI as a working tool inside tight iteration loops, and take responsibility for what ships.

## License

MIT. See [`LICENSE`](LICENSE).
