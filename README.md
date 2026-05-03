# optimizer_experiments

Research repository for **Soft-Orthogonal Trust Region (SOTR)** and Muon-family optimizer studies.

## Status

Phase 0 — repository scaffolding and methodology lock-in. No experiments run yet, no optimizer code written yet.

## What this is

Two papers planned (see [`PROTOCOL.md`](PROTOCOL.md) for the contract):

1. **SOTR (Paper 1)** — Partial Newton-Schulz orthogonalization, tunable α-blend with normalized gradient, per-matrix Frobenius trust region. Strictly contains Muon as `α=1, Δ=∞, q=5`.
2. **PSORL (Paper 2)** — Empirical study of Muon-family optimizers in RLHF/DPO/GRPO. Drafted as a protocol amendment after Paper 1 reaches Phase 2.

## Read these in order

1. [`PROTOCOL.md`](PROTOCOL.md) — pre-registered methodology. Hypotheses, success criteria, baselines, ablation grid, statistical tests, decision rules. **Read first** if evaluating this work.
2. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — repo layout and design rationale.
3. [`CONTRIBUTING.md`](CONTRIBUTING.md) — coding and testing standards. Single source of truth for how code is written.
4. [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) — how to define and run an experiment.
5. [`docs/CLUSTER.md`](docs/CLUSTER.md) — DRAC (Compute Canada) cluster specifics: account, modules, SLURM templates.
6. [`docs/PHASE1.md`](docs/PHASE1.md) — Phase 1 reproduction procedure (PROTOCOL §6 gate).
7. [`knowledge/00_index.md`](knowledge/00_index.md) — literature summaries from the source PDFs.

## Quick start

Requires Python ≥ 3.10 and CUDA-capable PyTorch ≥ 2.10.

```bash
git clone --recurse-submodules git@github.com:rsingla92/optimizer_experiments.git
cd optimizer_experiments
./scripts/setup.sh
```

This initializes submodules, creates a Python environment, installs dev deps, and registers pre-commit hooks. After it finishes:

```bash
make sanity     # PROTOCOL §7 gate (once optimizer code lands)
make test       # full test suite
make lint       # ruff check
```

## Layout

```
PROTOCOL.md              Pre-registered methodology
README.md                This file
CONTRIBUTING.md          Coding/testing standards
docs/                    Architecture, cluster, experiment guides
knowledge/               Literature summaries from source PDFs
external/                Pinned reference repos (read-only)
optimizers/              Our optimizer implementations
experiments/             Configs and training entry points
scripts/                 Setup + SLURM templates
tests/                   Sanity (gating) + unit tests
results/, checkpoints/   Run outputs (gitignored)
```

Full breakdown: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Hardware

We target UBC research computing (Sockeye, DRAC). No paid cloud GPUs needed. See [`docs/CLUSTER.md`](docs/CLUSTER.md). The methodology survives any reasonable single-/multi-GPU setup as long as direct A-vs-B comparisons run on identical hardware (PROTOCOL §5).

## License

MIT. See `LICENSE` (added before any external release).

## Contact

Rohit Singla — rsingla@ece.ubc.ca
