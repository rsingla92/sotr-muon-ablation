# optimizer_experiments

Research repository for **Soft-Orthogonal Trust Region (SOTR)** and Muon-family optimizer studies.

## Status

Phase 0 — repository scaffolding. No experiments run yet.

## What this is

Two papers planned:

1. **SOTR (Paper 1)** — A new optimizer combining partial Newton-Schulz orthogonalization, a tunable α blend with the normalized gradient, and a per-matrix Frobenius trust region. Strictly contains Muon as the corner case `α=1, Δ=∞, q=5`.
2. **PSORL (Paper 2)** — Empirical study of Muon-family optimizers in RLHF / DPO / GRPO. Drafted as a protocol amendment after Paper 1 reaches Phase 2.

The methodology is pre-registered in [`PROTOCOL.md`](PROTOCOL.md) — read that first if you're evaluating this work.

## Layout

```
optimizer_experiments/
├── PROTOCOL.md            Pre-registered methodology (READ FIRST)
├── README.md              This file
├── CLAUDE.md              Skill routing for Claude Code
├── knowledge/             Literature summaries from source PDFs
│   ├── 00_index.md
│   ├── 01_muon_landscape.md
│   ├── 02_muon_scalability.md
│   ├── 03_sotr_design.md
│   ├── 04_proposals_existing.md
│   ├── 05_open_directions.md
│   └── 06_lit_update_2026_05.md
├── external/              Git submodules of reference repos (see external/README.md)
├── optimizers/            Our optimizer implementations (empty in Phase 0)
├── experiments/           Configs, training scripts
├── scripts/               Repo-level utilities (setup, sanity, etc.)
├── tests/                 Unit + sanity tests (PROTOCOL.md §7)
│   └── sanity/            Limit-case checks; gating tests for Phase 1
├── results/               Run outputs (gitignored)
├── checkpoints/           Model checkpoints (gitignored)
├── data/                  Tokenized corpora (gitignored)
├── pyproject.toml
├── Makefile
└── *.pdf                  Reference PDFs supplied by the author
```

## Setup

Requires Python ≥ 3.10 and CUDA-capable PyTorch ≥ 2.4.

```bash
git clone --recurse-submodules git@github.com:rsingla92/optimizer_experiments.git
cd optimizer_experiments
make setup
```

Or, if you cloned without submodules:

```bash
make submodules
make deps
```

`make setup` initializes submodules under `external/` and installs Python deps via [uv](https://docs.astral.sh/uv/) (preferred) or pip fallback.

## Running sanity checks

Once Phase 0 code lands, the sanity gate (PROTOCOL.md §7) is:

```bash
make sanity
```

This must pass before any Phase 2 result can be reported.

## Reference repositories

We track upstream reference implementations as git submodules under `external/` so every reported number is reproducible against a known commit. See [`external/README.md`](external/README.md) for the list.

## License

MIT. Code released for community reproduction. See `LICENSE` (to be added).

## Contact

Rohit Singla — rsingla@ece.ubc.ca
