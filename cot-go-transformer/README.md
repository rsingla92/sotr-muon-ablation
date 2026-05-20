# CoT Go Transformer

A research project training a chain-of-thought transformer to play 9×9 Go, then
applying sparse autoencoder interpretability (Transcoders + Lorsa, following
Lin et al. 2026) to compare the model's self-reported reasoning to its actual
internal computation.

**Headline question:** when the model writes reasoning tokens before a move,
do those tokens correspond to the features that mechanistically drive the
move choice?

## Status

| Phase | Goal | Status |
| ----- | ---- | ------ |
| 0 | End-to-end pipeline at toy scale | In progress |
| 1 | Chain-of-thought injection (4-way ablation) | Not started |
| 2 | Test-time compute scaling | Not started |
| 3 | Mechanistic interpretability (Transcoders + Lorsa) | Not started |

## Compute target

One node of Compute Canada's Fir cluster: 4× H100 80GB, 7-day max job
duration, no Anaconda (virtualenv only).

## Layout

```
cot-go-transformer/
├── gogpt/                 # Python package
│   ├── katago.py          # KataGo analysis-engine subprocess wrapper
│   ├── tokenizer.py       # Hybrid prefix-LM vocab (board prefix + trajectory)
│   ├── model.py           # Prefix-LM transformer with bidirectional board attn
│   ├── data.py            # SGF → tensor dataloader
│   ├── train.py           # nanoGPT-style training loop (DDP across 4 GPUs)
│   ├── eval.py            # Match games vs KataGo, Elo estimation
│   └── inference.py       # CoT-aware sampling (filled in Phase 1+)
├── scripts/
│   ├── generate_selfplay.py
│   ├── smoke_test.sh
│   └── slurm/             # SLURM submission templates for Fir
├── configs/
│   ├── smoke.yaml         # 4-layer tiny model, 100 games, single GPU, ~1h
│   └── baseline_30m.yaml  # Phase 0 target: 12 layers, d_model=512, ~30M params
├── tests/                 # Unit tests (CPU-runnable)
├── docs/                  # Write-ups and audits
└── runs/<run_id>/         # Per-run manifests, checkpoints, SGFs
```

## Setup (Fir cluster)

```bash
module load python/3.11 cuda/12 gcc
python -m venv ~/venv-gogpt
source ~/venv-gogpt/bin/activate
pip install --upgrade pip
pip install -e .[dev]
# FlashAttention-2 needs a specific torch ABI; install per Fir's CUDA stack.
pip install flash-attn --no-build-isolation
```

Build KataGo from source (GPU build) and place the binary on `$PATH` as
`katago`. Download a strong 9×9-capable network and set the env var
`KATAGO_MODEL` to its path.

## Smoke test

The smallest viable pipeline: 100 self-play games → 4-layer model trained
1000 steps → 10 match games vs KataGo at 1 visit. Target: under 1 hour on
a single GPU.

```bash
bash scripts/smoke_test.sh
```

## Reproducibility

Every training run writes `runs/<run_id>/manifest.json` containing the git
commit hash, full config dict, dataset version hash, wandb run ID, and final
eval metrics.

## References

- Lin et al. 2026, *Tracing the Thought of a Grandmaster-level Chess-Playing
  Transformer*
- He et al. 2025, *Lorsa: Low-Rank Sparse Attention*
- KataGo analysis engine
  [docs](https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md)
- Leela-SAEs reference implementation
