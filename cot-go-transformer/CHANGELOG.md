# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Phase 0 scaffold
- Initial project layout
- KataGo analysis-engine subprocess wrapper (`gogpt/katago.py`)
- Hybrid prefix-LM tokenizer for 9×9 Go (`gogpt/tokenizer.py`)
- Prefix-LM transformer with bidirectional board attention (`gogpt/model.py`)
- nanoGPT-style training loop with DDP (`gogpt/train.py`)
- Match-game evaluator vs KataGo (`gogpt/eval.py`)
- Self-play data generator (`scripts/generate_selfplay.py`)
- SLURM templates for Fir
- CPU-runnable unit tests for tokenizer and attention mask

### Pre-Phase-1 (while waiting on cluster setup)
- Go-concept rule library (`gogpt/concepts.py`) -- groups, liberties,
  atari, eye/bamboo/tiger shapes, ladder reading, ko-capture, and
  territory/influence/life-death from KataGo ownership.
- Structured CoT vocab (`gogpt/cot_vocab.py`) -- ~55 think-block tokens
  in the previously-reserved slots.
- Mechanical CoT extractor (`gogpt/cot_extractor.py`) translating
  (board, KataGo analysis) into think-block tokens.
- Offline label pipeline (`scripts/extract_cot_labels.py`) writing
  sharded NPZ training tensors.
- Batched best-of-N sampler stub (`gogpt/inference.py`).
- Board renderers (`gogpt/render.py`) -- ASCII + SVG, dependency-free.
- Dev infra: GitHub Actions CI (CPU + torch jobs), pre-commit config,
  Makefile, ruff-clean across the codebase.
- Phase 1 plan doc (`docs/phase1_plan.md`).
- CPU test suite: 81 passing, 1 xfail (deferred ladder-breaker case).
