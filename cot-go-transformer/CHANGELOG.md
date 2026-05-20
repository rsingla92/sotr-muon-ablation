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

### CoT vocab + extractor review fixes
- TOP_MOVE now binds to the played move (not KataGo's top); tactics and
  shapes are evaluated against the played move.
- Token IDs clustered by category in contiguous blocks; expose
  `CATEGORY_RANGES` for Phase-3 SAE slicing.
- Phase token emitted after grounding facts (not first).
- Confidence switched from visit-count ratio to winrate-gap.
- `extract_cot_labels.py --mode {structured, empty, free}` for the
  A / B / D variants of the central four-way ablation. Loss-mask
  construction pinned with a regression test
  (`tests/test_extract_cot_labels.py`).

### Natural-language CoT rewriter (Phase 1 mode C)
- `gogpt/nl_rewriter.py` -- provider abstraction (Anthropic Claude /
  Google Gemini / Mock), structured-token decoder, retry wrapper.
- `scripts/rewrite_cot_natural.py` -- resumable CLI that samples
  positions from structured-CoT NPZ shards, calls the LLM, and writes
  JSONL. Defaults to Gemini Flash 2.5 (free tier); Anthropic available
  with `--provider anthropic --model claude-haiku-4-5` for ~$4 per 15k
  rewrites.
- `[nl-cot]` extra in `pyproject.toml` pulls `anthropic` and
  `google-genai`.
- `docs/nl_cot.md` documents the tokenization-bridge (custom BPE on
  the rewriter output) as the path to wiring NL-CoT into training.

- CPU test suite: 102 passing, 1 xfail.
