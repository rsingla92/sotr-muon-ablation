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
