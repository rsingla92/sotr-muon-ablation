# experiments/

Training pipeline + run configs. See `docs/EXPERIMENTS.md` for how to define and run an experiment.

## Files (planned)

| Path | Purpose |
|---|---|
| `_configs.py` | Typed config dataclasses (`TrainConfig`, `OptimizerConfig`, `ModelConfig`, ...) |
| `_run_id.py` | Run-ID generator (timestamp + hash + salt) |
| `_logging.py` | JSONL logger with stable schema |
| `train.py` | Training entry point: `python -m experiments.train --config <yaml>` |
| `eval.py` | Evaluation-only entry point (added when needed) |
| `configs/` | One YAML per actual run — gitted; the source of truth for "what was the config?" |
| `scripts/` | Helper scripts: `gen_phase2_configs.py`, `run_lr_sweep.py`, `aggregate.py` |

## Why YAML configs are gitted

Each YAML is a small, human-readable record of one experiment. Diffs across configs are meaningful (what changed between Phase 1 base and Phase 2 cell A). Generated configs (Phase 2 grid expansion) are gitignored — only the *generator* is committed.

## Naming convention

`<phase>_<purpose>.yaml`. Examples:

- `phase1_repro_muon.yaml`
- `phase1_repro_adamw.yaml`
- `phase2_base.yaml`              (template, generator input)
- `phase3_sotr_500m.yaml`
- `sanity_shakespeare.yaml`       (Phase 0 dev runs)
