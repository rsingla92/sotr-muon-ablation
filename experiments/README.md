# experiments/

Training pipeline + run configs. See [`docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md) for how to define and run an experiment.

## Files

| Path | Purpose |
|---|---|
| `_configs.py` | Typed `RunConfig` dataclass + `OptimizerKind` enum |
| `_run_id.py` | Deterministic run-ID generator |
| `_logging.py` | JSONL logger + PROTOCOL §8 stability-incident detection |
| `train.py` | Training entry point (vendored modded-nanogpt + optimizer dispatcher). Loads a config module via `--config <module.path>` |
| `configs/` | Python config modules — `@dataclass` instances, not YAML |
| `configs/phase2/` | Cell-level dirs. Per-run configs are gitignored (regenerable) |
| `scripts/gen_phase2_configs.py` | Code-as-source-of-truth for the 305-config ablation grid |
| `analysis/phase2_summary.py` | Offline aggregation: paired bootstrap + Holm–Bonferroni + decision-tree |

## Why Python configs, not YAML

- **Type checking.** A `RunConfig` dataclass gives us ruff/mypy coverage. YAML gives us runtime errors.
- **Composition.** `replace(base_config, muon_learning_rate=lr)` beats YAML anchors + overrides for grid expansion.
- **Generator symmetry.** `gen_phase2_configs.py` emits Python that imports the same base config, so generated files parse and type-check with no separate schema.

## Naming convention

`<phase>_<purpose>.py`. Examples:

- `phase1_repro_muon.py` — Phase 1 reproduction
- `_phase2_base.py` — shared Phase 2 base config (leading underscore = shared, not launched directly)
- `phase2/A_sotr_full/seed0_lr0p02.py` — generated per-run config (gitignored)

## Regenerating Phase 2 configs

Per-run modules under `configs/phase2/<cell>/seed<S>_lr<L>.py` are gitignored — the generator is the source of truth. Regenerate any time cell definitions, seeds, or LRs change:

```bash
python -m experiments.scripts.gen_phase2_configs
```
