# scripts/

Repo-level utilities. Cluster-specific scripts live in `scripts/slurm/`.

## Files (planned)

| Path | Purpose |
|---|---|
| `setup.sh` | One-shot: clone submodules, create venv, `pip install -e ".[dev]"`, install pre-commit hooks |
| `slurm/single_gpu.sh` | SLURM template for 1× GPU runs (sanity, Phase 1, Phase 2 cells) |
| `slurm/multi_gpu.sh` | SLURM template for single-node multi-GPU (Phase 3) |
| `slurm/array_ablation.sh` | SLURM job array for the Phase 2 200-run grid |

See `docs/CLUSTER.md` for SLURM conventions and what to fill in for UBC.

## When *not* to put something here

- One-off experiment scripts → `experiments/scripts/` instead
- Python modules → `optimizers/`, `experiments/`, `tests/`
- Anything that takes config arguments → write a Python entry point and call it from a shell wrapper, don't reimplement argument parsing in bash
