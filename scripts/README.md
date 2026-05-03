# scripts/

Repo-level utilities. Cluster-specific scripts live in `scripts/slurm/`.

## Files

| Path | Purpose |
|---|---|
| `setup.sh` | Local-dev environment setup (any machine with Python + GPU) |
| `setup_drac.sh` | DRAC login-node setup (modules + venv in `$SCRATCH` + FineWeb tokens + symlinks). Run once after cloning to a DRAC cluster. |
| `slurm/single_gpu.sh` | Generic 1× GPU template (`rrg-timsbc`). For Phase 0/1 dev runs. |
| `slurm/multi_gpu.sh` | 4× H100 single-node template (Fir). For Phase 3. |
| `slurm/array_ablation.sh` | Phase 2 SLURM array (250 jobs, %24 concurrent). Reads from `experiments/configs/phase2/index.txt`. |
| `slurm/phase1_modded_nanogpt.sh` | Phase 1 reproduction. Runs upstream `external/modded-nanogpt/train_gpt2.py` at single GPU. No code changes from us. |

## Phase 1 procedure

See `docs/PHASE1.md` for the full procedure. TL;DR:

```bash
# Login node, one-time:
git clone --recurse-submodules git@github.com:rsingla92/optimizer_experiments.git \
    ~/projects/rrg-timsbc/$USER/code/optimizer_experiments
cd ~/projects/rrg-timsbc/$USER/code/optimizer_experiments
./scripts/setup_drac.sh

# Submit the reproduction:
sbatch scripts/slurm/phase1_modded_nanogpt.sh
```

## When *not* to put something here

- One-off experiment scripts → `experiments/scripts/` instead
- Python modules → `optimizers/`, `experiments/`, `tests/`
- Anything that takes config arguments → write a Python entry point and call it from a shell wrapper, don't reimplement argument parsing in bash
