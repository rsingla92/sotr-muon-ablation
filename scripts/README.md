# scripts/

Repo-level utilities. Cluster-specific scripts live in `scripts/slurm/`.

## Files

| Path | Purpose |
|---|---|
| `setup.sh` | Local-dev environment setup (any machine with Python + GPU) |
| `setup_drac.sh` | DRAC login-node setup (modules + venv in `$SCRATCH` + FineWeb tokens + generated Phase 2 configs). Idempotent — safe to re-run. |
| `ablation_status.sh` | One-shot progress report for a Phase 2 array (state counts, sample val-losses, failure table, ETA). Auto-detects the most recent `ablation` array; explicit `<jobid>` also works. |
| `slurm/single_gpu.sh` | Generic 1× GPU template (`rrg-timsbc`). Phase 1 and dev runs. Launches via `torchrun --standalone --nproc_per_node=1`. |
| `slurm/multi_gpu.sh` | 4× H100 single-node template (Fir). For Phase 3. |
| `slurm/array_ablation.sh` | Phase 2 SLURM array (reads `experiments/configs/phase2/index.txt`, default `%24` concurrent). Also launches via `torchrun`. |
| `slurm/phase1_modded_nanogpt.sh` | Phase 1 reproduction. Runs upstream `external/modded-nanogpt/train_gpt2.py` at single GPU. No code changes from us. |

## Phase 1 procedure

See [`../docs/PHASE1.md`](../docs/PHASE1.md) for the full procedure. TL;DR:

```bash
# Login node, one-time:
git clone --recurse-submodules git@github.com:rsingla92/sotr-muon-ablation.git \
    ~/projects/rrg-timsbc/$USER/code/sotr-muon-ablation
cd ~/projects/rrg-timsbc/$USER/code/sotr-muon-ablation
./scripts/setup_drac.sh

# Submit the reproduction:
sbatch scripts/slurm/phase1_modded_nanogpt.sh
```

## When *not* to put something here

- One-off experiment scripts → `experiments/scripts/` instead
- Python modules → `optimizers/`, `experiments/`, `tests/`
- Anything that takes config arguments → write a Python entry point and call it from a shell wrapper, don't reimplement argument parsing in bash
