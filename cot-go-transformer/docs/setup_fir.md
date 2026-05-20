# Fir cluster setup (Compute Canada)

## One-time setup on the login node

```bash
ssh fir.alliancecan.ca
module load python/3.11 cuda/12 gcc
python -m venv ~/venv-gogpt
source ~/venv-gogpt/bin/activate
pip install --upgrade pip wheel

# Project
cd $SCRATCH
git clone <repo-url> cot-go-transformer
cd cot-go-transformer
pip install -e .[dev]

# FlashAttention-2 (must match the CUDA stack)
pip install flash-attn --no-build-isolation

# wandb (login once, then API key lives in ~/.netrc)
wandb login
```

## Build KataGo (GPU build)

```bash
cd $SCRATCH
git clone https://github.com/lightvector/KataGo.git
cd KataGo/cpp
module load cuda/12 cmake gcc
mkdir build && cd build
cmake .. -DUSE_BACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
make -j 16
# binary is at $SCRATCH/KataGo/cpp/build/katago
```

Download a strong 9x9-capable network from the KataGo releases page and
save its path:

```bash
export KATAGO_BIN=$SCRATCH/KataGo/cpp/build/katago
export KATAGO_MODEL=$SCRATCH/katago-models/g170-b30c320x2-s5824600320-d1736003787.bin.gz
```

Add the two exports to `~/.bashrc` so SLURM jobs inherit them.

## Smoke test before submitting any long job

From a login node, request an interactive 1-GPU session:

```bash
salloc --gres=gpu:h100:1 --cpus-per-task=8 --mem=64G --time=1:00:00 \
    --account=__YOUR_RAPI__
```

Inside the allocation:

```bash
source ~/venv-gogpt/bin/activate
cd $SCRATCH/cot-go-transformer
bash scripts/smoke_test.sh
```

This generates 100 games, trains a 4-layer model, and plays 10 eval
games. Target wall-clock: <1h. If anything errors, fix it before
submitting `train_baseline.slurm`.

## Submitting full jobs

```bash
sbatch scripts/slurm/generate_data.slurm   # ~12h, generates training data
sbatch scripts/slurm/train_baseline.slurm  # 24h, trains 30M baseline
```

Edit `__YOUR_RAPI__` in each SLURM script before submitting.

## Disk discipline

- `$HOME` is small and slow. Don't write data or checkpoints here.
- `$SCRATCH` is fast and large (multi-TB) but purged on a rolling
  schedule. Use it for persistent data and checkpoints.
- `$SLURM_TMPDIR` is node-local NVMe; use it during training for hot
  data. Stage data into it from `$SCRATCH` at job start (the SLURM
  scripts do this with `rsync`).

## Auto-resume on preemption

The training loop checkpoints every `save_every` steps to
`<run_dir>/<run_id>/latest.pt` and the best-by-val to `best.pt`. If a
job is preempted and re-scheduled, `train.py` automatically resumes from
`latest.pt`. To force a fresh run, delete the run directory or change
`run_id` in the config.
