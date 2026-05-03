# DRAC cluster — running experiments

Locked target: **Digital Research Alliance of Canada (DRAC)**, formerly Compute Canada. SLURM-managed national HPC. No paid cloud GPUs required.

## Account

| Field | Value |
|---|---|
| **User** | `rsingla` (Rohit Singla) |
| **CCI** | `dim-582` |
| **SLURM account** | `rrg-timsbc` (Tim Salcudean's RAPI group) |
| **Project ID** | `nyq-884-ab` |
| **Status** | Active |

Use `--account=rrg-timsbc` on every `sbatch` and `salloc`. (See also `~/.claude/projects/<project>/memory/drac_account.md` for the durable cross-session record.)

## Cluster choice

DRAC has several GPU-bearing clusters; we pick by phase:

| Cluster | GPUs | Best for |
|---|---|---|
| **Narval** | A100 40GB / A100 80GB SXM (`a100l`) | **Default for Phase 0–3.** A100 80GB is the sweet spot for our 124M speedrun and 300–500M validation. Large allocation pool. |
| **Trillium** | H100 (newest cluster) | Phase 1 reproduction & Phase 3 full-canonical runs *if access available*. Closest to the modded-nanogpt official 8× H100 setup. |
| **Cedar / Béluga** | V100, P100, T4 | Fallback for sanity / Phase 0 dev work if Narval is congested. Not for primary numbers (older GPUs → not apples-to-apples with H100-class records). |
| **Graham** | Mix (P100, V100, T4, A100 nodes) | Available but inconsistent; avoid for headline numbers. |

**Phase-by-phase recommendation:**

| Phase | Cluster | GPUs requested |
|---|---|---|
| Phase 0 sanity / dev | Narval (or any) | 1× A100 |
| Phase 1 reproduction | Narval (or Trillium if available) | 1× A100 80GB |
| Phase 2 ablation grid (250 runs) | Narval | 1× A100 per array job, `%24` concurrent cap |
| Phase 3 mid-scale validation | Narval | 4× A100 80GB single-node |
| Phase 4 release replication | Same as Phase 3 | one rerun for external check |

Cost: **zero** (allocation-based). We are bounded by GPU-hour quota and queue time, not dollars.

## Filesystem layout (DRAC standard)

```
$HOME                                      Small (~50GB). Code clone only. Never put data here.
~/projects/rrg-timsbc/rsingla/             Project space (group-shared, several TB quota)
├── code/optimizer_experiments/            Git working tree (this repo)
└── checkpoints/                           Long-term checkpoints (symlinked from repo)

~/scratch/                                 Scratch (~20TB, periodically purged)
├── optimizer_experiments/venv/            Python environment (rebuild on cluster, not synced)
├── optimizer_experiments/data/            FineWeb shards (~10–50GB)
└── optimizer_experiments/results/         Per-run outputs (large; gitignored, symlinked from repo)

$SLURM_TMPDIR                              Node-local SSD (per-job, fastest IO).
                                            Use for shuffled tokenized batches if data is the bottleneck.
```

`results/`, `checkpoints/`, and `data/` in the repo are symlinks to scratch / project so the train script doesn't need cluster-aware paths.

## First-time cluster setup

```bash
# On a DRAC login node:
cd ~/projects/rrg-timsbc/rsingla/code
git clone --recurse-submodules git@github.com:rsingla92/optimizer_experiments.git
cd optimizer_experiments

# Module stack — DRAC's StdEnv 2023 has Python 3.11/3.12 + CUDA available.
module load StdEnv/2023 python/3.12 cuda/12.6 gcc/12

# Build venv in scratch (HOME is too small for torch + tokens + checkpoints).
mkdir -p ~/scratch/optimizer_experiments
python -m venv ~/scratch/optimizer_experiments/venv
source ~/scratch/optimizer_experiments/venv/bin/activate

# Install our package + reference submodules.
./scripts/setup.sh

# Symlink results / checkpoints / data dirs to scratch.
ln -sf ~/scratch/optimizer_experiments/data data
ln -sf ~/scratch/optimizer_experiments/results results
ln -sf ~/projects/rrg-timsbc/rsingla/checkpoints checkpoints

# Pre-download FineWeb tokens (modded-nanogpt's first 900M).
cd external/modded-nanogpt && python data/cached_fineweb10B.py 9
```

Use `uv` instead of pip if it's available (faster):
```bash
pip install uv  # one-time
uv sync --extra dev   # in lieu of pip install -e ".[dev]"
```

## SLURM job templates

Templates live in `scripts/slurm/`. All use `--account=rrg-timsbc`.

### `scripts/slurm/single_gpu.sh`

```bash
#!/bin/bash
#SBATCH --job-name=optexp
#SBATCH --account=rrg-timsbc
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a100:1                # or a100l:1 for 80GB on Narval
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load StdEnv/2023 python/3.12 cuda/12.6 gcc/12
source ~/scratch/optimizer_experiments/venv/bin/activate

python experiments/train.py --config "$1"
```

Submit:
```bash
sbatch scripts/slurm/single_gpu.sh experiments/configs/phase1_repro_muon.yaml
```

### `scripts/slurm/multi_gpu.sh`

For Phase 3 (4–8× A100 80GB single-node on Narval):

```bash
#!/bin/bash
#SBATCH --job-name=optexp_multi
#SBATCH --account=rrg-timsbc
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100l:4               # Narval A100 80GB SXM
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load StdEnv/2023 python/3.12 cuda/12.6 gcc/12
source ~/scratch/optimizer_experiments/venv/bin/activate

torchrun --standalone --nproc_per_node=4 \
    experiments/train.py --config "$1"
```

### `scripts/slurm/array_ablation.sh`

For Phase 2's 250-run ablation grid (10 cells × 5 seeds × 5 LRs):

```bash
#!/bin/bash
#SBATCH --job-name=ablation
#SBATCH --account=rrg-timsbc
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-249%24                 # 250 jobs, max 24 concurrent
#SBATCH --output=results/slurm/%x-%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load StdEnv/2023 python/3.12 cuda/12.6 gcc/12
source ~/scratch/optimizer_experiments/venv/bin/activate

# Map array index to a generated config file.
CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" experiments/configs/phase2/index.txt)
python experiments/train.py --config "$CONFIG"
```

`experiments/configs/phase2/index.txt` is regenerated by `experiments/scripts/gen_phase2_configs.py` — see PROTOCOL.md §9 for the cell definitions.

## DRAC-specific gotchas

1. **`a100` vs `a100l` on Narval.** `a100` = 40GB PCIe; `a100l` = 80GB SXM. Always use `a100l` for our scale (124M with bf16 + activations easily fits in 40GB but 80GB gives headroom for Phase 3).
2. **`module load StdEnv/2023`** before anything else. Older `StdEnv/2020` doesn't have a current Python.
3. **No internet on compute nodes** by default. Pre-download FineWeb tokens on the login node (`cached_fineweb10B.py`). `pip install` must happen on login nodes.
4. **`$SLURM_TMPDIR` is per-job**, wiped after the job ends. Don't put checkpoints there. Use it for read-only data shards if I/O bandwidth matters.
5. **Time format**: `--time=04:00:00` (HH:MM:SS) or `--time=2-00:00:00` (days-HH:MM:SS). Max wallclock per cluster varies; check `sinfo` or DRAC docs.
6. **Apptainer/Singularity** is available everywhere on DRAC if we want to build a reproducible container off `external/modded-nanogpt/Dockerfile`. Recommend doing this once Phase 1 reproduction is locked.
7. **Submit from project space, not `$HOME`.** `$SLURM_SUBMIT_DIR` becomes the working dir; we want it on a fast, large-quota filesystem.

## Reproducibility within DRAC

Even on a single cluster, results can drift across:
- GPU model (A100-40GB vs A100-80GB vs H100)
- CUDA version
- Driver version
- NCCL build (multi-GPU jobs)

Mitigations:

1. **Record everything per run.** `results/<run_id>/env.txt` captures `nvidia-smi`, `python -V`, `pip freeze`, `torch.__config__.show()`, CUDA version, NCCL version, hostname, SLURM_JOB_ID, SLURM cluster name.
2. **Pin GPU type per comparison.** Within a single A-vs-B comparison (SOTR vs Muon), restrict to one GPU model via `--gres=gpu:a100l:1` (not just `gpu:1`).
3. **Containerize when ready.** DRAC supports Apptainer; build off `external/modded-nanogpt/Dockerfile`.

## What to do when a job fails

- SLURM logs at `results/slurm/<jobname>-<jobid>.err`
- Per-run logs at `results/<run_id>/train.log`
- Checkpoint last saved at `checkpoints/<run_id>/step_<N>.pt`

Resume from latest checkpoint:
```bash
python experiments/train.py --config <config> --resume checkpoints/<run_id>/step_<N>.pt
```

If the failure is a NaN or stability incident: PROTOCOL §8 says it counts as an incident regardless. Do not silently retry. Log the incident and decide explicitly.

## Etiquette

- Don't request more resources than needed. A 124M-param model fits on 1× A100 80GB with margin; don't request 4 GPUs for it.
- Realistic `--time` limits. Jobs that hit the wall waste both allocation and queue priority.
- Job arrays for batch sweeps, not 250 separate `sbatch` calls.
- Check queue health (`squeue -u rsingla`, `sinfo`) before assuming a job is stuck.
- Avoid writing many small files to GPFS (project / scratch parallel filesystems). Aggregate logs into JSONL streams; checkpoint at coarse intervals (every ~1000 steps, not every step).

## Useful one-liners

```bash
# How many GPU-hours have we used this allocation period?
sshare --account=rrg-timsbc --user=rsingla --format=Account,User,RawUsage

# Check pending and running jobs
squeue -u rsingla

# Job efficiency report (after completion)
seff <jobid>

# Cancel everything
scancel -u rsingla

# Look up cluster-wide GPU availability
sinfo -p gpu --Format=Nodes,Gres,GresUsed
```
