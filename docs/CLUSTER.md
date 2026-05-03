# DRAC cluster — running experiments

Locked target: **Fir** at Simon Fraser University (Digital Research Alliance of Canada). SLURM-managed, 640× NVIDIA H100, AMD Epyc 9454, ~20 PF FP64. Online since late 2025; replaced Cedar at the Cedar Supercomputing Centre. **The H100 hardware matches modded-nanogpt's official speedrun environment exactly** — our Phase 1 reproduction is directly apples-to-apples with the published records.

## Account

| Field | Value |
|---|---|
| **User** | `rsingla` (Rohit Singla) |
| **CCI** | `dim-582` |
| **SLURM account** | `rrg-timsbc` (Tim Salcudean's RAPI group) |
| **Project ID** | `nyq-884-ab` |
| **Cluster** | Fir (SFU) |
| **Status** | Active |

Use `--account=rrg-timsbc --gres=gpu:h100:N` on every `sbatch` and `salloc`. The durable cross-session record lives in `~/.claude/projects/<project>/memory/drac_account.md`.

## Cluster choice

**Default: Fir.** H100s match modded-nanogpt's published hardware. No need to consider other DRAC clusters for our scope. If Fir is congested or unavailable, fallbacks (in priority order):

| Fallback | GPUs | Caveat |
|---|---|---|
| **Trillium** (U of T) | H100 | Same GPU class; cross-cluster comparisons of headline numbers OK if hardware matches. |
| **Narval** (Calcul Québec) | A100 (`a100l` = 80GB SXM) | Different GPU; flag any cross-cluster comparison explicitly per PROTOCOL §5. |
| **Cedar / Béluga / Graham** | V100, P100 | Older — only for non-headline runs. |

**Phase-by-phase plan on Fir:**

| Phase | Resource ask | Why |
|---|---|---|
| Phase 0 sanity / dev | 1× H100 | Limit-case unit tests (`make sanity`) — actually fast on login node CPU too |
| Phase 1 reproduction | 1× H100, ~6h | Single-GPU modded-nanogpt with `grad_accum=8`. H100 is ~2× faster than A100 → faster turnaround than original Narval estimate |
| Phase 2 ablation grid (250 runs) | 1× H100 per array job, `%24` concurrent | Reduced-scale model; each cell completes in ~30–60 min |
| Phase 3 mid-scale validation | 4× H100 single-node (full canonical config) | Closest reproduction of modded-nanogpt's 8× H100 official record |
| Phase 4 release replication | Same as Phase 3 | One rerun for external check |

Cost: **zero** (allocation-based). Bound is GPU-hour quota and queue time.

## Filesystem layout (Fir / DRAC standard)

```
$HOME                                      Small (~50GB). Code clone only. Never put data here.
~/projects/rrg-timsbc/rsingla/             Project space (group-shared, several TB quota)
├── code/optimizer_experiments/            Git working tree (this repo)
└── checkpoints/                           Long-term checkpoints (symlinked from repo)

~/scratch/                                 Scratch (large quota, periodically purged)
├── optimizer_experiments/venv/            Python environment (rebuild on cluster, not synced)
├── optimizer_experiments/data/            FineWeb shards (~5–10GB)
└── optimizer_experiments/results/         Per-run outputs (large; gitignored, symlinked from repo)

$SLURM_TMPDIR                              Node-local NVMe (per-job, fastest IO).
                                            Use for shuffled tokenized batches if data is the bottleneck.
```

> **Verify on first login:** `quota -s` to confirm $HOME / scratch / project sizes; layout above is the standard DRAC pattern but Fir may differ slightly. If `~/projects/rrg-timsbc/` doesn't exist for your account, contact DRAC support; the rrg-* group dir is auto-provisioned for active allocations.

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
#SBATCH --gres=gpu:h100:1
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

For Phase 3 (4× H100 single-node on Fir):

```bash
#!/bin/bash
#SBATCH --job-name=optexp_multi
#SBATCH --account=rrg-timsbc
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:4
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
#SBATCH --gres=gpu:h100:1
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

## DRAC-specific gotchas (Fir)

1. **GPU type:** Fir has H100 80GB only. Use `--gres=gpu:h100:1` (or `:4`, `:8` for multi-GPU). Verify the exact label with `sinfo --Format=Gres -p gpu` on first login — DRAC sometimes uses suffixes like `h100:80gb` or `h100_80g` depending on cluster generation.
2. **`module load StdEnv/2023`** before anything else. Newer Fir may default to `StdEnv/2025` — if `StdEnv/2023` isn't available, try `StdEnv/2025` and update `setup_drac.sh` accordingly.
3. **No internet on compute nodes.** Pre-download FineWeb tokens on the login node (`cached_fineweb10B.py 9` from inside `external/modded-nanogpt/`). `pip install` must happen on login nodes too.
4. **`$SLURM_TMPDIR` is per-job**, wiped after the job ends. Don't put checkpoints there. Use it for read-only data shards if I/O bandwidth matters.
5. **Time format:** `--time=04:00:00` (HH:MM:SS) or `--time=2-00:00:00` (days-HH:MM:SS). Max wallclock varies; on Fir the standard limit is 7 days for `rrg-*` accounts but check `scontrol show partition` to confirm.
6. **Apptainer/Singularity** is available on DRAC if we want a reproducible container off `external/modded-nanogpt/Dockerfile`. Recommended once Phase 1 reproduction is locked.
7. **Submit from project space, not `$HOME`.** `$SLURM_SUBMIT_DIR` becomes the working dir; project space has the large quota.
8. **AMD Epyc CPU host on Fir.** Different from older Intel-host clusters; if any pinned wheel was built only for x86_64-AVX-512-Intel, it may fail. Build from source or use DRAC's `--no-index` wheelhouse.

## Reproducibility within DRAC

Even on a single cluster, results can drift across:
- GPU model (A100-40GB vs A100-80GB vs H100)
- CUDA version
- Driver version
- NCCL build (multi-GPU jobs)

Mitigations:

1. **Record everything per run.** `results/<run_id>/env.txt` captures `nvidia-smi`, `python -V`, `pip freeze`, `torch.__config__.show()`, CUDA version, NCCL version, hostname, SLURM_JOB_ID, SLURM cluster name.
2. **Pin GPU type per comparison.** Within a single A-vs-B comparison (SOTR vs Muon), restrict to one GPU model via `--gres=gpu:h100:1` (not just `gpu:1`).
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
