# DRAC cluster, running experiments

**Status:** validated 2026-05-02 on Fir login node. `scripts/setup_drac.sh` runs to completion under `rrg-timsbc`, `make sanity` passes (30 passed, 1 GPU test skipped on login node), and SLURM accepts Phase 1 submission with `--gres=gpu:h100:1` (job 38333414, queued `PD (Priority)`).

Locked target: **Fir** at Simon Fraser University (Digital Research Alliance of Canada). SLURM-managed, 640× NVIDIA H100, AMD Epyc 9454, ~20 PF FP64. Online since late 2025; replaced Cedar at the Cedar Supercomputing Centre. **The H100 hardware matches modded-nanogpt's official speedrun environment exactly**, our Phase 1 reproduction is directly apples-to-apples with the published records.

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
├── code/sotr-muon-ablation/            Git working tree (this repo)
└── checkpoints/                           Long-term checkpoints (symlinked from repo)

~/scratch/                                 Scratch (large quota, periodically purged)
├── sotr-muon-ablation/venv/            Python environment (rebuild on cluster, not synced)
├── sotr-muon-ablation/data/            FineWeb shards (~5–10GB)
└── sotr-muon-ablation/results/         Per-run outputs (large; gitignored, symlinked from repo)

$SLURM_TMPDIR                              Node-local NVMe (per-job, fastest IO).
                                            Use for shuffled tokenized batches if data is the bottleneck.
```

> Run `quota -s` once to see your actual `$HOME` / scratch / project ceilings. The `~/projects/rrg-timsbc/$USER/` directory is auto-provisioned for active rrg-* allocations and is present for `rsingla`. `setup_drac.sh` accepts the repo cloned into either `~/projects/rrg-timsbc/$USER/code/` or `~/scratch/` and routes symlinks accordingly.

`results/`, `checkpoints/`, and `data/` in the repo are symlinks to scratch / project so the train script doesn't need cluster-aware paths.

## First-time cluster setup

`scripts/setup_drac.sh` is the canonical procedure. Run it once on a Fir login node and it handles modules, venv, torch from the DRAC wheelhouse, our package + submodules, symlinks (scratch and project), and the FineWeb token download.

```bash
# On a Fir login node:
git clone --recurse-submodules git@github.com:rsingla92/sotr-muon-ablation.git \
    ~/projects/rrg-timsbc/$USER/code/sotr-muon-ablation
cd ~/projects/rrg-timsbc/$USER/code/sotr-muon-ablation
./scripts/setup_drac.sh
```

The script is idempotent. Re-running it repairs broken symlinks and skips the FineWeb download if shards already exist. First run takes ~5 minutes (mostly the 900M-token download).

For interactive work after setup, the module stack must be reloaded before activating the venv:

```bash
module load StdEnv/2023 python/3.12 cuda/12.6 gcc/12
source ~/scratch/sotr-muon-ablation/venv/bin/activate
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
source ~/scratch/sotr-muon-ablation/venv/bin/activate

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
source ~/scratch/sotr-muon-ablation/venv/bin/activate

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
source ~/scratch/sotr-muon-ablation/venv/bin/activate

# Map array index to a generated config file.
CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" experiments/configs/phase2/index.txt)
python experiments/train.py --config "$CONFIG"
```

`experiments/configs/phase2/index.txt` is regenerated by `experiments/scripts/gen_phase2_configs.py` — see PROTOCOL.md §9 for the cell definitions.

## DRAC-specific gotchas (Fir)

1. **GPU type:** Fir has H100 80GB only. Use `--gres=gpu:h100:1`, confirmed accepted by SLURM (job 38333414 shows `gres/gpu:h` after `squeue -o '%b'`). Multi-GPU `:4` and `:8` syntax follows the same pattern but has not yet been exercised.
2. **`module load StdEnv/2023`** before anything else. The full line `StdEnv/2023 python/3.12 cuda/12.6 gcc/12` resolves cleanly on Fir as of 2026-05.
3. **`setuptools<82` for torch compatibility.** DRAC's `torch 2.11.0+computecanada` declares `setuptools<82` in its metadata. `setup_drac.sh` pins `setuptools<82` explicitly so pip doesn't print resolver warnings (the warnings are harmless but noisy).
4. **No internet on compute nodes.** Pre-download FineWeb tokens on the login node (`cached_fineweb10B.py 9` from inside `external/modded-nanogpt/`). `pip install` must happen on login nodes too.
5. **`$SLURM_TMPDIR` is per-job**, wiped after the job ends. Don't put checkpoints there. Use it for read-only data shards if I/O bandwidth matters.
6. **Time format:** `--time=04:00:00` (HH:MM:SS) or `--time=2-00:00:00` (days-HH:MM:SS). On Fir the documented standard limit for `rrg-*` accounts is 7 days; `scontrol show partition` is the authoritative source. We've only submitted up to `--time=06:00:00` so far.
7. **Apptainer/Singularity** is available on DRAC. We have not built or run a container yet, that's a Phase-3-or-later item. See "Still to verify under load" below.
8. **Submit from project space, not `$HOME`.** `$SLURM_SUBMIT_DIR` becomes the working dir; project space has the large quota.
9. **AMD Epyc CPU host on Fir.** Different from older Intel-host clusters; if any pinned wheel was built only for x86_64-AVX-512-Intel, it may fail. Build from source or use DRAC's `--no-index` wheelhouse.
10. **`arrow` module is *not* needed at the current submodule pin.** Earlier revisions required `module load arrow` because `datasets` was a transitive dep (it ships a "noinstall" stub pyarrow wheel — `pyarrow_noinstall-9999+dummy` — that fails `pip install` on purpose, pointing at `module load arrow`). The 2024-10-29 pin (`dd2224b`) of modded-nanogpt only needs numpy/tqdm/torch/huggingface-hub. Re-add `arrow` to the module-load lines if we ever bump the submodule forward to a `datasets`-using revision. See <https://docs.alliancecan.ca/wiki/Arrow>.

## Reproducibility within DRAC

Even on a single cluster, results can drift across:
- GPU model (A100-40GB vs A100-80GB vs H100)
- CUDA version
- Driver version
- NCCL build (multi-GPU jobs)

Mitigations:

1. **Record everything per run.** `results/<run_id>/env.txt` captures `nvidia-smi`, `python -V`, `pip freeze`, `torch.__config__.show()`, CUDA version, NCCL version, hostname, SLURM_JOB_ID, SLURM cluster name.
2. **Pin GPU type per comparison.** Within a single A-vs-B comparison (SOTR vs Muon), restrict to one GPU model via `--gres=gpu:h100:1` (not just `gpu:1`).
3. **Containerize when ready.** DRAC supports Apptainer; build off `external/modded-nanogpt/Dockerfile`. Tracked under "Still to verify under load".

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

- Don't request more resources than needed. A 124M-param model fits on 1× H100 80GB with margin; don't request 4 GPUs for it.
- Realistic `--time` limits. Jobs that hit the wall waste both allocation and queue priority.
- Job arrays for batch sweeps, not 250 separate `sbatch` calls.
- Check queue health (`squeue -u rsingla`, `sinfo`) before assuming a job is stuck.
- Avoid writing many small files to GPFS (project / scratch parallel filesystems). Aggregate logs into JSONL streams; checkpoint at coarse intervals (every ~1000 steps, not every step).
- Lint runs in GitHub Actions (`.github/workflows/lint.yml`), not as a DRAC pre-commit hook. Pushing from the login node triggers ruff check + format check on CI; see `CONTRIBUTING.md` §"Formatting and linting" for the local workflow.

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

## Confirmed Fir specifics (validated 2026-05-02)

| Item | Confirmed value |
|---|---|
| Account string | `--account=rrg-timsbc` |
| GPU label | `--gres=gpu:h100:1` (single-GPU); SLURM resolves to `gres/gpu:h` in `squeue` output |
| Module stack | `module load StdEnv/2023 python/3.12 cuda/12.6 gcc/12` |
| `setuptools` pin | `setuptools<82` satisfies torch 2.11.0+computecanada metadata |
| Project space | `~/projects/rrg-timsbc/$USER/` is auto-provisioned and writable |
| Repo location | `setup_drac.sh` accepts the repo under `~/projects/rrg-timsbc/$USER/code/` or `~/scratch/`; symlinks are routed to keep results / data in scratch and checkpoints in project |
| Sanity gate | `make sanity` on the login node: 30 passed, 1 GPU test skipped (login nodes have no GPU) |
| Phase 1 ask | `--gres=gpu:h100:1 --time=06:00:00 --cpus-per-task=12 --mem=64G` accepted |

## Still to verify under load

- Phase 1 actually completing (job 38333414 was `PD (Priority)` at last check, not yet `R` or `CD`).
- Multi-GPU GRES syntax (`--gres=gpu:h100:4`, `:8`); only `:1` has been submitted.
- 7-day max wallclock for `rrg-*` accounts; longest submission so far is 6 hours.
- Apptainer container build off `external/modded-nanogpt/Dockerfile`; not attempted yet.
- DRAC `--no-index` torch wheelhouse path on Fir specifically (`setup_drac.sh` falls back to PyPI if `--no-index` fails, so this is silent until inspected).
