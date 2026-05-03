# UBC cluster — running experiments

This repo is designed to run on UBC's research computing infrastructure (Sockeye, Compute Canada / DRAC, or any SLURM-managed cluster). No paid cloud GPUs required for the planned scope.

> **Status:** Template. Specific partition/queue names need to be filled in once we confirm which UBC cluster account is being used. Where this doc says `<TODO>`, replace with the real value.

## Available clusters at UBC (overview)

The user (Rohit) has access to one or more of:

| Cluster | Institution | Typical GPUs | Notes |
|---|---|---|---|
| **Sockeye** (ARC) | UBC | A100, V100, some H100 | On-campus; PI: Tim Salcudean's group account |
| **DRAC: Vulcan** | Compute Canada | H100 | Newest national cluster |
| **DRAC: Narval** | Compute Canada | A100 | Large national A100 pool |
| **DRAC: Cedar / Béluga / Graham** | Compute Canada | V100 mostly, some A100 | Older but available |

For Paper 1 (SOTR), the practical recommendation:

- **Phase 0–1 (sanity, reproduction):** any single A100 or H100. Sockeye or Narval.
- **Phase 2 ablations (200 reduced-scale runs):** single A100 or H100 per run; submit as a job array. Sockeye or Narval.
- **Phase 3 mid-scale validation:** 4–8× A100 or H100 single-node. Vulcan, Narval-large, or Sockeye GPU partition.

Cost: zero (allocation-based). We are bounded by *queue time and GPU-hour allocation*, not dollars.

## Cluster-specific setup

### Account / allocation

- **PI account:** `<TODO: Salcudean group account ID>`
- **Storage quota:** project space (`/project/...`) for code + checkpoints; scratch (`/scratch/...`) for intermediate data
- **GPU-hour allocation:** `<TODO: yearly allocation>` — track usage with the cluster's `seff` or equivalent

### Module / environment

UBC clusters use Lmod modules. Typical stack for this project:

```bash
module load python/3.12 cuda/12.6 gcc/13
# or whatever's available; see scripts/setup.sh
```

Conda or venv inside `$SCRATCH` (project space has size limits):

```bash
python -m venv $SCRATCH/optimizer_experiments/venv
source $SCRATCH/optimizer_experiments/venv/bin/activate
pip install -e ".[dev,logging]"
```

Use `uv` if available — much faster.

### Filesystem layout

Recommended:

```
$HOME/optimizer_experiments/        Code clone (git working tree)
$SCRATCH/optimizer_experiments/
├── venv/                           Python env (rebuild on cluster, not synced)
├── data/                           FineWeb shards (~2-10 GB)
└── results/                        Per-run outputs (large; gitignored)
$PROJECT/optimizer_experiments/
└── checkpoints/                    Long-term checkpoints
```

`results/` and `checkpoints/` are symlinked from the working tree to scratch/project so the train script doesn't need cluster-aware paths.

## SLURM job templates

Templates live in `scripts/slurm/`. Shape:

### `scripts/slurm/single_gpu.sh`

```bash
#!/bin/bash
#SBATCH --job-name=optexp
#SBATCH --account=<TODO>
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a100:1          # adjust per cluster
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err

set -euo pipefail
cd $SLURM_SUBMIT_DIR

source $SCRATCH/optimizer_experiments/venv/bin/activate
module load cuda/12.6

python experiments/train.py --config "$1"
```

Submit:

```bash
sbatch scripts/slurm/single_gpu.sh experiments/configs/phase1_repro_muon.yaml
```

### `scripts/slurm/multi_gpu.sh`

```bash
#!/bin/bash
#SBATCH --job-name=optexp_multi
#SBATCH --account=<TODO>
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:8          # or a100:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err

set -euo pipefail
cd $SLURM_SUBMIT_DIR

source $SCRATCH/optimizer_experiments/venv/bin/activate
module load cuda/12.6

torchrun --standalone --nproc_per_node=8 \
    experiments/train.py --config "$1"
```

### `scripts/slurm/array_ablation.sh`

For Phase 2's 200-run ablation grid:

```bash
#!/bin/bash
#SBATCH --job-name=ablation
#SBATCH --account=<TODO>
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-199%20            # 200 jobs, max 20 concurrent
#SBATCH --output=results/slurm/%x-%A_%a.out

set -euo pipefail
cd $SLURM_SUBMIT_DIR
source $SCRATCH/optimizer_experiments/venv/bin/activate

# Map array index to a config file
CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" experiments/configs/ablation_index.txt)
python experiments/train.py --config "$CONFIG"
```

`experiments/configs/ablation_index.txt` is generated once and lists the 200 (config, seed) combinations.

## Reproducibility within a cluster

Even on a single cluster, results can drift across:

- GPU model (A100-40GB vs A100-80GB vs H100)
- CUDA version
- Driver version
- NCCL build (multi-GPU jobs)

Mitigations:

1. **Record everything per run.** `results/<run_id>/env.txt` captures `nvidia-smi`, `python -V`, `pip freeze`, `torch.__config__.show()`, CUDA version, NCCL version, hostname.
2. **Pin GPU type per comparison.** Within a single A-vs-B comparison (e.g., SOTR vs Muon), restrict to one GPU model via SLURM's `--constraint` or specific `--gres` flag.
3. **Use containers when feasible.** Sockeye and DRAC both support Apptainer/Singularity; we can build off the modded-nanogpt Dockerfile.

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

- Don't request more resources than you need. A 124M-parameter model fits on 1× A100 with margin; don't request 4 GPUs for it.
- Free GPU-hours by setting realistic `--time` limits. Jobs that hit the wall waste both your allocation and queue priority.
- Use job arrays for batch sweeps, not 200 separate `sbatch` calls.
- Check queue health (`squeue -u $USER`) before assuming a job is "stuck."

## Open `<TODO>` items to fill in once cluster confirmed

- [ ] Account ID for SLURM `--account=`
- [ ] Default partition / `--gres` syntax (cluster-dependent)
- [ ] Module names for python and CUDA
- [ ] Storage quotas (home / scratch / project)
- [ ] Whether singularity/apptainer is available
- [ ] Networked filesystem semantics for checkpoint writes (don't write 1000 files/sec to GPFS)
