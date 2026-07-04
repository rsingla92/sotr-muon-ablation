# Experiments — how to define and run one

Read after [`PROTOCOL.md`](../PROTOCOL.md) (the contract) and [`CONTRIBUTING.md`](../CONTRIBUTING.md) (the rules).

## Anatomy of an experiment

An experiment is fully specified by:

1. A **Python config module** — a small file under `experiments/configs/` exposing `config = RunConfig(...)`.
2. A **run ID** generated deterministically from the config + a salt.
3. A **results directory** `results/<phase>/<run_id>/` containing all outputs.

There's no magic: running the same config on the same hardware with the same seed reproduces results bit-for-bit (modulo CUDA non-determinism, which is flagged in `env.txt` if it's on).

## Config schema (Python, not YAML)

Every config file is a Python module that constructs a single `RunConfig` dataclass from `experiments/_configs.py`. Example (`experiments/configs/phase1_repro_muon.py`):

```python
from experiments._configs import OptimizerKind, RunConfig, TrainHParams

config = RunConfig(
    name="phase1_repro_muon",
    purpose="phase1",
    description="Reproduce Muon val_loss on modded-nanogpt at single-H100.",
    hparams=TrainHParams(
        muon_learning_rate=0.02,
        adam_learning_rate=0.0036,
        num_iters=5100,
        batch_size=512,
        # ...
    ),
    seed=0,
    optimizer_name=OptimizerKind.MUON,
)
```

Validation happens in `RunConfig.__post_init__` — missing fields, type mismatches, or invalid combinations (e.g., `optimizer_name=SOTR` with `sotr_alpha=None`) raise immediately. Every config that lands on disk is a config the type checker has already vetted.

## Running an experiment

### Single GPU (local dev)

```bash
torchrun --standalone --nproc_per_node=1 \
    experiments/train.py --config experiments.configs.phase1_repro_muon
```

`torchrun` is required because `experiments/train.py` (vendored from `modded-nanogpt`) calls `dist.init_process_group` at import time.

### On the cluster (DRAC Fir)

See [`CLUSTER.md`](CLUSTER.md) for setup. The single-GPU template wraps the same command:

```bash
sbatch scripts/slurm/single_gpu.sh experiments.configs.phase1_repro_muon
```

The Phase 2 ablation uses a SLURM array, not `single_gpu.sh` — see [`PHASE2.md`](PHASE2.md).

## Output structure

```
results/<phase>/<run_id>/
├── env.txt                     GPU / git / module / pip provenance
├── train.log                   Full stdout mirror
├── train.jsonl                 One JSON row per logged step
├── eval.jsonl                  Validation loss at each eval step
├── final_metrics.json          Summary: final loss, wallclock, hardware, incident counts
├── stability_incidents.jsonl   PROTOCOL §8 incidents (NaN, spike, plateau, …)
└── checkpoints/                Periodic saves (also symlinked to checkpoints/<run_id>/)
```

The run is fully self-describing — anyone can read `env.txt` + the git commit encoded in the run_id and know what to run to reproduce.

## Defining the ablation grid (Phase 2)

Per PROTOCOL §9, Phase 2 runs an 11-cell × 5-seed × 5-LR grid plus a 3-cell × 5-seed × 2-LR extension = **305 runs**. Don't write 305 Python files by hand.

Pattern (already implemented):

1. **Base config** in `experiments/configs/_phase2_base.py` — shared hparams.
2. **Cell definitions** in `experiments/scripts/gen_phase2_configs.py` — one `dict` per cell listing `(name, optimizer_name, sotr_alpha, sotr_delta, sotr_ns_steps)`.
3. **Generator** emits `experiments/configs/phase2/<cell>/seed<S>_lr<L>.py` for every combination + writes the flat index at `experiments/configs/phase2/index.txt`.
4. Generated per-run configs are **gitignored** — the generator is the source of truth. Regenerate on demand:

```bash
python -m experiments.scripts.gen_phase2_configs
# emitted 305 configs to experiments/configs/phase2
# index → experiments/configs/phase2/index.txt (305 entries; expected 305)
```

5. The SLURM array reads `index.txt` and launches one task per line.

If you need to modify the grid — add a cell, change LRs, add seeds — modify the generator, regenerate, commit the generator's diff (not 305 config diffs).

## Logging conventions

- **`train.jsonl`** — one JSON object per line, fields documented in `experiments/_logging.py`. Stable schema — adding fields is safe, renaming breaks downstream tooling.
- **Units in names.** `val_loss_nats`, `time_s`, `tokens_per_s`, `lr_step`. No ambiguity about whether "throughput" is tokens/sec or steps/sec.
- **Stability incidents** (PROTOCOL §8) logged immediately, separately, and to a *different* JSONL so they're trivially greppable: `stability_incidents.jsonl`.

## Result aggregation

Aggregation is offline, not in the train loop. The Phase 2 pipeline:

```bash
python -m experiments.analysis.phase2_summary
```

does:

- Walks `results/slurm/ablation-*.out` for the `[SOTR]` header emitted by `train.py` → recovers `(run_id, cell, seed, lr)` for each task.
- Reads `results/phase2/<run_id>/eval.jsonl` → final `val_loss_nats`.
- Per-cell best-LR selection by median across seeds.
- Paired bootstrap (10k resamples) over per-seed pairs for each pre-registered comparison.
- Holm–Bonferroni correction across the H2 family.
- Emits `_summary_<jobid>.md` (decision-tree narrative) and `_per_run_<jobid>.csv`.

Unit tests: [`tests/unit/test_phase2_analysis.py`](../tests/unit/test_phase2_analysis.py) (17 cases including Holm–Bonferroni step-down, bootstrap null case, and end-to-end `main()` on synthetic data).

## Pre-experiment checklist

Before running any experiment that will appear in a paper:

- [ ] Sanity gate passes: `make sanity`
- [ ] Config imports cleanly: `python -c "from experiments.configs.<path> import config; print(config)"`
- [ ] You're on a clean git tree (no uncommitted changes that would make provenance misleading)
- [ ] Output destination has space: `results/` on scratch, not on home
- [ ] You've checked your GPU allocation status

## Post-experiment checklist

- [ ] `final_metrics.json` exists and looks sane
- [ ] No unexpected `stability_incidents.jsonl` entries (or if there were, they're documented in the next commit's message)
- [ ] Run rolled into `_per_run_*.csv` via `phase2_summary`
- [ ] If results contradict prior runs, `git diff` between configs to find the cause

## What we don't do

- We don't tune hyperparameters inside the train loop. Tuning is a sweep, separate.
- We don't auto-resubmit failed runs. Failures get inspected (bad node? config bug? real instability?).
- We don't aggregate "the best 3 of 5 seeds" — all seeds are reported (PROTOCOL §13).
- We don't use the test set during training or hyperparameter selection. Validation only.
