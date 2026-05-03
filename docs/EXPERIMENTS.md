# Experiments — how to define and run one

Read after `PROTOCOL.md` (the contract) and `CONTRIBUTING.md` (the rules).

## Anatomy of an experiment

An experiment is fully specified by:

1. A **YAML config** in `experiments/configs/<name>.yaml` — every hyperparameter, seed, and hardware tag
2. A **run ID** generated automatically from timestamp + config hash + 6-char salt
3. A **results directory** `results/<run_id>/` containing all outputs

There is no "magic": running the same config on the same hardware with the same seed reproduces results bit-for-bit (modulo CUDA non-determinism flagged in the env).

## YAML config schema

Every config file maps 1:1 to a `TrainConfig` dataclass in `experiments/_configs.py`. Fields:

```yaml
# experiments/configs/example.yaml
name: phase1_repro_muon                    # human-readable; appears in run_id
purpose: phase1                             # one of {sanity, phase1, phase2, phase3, phase4}
description: >
  Reproduce KellerJordan/Muon's published Shakespeare-char loss-to-target
  on single H100. PROTOCOL Phase 1 gate.

# Model
model:
  arch: gpt_modded                          # match modded-nanogpt's train_gpt2.py
  n_layer: 12
  n_embd: 768
  n_head: 6
  vocab_size: 50304

# Data
data:
  dataset: fineweb_edu_10b
  shard: 0..9                               # first 900M tokens (modded-nanogpt convention)
  context_length: 1024
  batch_size_tokens: 524288                 # 512K tokens per batch

# Training
train:
  total_tokens: 5000000000                   # 5B
  warmup_tokens: 256000000                   # 256M
  decay_tokens: 1000000000                   # 1B (trapezoidal schedule)
  precision: bfloat16

# Optimizer
optimizer:
  name: muon                                # one of {adamw, lion, muon, muon_like, sotr}
  hidden_lr: 0.02
  hidden_weight_decay: 0.0
  aux_lr: 0.0018                            # for embeddings/head
  aux_weight_decay: 0.0
  # SOTR-only fields ignored if optimizer.name != sotr
  sotr_alpha: null
  sotr_delta: null
  sotr_ns_iters: null

# Reproducibility
seed: 0
deterministic: true                          # affects torch.use_deterministic_algorithms

# Logging
log:
  log_every_steps: 50
  eval_every_steps: 500
  checkpoint_every_steps: 2000
  wandb: false
```

The dataclass `TrainConfig` validates this on load — missing fields, type mismatches, or invalid combinations (e.g., `optimizer.name == sotr` with no `sotr_alpha`) raise immediately.

## Running an experiment

### Single GPU

```bash
python experiments/train.py --config experiments/configs/phase1_repro_muon.yaml
```

### Multi-GPU (single node)

```bash
torchrun --standalone --nproc_per_node=8 \
    experiments/train.py --config experiments/configs/phase3_sotr_500m.yaml
```

### On the cluster

See `docs/CLUSTER.md`. Submit via:

```bash
sbatch scripts/slurm/single_gpu.sh experiments/configs/phase1_repro_muon.yaml
```

## Output structure

```
results/<run_id>/
├── config.yaml             Exact config used (resolved dataclass → yaml)
├── commit.txt              git rev-parse HEAD at run start
├── env.txt                 nvidia-smi, pip freeze, torch.__config__
├── train.log               Human-readable log (stdout mirror)
├── train.jsonl             One JSON line per logged step: {step, loss, lr, grad_norm, ...}
├── eval.jsonl              Validation loss at each eval step
├── final_metrics.json      Summary: final loss, time, hardware, incident counts
├── stability_incidents.jsonl   Pre-registered incidents (PROTOCOL §8)
└── checkpoints/            Periodic saves (also symlinked to checkpoints/<run_id>/)
```

The run is **fully self-describing** — anyone can read `config.yaml` + `commit.txt` + `env.txt` and know what to run to reproduce.

## Defining the ablation grid (Phase 2)

Per PROTOCOL §9, Phase 2 runs an 8-cell × 5-seed × 5-LR grid = 200 runs. Don't write 200 YAML files by hand.

Pattern:

1. Write a **base config** `experiments/configs/_phase2_base.yaml`
2. Write a **grid generator** `experiments/scripts/gen_phase2_configs.py` that emits all 200 derivative configs
3. Generated configs go in `experiments/configs/phase2/<cell_name>/<seed>_<lr>.yaml` and are gitignored (regenerable)
4. The SLURM array index points at a flat `experiments/configs/phase2/index.txt`

The generator is the source of truth. If you need to modify the grid, modify the generator, regenerate, commit the diff to the generator (not to 200 YAML files).

## Logging conventions

- `train.jsonl`: one JSON object per line, fields documented in `experiments/_logging.py`. Stable schema — adding fields OK, renaming fields breaks downstream tooling.
- Each metric has a units suffix: `loss_nats`, `time_s`, `tokens_per_s`, `lr_step`. No ambiguity about whether "throughput" is tokens/sec or steps/sec.
- Stability incidents (PROTOCOL §8) logged immediately, separately, and to a *different* JSONL so they're trivially greppable: `stability_incidents.jsonl`.

## Hyperparameter sweeps

For per-baseline LR sweeps (PROTOCOL §6 — 5 LRs × 3 seeds = 15 runs per baseline):

```bash
python experiments/scripts/run_lr_sweep.py \
    --base experiments/configs/_phase1_base.yaml \
    --optimizer muon \
    --lrs 0.005,0.01,0.02,0.04,0.08 \
    --seeds 0,1,2 \
    --cluster slurm
```

This generates configs and submits the SLURM array. Output index: `results/sweeps/<sweep_id>/`.

## Result aggregation

Don't aggregate in the train script. Aggregation is a separate offline step:

```bash
python experiments/scripts/aggregate.py results/sweeps/<sweep_id>/ \
    --output results/sweeps/<sweep_id>/summary.csv
```

Aggregation:

- Reads each run's `final_metrics.json` and `stability_incidents.jsonl`
- Computes mean/std/min/max across seeds per (method × LR)
- Reports stability incident rate per condition
- Outputs a CSV that can be loaded into a notebook for plotting (notebook lives outside the repo)

## Pre-experiment checklist

Before running any experiment that will appear in a paper:

- [ ] Sanity gate passes: `make sanity`
- [ ] Config has been linted: load the YAML in Python and validate against the dataclass
- [ ] You're on a clean git tree (no uncommitted changes that would make `commit.txt` misleading)
- [ ] Output destination has space: `results/` on scratch, not on home
- [ ] You've checked your GPU allocation status
- [ ] If this is a re-run: the original `run_id`'s outputs are not at risk of being overwritten (run IDs include a salt to avoid collisions, but be careful)

## Post-experiment checklist

- [ ] `final_metrics.json` exists and looks sane
- [ ] No unexpected stability incidents (or if there were, they're documented in your lab notes / commit message of the next phase)
- [ ] Run committed to results CSV / aggregator
- [ ] If results contradict prior runs, `git diff` between configs to find the cause

## What we don't do

- We don't tune hyperparameters in the train script. Tuning is a sweep, separate.
- We don't auto-resubmit failed runs. Failures get inspected.
- We don't aggregate "the best 3 of 5 seeds" — all seeds are reported (PROTOCOL §13).
- We don't use the test set during training or hyperparameter selection. Validation set only.
