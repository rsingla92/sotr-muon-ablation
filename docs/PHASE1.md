# Phase 1 — Reproduction

The PROTOCOL §6 reproduction gate. **No new code from us.** We run upstream `external/modded-nanogpt/train_gpt.py` at single-GPU on DRAC and verify our setup reproduces a published Muon number within ±5%.

If this passes, we know:

1. The cluster setup is correct.
2. The Muon implementation we'll be benchmarking against is canonical.
3. Our environment doesn't introduce mysterious drift.

If it fails, **HALT** per PROTOCOL §11 — debugging the infrastructure comes before any Phase 2 work.

## Reproduction target

We're matching what the **Newton-Muon paper** (Du & Su, `arXiv:2604.01472`, April 2026) used as their Phase-1-equivalent baseline:

> "We begin with the short track Record #4 with a single NVIDIA H100 GPU."

That's `external/modded-nanogpt/records/track_1_short/2024-10-10_Muon/eb5659d0-fb6a-49e5-a311-f1f89412f726.txt`, run on a single H100 instead of 8× via `--nproc_per_node=1`. The published 8× H100 record reaches **≤3.28 cross-entropy on FineWeb validation** in ~22 minutes; on a single GPU with `grad_accum=8`, expected wallclock is **~2.5–3.5 hours** depending on whether we get an A100 80GB SXM (`a100l`) or 40GB PCIe (`a100`).

**Our gate:** final FineWeb validation loss within ±5% of 3.28 → loss between 3.12 and 3.44.

## Procedure

### One-time setup (login node)

```bash
# Choose a DRAC cluster. Narval recommended (large A100 pool).
ssh narval.alliancecan.ca

# Clone into project space (NOT $HOME — too small for this work).
mkdir -p ~/projects/rrg-timsbc/$USER/code
cd ~/projects/rrg-timsbc/$USER/code
git clone --recurse-submodules git@github.com:rsingla92/optimizer_experiments.git
cd optimizer_experiments

# Run the DRAC setup helper. Takes ~5 minutes (mostly FineWeb download).
./scripts/setup_drac.sh
```

`setup_drac.sh` is idempotent — safe to re-run if anything fails partway.

### Sanity gate (offline, fast)

Before submitting any compute job, verify the local sanity tests still pass:

```bash
module load StdEnv/2023 python/3.12 cuda/12.6 gcc/12
source ~/scratch/optimizer_experiments/venv/bin/activate
make sanity
```

Expected: **30 passed, 1 skipped** (GPU determinism test skips if you're on a login node without CUDA visible).

### Submit the Phase 1 job

```bash
sbatch scripts/slurm/phase1_modded_nanogpt.sh
```

Returns a job ID. Watch with:

```bash
squeue -u $USER
```

Typical wait: minutes to hours depending on Narval queue. Compute itself: ~3 hours.

### Inspect the result

```bash
JOB=<your-job-id>
ls results/phase1/phase1_modded_nanogpt-$JOB/
#   env.txt                  GPU/git/module/pip provenance
#   train.log                full stdout from train_gpt.py
#   modded_nanogpt_logs/     modded-nanogpt's own log dir (their convention)

# Tail the training loss curve
grep "step" results/phase1/phase1_modded_nanogpt-$JOB/train.log | tail -30

# Final loss
grep -i "val_loss\|validation" results/phase1/phase1_modded_nanogpt-$JOB/train.log | tail -5
```

## Pass/fail criteria

| Outcome | Action |
|---|---|
| Final val loss ∈ [3.12, 3.44], no NaN/Inf, completed normally | **PASS** — Phase 1 gate clears. Move to Phase 2 prep. |
| Final val loss outside [3.12, 3.44] but completed | Investigate. Likely a hardware mismatch (A100 vs H100 numerical drift) or an env issue. **Halt before Phase 2.** |
| NaN / Inf / crash | Implementation or env bug. **Halt** per PROTOCOL §11. Inspect `env.txt`, share details. |
| OOM | Job needs more memory; bump `--mem` in the SLURM script and retry. Not a reproduction failure. |
| Hits wall time (8h) | Single-GPU was slower than expected. Either (a) bump `--time` to 12h, or (b) try `a100l` (80GB SXM) which is faster than `a100` (40GB PCIe). |

## What this Phase 1 does *not* prove

- It does not validate SOTR (we're running stock modded-nanogpt — no SOTR involved).
- It does not validate our `experiments/train.py` — that doesn't exist yet.
- It does not validate Phase 2 ablation infrastructure — that's the next step.

It only validates: **environment + canonical Muon implementation reproduce on our hardware**. That's the necessary precondition for everything else.

## After Phase 1 passes

Phase 2 prep begins. Per the conversation in this repo's history, that means:

1. Vendor `external/modded-nanogpt/train_gpt.py` into `experiments/train.py` with a header per `CONTRIBUTING.md`. Minimal patch (~30-line diff) to the optimizer construction so `--optimizer-name ∈ {adamw, lion, muon, sotr}` works.
2. Add `experiments/_logging.py` with PROTOCOL §8 stability incident detection and JSONL writer (~80 lines).
3. Add Python config files in `experiments/configs/` (one per actual run; `@dataclass` instances, no YAML).
4. `experiments/scripts/gen_phase2_configs.py` to emit the 250-cell ablation index.
5. Submit `scripts/slurm/array_ablation.sh`.

That's the next session, after Phase 1 lands a number.
