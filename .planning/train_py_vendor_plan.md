# Vendor + patch plan for `external/modded-nanogpt/train_gpt.py`

**Status:** planning only. Author: research agent (Opus). Date: 2026-05-02.
**Source SHA reference:** capture `git -C external/modded-nanogpt rev-parse HEAD`
at vendor time and put it in the header (we have not pinned a specific SHA in
this plan; the integrator must record the actual one).
**Target output:** `experiments/train.py` (new file, vendored copy + minimal
patch). Loads a config dataclass passed via `--config <python.module.path>`.

---

## A. Structural inventory of upstream `train_gpt.py` (2022 lines)

> All citations are `train_gpt.py:NNN`.

### A.1 The "`Hyperparameters`" dataclass — `:1558-1578`

This is **not** what it was at the time of Record #4. In current head:

| Field | Default | Controls |
|---|---|---|
| `data_path` | `os.environ.get("DATA_PATH", ".")` (class-level, not field) | base for train/val globs |
| `train_files` | `<data_path>/data/fineweb10B/fineweb_train_*.bin` | glob for train shards |
| `val_files`   | `<data_path>/data/fineweb10B/fineweb_val_*.bin`   | glob for val shards |
| `val_tokens`  | `10485760` | val token budget (held fixed for comparability) |
| `val_batch_size` | `4 * 64 * 1024 * 8 = 2_097_152` | val batch in tokens |
| `num_scheduled_iterations` | `1440` | total scheduled steps |
| `num_extension_iterations` | `40`  | extra steps after schedule |
| `run_id` | `uuid4()` | log/checkpoint folder name |
| `val_loss_every` | `250` | val cadence in steps |
| `save_checkpoint` | `False` | save final state |
| `run_evals` | `False` | run HellaSwag after train |
| `bigram_vocab_size` | `50304 * 5` | size of bigram embedding |

Crucially **all optimizer hyperparameters are no longer here**. They live in
`TrainingManager.__init__` `:1721-1732`:

```text
adam_defaults    = lr=0.008, eps=1e-10, weight_decay=0.005
normuon_defaults = lr=0.023, momentum=0.95, beta2=0.9, weight_decay=1.2
```

The instance is constructed at module top level: `args = Hyperparameters()` `:1578`.
`args` is then read directly from many places (e.g. `args.bigram_vocab_size` at
`:1233`, `:1466`, `:1817` — including from inside `GPT.__init__`). Replacing
`args` is therefore not a local change.

### A.2 Model class hierarchy — *do not touch*

- `CastedLinearT` `:949-976` — bf16 linear with FP8 path.
- `Yarn` `:981-1048` — RoPE with stage-by-stage frequency rescaling.
- `AttnArgs` (`@dataclass`) `:1050-1060`.
- `CausalSelfAttention` `:1064-1134` — bank-fed; `forward(x, attn_args, qkvo_w)`.
- `ForwardScheduleConfig` `:1143-1148`.
- `GPT` `:1150-1371` — owns `value_embeds`, `attn_gate_bank`, `ve_gate_bank`,
  `qk_bank`, `vo_bank`, `mlp_bank`, `lm_head` (`CastedLinearT`), `embed`,
  `bigram_embed`, `post_lambdas`, `x0_lambdas`, `bigram_lambdas`,
  `resid_lambdas`, `scalars`. Auto-labels every parameter at `:1259-1260`
  with `param.label = name.replace('.weight','')`. Banks carry a `.reshape`
  attribute (`:1191`, `:1197`, `:1202`).

We must not alter any of this; the labels are load-bearing for `NorMuonAndAdam`.

### A.3 Optimizer construction — *the only thing we patch*

- The optimizer is **a single object** of type `NorMuonAndAdam` defined at
  `:367-940`. It is not a list; it dispatches per-parameter via `param_table`
  (`:1693-1710`) and uses a `scatter_order` (dict insertion order) plus
  `work_order` (`:1714-1719`) to overlap reduce_scatter / all_gather with
  compute. There is no `torch.optim.Optimizer` here, no `param_groups` in the
  usual sense, no list of multiple optimizers.
- **Param-group convention** (`:1693-1710`):
  - `qk_bank`, `vo_bank`, `mlp_bank` → `optim="normuon"`, `comms="sharded"`.
  - `lm_head`, `embed`, `value_embeds` → `optim="adam"`, `comms="sharded"`.
  - `bigram_embed` → `optim="adam"`, `comms="sharded_sparse"`.
  - All scalar / gate params (`scalars`, `*_lambdas`, `*_gate_bank`,
    `smear_gate`, `skip_gate`) → `optim="adam"`, `comms="replicated"`.
  - `lr_mul` and `wd_mul` carry the per-bucket scaling Muon papers describe.
- `TrainingManager.step_optimizers(step)` `:1785-1801` writes the per-step LR
  multiplier and Muon momentum back into each `ParamConfig` and then calls
  `self.optimizer.step(do_adam=is_adam_step)` (Adam updates only on odd steps).

### A.4 Training loop — `:1957-2009`

- Warmup pass `:1925-1944` runs ~10 steps to compile kernels, then resets
  weights/state from `initial_state`.
- Main loop `:1961-2009`. Per step: `advance_schedule(step)`, optional val,
  `grad_accum_steps` micro-batches each running `model(...).sum() * grad_scale`
  + `loss.backward()`, then `training_manager.step_optimizers(step)`.
- Per-step log line `:2009`:
  `print0(f"step:{step+1}/{train_steps} train_time:{...}ms step_avg:{...}ms")`.
- Val log line `:1983`:
  `print0(f"step:{step}/{train_steps} val_loss:{...} train_time:{...}ms step_avg:{...}ms")`.
- Final memory line `:2020-2021`. `dist.destroy_process_group()` `:2022`.

### A.5 Distributed setup — `:47-57`

Top-level, executed at import:
```text
rank, world_size  = int(os.environ["RANK|WORLD_SIZE"])
LOCAL_RANK pulled from env                        :28, :53
torch.empty(...).backward()                       :27 (workaround)
dist.init_process_group(backend="cuda:nccl,cpu:gloo", device_id=device)  :55
dist.barrier()                                    :56
master_process = (rank == 0)                      :57
```

`grad_accum_steps = 8 // world_size` `:50` (so single-GPU → 8 micro-batches).

### A.6 Things that run at import time (this is the trap)

Sorted by line:

1. **Self-source slurp** `:4-9`: opens `sys.argv[0]` and `triton_kernels.py`
   into a single `code` string — used at `:1875` to prepend the script's own
   text to the log file.
2. `os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"` `:22` —
   must be set **before** `import torch`.
3. `torch.empty(..., device="cuda:LOCAL_RANK").backward()` `:27` — bug
   workaround; needs `LOCAL_RANK` already set.
4. `dynamo.config.recompile_limit = 64` `:43`.
5. Distributed init `:47-57` (above).
6. `@torch.library.custom_op` registration `:63-155` for FP8 `mm_t`.
7. `polar_express_coeffs` literal `:161-167`.
8. `flash_attn_interface = get_kernel('varunneal/flash-attention-3')...` `:1062`
   — pulls a kernel from HF Hub at import.
9. `args = Hyperparameters()` `:1578`.
10. `training_schedule = TrainingSchedule(...)` `:1659` — references
    `args.num_scheduled_iterations` etc.
11. Logging file open `:1861-1875`, writes the source dump.
12. Model build, FP8 cast, `torch.compile(model)` `:1888-1907`.
13. `training_manager = TrainingManager(model)` `:1908`.
14. Warmup loop `:1925-1944`.
15. Main loop `:1961-2009` (yes — *the training loop runs at import*).

This last point is decisive for the patch: **there is no `if __name__ == "__main__":` and no `main()` function**. The script is imperative top-to-bottom.

### A.7 Top-of-file env vars / torch settings

| Where | What | Why |
|---|---|---|
| `:22` | `PYTORCH_ALLOC_CONF=expandable_segments:True` | mem allocator |
| `:43` | `dynamo.config.recompile_limit = 64` | prevent recomp blow-up |
| `:1223` | `os.environ.get("DISABLE_FP8", False)` toggles FP8 lm_head | test escape hatch |
| `:55` | `init_process_group("cuda:nccl,cpu:gloo")` | mixed backend |
| `:1907` | `torch.compile(model, dynamic=False, fullgraph=True)` | compiled model |

---

## B. Vendoring strategy

### B.1 File location

Vendor target: **`experiments/train.py`** (per `docs/PHASE1.md` §"After Phase 1
passes" line 110, and per `experiments/README.md` if not already saying so).
This matches the layout in `CONTRIBUTING.md` §"Repository layout".

### B.2 Header (mandatory, per `CONTRIBUTING.md:75-83`)

```python
# Adapted from KellerJordan/modded-nanogpt @ commit <SHA>
# Source: external/modded-nanogpt/train_gpt.py (vendored 2026-05-02)
# Changes:
#   - Added argparse `--config <module.path>` selecting an experiments/_configs.py
#     RunConfig dataclass; resolved BEFORE any heavy imports.
#   - Replaced module-level `args = Hyperparameters()` with `args = cfg.to_hparams()`.
#   - `TrainingManager.__init__` now reads optimizer-section of cfg and
#     dispatches optimizer construction to {adamw, lion, muon, sotr, normuon}.
#   - Inserted RunLogger.log_step(...) call after training_manager.step_optimizers.
#   - Inserted StabilityMonitor.check(...) immediately after loss.backward().
#   - All other behaviour (model, schedule, FP8, sparse comms, warmup,
#     val cadence, print0 lines) preserved verbatim.
```

### B.3 Monkey-patch vs full vendor — **vendor**

Argued options:

| Option | Verdict |
|---|---|
| (a) `from external.modded_nanogpt import train_gpt` and override symbols | **Reject.** The script *runs the training loop at import* (§A.6). There is nothing to override before it executes. Any monkey-patch fires too late. |
| (b) Import individual classes (`GPT`, `NorMuonAndAdam`, helpers) and write our own glue | **Reject.** `train_gpt.py` has no `main()`; classes are interleaved with executable side-effect statements (custom_op registration, `args = Hyperparameters()`, kernel slurping, distributed init). You cannot import class definitions without triggering all of it. |
| (c) **Full vendored copy with surgical patches** | **Accept.** Diff is small (~30 lines real change + header). We get reproducibility (frozen text in our git history) and the "single config + commit-hash → run" guarantee `CONTRIBUTING.md:11` requires. |

Even though the file is 2022 lines, the integrator must `cp` the whole thing
once and apply the patches in §C; nothing else changes. We pin the upstream
SHA in the header and re-vendor manually if upstream moves.

### B.4 Triton kernels

`train_gpt.py:7-9` opens `triton_kernels.py` (sibling file, 34510 bytes). It
is also imported as a module at `:38`. Two viable options:

- **(Recommended) Copy `triton_kernels.py` into `experiments/` alongside
  `train.py`.** The self-source slurp at `:4-9` resolves it via
  `os.path.dirname(sys.argv[0])`, which after vendoring will be
  `experiments/`. Importing it as `from triton_kernels import ...` requires
  `experiments/` on `sys.path` — which it already will be when launched as
  `python -m experiments.train` or `python experiments/train.py`. Add a
  matching header to the vendored copy.
- (Alternative) Keep the canonical copy under `external/` and add
  `sys.path.insert(0, "external/modded-nanogpt")` in our header. Rejected:
  brittle (breaks `python -m`), and silently couples our results to upstream
  drift in `triton_kernels.py`. Reproducibility wins.

---

## C. Proposed minimal patch

### C.1 `--config` argument parsing — **before any other import**

Insert at the very top of `experiments/train.py`, *before* `import os`/`sys`
side effects (`train_gpt.py:1`):

```text
# Pseudocode
import argparse, importlib, sys
_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--config", required=True,
                help="dotted module path resolving to a RunConfig instance, "
                     "e.g. 'experiments.configs.muon_baseline'")
_args, _rest = _p.parse_known_args()
sys.argv = [sys.argv[0], *_rest]      # hide --config from upstream's `with open(sys.argv[0])`
cfg = importlib.import_module(_args.config).config   # convention: each config module exposes `config`

# Then the upstream block runs unchanged:
import os
import sys
with open(sys.argv[0], 'r') as f:
    code = f.read()
...
```

Why argparse over a `sys.argv` hack: argparse fails loud on typos (`ValueError`
per `CONTRIBUTING.md:124`). The `parse_known_args` + `sys.argv` rewrite is
necessary because `:5` reads `sys.argv[0]` for self-logging; we must leave
exactly that one element behind so the dump still works.

### C.2 Replace `args = Hyperparameters()` `:1578`

```text
# Pseudocode
args = cfg.to_hparams()       # returns a populated upstream-compatible Hyperparameters
                              # so all downstream `args.foo` reads keep working
```

`cfg.to_hparams()` lives in `experiments/_configs.py` and is the only
adaptation seam. This keeps `:1233`, `:1466`, `:1817`, etc. unchanged.

### C.3 Optimizer dispatcher — inside `TrainingManager.__init__`

The cleanest insertion point is **just before `:1734`** (the
`self.optimizer = NorMuonAndAdam(...)` call), keyed on `cfg.optimizer_name`:

```text
# Pseudocode in TrainingManager.__init__
opt_name = cfg.optimizer.name      # one of {"adamw", "lion", "muon", "sotr", "normuon"}
opt_hp   = cfg.optimizer.hparams   # dataclass field

if opt_name == "normuon":
    # Upstream behavior — apples-to-apples Muon/NorMuon baseline.
    self.optimizer = NorMuonAndAdam(model.named_parameters(),
                                    param_table=self.param_table,
                                    scatter_order=list(self.param_table),
                                    work_order=self.work_order,
                                    adam_defaults=adam_defaults,
                                    normuon_defaults=normuon_defaults)

elif opt_name in {"adamw", "lion", "muon", "sotr"}:
    # Build a vanilla torch.optim.Optimizer (or our optimizers/sotr.SOTR / lion_pytorch.Lion / external Muon).
    # We MUST adapt to upstream's no-param_groups convention. Simplest: build classic
    # param groups by walking model.named_parameters() and bucketing by `param.label`
    # into {matrix-bank, embed/lm_head, scalars/gates}. Use opt_hp.lr / wd for the
    # primary group; reuse the upstream lr_mul/wd_mul ratios per bucket so other
    # baselines remain comparable.
    self.optimizer = build_simple_optimizer(model, opt_name, opt_hp)
    self._is_simple_optimizer = True
else:
    raise ValueError(f"unknown optimizer: {opt_name}")
```

Then guard `step_optimizers` `:1785`:

```text
def step_optimizers(self, step):
    step_lr = training_schedule.get_lr(step)
    if getattr(self, "_is_simple_optimizer", False):
        for g in self.optimizer.param_groups:
            g["lr"] = g["initial_lr"] * step_lr
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
    else:
        # original upstream block, verbatim
        ...
```

**Caveat to flag for the writer**: the simple-optimizer path *will not*
include sharded comms, sparse bigram comms, FP8-aware update, mantissa
tracking, MTP `do_adam` parity, or the embed↔lm_head copy at split step.
That is deliberate — adamw/lion/muon/sotr baselines are not meant to share
NorMuon's communication scheduling. We accept that the four non-Muon
baselines run with `world_size=1` semantics regardless of `WORLD_SIZE`. The
PROTOCOL says single-GPU, so this is fine; document it in the header.

### C.4 Logger / monitor hooks (locations only)

- `RunLogger.log_step(step, lr, momentum, val_loss=None, train_time_ms, ...)`:
  place **right after `:2009`** (the per-step `print0`) so we get the same
  cadence as upstream and can correlate with their log lines for free.
  Validation-side: also call after `:1983` with `val_loss` set.
- `StabilityMonitor.check(loss_value, grad_norm)`: call **immediately after
  `:1938` and `:2003`** (the two `loss.backward()` calls — warmup and main).
  PROTOCOL §8 incident detection wants this pre-step so a NaN doesn't get
  swept into the optimizer state; the monitor can raise and abort cleanly
  before `step_optimizers`.
- Both objects constructed once near `:1908` (alongside `training_manager`)
  using `cfg.logging` / `cfg.stability` sub-configs. Master-rank only.

### C.5 No CLI changes beyond `--config`

Upstream takes no args. Adding only `--config` (not bare positional) keeps
their `sys.argv[0]`-based self-logging intact and lets us launch with
`torchrun --nproc_per_node=1 experiments/train.py --config experiments.configs.muon_baseline`.

---

## D. Risks and gotchas

1. **Self-source slurp (`:4-9`) reads our edited file.** The hash logged inside
   the run differs from upstream's. *Doesn't matter for correctness* — used
   only for self-documenting logs. Mention in the header so a reader doesn't
   chase a phantom drift.
2. **`triton_kernels.py` import-by-name.** Two reads happen: `open(...)` for
   logging at `:7-9`, and `from triton_kernels import ...` at `:38`. Both
   resolve relative to the script's directory. Vendor the file (recommended
   path B.4) to make this trivial.
3. **`init_process_group` runs at import (`:55`).** `--config` parsing must
   complete *before* `import torch` (because `:22` sets the allocator env
   var that PyTorch reads at import). Pseudocode in C.1 handles this by
   doing argparse with the bare stdlib (`argparse, importlib, sys`) at file
   top, then letting the upstream import block run. Do not reorder.
4. **`nproc_per_node=1` path** (Phase 1 / single-H100). Exercised by `:50`
   (`grad_accum_steps = 8`), `:254` (`_sparse_comms_active` returns False),
   `:445-447` (NorMuon comms forced to `"none"`), `:670/681` branches in
   `copy_lm_state_to_embed`. Our patch must not introduce a `world_size>1`
   assumption; the simple-optimizer dispatcher in C.3 has no comms code at
   all, which is correct.
5. **bf16 / FP8 / `torch.compile` / Triton.** All wired into the model
   forward and into the FP8 lm_head (`:1225`, `:954-976`, `:1365`). Our patch
   touches only optimizer construction and two hook-call sites — none of
   these paths are altered. FP8 can be disabled by `DISABLE_FP8=1` (`:1223`)
   if a future ablation needs it.
6. **`param.label` requirement.** `NorMuonAndAdam` requires every parameter
   to carry a `.label` (`:434-436`). Auto-set at `:1259-1260` for the upstream
   model. The simple-optimizer path doesn't need labels; safe to ignore.
7. **`LOCAL_RANK` env var requirement at import (`:28, :53`).** Even on a
   single GPU, launch must set `LOCAL_RANK=0 RANK=0 WORLD_SIZE=1` (or use
   `torchrun`). The Phase 1 SLURM scripts already do this; verify.
8. **The training loop is at module top-level.** Patch behaviour is preserved
   only if our config import has zero side effects on `torch.cuda` /
   `torch.distributed`. Keep `experiments/_configs.py` pure-Python (no torch
   imports beyond `dataclass` annotations).
9. **`run_id` default is `uuid4()` `:1571`.** Override in `cfg.to_hparams()`
   to use `experiments/_run_id.py` (PROTOCOL convention: `YYYY-MM-DD_HHMMSS_<6char>`).

---

## E. Checklist for the integrator

- [ ] `mkdir -p experiments` (already exists; confirm).
- [ ] Capture upstream SHA: `git -C external/modded-nanogpt rev-parse HEAD`. Put in header.
- [ ] `cp external/modded-nanogpt/train_gpt.py experiments/train.py`.
- [ ] `cp external/modded-nanogpt/triton_kernels.py experiments/triton_kernels.py`.
  Add a one-line vendor header to `triton_kernels.py` per `CONTRIBUTING.md:75`.
- [ ] Insert §B.2 header at top of `experiments/train.py`.
- [ ] Insert §C.1 argparse block at the very top of `experiments/train.py`,
      *before* `import os` (currently `:1`). Use only stdlib imports here.
- [ ] Replace `args = Hyperparameters()` (currently `:1578`) with
      `args = cfg.to_hparams()`.
- [ ] In `TrainingManager.__init__` (currently `:1686-1746`), wrap the
      `self.optimizer = NorMuonAndAdam(...)` call (currently `:1734-1741`)
      with the §C.3 dispatcher. Add `build_simple_optimizer(...)` helper
      defined just above the `TrainingManager` class.
- [ ] In `TrainingManager.step_optimizers` (currently `:1785-1801`), add the
      simple-optimizer branch from §C.3.
- [ ] Construct `RunLogger` and `StabilityMonitor` near current `:1908`
      (right after `training_manager = TrainingManager(model)`), guarded by
      `master_process`.
- [ ] Insert `stability_monitor.check(loss.item(), ...)` right after
      `loss.backward()` at current `:1938` (warmup) and `:2003` (main loop).
- [ ] Insert `run_logger.log_step(step+1, ...)` right after the per-step
      `print0` at current `:2009` and `run_logger.log_step(step, ..., val_loss=val_loss)`
      after current `:1983`.
- [ ] Verify `python -c "import ast; ast.parse(open('experiments/train.py').read())"`
      parses (cheap structural check; no execution needed).
- [ ] Run `make lint` (ruff). Expect zero diff.
- [ ] Run `make sanity` if `experiments/` is in scope. Expect 30 passed / 1 skipped.
- [ ] Smoke test with a placeholder config that sets `num_scheduled_iterations=2,
      num_extension_iterations=0, val_loss_every=0, save_checkpoint=False` —
      should reach the val-skip path and exit cleanly without OOM on a CPU
      sanity build (will of course fail without CUDA; just check parsing/import).
- [ ] Commit with message: `experiments: vendor train_gpt.py with optimizer dispatcher`.
      Body: link to this plan and the upstream SHA.

---

## Critical files for implementation

- `/Users/rsingla/Desktop/optimzer_experiments/external/modded-nanogpt/train_gpt.py` (the source)
- `/Users/rsingla/Desktop/optimzer_experiments/external/modded-nanogpt/triton_kernels.py` (must be co-vendored)
- `/Users/rsingla/Desktop/optimzer_experiments/experiments/train.py` (target — to be created)
- `/Users/rsingla/Desktop/optimzer_experiments/experiments/_configs.py` (defines `RunConfig` and `to_hparams()` — separate task #43)
- `/Users/rsingla/Desktop/optimzer_experiments/CONTRIBUTING.md` (header format §"Comments", config rules §"Configs")
