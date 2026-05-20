"""nanoGPT-style training loop with DDP for the prefix-LM Go transformer.

Run on Fir:
    torchrun --standalone --nproc_per_node=4 -m gogpt.train --config configs/baseline_30m.yaml

Or for a smoke test on one GPU (or even CPU):
    python -m gogpt.train --config configs/smoke.yaml

Key features:
- Single-node DDP via torchrun.
- AdamW + cosine schedule with warmup.
- bf16 mixed precision (autocast).
- Gradient clipping at 1.0.
- Checkpoint every ``save_every`` steps; retain best-by-val-loss and latest.
- wandb optional (off when WANDB_DISABLED or run with --no-wandb).
- Auto-resume from latest checkpoint in the run directory.
- Per-step assertion that loss is finite -- crash early if NaN.
- runs/<run_id>/manifest.json with git hash, config, dataset hash, wandb id.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .data import SgfShardDataset, collate
from .model import GoGPT, GoGPTConfig

log = logging.getLogger("gogpt.train")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Data
    train_glob: str = "data/train/*.sgf"
    val_glob: str = "data/val/*.sgf"
    max_trajectory_len: int = 64
    max_examples_per_game: int = 4

    # Model
    model: dict[str, Any] = field(default_factory=lambda: {})

    # Optim
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_steps: int = 200
    max_steps: int = 1000
    batch_size: int = 16
    grad_accum: int = 1
    num_workers: int = 4

    # Loop
    log_every: int = 10
    val_every: int = 200
    val_batches: int = 50
    save_every: int = 500

    # Run
    run_dir: str = "runs/smoke"
    run_id: str | None = None  # auto-set if None
    seed: int = 0
    bf16: bool = True
    wandb_project: str = "cot-go-transformer"
    wandb_run_name: str | None = None

    # Misc
    debug_no_data: bool = False  # use synthetic data instead of SGFs


def load_config(path: str) -> TrainConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return TrainConfig(**raw)


# ---------------------------------------------------------------------------
# DDP setup
# ---------------------------------------------------------------------------

def setup_ddp() -> tuple[int, int, int, torch.device]:
    """Initialize DDP if launched via torchrun, else single-process."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return 0, 0, 1, device


def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def dataset_version_hash(paths: list[str]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.encode())
        try:
            st = os.stat(p)
            h.update(str(st.st_size).encode())
            h.update(str(int(st.st_mtime)).encode())
        except OSError:
            continue
    return h.hexdigest()[:16]


def cosine_lr(step: int, warmup: int, max_steps: int, base_lr: float, min_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    progress = min(progress, 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def list_shards(glob: str) -> list[str]:
    from glob import glob as _glob
    return sorted(_glob(glob))


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, step: int, val_loss: float | None) -> None:
    state = {
        "step": step,
        "val_loss": val_loss,
        "model": _unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer | None) -> int:
    state = torch.load(path, map_location="cpu")
    _unwrap(model).load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return int(state.get("step", 0))


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig) -> None:
    rank, local_rank, world_size, device = setup_ddp()
    torch.manual_seed(cfg.seed + rank)

    if cfg.run_id is None:
        cfg.run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg.run_dir) / cfg.run_id
    if is_main_process():
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(asdict(cfg)))

    # ---- Model
    model_cfg = GoGPTConfig(**cfg.model)
    model = GoGPT(model_cfg).to(device)
    if cfg.bf16 and device.type == "cuda":
        # Keep model params in fp32; autocast handles compute dtype.
        pass
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    # ---- Optimizer
    decay, no_decay = [], []
    for n, p in _unwrap(model).named_parameters():
        if p.ndim < 2 or n.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=cfg.betas,
        eps=1e-8,
    )

    # ---- Data
    train_shards = [] if cfg.debug_no_data else list_shards(cfg.train_glob)
    val_shards = [] if cfg.debug_no_data else list_shards(cfg.val_glob)
    if not cfg.debug_no_data and not train_shards:
        raise RuntimeError(f"no training shards matched {cfg.train_glob!r}")

    train_loader, val_loader = None, None
    if not cfg.debug_no_data:
        train_ds = SgfShardDataset(
            train_shards,
            max_trajectory_len=cfg.max_trajectory_len,
            max_examples_per_game=cfg.max_examples_per_game,
            seed=cfg.seed + rank,
        )
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, collate_fn=collate,
            num_workers=cfg.num_workers, persistent_workers=cfg.num_workers > 0,
            pin_memory=device.type == "cuda",
        )
        if val_shards:
            val_ds = SgfShardDataset(
                val_shards,
                max_trajectory_len=cfg.max_trajectory_len,
                max_examples_per_game=cfg.max_examples_per_game,
                seed=cfg.seed + 9999,
                shuffle_shards=False,
            )
            val_loader = DataLoader(
                val_ds, batch_size=cfg.batch_size, collate_fn=collate,
                num_workers=max(1, cfg.num_workers // 2),
            )

    # ---- Resume
    latest_ckpt = run_dir / "latest.pt"
    start_step = 0
    if latest_ckpt.exists():
        start_step = load_checkpoint(latest_ckpt, model, optimizer) + 1
        log.info("resumed from step %d", start_step)

    # ---- wandb (best-effort)
    wandb_run = None
    if is_main_process() and not os.environ.get("WANDB_DISABLED"):
        try:
            import wandb
            wandb_run = wandb.init(
                project=cfg.wandb_project,
                name=cfg.wandb_run_name or cfg.run_id,
                config=asdict(cfg),
                dir=str(run_dir),
                resume="allow",
            )
        except Exception as e:
            log.warning("wandb init failed: %s", e)

    # ---- Manifest
    if is_main_process():
        manifest = {
            "git_commit": git_commit_hash(),
            "config": asdict(cfg),
            "model_params": _unwrap(model).num_parameters(),
            "train_dataset_hash": dataset_version_hash(train_shards),
            "wandb_run_id": getattr(wandb_run, "id", None) if wandb_run else None,
            "world_size": world_size,
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # ---- Train
    best_val: float | None = None
    step = start_step
    t0 = time.time()
    autocast_dtype = torch.bfloat16 if (cfg.bf16 and device.type == "cuda") else torch.float32
    model.train()
    grad_accum_buf = 0

    data_iter = _make_data_iter(train_loader, cfg, device)

    while step < cfg.max_steps:
        for g in optimizer.param_groups:
            g["lr"] = cosine_lr(step, cfg.warmup_steps, cfg.max_steps, cfg.lr, cfg.min_lr)
        optimizer.zero_grad(set_to_none=True)

        loss_accum = 0.0
        for _ in range(cfg.grad_accum):
            batch = next(data_iter)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=device.type == "cuda"):
                out = model(
                    tokens=batch["tokens"],
                    state_categories=batch["state_categories"],
                    labels=batch["labels"],
                    loss_mask=batch["loss_mask"],
                )
                loss = out["loss"] / cfg.grad_accum
            loss.backward()
            loss_accum += loss.detach().float().item()
            grad_accum_buf += 1

        if not math.isfinite(loss_accum):
            raise FloatingPointError(f"non-finite loss at step {step}: {loss_accum}")

        gnorm = torch.nn.utils.clip_grad_norm_(_unwrap(model).parameters(), cfg.grad_clip)
        optimizer.step()

        if is_main_process() and (step % cfg.log_every == 0):
            elapsed = time.time() - t0
            log.info(
                "step=%d loss=%.4f grad_norm=%.3f lr=%.2e (%.1fs since start)",
                step, loss_accum, float(gnorm), optimizer.param_groups[0]["lr"], elapsed,
            )
            if wandb_run is not None:
                wandb_run.log({
                    "train/loss": loss_accum,
                    "train/grad_norm": float(gnorm),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "step": step,
                })

        if val_loader is not None and step > 0 and step % cfg.val_every == 0:
            vloss = run_validation(model, val_loader, device, autocast_dtype, cfg.val_batches)
            if is_main_process():
                log.info("step=%d val_loss=%.4f", step, vloss)
                if wandb_run is not None:
                    wandb_run.log({"val/loss": vloss, "step": step})
                if best_val is None or vloss < best_val:
                    best_val = vloss
                    save_checkpoint(run_dir / "best.pt", model, optimizer, step, vloss)

        if is_main_process() and step > 0 and step % cfg.save_every == 0:
            save_checkpoint(run_dir / "latest.pt", model, optimizer, step, best_val)

        step += 1

    if is_main_process():
        save_checkpoint(run_dir / "latest.pt", model, optimizer, step - 1, best_val)
        if wandb_run is not None:
            wandb_run.finish()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _make_data_iter(loader: DataLoader | None, cfg: TrainConfig, device: torch.device):
    """Return an infinite iterator over training batches, or synth in debug mode."""
    if loader is not None:
        def _gen():
            while True:
                for b in loader:
                    yield {k: v.to(device, non_blocking=True) for k, v in b.items()}
        return _gen()
    # Synthetic batches for debug/CPU smoke testing.
    from .tokenizer import NUM_POINTS, PASS_TOKEN, BOS_TOKEN, SEP_POS_TOKEN, EOS_TOKEN
    def _synth():
        while True:
            T = 1 + NUM_POINTS + 1 + cfg.max_trajectory_len + 1
            tokens = torch.randint(0, 82, (cfg.batch_size, T), device=device)
            tokens[:, 0] = BOS_TOKEN
            tokens[:, 1 : 1 + NUM_POINTS] = PASS_TOKEN
            tokens[:, 1 + NUM_POINTS] = SEP_POS_TOKEN
            tokens[:, -1] = EOS_TOKEN
            labels = torch.full_like(tokens, -100)
            sep_idx = 1 + NUM_POINTS
            labels[:, sep_idx:-1] = tokens[:, sep_idx + 1:]
            loss_mask = torch.zeros_like(tokens, dtype=torch.int8)
            loss_mask[:, sep_idx:-1] = 1
            state_cats = torch.randint(0, 5, (cfg.batch_size, NUM_POINTS), device=device)
            yield {"tokens": tokens, "labels": labels, "loss_mask": loss_mask, "state_categories": state_cats}
    return _synth()


@torch.no_grad()
def run_validation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    autocast_dtype: torch.dtype,
    max_batches: int,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=device.type == "cuda"):
            out = model(
                tokens=b["tokens"],
                state_categories=b["state_categories"],
                labels=b["labels"],
                loss_mask=b["loss_mask"],
            )
        total += float(out["loss"].item())
        n += 1
    model.train()
    return total / max(1, n)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "1"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
