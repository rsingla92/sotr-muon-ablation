"""Phase 2 base config — reduced-scale model used by the ablation grid.

The Phase 2 ablation runs a reduced-scale variant of the canonical
modded-nanogpt setup so 250 cells (10 × 5 seeds × 5 LRs, see PROTOCOL §9)
fit in a single SLURM array. We keep the architecture identical (so
reproduction-style claims still apply) but cut the iteration count
substantially.

This module is **not** a runnable config on its own; it exposes a
``base_hparams`` and ``base_config`` that ``gen_phase2_configs.py`` derives
its 250 cells from. It is also imported by tests verifying the generator
emits sensible cells.
"""

from __future__ import annotations

from experiments._configs import OptimizerKind, RunConfig, UpstreamHparams

# Reduced-scale Phase 2 hparams: smaller model, fewer iterations than Phase 1.
# Keeps Chinchilla-like ratio (~20 tokens/param) and trapezoidal schedule shape.
# Phase 1 reproduction (full-scale) is the apples-to-apples Muon baseline; the
# Phase 2 grid filters configurations cheaply, then the winner promotes to
# Phase 3 at full scale.
_PHASE2_HPARAMS = UpstreamHparams(
    input_bin="external/modded-nanogpt/data/fineweb10B/fineweb_train_*.bin",
    input_val_bin="external/modded-nanogpt/data/fineweb10B/fineweb_val_*.bin",
    batch_size=8 * 16,  # 128 sequences (1/4 of Phase 1's 512)
    device_batch_size=16,
    sequence_length=1024,
    num_iterations=1500,  # ~30% of Phase 1's 5100 — tractable per cell
    embed_learning_rate=0.0036,
    muon_learning_rate=0.02,  # default; will be swept per cell
    warmup_iters=0,
    warmdown_iters=400,
    weight_decay=0.0,
    val_loss_every=100,
    val_tokens=2 * 1024 * 1024,  # 2M tokens — quicker val
    save_every=0,
)

base_config = RunConfig(
    name="phase2_base",
    purpose="phase2",
    description="Reduced-scale Phase 2 base. Used by gen_phase2_configs.py.",
    hparams=_PHASE2_HPARAMS,
    optimizer_name=OptimizerKind.MUON,
    momentum=0.95,
    seed=0,
    log_every_steps=25,
)
