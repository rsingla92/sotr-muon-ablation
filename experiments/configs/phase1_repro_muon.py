"""Phase 1 reproduction config — single-GPU Muon on FineWeb.

Mirrors `external/modded-nanogpt/train_gpt2.py`'s default `Hyperparameters`
verbatim. PROTOCOL §6 reproduction gate: final FineWeb val loss within ±5%
of 3.28.

Submit on Fir:
    sbatch scripts/slurm/single_gpu.sh experiments.configs.phase1_repro_muon
"""

from experiments._configs import OptimizerKind, RunConfig, UpstreamHparams

config = RunConfig(
    name="phase1_repro_muon",
    purpose="phase1",
    description="Reproduce KellerJordan/Muon's published number on FineWeb at "
    "single-GPU H100 with grad_accum=8. Vendored harness at upstream commit "
    "dd2224b. Target: val loss within [3.12, 3.44].",
    hparams=UpstreamHparams(
        # Exactly the upstream defaults.
        input_bin="external/modded-nanogpt/data/fineweb10B/fineweb_train_*.bin",
        input_val_bin="external/modded-nanogpt/data/fineweb10B/fineweb_val_*.bin",
        batch_size=8 * 64,
        device_batch_size=64,
        sequence_length=1024,
        num_iterations=5100,
        embed_learning_rate=0.0036,
        muon_learning_rate=0.02,
        warmup_iters=0,
        warmdown_iters=1450,
        weight_decay=0.0,
        val_loss_every=125,
        val_tokens=10485760,
        save_every=0,
    ),
    optimizer_name=OptimizerKind.MUON,
    momentum=0.95,
    seed=0,
    log_every_steps=50,
)
