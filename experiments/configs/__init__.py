"""Run configs.

Each module in this package exposes a single top-level ``config`` attribute
of type :class:`experiments._configs.RunConfig`. Pass the dotted path on the
command line: ``python experiments/train.py --config experiments.configs.<name>``.

Generated configs (Phase 2 ablation grid) live under ``configs/phase2/`` and
are gitignored — see ``experiments/scripts/gen_phase2_configs.py``.
"""
