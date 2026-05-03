"""Optimizers for the SOTR / Muon-family research project.

Public API:
    SOTR — the only optimizer we implement. See ``optimizers/sotr.py`` and
        ``PROTOCOL.md`` §7.
    Lion — re-exported from ``lion_pytorch`` (lucidrains, MIT). Imported here
        for convenience; ``from optimizers import Lion`` is equivalent to
        ``from lion_pytorch import Lion``.

We deliberately do NOT re-export Muon. Use it directly:
    ``from muon import Muon, MuonWithAuxAdam, SingleDeviceMuon``

This makes the lineage of every baseline explicit at the import site. See
``optimizers/README.md``.
"""

from lion_pytorch import Lion

from optimizers.sotr import SOTR, sotr_update

__all__ = ["SOTR", "Lion", "sotr_update"]
