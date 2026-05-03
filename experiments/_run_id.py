"""Deterministic-prefix run-ID generator.

A run ID is ``YYYY-MM-DD_HHMMSS_<6-hex>``. The 6-hex tail is salted by
``os.urandom`` for collision avoidance, optionally mixed with a config hash
for traceability.

Single function: :func:`make_run_id`. See ``experiments/_logging.make_run_id``
for the equivalent inside the logging module — kept here as a separate file
because :mod:`experiments.train` imports run-ID generation from a path that
is independent of logging (so it works even if logging is disabled, e.g.
during smoke tests).
"""

from __future__ import annotations

import hashlib
import os
import time

__all__ = ["make_run_id"]


def make_run_id(config_hash_input: object | None = None) -> str:
    """Return ``YYYY-MM-DD_HHMMSS_<6-hex>``.

    Args:
        config_hash_input: optional object whose ``repr`` is mixed into the
            tail hex. Useful for making the run id deterministic per-config
            (so re-running the same config in the same second collides;
            re-running with a different config does not).
    """
    timestamp = time.strftime("%Y-%m-%d_%H%M%S", time.localtime())
    salt = os.urandom(8)
    if config_hash_input is not None:
        cfg_bytes = repr(config_hash_input).encode("utf-8")
        salt = bytes(
            a ^ b for a, b in zip(salt, hashlib.sha256(cfg_bytes).digest()[:8], strict=True)
        )
    tail = hashlib.sha256(salt).hexdigest()[:6]
    return f"{timestamp}_{tail}"
