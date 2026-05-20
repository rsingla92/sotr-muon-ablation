"""KataGo wrapper smoke test. Skipped unless KATAGO_BIN and KATAGO_MODEL
are set (i.e. only runs on the cluster or a dev box with KataGo installed).

The Phase-0 verification gate requires: 100 calls to analyze() return
well-formed results with no subprocess leaks. This test runs that check.
"""

from __future__ import annotations

import subprocess

import pytest


pytestmark = pytest.mark.requires_katago


def _pgrep_katago() -> list[int]:
    try:
        out = subprocess.check_output(["pgrep", "-f", "katago"], text=True)
    except subprocess.CalledProcessError:
        return []
    return [int(line) for line in out.strip().splitlines() if line.strip()]


def test_katago_wrapper_no_subprocess_leak():
    from gogpt.katago import KataGo, KataGoConfig, default_model_path, find_katago_binary

    pids_before = set(_pgrep_katago())

    cfg = KataGoConfig(
        binary=find_katago_binary(),
        model=default_model_path(),
        default_visits=8,
        request_ownership=False,
    )

    with KataGo(cfg) as kg:
        for i in range(100):
            r = kg.analyze(num_visits=2)
            assert r.top_move is not None
            assert r.to_move in ("B", "W")
        # After 100 queries, the subprocess should still be alive.
        new_pids = set(_pgrep_katago()) - pids_before
        assert new_pids, "expected the KataGo child process to be visible"

    # After context exit, no new katago PIDs should remain.
    pids_after = set(_pgrep_katago())
    leaked = (pids_after - pids_before)
    assert not leaked, f"KataGo subprocess(es) leaked: {leaked}"
