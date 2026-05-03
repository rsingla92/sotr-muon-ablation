"""Unit tests for ``experiments/_logging.py``.

Covers:

- ``RunLogger`` writes JSONL, respects ``log_every_steps``, eval always logs,
  ``close`` is idempotent, context-manager works.
- ``StabilityMonitor`` fires the five PROTOCOL §8 incident types under their
  documented thresholds and stays quiet during smooth training.
- ``make_run_id`` returns the documented format and is unique across calls.
- ``capture_env`` writes a non-empty file and tolerates missing tools.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

from experiments._logging import (
    Incident,
    RunLogger,
    StabilityMonitor,
    capture_env,
    make_run_id,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# RunLogger
# ---------------------------------------------------------------------------
class TestRunLogger:
    def test_log_step_writes_jsonl_with_step_and_ts(self, tmp_path: Path) -> None:
        """Each log_step row has step, ts, and the supplied metrics."""
        log = RunLogger(tmp_path, log_every_steps=1)
        log.log_step(0, loss_nats=2.5, lr_step=1e-3)
        log.close()

        rows = _read_jsonl(tmp_path / "train.jsonl")
        assert len(rows) == 1
        assert rows[0]["step"] == 0
        assert rows[0]["loss_nats"] == 2.5
        assert rows[0]["lr_step"] == 1e-3
        assert isinstance(rows[0]["ts"], float)

    def test_log_step_respects_log_every_steps(self, tmp_path: Path) -> None:
        """Only step 0 and steps divisible by log_every_steps are written."""
        log = RunLogger(tmp_path, log_every_steps=10)
        for step in range(25):
            log.log_step(step, loss_nats=float(step))
        log.close()

        rows = _read_jsonl(tmp_path / "train.jsonl")
        assert [r["step"] for r in rows] == [0, 10, 20]

    def test_log_eval_always_writes(self, tmp_path: Path) -> None:
        """eval rows are written regardless of log_every_steps."""
        log = RunLogger(tmp_path, log_every_steps=1000)
        for step in (1, 2, 7):
            log.log_eval(step, val_loss_nats=float(step))
        log.close()

        rows = _read_jsonl(tmp_path / "eval.jsonl")
        assert [r["step"] for r in rows] == [1, 2, 7]
        # Train file should be empty (none of these steps hit log_every_steps).
        assert not (tmp_path / "train.jsonl").read_text().strip()

    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        """Calling close() twice does not raise."""
        log = RunLogger(tmp_path)
        log.log_step(0, loss_nats=1.0)
        log.close()
        log.close()  # must not raise

    def test_context_manager(self, tmp_path: Path) -> None:
        """`with RunLogger(...)` closes the file on exit and flushes contents."""
        with RunLogger(tmp_path, log_every_steps=1) as log:
            log.log_step(0, loss_nats=1.0)
        # File should be closed and contain the row.
        rows = _read_jsonl(tmp_path / "train.jsonl")
        assert len(rows) == 1

    def test_log_after_close_raises(self, tmp_path: Path) -> None:
        """Logging after close raises RuntimeError."""
        log = RunLogger(tmp_path)
        log.close()
        with pytest.raises(RuntimeError):
            log.log_step(0, loss_nats=1.0)

    def test_creates_run_dir(self, tmp_path: Path) -> None:
        """run_dir is created if it does not exist."""
        target = tmp_path / "newly" / "nested" / "run"
        log = RunLogger(target)
        log.close()
        assert target.is_dir()

    def test_invalid_log_every_steps(self, tmp_path: Path) -> None:
        """log_every_steps <= 0 is rejected."""
        with pytest.raises(ValueError):
            RunLogger(tmp_path, log_every_steps=0)

    def test_flushes_on_every_write(self, tmp_path: Path) -> None:
        """Crashed-run-safety: each log_step flushes immediately to disk."""
        log = RunLogger(tmp_path, log_every_steps=1)
        log.log_step(0, loss_nats=1.0)
        # Read without closing — flush() should have made the row visible.
        rows = _read_jsonl(tmp_path / "train.jsonl")
        assert len(rows) == 1
        log.close()


# ---------------------------------------------------------------------------
# StabilityMonitor
# ---------------------------------------------------------------------------
class TestStabilityMonitor:
    def _mk(self, tmp_path: Path, **kwargs) -> StabilityMonitor:
        return StabilityMonitor(tmp_path / "stability_incidents.jsonl", **kwargs)

    def test_spike_fires_on_loss_above_2x_rolling_mean(self, tmp_path: Path) -> None:
        """Spike fires when loss > 2× the rolling-window mean of prior losses."""
        mon = self._mk(tmp_path, rolling_window=10)
        for step in range(10):
            assert mon.check_step(step, loss=1.0) == []
        # Mean of prior window = 1.0; threshold = 2.0; 5.0 > 2.0 → spike.
        fired = mon.check_step(10, loss=5.0)
        assert [i.type for i in fired] == ["spike"]
        assert fired[0].severity == "recoverable"
        assert fired[0].value == 5.0
        assert mon.incident_counts["spike"] == 1

    def test_crash_fires_on_nan_or_inf(self, tmp_path: Path) -> None:
        """NaN or Inf in any signal fires a terminal crash."""
        mon = self._mk(tmp_path)
        fired_nan = mon.check_step(0, loss=float("nan"))
        fired_inf = mon.check_step(1, loss=1.0, grad_norm=float("inf"))
        assert any(i.type == "crash" and i.severity == "terminal" for i in fired_nan)
        assert any(i.type == "crash" for i in fired_inf)

    def test_blowup_fires_on_update_norm_above_10x_mean(self, tmp_path: Path) -> None:
        """Update norm > 10× rolling mean fires blowup."""
        mon = self._mk(tmp_path, rolling_window=10)
        for step in range(10):
            mon.check_step(step, loss=1.0, update_norm=1.0)
        fired = mon.check_step(10, loss=1.0, update_norm=20.0)
        assert any(i.type == "blowup" for i in fired)
        blowup = next(i for i in fired if i.type == "blowup")
        assert blowup.value == 20.0
        assert blowup.threshold == pytest.approx(10.0)

    def test_grad_spike_fires_on_grad_norm_above_100x_median(self, tmp_path: Path) -> None:
        """Grad norm > 100× rolling median fires grad_spike."""
        mon = self._mk(tmp_path, rolling_window=10, grad_window=20)
        for step in range(20):
            mon.check_step(step, loss=1.0, grad_norm=1.0)
        fired = mon.check_step(20, loss=1.0, grad_norm=200.0)
        assert any(i.type == "grad_spike" for i in fired)

    def test_rank_collapse_fires_on_50_percent_drop(self, tmp_path: Path) -> None:
        """check_stable_rank fires when current < 50% of initial."""
        mon = self._mk(tmp_path)
        # 60% of initial → no fire (drop is 40%, threshold is 50% drop).
        assert mon.check_stable_rank(100, "h.0.attn", stable_rank=6.0, initial=10.0) is None
        # 40% of initial → fire (drop is 60%).
        inc = mon.check_stable_rank(200, "h.0.attn", stable_rank=4.0, initial=10.0)
        assert inc is not None
        assert inc.type == "rank_collapse"
        assert inc.severity == "concerning"
        assert "h.0.attn" in inc.details

    def test_no_false_positives_during_stable_training(self, tmp_path: Path) -> None:
        """Smooth, slowly-decreasing loss + steady grad/update norms fire nothing."""
        mon = self._mk(tmp_path, rolling_window=50, grad_window=200)
        for step in range(500):
            loss = 3.0 * math.exp(-step / 1000.0) + 0.1
            grad = 1.0 + 0.05 * math.sin(step / 10.0)
            update = 0.5 + 0.02 * math.cos(step / 7.0)
            fired = mon.check_step(step, loss=loss, grad_norm=grad, update_norm=update)
            assert fired == [], f"unexpected incident at step {step}: {fired}"
        assert sum(mon.incident_counts.values()) == 0

    def test_rolling_window_not_yet_full_no_false_positives(self, tmp_path: Path) -> None:
        """Before the window has any prior samples, threshold checks must skip."""
        mon = self._mk(tmp_path, rolling_window=100)
        # Very first sample — no prior data, must not fire spike/blowup/grad_spike.
        fired = mon.check_step(0, loss=42.0, grad_norm=42.0, update_norm=42.0)
        assert fired == []
        # Next sample even if much larger: only one prior datapoint exists,
        # but the rolling mean check is well-defined and this is a real spike.
        # We just verify no crash.
        mon.check_step(1, loss=200.0, grad_norm=200.0, update_norm=200.0)

    def test_outlier_does_not_contaminate_its_own_baseline(self, tmp_path: Path) -> None:
        """A spike step is checked against the *prior* window, not the post-append window."""
        mon = self._mk(tmp_path, rolling_window=5)
        for step in range(5):
            mon.check_step(step, loss=1.0)
        fired = mon.check_step(5, loss=10.0)
        spikes = [i for i in fired if i.type == "spike"]
        assert len(spikes) == 1
        # Threshold should be 2× the mean of the 5 prior 1.0s = 2.0, not contaminated.
        assert spikes[0].threshold == pytest.approx(2.0)

    def test_incidents_written_to_jsonl(self, tmp_path: Path) -> None:
        """Each fired incident appends a JSON line to the incidents file."""
        path = tmp_path / "stability_incidents.jsonl"
        mon = StabilityMonitor(path, rolling_window=5)
        for step in range(5):
            mon.check_step(step, loss=1.0)
        mon.check_step(5, loss=10.0)
        mon.close()

        rows = _read_jsonl(path)
        assert len(rows) == 1
        assert rows[0]["type"] == "spike"
        assert rows[0]["step"] == 5

    @pytest.mark.parametrize(
        "incident_type,expected_severity",
        [
            ("spike", "recoverable"),
            ("crash", "terminal"),
            ("blowup", "recoverable"),
            ("grad_spike", "recoverable"),
            ("rank_collapse", "concerning"),
        ],
    )
    def test_severity_per_protocol(self, incident_type: str, expected_severity: str) -> None:
        """Severity assignments match PROTOCOL §8 exactly."""
        from experiments._logging import _SEVERITY

        assert _SEVERITY[incident_type] == expected_severity

    def test_summary_has_counts_and_total(self, tmp_path: Path) -> None:
        """summary() reports per-type counts and aggregate total."""
        mon = self._mk(tmp_path, rolling_window=5)
        for step in range(5):
            mon.check_step(step, loss=1.0)
        mon.check_step(5, loss=10.0)
        mon.check_step(6, loss=float("nan"))

        s = mon.summary()
        assert s["counts"]["spike"] == 1
        assert s["counts"]["crash"] == 1
        assert s["total"] == 2

    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        """close() is safe to call repeatedly."""
        mon = self._mk(tmp_path)
        mon.close()
        mon.close()


# ---------------------------------------------------------------------------
# make_run_id
# ---------------------------------------------------------------------------
class TestMakeRunId:
    _PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}_[0-9a-f]{6}$")

    def test_format(self) -> None:
        """Run id matches YYYY-MM-DD_HHMMSS_<6-hex>."""
        rid = make_run_id()
        assert self._PATTERN.match(rid), rid

    def test_consecutive_calls_are_unique(self) -> None:
        """Two back-to-back calls return different ids (salt prevents collision)."""
        ids = {make_run_id() for _ in range(10)}
        assert len(ids) == 10

    def test_with_config_hash_input_still_unique(self) -> None:
        """Even with the same config input, salt keeps ids distinct."""
        cfg = {"lr": 1e-3, "alpha": 0.5}
        ids = {make_run_id(cfg) for _ in range(5)}
        assert len(ids) == 5

    def test_with_config_hash_input_matches_format(self) -> None:
        """Config-mixin id still matches the documented format."""
        rid = make_run_id({"any": "config"})
        assert self._PATTERN.match(rid), rid


# ---------------------------------------------------------------------------
# capture_env
# ---------------------------------------------------------------------------
class TestCaptureEnv:
    def test_writes_non_empty_file(self, tmp_path: Path) -> None:
        """env.txt is written and is non-empty."""
        out = tmp_path / "env.txt"
        capture_env(out)
        assert out.exists()
        text = out.read_text()
        assert len(text) > 0

    def test_contains_expected_sections(self, tmp_path: Path) -> None:
        """env.txt has the documented section headers."""
        out = tmp_path / "env.txt"
        capture_env(out)
        text = out.read_text()
        for section in ("git", "python", "torch", "cuda", "pip", "host", "modules"):
            assert f"## {section}" in text, f"missing section: {section}"

    def test_does_not_crash_when_tools_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """capture_env tolerates missing nvidia-smi / git / module / pip."""
        # Force every subprocess call to behave as if the binary is missing.
        import subprocess as sp

        def _missing(*args, **kwargs):
            raise FileNotFoundError("simulated missing binary")

        monkeypatch.setattr(sp, "run", _missing)
        out = tmp_path / "env.txt"
        capture_env(out)  # must not raise
        assert out.exists()
        assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# Incident dataclass
# ---------------------------------------------------------------------------
def test_incident_is_frozen() -> None:
    """Incident is a frozen dataclass (immutable post-construction)."""
    from dataclasses import FrozenInstanceError

    inc = Incident(step=1, type="spike", severity="recoverable", value=2.0, threshold=1.0)
    with pytest.raises(FrozenInstanceError):
        inc.value = 5.0  # type: ignore[misc]
