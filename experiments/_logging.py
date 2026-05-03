"""Per-run logging, stability monitoring, and provenance capture.

Phase 2 logging module. See:

- ``PROTOCOL.md`` §8 for the locked stability incident definitions and severities
  (spike, crash, blowup, grad_spike, rank_collapse).
- ``PROTOCOL.md`` §12 for what every result table must report (and therefore
  what each run directory must contain).
- ``CONTRIBUTING.md`` §"Logging" for the per-run directory layout
  (``train.jsonl``, ``eval.jsonl``, ``stability_incidents.jsonl``, ``env.txt``)
  and the unit-suffixed metric naming convention.

Public API:

- :class:`RunLogger` — JSONL writer for per-step / per-eval metrics.
- :class:`StabilityMonitor` — rolling-window incident detector matching §8.
- :class:`Incident` — frozen dataclass describing a single fired incident.
- :func:`capture_env` — write ``env.txt`` (git, torch, CUDA, hostname, SLURM, ...).
- :func:`make_run_id` — sortable ``YYYY-MM-DD_HHMMSS_<6-hex>`` run id.

Metric naming convention (units suffix; see CONTRIBUTING.md §"Logging"):
``loss_nats``, ``time_s``, ``tokens_per_s``, ``lr_step``, ``grad_norm_l2``,
``update_norm_f``, ``clip_rate``, etc. ``RunLogger`` does not enforce names;
callers should follow the convention.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType

__all__ = [
    "Incident",
    "RunLogger",
    "StabilityMonitor",
    "capture_env",
    "make_run_id",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RunLogger
# ---------------------------------------------------------------------------
class RunLogger:
    """Append-only JSONL writer for per-step training and eval metrics.

    Writes two files into ``run_dir``:

    - ``train.jsonl`` — one JSON line per logged training step.
    - ``eval.jsonl`` — one JSON line per evaluation event (always written).

    Each line is ``{"step": int, "ts": float, **metrics}``. ``ts`` is wall-clock
    time as ``time.time()``. The file is flushed after every write so a crashed
    run still has data on disk.

    Metric-name convention (see ``CONTRIBUTING.md`` §"Logging"): every metric
    carries a units suffix, e.g. ``loss_nats``, ``time_s``, ``tokens_per_s``,
    ``lr_step``, ``grad_norm_l2``, ``update_norm_f``, ``clip_rate``. The class
    does not enforce these names — it accepts arbitrary ``**metrics`` — but
    callers are expected to follow the convention.

    Args:
        run_dir: directory to write logs into. Created if missing.
        log_every_steps: write a training row every N steps. Step 0 always logs.
            Eval rows are unaffected and always log.
    """

    def __init__(self, run_dir: Path, *, log_every_steps: int = 50) -> None:
        if log_every_steps <= 0:
            raise ValueError(f"log_every_steps must be > 0, got {log_every_steps}")

        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_every_steps = log_every_steps

        self._train_path = self.run_dir / "train.jsonl"
        self._eval_path = self.run_dir / "eval.jsonl"
        self._train_fh = self._train_path.open("a", encoding="utf-8")
        self._eval_fh = self._eval_path.open("a", encoding="utf-8")
        self._closed = False

    def log_step(self, step: int, **metrics: object) -> None:
        """Write a training row if ``step == 0`` or ``step % log_every_steps == 0``."""
        if self._closed:
            raise RuntimeError("RunLogger is closed.")
        if step != 0 and step % self.log_every_steps != 0:
            return
        self._write(self._train_fh, step, metrics)

    def log_eval(self, step: int, **metrics: object) -> None:
        """Write an eval row. Always writes, regardless of ``log_every_steps``."""
        if self._closed:
            raise RuntimeError("RunLogger is closed.")
        self._write(self._eval_fh, step, metrics)

    def close(self) -> None:
        """Flush and close file handles. Idempotent."""
        if self._closed:
            return
        for fh in (self._train_fh, self._eval_fh):
            try:
                fh.flush()
                fh.close()
            except Exception:
                logger.exception("error closing log file %s", getattr(fh, "name", "?"))
        self._closed = True

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        # Best-effort; suppress errors at interpreter shutdown.
        with contextlib.suppress(Exception):
            self.close()

    @staticmethod
    def _write(fh, step: int, metrics: dict[str, object]) -> None:
        row = {"step": step, "ts": time.time(), **metrics}
        fh.write(json.dumps(row, default=_json_default) + "\n")
        fh.flush()


def _json_default(obj: object) -> object:
    """Fall back to ``float()`` / ``str()`` for non-JSON-native scalars (e.g. tensors)."""
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


# ---------------------------------------------------------------------------
# StabilityMonitor — PROTOCOL §8
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Incident:
    """A single stability incident, persisted to ``stability_incidents.jsonl``.

    Severity levels follow PROTOCOL §8: ``recoverable`` (training continues),
    ``terminal`` (NaN/Inf — run is dead), ``concerning`` (worth flagging but
    not actionable mid-run, e.g. rank collapse).
    """

    step: int
    type: str
    severity: str
    value: float
    threshold: float
    details: str = ""


_SEVERITY = {
    "spike": "recoverable",
    "crash": "terminal",
    "blowup": "recoverable",
    "grad_spike": "recoverable",
    "rank_collapse": "concerning",
}


@dataclass
class _RollingStats:
    """Fixed-size deque with mean and median helpers (recomputed on demand)."""

    maxlen: int
    values: deque[float] = field(init=False)

    def __post_init__(self) -> None:
        self.values = deque(maxlen=self.maxlen)

    def append(self, x: float) -> None:
        self.values.append(x)

    def mean(self) -> float | None:
        return statistics.fmean(self.values) if self.values else None

    def median(self) -> float | None:
        return statistics.median(self.values) if self.values else None

    def __len__(self) -> int:
        return len(self.values)


class StabilityMonitor:
    """Detect PROTOCOL §8 stability incidents from streaming step metrics.

    Maintains rolling buffers of training loss (window=``rolling_window``,
    default 100), update Frobenius norm (same window), and gradient norm
    (window=``grad_window``, default 1000). Each call to :meth:`check_step`
    returns a list of incidents fired this step (often empty) and writes any
    fired incidents to ``stability_incidents.jsonl`` in append mode.

    Incident definitions (from PROTOCOL §8):

    - ``spike``: loss > 2× rolling-window mean of loss.
    - ``crash``: NaN or Inf in loss, grad_norm, or update_norm. Terminal.
    - ``blowup``: update_norm > 10× rolling-window mean of update_norm.
    - ``grad_spike``: grad_norm > 100× rolling-``grad_window`` median of grad_norm.
    - ``rank_collapse``: stable_rank dropped >50% from initialization (separate
      method, :meth:`check_stable_rank` — caller tracks initial values).

    Comparisons require the rolling window to be non-empty for the relevant
    metric; the new sample is checked against the *prior* window state and
    only then appended (so a fresh monitor never falsely fires on its first
    sample).

    Args:
        out_path: path to ``stability_incidents.jsonl``. Parent dir is created.
        rolling_window: window for spike (loss) and blowup (update norm).
        grad_window: window for grad_spike (grad norm median).
    """

    SPIKE_RATIO = 2.0
    BLOWUP_RATIO = 10.0
    GRAD_SPIKE_RATIO = 100.0
    RANK_COLLAPSE_DROP = 0.5

    def __init__(
        self,
        out_path: Path,
        *,
        rolling_window: int = 100,
        grad_window: int = 1000,
    ) -> None:
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.out_path.open("a", encoding="utf-8")

        self._loss = _RollingStats(maxlen=rolling_window)
        self._update = _RollingStats(maxlen=rolling_window)
        self._grad = _RollingStats(maxlen=grad_window)
        self._counts: dict[str, int] = dict.fromkeys(_SEVERITY, 0)
        self._closed = False

    def check_step(
        self,
        step: int,
        *,
        loss: float,
        grad_norm: float | None = None,
        update_norm: float | None = None,
    ) -> list[Incident]:
        """Examine one new sample, return incidents fired this step.

        Order of checks: crash (NaN/Inf), spike, blowup, grad_spike. A single
        step can fire multiple incidents (e.g., crash + spike). Each fired
        incident is written to ``stability_incidents.jsonl`` immediately.

        After all checks, the new sample is appended to its rolling buffer so
        future steps see the updated window.
        """
        fired: list[Incident] = []

        # crash: NaN/Inf in any provided signal.
        for name, val in (("loss", loss), ("grad_norm", grad_norm), ("update_norm", update_norm)):
            if val is not None and not math.isfinite(val):
                fired.append(
                    Incident(
                        step=step,
                        type="crash",
                        severity=_SEVERITY["crash"],
                        value=float(val),
                        threshold=math.inf,
                        details=f"non-finite {name}",
                    )
                )

        # spike: loss > 2× rolling mean. Requires non-empty prior window.
        loss_mean = self._loss.mean()
        if loss_mean is not None and math.isfinite(loss) and math.isfinite(loss_mean):
            threshold = self.SPIKE_RATIO * loss_mean
            if loss > threshold:
                fired.append(
                    Incident(
                        step=step,
                        type="spike",
                        severity=_SEVERITY["spike"],
                        value=float(loss),
                        threshold=float(threshold),
                        details=f"rolling-{self._loss.maxlen} mean={loss_mean:.6g}",
                    )
                )

        # blowup: update_norm > 10× rolling mean of update_norm.
        if update_norm is not None and math.isfinite(update_norm):
            up_mean = self._update.mean()
            if up_mean is not None and math.isfinite(up_mean) and up_mean > 0:
                threshold = self.BLOWUP_RATIO * up_mean
                if update_norm > threshold:
                    fired.append(
                        Incident(
                            step=step,
                            type="blowup",
                            severity=_SEVERITY["blowup"],
                            value=float(update_norm),
                            threshold=float(threshold),
                            details=f"rolling-{self._update.maxlen} mean={up_mean:.6g}",
                        )
                    )

        # grad_spike: grad_norm > 100× rolling-grad_window median.
        if grad_norm is not None and math.isfinite(grad_norm):
            g_med = self._grad.median()
            if g_med is not None and math.isfinite(g_med) and g_med > 0:
                threshold = self.GRAD_SPIKE_RATIO * g_med
                if grad_norm > threshold:
                    fired.append(
                        Incident(
                            step=step,
                            type="grad_spike",
                            severity=_SEVERITY["grad_spike"],
                            value=float(grad_norm),
                            threshold=float(threshold),
                            details=f"rolling-{self._grad.maxlen} median={g_med:.6g}",
                        )
                    )

        for inc in fired:
            self._record(inc)

        # Append finite samples after checking, so this step's outlier doesn't
        # contaminate its own rolling baseline.
        if math.isfinite(loss):
            self._loss.append(loss)
        if update_norm is not None and math.isfinite(update_norm):
            self._update.append(update_norm)
        if grad_norm is not None and math.isfinite(grad_norm):
            self._grad.append(grad_norm)

        return fired

    def check_stable_rank(
        self, step: int, layer_name: str, stable_rank: float, *, initial: float
    ) -> Incident | None:
        """Fire ``rank_collapse`` if stable_rank dropped >50% from ``initial``.

        Caller is responsible for tracking the initialization value per layer
        (typically captured once, before training, on the parameter at step 0).
        """
        if initial <= 0 or not math.isfinite(initial) or not math.isfinite(stable_rank):
            return None
        threshold = (1.0 - self.RANK_COLLAPSE_DROP) * initial
        if stable_rank >= threshold:
            return None
        inc = Incident(
            step=step,
            type="rank_collapse",
            severity=_SEVERITY["rank_collapse"],
            value=float(stable_rank),
            threshold=float(threshold),
            details=f"layer={layer_name} initial={initial:.6g}",
        )
        self._record(inc)
        return inc

    @property
    def incident_counts(self) -> dict[str, int]:
        return dict(self._counts)

    def summary(self) -> dict[str, object]:
        return {
            "counts": self.incident_counts,
            "total": sum(self._counts.values()),
            "rolling_window": self._loss.maxlen,
            "grad_window": self._grad.maxlen,
        }

    def close(self) -> None:
        """Flush and close the incidents file. Idempotent."""
        if self._closed:
            return
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            logger.exception("error closing %s", self.out_path)
        self._closed = True

    def __enter__(self) -> StabilityMonitor:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()

    def _record(self, inc: Incident) -> None:
        self._counts[inc.type] = self._counts.get(inc.type, 0) + 1
        self._fh.write(json.dumps(asdict(inc)) + "\n")
        self._fh.flush()


# ---------------------------------------------------------------------------
# Provenance: capture_env
# ---------------------------------------------------------------------------
_PIP_FREEZE_KEEP = {
    "torch",
    "numpy",
    "tiktoken",
    "datasets",
    "huggingface-hub",
    "kernels",
    "muon-optimizer",
    "lion-pytorch",
    "optimizer-experiments",
}


def capture_env(out_path: Path) -> None:
    """Write a structured ``env.txt`` snapshot for reproducibility (PROTOCOL §12).

    Captures git SHA + dirty status, Python and torch versions, CUDA + GPU info
    (via ``nvidia-smi`` if available), filtered ``pip freeze`` (key packages
    only), hostname, SLURM env vars, and any loaded modules from ``module list``.
    Subprocess calls use ``check=False`` and a short timeout so missing tools
    do not crash the snapshot.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sections: list[str] = []

    sections.append("# env.txt — captured " + datetime.now(timezone.utc).isoformat())

    git_sha = _run(["git", "rev-parse", "HEAD"])
    git_status = _run(["git", "status", "--short"])
    sections.append(_section("git", f"sha: {git_sha or '<unknown>'}\nstatus:\n{git_status}"))

    py = (
        f"version: {sys.version.replace(chr(10), ' ')}\n"
        f"executable: {sys.executable}\n"
        f"platform: {platform.platform()}"
    )
    sections.append(_section("python", py))

    sections.append(_section("torch", _torch_info()))

    sections.append(_section("cuda", _cuda_info()))

    sections.append(_section("pip (filtered)", _pip_freeze_filtered()))

    host_lines = [f"hostname: {socket.gethostname()}"]
    for k in ("SLURM_JOB_ID", "SLURM_CLUSTER_NAME", "SLURM_NODELIST", "SLURM_JOB_NAME"):
        v = os.environ.get(k)
        if v:
            host_lines.append(f"{k}: {v}")
    sections.append(_section("host", "\n".join(host_lines)))

    sections.append(_section("modules", _module_list()))

    out.write_text("\n\n".join(sections) + "\n", encoding="utf-8")


def _section(title: str, body: str) -> str:
    return f"## {title}\n{body.rstrip()}"


def _run(cmd: list[str], *, timeout: float = 5.0) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return ""
    return (r.stdout or "").strip()


def _torch_info() -> str:
    try:
        import torch
    except ImportError:
        return "torch: not installed"
    try:
        build = torch.__config__.show()
    except Exception:
        build = "<torch.__config__.show() failed>"
    return f"version: {torch.__version__}\nbuild:\n{build}"


def _cuda_info() -> str:
    lines: list[str] = []
    try:
        import torch

        lines.append(f"available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"device_count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                lines.append(f"device[{i}]: {torch.cuda.get_device_name(i)}")
            with contextlib.suppress(Exception):
                lines.append(f"cuda_version: {torch.version.cuda}")
    except ImportError:
        lines.append("torch: not installed")

    smi = _run(["nvidia-smi", "-L"])
    if smi:
        lines.append("nvidia-smi -L:")
        lines.append(smi)
    return "\n".join(lines)


def _pip_freeze_filtered() -> str:
    raw = _run([sys.executable, "-m", "pip", "freeze"], timeout=15.0)
    if not raw:
        return "<pip freeze unavailable>"
    kept = []
    for line in raw.splitlines():
        # Match against the package name (before any '==' / ' @ ').
        name = line.split("==", 1)[0].split(" @ ", 1)[0].strip().lower()
        if name in _PIP_FREEZE_KEEP:
            kept.append(line)
    return "\n".join(kept) if kept else "<no tracked packages found>"


def _module_list() -> str:
    # `module` is a shell function on most clusters, not a binary; route through bash.
    try:
        r = subprocess.run(
            ["bash", "-lc", "module list 2>&1"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return "<module command unavailable>"
    out = (r.stdout or "").strip()
    if not out or "command not found" in out.lower() or "module: not found" in out.lower():
        return "<module command unavailable>"
    return out


# ---------------------------------------------------------------------------
# Run IDs
# ---------------------------------------------------------------------------
def make_run_id(config_hash_input: object | None = None) -> str:
    """Return a sortable unique run id ``YYYY-MM-DD_HHMMSS_<6-hex>``.

    The 6-hex suffix is for collision avoidance. If ``config_hash_input`` is
    given, it is mixed in (sha256 of ``repr(config_hash_input)`` xor'd against
    8 random bytes), so two runs of the *same* config get distinct ids while
    still embedding a deterministic component. If ``None``, the suffix is
    pure ``os.urandom(8)``.
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    salt = os.urandom(8)
    if config_hash_input is not None:
        h = hashlib.sha256(repr(config_hash_input).encode("utf-8")).digest()[:8]
        salt = bytes(a ^ b for a, b in zip(salt, h, strict=True))
    return f"{ts}_{salt.hex()[:6]}"
