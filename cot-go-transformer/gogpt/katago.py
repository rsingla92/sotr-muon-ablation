"""KataGo analysis-engine subprocess wrapper.

Spawns ``katago analysis -config ... -model ...`` as a long-lived subprocess
and speaks the line-delimited JSON protocol documented at
https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md.

Design notes:
- One ``KataGo`` instance corresponds to one subprocess. The wrapper is *not*
  thread-safe; for multi-process data generation, spawn one wrapper per worker.
- The wrapper retries up to ``max_restarts`` times on subprocess crash or
  timeout. State (current game's moves) is replayed on restart so the engine
  is consistent with where the wrapper thinks the game is.
- Use as a context manager so the subprocess is reliably killed (KataGo
  sometimes ignores SIGTERM; we escalate to SIGKILL after a grace period).
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from typing import Any, Iterable

log = logging.getLogger(__name__)


@dataclass
class MoveAnalysis:
    move: str          # GTP vertex, e.g. 'D4' or 'pass'
    visits: int
    winrate: float     # from the perspective of the side to move
    score_lead: float  # likewise
    prior: float
    order: int         # KataGo's rank for this move at the current visit count


@dataclass
class AnalysisResult:
    """Parsed analysis response for one position."""

    query_id: str
    to_move: str               # 'B' or 'W'
    root_winrate: float
    root_score_lead: float
    move_infos: list[MoveAnalysis] = field(default_factory=list)
    ownership: list[float] | None = None      # length 81, [-1, 1], black-positive
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def top_move(self) -> MoveAnalysis | None:
        return self.move_infos[0] if self.move_infos else None


@dataclass
class KataGoConfig:
    binary: str = "katago"
    model: str = ""             # path to a 9x9-capable .bin.gz network
    config: str | None = None   # path to analysis_example.cfg or similar
    rules: str = "tromp-taylor"  # ko / scoring; see KataGo docs
    komi: float = 7.0           # standard 9x9 komi
    default_visits: int = 100
    request_ownership: bool = True
    max_visits_cap: int = 4000  # hard cap to keep per-query latency bounded
    startup_timeout_s: float = 60.0
    query_timeout_s: float = 120.0
    shutdown_grace_s: float = 5.0
    max_restarts: int = 3


class KataGoError(RuntimeError):
    pass


class KataGo:
    """Long-lived KataGo analysis-engine wrapper.

    Usage:

        with KataGo(KataGoConfig(model="...")) as kg:
            result = kg.analyze(moves=[('B', 'E5')], num_visits=200)
    """

    def __init__(self, cfg: KataGoConfig):
        if not cfg.model:
            raise ValueError("KataGoConfig.model must be set")
        self.cfg = cfg
        self._proc: subprocess.Popen[bytes] | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stderr_lines: Queue[str] = Queue(maxsize=2048)
        self._restart_count = 0
        # Game state replayed on restart: list of (color, GTP vertex).
        self._game_moves: list[tuple[str, str]] = []

    # -- lifecycle ----------------------------------------------------------

    def __enter__(self) -> "KataGo":
        self._spawn()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _spawn(self) -> None:
        argv = [self.cfg.binary, "analysis", "-model", self.cfg.model]
        if self.cfg.config:
            argv += ["-config", self.cfg.config]
        log.info("spawning KataGo: %s", " ".join(argv))
        self._proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            close_fds=True,
            preexec_fn=os.setsid,  # so we can kill the whole process group
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True, name="katago-stderr"
        )
        self._stderr_thread.start()
        # KataGo prints startup banner; wait for it to become quiescent by
        # sending a trivial query and waiting for a response.
        self._wait_until_ready()

    def _drain_stderr(self) -> None:
        assert self._proc is not None
        assert self._proc.stderr is not None
        for raw_line in self._proc.stderr:
            try:
                line = raw_line.decode("utf-8", errors="replace").rstrip()
            except Exception:
                continue
            try:
                self._stderr_lines.put_nowait(line)
            except Exception:
                pass

    def _wait_until_ready(self) -> None:
        # Send a tiny, cheap query and discard the response.
        probe_id = f"probe-{uuid.uuid4().hex[:8]}"
        probe = {
            "id": probe_id,
            "moves": [],
            "rules": self.cfg.rules,
            "komi": self.cfg.komi,
            "boardXSize": 9,
            "boardYSize": 9,
            "maxVisits": 1,
            "includeOwnership": False,
        }
        self._send_raw(probe)
        deadline = time.monotonic() + self.cfg.startup_timeout_s
        while time.monotonic() < deadline:
            resp = self._read_one(timeout=deadline - time.monotonic())
            if resp is None:
                continue
            if resp.get("id") == probe_id:
                return
        raise KataGoError("KataGo failed to respond to startup probe in time")

    def close(self) -> None:
        if self._proc is None:
            return
        proc = self._proc
        self._proc = None
        try:
            if proc.stdin and not proc.stdin.closed:
                try:
                    proc.stdin.close()
                except BrokenPipeError:
                    pass
            try:
                proc.wait(timeout=self.cfg.shutdown_grace_s)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=self.cfg.shutdown_grace_s)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait(timeout=self.cfg.shutdown_grace_s)
        finally:
            try:
                if proc.stdout:
                    proc.stdout.close()
                if proc.stderr:
                    proc.stderr.close()
            except Exception:
                pass

    # -- public API ---------------------------------------------------------

    def reset_game(self) -> None:
        """Forget the played-move history (next analyze() starts from empty)."""
        self._game_moves = []

    def play_move(self, color: str, vertex: str) -> None:
        """Record a played move so subsequent analyze() includes it."""
        color = color.upper()
        if color not in ("B", "W"):
            raise ValueError(f"color must be 'B' or 'W', got {color!r}")
        self._game_moves.append((color, vertex))

    def analyze(
        self,
        moves: Iterable[tuple[str, str]] | None = None,
        num_visits: int | None = None,
        include_ownership: bool | None = None,
    ) -> AnalysisResult:
        """Analyze the position reached by ``moves`` (or current game state).

        ``moves`` is an iterable of (color, GTP-vertex) pairs played from the
        empty board. If None, uses moves recorded via play_move().
        """
        if moves is None:
            move_list = list(self._game_moves)
        else:
            move_list = list(moves)
        visits = num_visits or self.cfg.default_visits
        if visits > self.cfg.max_visits_cap:
            visits = self.cfg.max_visits_cap
        include_own = (
            self.cfg.request_ownership if include_ownership is None else include_ownership
        )
        query_id = f"q-{uuid.uuid4().hex[:12]}"
        query = {
            "id": query_id,
            "moves": [[c, v] for c, v in move_list],
            "rules": self.cfg.rules,
            "komi": self.cfg.komi,
            "boardXSize": 9,
            "boardYSize": 9,
            "maxVisits": visits,
            "includeOwnership": include_own,
        }
        resp = self._query_with_retry(query)
        return self._parse(resp, query_id, move_list)

    # -- internals ----------------------------------------------------------

    def _query_with_retry(self, query: dict) -> dict:
        last_err: Exception | None = None
        for attempt in range(self.cfg.max_restarts + 1):
            try:
                self._send_raw(query)
                deadline = time.monotonic() + self.cfg.query_timeout_s
                while time.monotonic() < deadline:
                    resp = self._read_one(timeout=deadline - time.monotonic())
                    if resp is None:
                        continue
                    if resp.get("id") == query["id"]:
                        if "error" in resp:
                            raise KataGoError(f"KataGo error: {resp['error']}")
                        return resp
                raise KataGoError("KataGo query timed out")
            except (BrokenPipeError, KataGoError) as e:
                last_err = e
                log.warning("KataGo query failed (attempt %d): %s", attempt, e)
                if attempt >= self.cfg.max_restarts:
                    break
                self._restart()
        raise KataGoError(f"KataGo query failed after restarts: {last_err}")

    def _restart(self) -> None:
        self._restart_count += 1
        if self._restart_count > self.cfg.max_restarts:
            raise KataGoError("exceeded max KataGo restarts")
        self.close()
        self._spawn()

    def _send_raw(self, obj: dict) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise KataGoError("KataGo subprocess is not running")
        payload = (json.dumps(obj) + "\n").encode("utf-8")
        try:
            self._proc.stdin.write(payload)
            self._proc.stdin.flush()
        except BrokenPipeError:
            raise
        except OSError as e:
            raise KataGoError(f"failed to write to KataGo: {e}") from e

    def _read_one(self, timeout: float) -> dict | None:
        """Read one JSON object from stdout, or None on timeout."""
        if self._proc is None or self._proc.stdout is None:
            raise KataGoError("KataGo subprocess is not running")
        if timeout <= 0:
            return None
        # readline blocks; rely on the per-query deadline at the caller for
        # the overall timeout. We approximate with a poll loop on a thread
        # in the rare case we need to time out a slow query.
        result: dict[str, dict | Exception | None] = {"value": None}
        done = threading.Event()

        def _reader() -> None:
            try:
                line = self._proc.stdout.readline()  # type: ignore[union-attr]
                if not line:
                    result["value"] = KataGoError("KataGo stdout closed unexpectedly")
                else:
                    result["value"] = json.loads(line.decode("utf-8"))
            except Exception as e:
                result["value"] = e
            finally:
                done.set()

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        if not done.wait(timeout=timeout):
            return None
        v = result["value"]
        if isinstance(v, Exception):
            raise v if isinstance(v, KataGoError) else KataGoError(str(v))
        return v  # type: ignore[return-value]

    def _parse(
        self,
        resp: dict,
        query_id: str,
        move_list: list[tuple[str, str]],
    ) -> AnalysisResult:
        # KataGo reports rootInfo and an ordered moveInfos array.
        root_info = resp.get("rootInfo", {})
        to_move = "B" if len(move_list) % 2 == 0 else "W"
        # KataGo's winrate is from the side to move. Keep that convention.
        infos: list[MoveAnalysis] = []
        for mi in resp.get("moveInfos", []):
            infos.append(
                MoveAnalysis(
                    move=mi["move"],
                    visits=int(mi.get("visits", 0)),
                    winrate=float(mi.get("winrate", 0.0)),
                    score_lead=float(mi.get("scoreLead", 0.0)),
                    prior=float(mi.get("prior", 0.0)),
                    order=int(mi.get("order", 0)),
                )
            )
        # Ownership: KataGo returns 81 floats in row-major board order, each in
        # [-1, 1] where positive favors black.
        ownership = resp.get("ownership")
        return AnalysisResult(
            query_id=query_id,
            to_move=to_move,
            root_winrate=float(root_info.get("winrate", 0.0)),
            root_score_lead=float(root_info.get("scoreLead", 0.0)),
            move_infos=infos,
            ownership=list(ownership) if ownership is not None else None,
            raw=resp,
        )


def find_katago_binary() -> str:
    """Locate the KataGo binary on $PATH or in common cluster locations."""
    candidates = [
        os.environ.get("KATAGO_BIN"),
        "katago",
        str(Path.home() / "katago" / "katago"),
        "/opt/katago/katago",
    ]
    for c in candidates:
        if not c:
            continue
        if os.path.isabs(c) and os.access(c, os.X_OK):
            return c
        # PATH lookup
        try:
            r = subprocess.run(["which", c], capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
        except Exception:
            continue
    raise FileNotFoundError(
        "KataGo binary not found; set KATAGO_BIN or place 'katago' on $PATH"
    )


def default_model_path() -> str:
    """Return the configured KataGo model path (env var KATAGO_MODEL)."""
    p = os.environ.get("KATAGO_MODEL")
    if not p:
        raise FileNotFoundError("set KATAGO_MODEL to the path of a 9x9-capable .bin.gz network")
    return p


__all__ = [
    "KataGo",
    "KataGoConfig",
    "KataGoError",
    "AnalysisResult",
    "MoveAnalysis",
    "find_katago_binary",
    "default_model_path",
]
