"""NL rewriter tests. Uses the deterministic MockProvider; no API calls."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from gogpt import cot_vocab as V
from gogpt.nl_rewriter import (
    MockProvider,
    RewriteResult,
    build_provider,
    decode_think_tokens,
    rewrite_with_retry,
)
from gogpt.tokenizer import (
    PASS_TOKEN,
    THINK_CLOSE_TOKEN,
    THINK_OPEN_TOKEN,
    point_to_token,
)


def test_decode_basic():
    tokens = [V.WR_BINS[10], V.PH_OPENING, V.SEP_FACTS]
    out = decode_think_tokens(tokens)
    assert out == "WR_EVEN PH_OPENING SEP_FACTS"


def test_decode_tag_at_vertex():
    """<GRP_*> AT_VERTEX <vertex> renders as TAG@<coord>."""
    tokens = [
        V.GRP_WEAK_2,
        V.AT_VERTEX,
        point_to_token(1, 1),  # B8 (col 1 = 'B', row 1 = '8')
    ]
    out = decode_think_tokens(tokens)
    assert "GRP_WEAK_2@" in out
    assert out.split("@")[1] == "B8"


def test_decode_top_move_with_vertex():
    tokens = [V.TOP_MOVE, point_to_token(4, 4), V.CONF_HIGH]
    out = decode_think_tokens(tokens)
    assert "TOP_MOVE E5" in out
    assert "CONF_HIGH" in out


def test_decode_top_move_with_pass():
    tokens = [V.TOP_MOVE, PASS_TOKEN, V.CONF_LOW]
    out = decode_think_tokens(tokens)
    assert "TOP_MOVE pass" in out


def test_decode_realistic_full_block():
    """Decode a realistic structured CoT (matches the extractor's output)."""
    tokens = [
        V.WR_BINS[10],            # WR_EVEN
        V.SL_B_TINY,
        V.GRP_WEAK_2,
        V.AT_VERTEX,
        point_to_token(1, 1),     # B8
        V.TAC_ATARI,
        V.PH_MIDGAME,
        V.SEP_FACTS,
        V.TOP_MOVE,
        point_to_token(2, 2),     # C7
        V.CONF_HIGH,
    ]
    out = decode_think_tokens(tokens)
    expected = (
        "WR_EVEN SL_B_TINY GRP_WEAK_2@B8 TAC_ATARI PH_MIDGAME "
        "SEP_FACTS TOP_MOVE C7 CONF_HIGH"
    )
    assert out == expected


def test_decode_handles_orphan_at_vertex():
    """A stray AT_VERTEX (no preceding tag) is skipped, not crashed on."""
    tokens = [V.WR_BINS[0], V.AT_VERTEX, V.PH_OPENING]
    out = decode_think_tokens(tokens)
    # AT_VERTEX should drop out; we just see WR_00 and PH_OPENING
    assert "AT_VERTEX" not in out
    assert "WR_00" in out
    assert "PH_OPENING" in out


def test_mock_provider_round_trip():
    p = MockProvider()
    result = p.rewrite("WR_EVEN PH_OPENING")
    assert isinstance(result, RewriteResult)
    assert result.provider == "mock"
    assert "WR_EVEN PH_OPENING" in result.nl_text


def test_build_provider_mock():
    p = build_provider("mock")
    assert p.name == "mock"


def test_build_provider_unknown_raises():
    with pytest.raises(ValueError):
        build_provider("nonexistent")


def test_rewrite_with_retry_succeeds_after_failures():
    """Retry wrapper succeeds when the provider eventually returns."""
    calls = {"n": 0}

    class Flaky:
        name = "flaky"
        model = "flaky"

        def rewrite(self, text: str) -> RewriteResult:
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("transient")
            return RewriteResult(nl_text="ok", provider="flaky", model="flaky")

    result = rewrite_with_retry(Flaky(), "input", max_attempts=5, base_delay=0.001)
    assert result.nl_text == "ok"
    assert calls["n"] == 3


def test_rewrite_with_retry_gives_up():
    class AlwaysFails:
        name = "f"
        model = "f"

        def rewrite(self, text: str) -> RewriteResult:
            raise RuntimeError("permanent")

    with pytest.raises(RuntimeError, match="failed after"):
        rewrite_with_retry(AlwaysFails(), "input", max_attempts=2, base_delay=0.001)


# ---------------------------------------------------------------------------
# End-to-end script test: build a fake shard, run the rewriter against a mock
# provider, verify the JSONL output.
# ---------------------------------------------------------------------------

def test_rewrite_script_against_mock(tmp_path: Path):
    import importlib.util

    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "rewrite_cot_natural", repo / "scripts" / "rewrite_cot_natural.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(repo))
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    # Build a tiny shard with one think-block sandwiched between markers.
    shard_dir = tmp_path / "structured"
    shard_dir.mkdir()
    out_path = tmp_path / "natural.jsonl"

    # Synthesize one sequence: [SEP, THINK_OPEN, WR_EVEN, PH_OPENING, SEP_FACTS,
    #                          TOP_MOVE, E5, CONF_HIGH, THINK_CLOSE, MOVE, EOS]
    seq = [
        0,                       # placeholder
        THINK_OPEN_TOKEN,
        V.WR_BINS[10],
        V.PH_OPENING,
        V.SEP_FACTS,
        V.TOP_MOVE,
        point_to_token(4, 4),
        V.CONF_HIGH,
        THINK_CLOSE_TOKEN,
        point_to_token(4, 4),
        0,
    ]
    tokens = np.array([seq], dtype=np.int32)
    # The script only consumes the `tokens` key.
    np.savez_compressed(shard_dir / "shard_000000.npz", tokens=tokens)

    # Invoke main() programmatically.
    sys.argv = [
        "rewrite_cot_natural.py",
        "--shard-dir", str(shard_dir),
        "--output", str(out_path),
        "--provider", "mock",
        "--sample-rate", "1.0",
        "--yes",
        "--seed", "0",
    ]
    mod.main()

    assert out_path.exists()
    lines = out_path.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["shard"] == "shard_000000.npz"
    assert rec["row"] == 0
    assert "WR_EVEN" in rec["structured_text"]
    assert "PH_OPENING" in rec["structured_text"]
    assert "TOP_MOVE E5" in rec["structured_text"]
    assert rec["nl_text"].startswith("PROSE:")
    assert rec["provider"] == "mock"


def test_rewrite_script_is_resumable(tmp_path: Path):
    """A second run should skip already-processed (shard, row) pairs."""
    import importlib.util

    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "rewrite_cot_natural", repo / "scripts" / "rewrite_cot_natural.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(repo))
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    shard_dir = tmp_path / "structured"
    shard_dir.mkdir()
    out_path = tmp_path / "natural.jsonl"

    # Two positions in one shard.
    seq = [
        0, THINK_OPEN_TOKEN, V.WR_BINS[10], V.PH_OPENING, V.SEP_FACTS,
        V.TOP_MOVE, point_to_token(4, 4), V.CONF_HIGH, THINK_CLOSE_TOKEN,
        point_to_token(4, 4), 0,
    ]
    tokens = np.array([seq, seq], dtype=np.int32)
    np.savez_compressed(shard_dir / "shard_000000.npz", tokens=tokens)

    base_argv = [
        "rewrite_cot_natural.py",
        "--shard-dir", str(shard_dir),
        "--output", str(out_path),
        "--provider", "mock",
        "--sample-rate", "1.0",
        "--yes",
        "--seed", "0",
    ]
    sys.argv = base_argv
    mod.main()
    first_count = len(out_path.read_text().strip().splitlines())
    assert first_count == 2

    # Second run: same args, should write nothing new.
    sys.argv = base_argv
    mod.main()
    second_count = len(out_path.read_text().strip().splitlines())
    assert second_count == 2
