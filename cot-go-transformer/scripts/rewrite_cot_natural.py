#!/usr/bin/env python3
"""Rewrite a sample of structured CoTs into natural-language prose.

Reads NPZ shards produced by ``scripts/extract_cot_labels.py --mode structured``,
selects a sample of positions, and calls the configured LLM provider to rewrite
each structured CoT as 1-2 sentences. Outputs JSONL.

Usage (Gemini Flash, free tier):
    export GOOGLE_API_KEY=...
    python scripts/rewrite_cot_natural.py \\
        --shard-dir data/cot/structured \\
        --output data/cot/natural.jsonl \\
        --sample-rate 0.15

Usage (Anthropic Claude Haiku 4.5, recommended for bulk):
    export ANTHROPIC_API_KEY=...
    python scripts/rewrite_cot_natural.py \\
        --shard-dir data/cot/structured \\
        --output data/cot/natural.jsonl \\
        --provider anthropic \\
        --model claude-haiku-4-5

The output JSONL has one record per rewritten position:

    {
      "shard": "shard_000000.npz",
      "row": 42,
      "structured_tokens": [97, 103, 140, 109, ...],
      "structured_text": "WR_EVEN SL_B_TINY NO_FACTS PH_OPENING ...",
      "nl_text": "The opening is roughly even ...",
      "provider": "gemini",
      "model": "gemini-2.5-flash"
    }

Resumable: the script reads any existing output and skips (shard, row) pairs
already present. Safe to ctrl-C and rerun.

Cost estimate is printed before any API calls; pass ``--yes`` to skip the
confirmation prompt (useful for SLURM).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from gogpt.nl_rewriter import (
    build_provider,
    decode_think_tokens,
    rewrite_with_retry,
)
from gogpt.tokenizer import THINK_CLOSE_TOKEN, THINK_OPEN_TOKEN

log = logging.getLogger("rewrite_cot_natural")


# Rough cost per call at typical settings (~30 input tokens uncached, ~50
# output tokens, system prompt cached at 0.1x). Used only for the
# pre-flight estimate; actual cost is reported per shard from response usage.
COST_PER_1K_OUTPUT: dict[str, float] = {
    # Anthropic
    "claude-opus-4-7": 25.0,
    "claude-opus-4-6": 25.0,
    "claude-sonnet-4-6": 15.0,
    "claude-haiku-4-5": 5.0,
    # Gemini (Flash 2.5 free tier covers reasonable usage)
    "gemini-2.5-flash": 0.0,
    "gemini-2.5-pro": 10.0,
}


def estimated_cost_usd(model: str, n_calls: int, avg_output_tokens: int = 50) -> float:
    rate = COST_PER_1K_OUTPUT.get(model)
    if rate is None:
        return float("nan")
    return rate * n_calls * avg_output_tokens / 1_000_000


def _extract_think_block(row_tokens: np.ndarray) -> list[int] | None:
    """Slice out the tokens between THINK_OPEN and THINK_CLOSE."""
    tokens = row_tokens.tolist()
    try:
        start = tokens.index(THINK_OPEN_TOKEN)
        end = tokens.index(THINK_CLOSE_TOKEN, start + 1)
    except ValueError:
        return None
    return tokens[start + 1 : end]


def iter_positions(shard_dir: Path) -> Iterator[tuple[str, int, list[int]]]:
    """Yield (shard_name, row_idx, think_tokens) across every shard."""
    for shard_path in sorted(shard_dir.glob("shard_*.npz")):
        data = np.load(shard_path)
        tokens_arr = data["tokens"]
        for i in range(tokens_arr.shape[0]):
            think = _extract_think_block(tokens_arr[i])
            if think is None or not think:
                continue
            yield shard_path.name, i, think


def load_already_done(output_path: Path) -> set[tuple[str, int]]:
    if not output_path.exists():
        return set()
    done: set[tuple[str, int]] = set()
    with output_path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
                done.add((rec["shard"], int(rec["row"])))
            except Exception:
                continue
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shard-dir", required=True, help="Directory of structured-CoT NPZ shards.")
    parser.add_argument("--output", required=True, help="JSONL output path (resumable).")
    parser.add_argument(
        "--provider",
        choices=["gemini", "anthropic", "mock"],
        default="gemini",
        help="LLM provider (default: gemini -- has a free tier).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Provider model. Default: gemini-2.5-flash for gemini; claude-opus-4-7 "
            "for anthropic. For bulk rewriting on Anthropic, claude-haiku-4-5 is "
            "5x cheaper and recommended."
        ),
    )
    parser.add_argument("--sample-rate", type=float, default=0.15, help="Fraction of positions to rewrite.")
    parser.add_argument("--max-positions", type=int, default=None, help="Hard cap on rewrites.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rps", type=float, default=0.0, help="Cap requests/sec (0 = no cap).")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max output tokens per call.")
    parser.add_argument("--yes", action="store_true", help="Skip the cost-estimate prompt.")
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    shard_dir = Path(args.shard_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not shard_dir.exists():
        log.error("shard directory not found: %s", shard_dir)
        sys.exit(1)

    rng = random.Random(args.seed)
    already_done = load_already_done(output_path)
    log.info("resuming with %d positions already rewritten in %s", len(already_done), output_path)

    # Pre-pass: enumerate positions and apply sampling.
    candidates: list[tuple[str, int, list[int]]] = []
    for shard, row, tokens in iter_positions(shard_dir):
        if (shard, row) in already_done:
            continue
        if rng.random() > args.sample_rate:
            continue
        candidates.append((shard, row, tokens))
        if args.max_positions is not None and len(candidates) >= args.max_positions:
            break
    log.info("selected %d positions to rewrite (sample-rate=%.3f)", len(candidates), args.sample_rate)
    if not candidates:
        log.info("nothing to do")
        return

    # Cost estimate
    provider = build_provider(args.provider, args.model, max_tokens=args.max_tokens)
    est = estimated_cost_usd(provider.model, len(candidates))
    if est == est:  # not NaN
        log.info("estimated cost: ~$%.2f (%d calls)", est, len(candidates))
    else:
        log.info("estimated cost: unknown for model %s", provider.model)
    if not args.yes and est > 5.0:
        resp = input(f"This will cost ~${est:.2f}. Continue? [y/N] ")
        if resp.strip().lower() not in ("y", "yes"):
            log.info("aborted by user")
            return

    # Run
    min_interval = 1.0 / args.rps if args.rps > 0 else 0.0
    last_call = 0.0
    total_in = 0
    total_out = 0
    written = 0

    with output_path.open("a") as out_f:
        for n, (shard, row, tokens) in enumerate(candidates):
            structured_text = decode_think_tokens(tokens)
            now = time.monotonic()
            if min_interval and (now - last_call) < min_interval:
                time.sleep(min_interval - (now - last_call))
            try:
                result = rewrite_with_retry(provider, structured_text)
            except Exception as e:
                log.warning("permanent failure at %s:%d: %s", shard, row, e)
                continue
            last_call = time.monotonic()
            rec: dict[str, Any] = {
                "shard": shard,
                "row": row,
                "structured_tokens": tokens,
                "structured_text": structured_text,
                "nl_text": result.nl_text,
                "provider": result.provider,
                "model": result.model,
            }
            if result.input_tokens is not None:
                rec["input_tokens"] = result.input_tokens
                total_in += result.input_tokens
            if result.output_tokens is not None:
                rec["output_tokens"] = result.output_tokens
                total_out += result.output_tokens
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
            written += 1
            if (n + 1) % args.log_every == 0:
                log.info(
                    "progress %d/%d  tokens in=%d out=%d  last_nl=%r",
                    n + 1, len(candidates), total_in, total_out, result.nl_text[:80],
                )
    log.info("done: wrote %d records (tokens in=%d, out=%d)", written, total_in, total_out)


if __name__ == "__main__":
    main()
