#!/usr/bin/env python3
"""KataGo self-play data generator.

Generates ``--num-games`` self-play games in parallel, writing one SGF per
game and a parallel JSONL file with per-position analysis (top moves,
score lead, ownership). For Phase 0 we use only the SGFs; the JSONL is
used in Phase 1 to build the structured-CoT labels.

Usage (Fir compute node):
    python scripts/generate_selfplay.py \
        --num-games 1000 \
        --visits 200 \
        --output data/train \
        --workers 16

Performance hints:
- KataGo's analysis engine is fast on H100 at low visit counts, but each
  subprocess takes a few GB of GPU memory. On a 4-GPU node, run ~4-8
  workers per GPU and pin each subprocess to a specific GPU via the env
  var ``CUDA_VISIBLE_DEVICES``.
- Set ``--temp-dir $SLURM_TMPDIR/selfplay`` to keep the writes local.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import random
import time
from pathlib import Path

from gogpt.katago import KataGo, KataGoConfig, default_model_path, find_katago_binary
from gogpt.tokenizer import token_to_point

log = logging.getLogger("selfplay")
BOARD_SIZE = 9
MAX_PLIES = 200


def _play_one_game(args: tuple[int, dict]) -> tuple[int, str, list[dict]]:
    game_idx, cfg_dict = args
    analysis_visits = cfg_dict["analysis_visits"]
    rng = random.Random(cfg_dict["seed"] + game_idx)
    katago_cfg = KataGoConfig(**cfg_dict["katago"])
    moves_played: list[tuple[str, str]] = []
    analyses: list[dict] = []
    passes = 0
    ply = 0
    to_move = "B"
    with KataGo(katago_cfg) as kg:
        while ply < MAX_PLIES and passes < 2:
            # Strong analysis for the label; lightweight policy sample for play.
            label = kg.analyze(num_visits=analysis_visits, include_ownership=True)
            analyses.append({
                "ply": ply,
                "to_move": label.to_move,
                "root_winrate": label.root_winrate,
                "root_score_lead": label.root_score_lead,
                "top_moves": [
                    {"move": m.move, "visits": m.visits, "winrate": m.winrate,
                     "score_lead": m.score_lead, "prior": m.prior, "order": m.order}
                    for m in label.move_infos[:8]
                ],
                "ownership": label.ownership,
            })
            # Pick a move; mix of top + temperature for diversity.
            top = label.move_infos[: max(1, len(label.move_infos))]
            if not top:
                chosen = "pass"
            else:
                if ply < 8:
                    # Early-game diversity: weight by visit count over top-5.
                    pool = top[:5]
                    weights = [max(1, m.visits) for m in pool]
                    chosen = rng.choices(pool, weights=weights, k=1)[0].move
                else:
                    chosen = top[0].move
            kg.play_move(to_move, chosen)
            moves_played.append((to_move, chosen))
            passes = passes + 1 if chosen.lower() == "pass" else 0
            to_move = "W" if to_move == "B" else "B"
            ply += 1
        final = kg.analyze(num_visits=analysis_visits, include_ownership=True)
    sgf = _to_sgf(moves_played, komi=katago_cfg.komi, final=final)
    return game_idx, sgf, analyses


def _to_sgf(moves: list[tuple[str, str]], komi: float, final) -> str:
    letters = "abcdefghi"
    parts = [f"(;GM[1]FF[4]SZ[9]KM[{komi}]"]
    lead = final.root_score_lead
    if final.to_move == "W":
        lead = -lead
    if abs(lead) >= 0.5:
        winner = "B" if lead > 0 else "W"
        parts.append(f"RE[{winner}+{abs(lead):.1f}]")
    for c, v in moves:
        if v.lower() == "pass":
            parts.append(f";{c}[]")
        else:
            from gogpt.tokenizer import gtp_vertex_to_token
            tok = gtp_vertex_to_token(v)
            row, col = token_to_point(tok)  # type: ignore[misc]
            parts.append(f";{c}[{letters[col]}{letters[row]}]")
    parts.append(")")
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-games", type=int, required=True)
    parser.add_argument("--visits", type=int, default=200, help="visits used during play")
    parser.add_argument("--analysis-visits", type=int, default=400, help="visits used for per-position labels")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 4))
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--komi", type=float, default=7.0)
    parser.add_argument("--rules", default="tromp-taylor")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    sgf_dir = out_dir / "sgf"
    jsonl_dir = out_dir / "analysis"
    sgf_dir.mkdir(exist_ok=True)
    jsonl_dir.mkdir(exist_ok=True)

    katago_kwargs = dict(
        binary=find_katago_binary(),
        model=default_model_path(),
        rules=args.rules,
        komi=args.komi,
        default_visits=args.visits,
        request_ownership=True,
    )
    cfg = {
        "visits": args.visits,
        "analysis_visits": args.analysis_visits,
        "seed": args.seed,
        "katago": katago_kwargs,
    }

    t0 = time.time()
    with mp.Pool(processes=args.workers) as pool:
        for game_idx, sgf, analyses in pool.imap_unordered(
            _play_one_game, [(i, cfg) for i in range(args.num_games)]
        ):
            (sgf_dir / f"game_{game_idx:06d}.sgf").write_text(sgf)
            with (jsonl_dir / f"game_{game_idx:06d}.jsonl").open("w") as f:
                for a in analyses:
                    f.write(json.dumps(a) + "\n")
            if (game_idx + 1) % 10 == 0:
                rate = (game_idx + 1) / (time.time() - t0)
                log.info("done %d games (%.2f games/s)", game_idx + 1, rate)
    log.info("finished %d games in %.1fs", args.num_games, time.time() - t0)


if __name__ == "__main__":
    main()
