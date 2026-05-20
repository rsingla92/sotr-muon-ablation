"""Match-game evaluation: model vs KataGo at configurable visit counts.

Plays both colors equally, reports win rate, logs all games to SGF for
post-hoc inspection. Elo is estimated with a simple logistic update; for
serious comparisons across many models use BayesElo offline.
"""

from __future__ import annotations

import argparse
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .data import play_stone
from .inference import generate_move
from .katago import KataGo, KataGoConfig, default_model_path, find_katago_binary
from .model import GoGPT, GoGPTConfig
from .tokenizer import PASS_TOKEN, point_to_token, token_to_gtp_vertex, token_to_point

log = logging.getLogger("gogpt.eval")
BOARD_SIZE = 9
MAX_PLIES = 200  # safety cap for 9x9


@dataclass
class GameResult:
    model_color: str        # 'B' or 'W'
    winner: str             # 'B', 'W', or 'draw'
    plies: int
    illegal_moves: int
    sgf: str


@dataclass
class MatchSummary:
    wins: int = 0
    losses: int = 0
    draws: int = 0
    illegal_total: int = 0
    games: list[GameResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        n = self.total
        if n == 0:
            return 0.0
        # draws count as half a win
        return (self.wins + 0.5 * self.draws) / n

    def elo_estimate(self) -> float:
        """Estimated Elo difference (model - opponent) from win rate."""
        wr = self.win_rate
        wr = min(max(wr, 1e-3), 1 - 1e-3)
        return -400.0 * math.log10(1.0 / wr - 1.0)


def _legal_mask(board: np.ndarray, ko: tuple[int, int] | None, color: int) -> np.ndarray:
    """Return a (82,) bool mask of legal moves for ``color``."""
    mask = np.zeros(82, dtype=bool)
    mask[PASS_TOKEN] = True
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] != 0:
                continue
            if ko is not None and (r, c) == ko:
                continue
            # Trial play: legal iff play_stone leaves a non-empty board with
            # the stone present (suicide rule is enforced inside play_stone).
            trial, _ = play_stone(board, color, (r, c))
            if trial[r, c] == color:
                mask[point_to_token(r, c)] = True
    return mask


def _swap_colors(board: np.ndarray) -> np.ndarray:
    out = board.copy()
    out[board == 1] = 2
    out[board == 2] = 1
    return out


def play_one_game(
    model: GoGPT,
    katago: KataGo,
    model_color: str,
    *,
    katago_visits: int,
    temperature: float,
    device: torch.device,
    rng: random.Random,
) -> GameResult:
    """Play one game; model plays ``model_color``."""
    katago.reset_game()
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    ko: tuple[int, int] | None = None
    last_move: tuple[int, int] | None = None
    history_for_model: list[int] = []  # tokens (always-as-black perspective)
    history_for_katago: list[tuple[str, str]] = []

    passes_in_row = 0
    illegal = 0
    ply = 0
    to_move = "B"
    while ply < MAX_PLIES and passes_in_row < 2:
        if to_move == model_color:
            # Build model-perspective board (always black-to-move)
            model_board = board if to_move == "B" else _swap_colors(board)
            model_color_val = 1
            legal = _legal_mask(model_board, ko, model_color_val)
            gen = generate_move(
                model, model_board,
                ko_point=ko, last_move=last_move,
                history_tokens=history_for_model,
                temperature=temperature,
                legal_mask=legal,
                device=device,
            )
            chosen = gen.token
            if not gen.is_legal_move:
                illegal += 1
                chosen = PASS_TOKEN
            history_for_model.append(chosen)
            rc = token_to_point(chosen)
            vertex = token_to_gtp_vertex(chosen)
        else:
            # KataGo's turn
            kr = katago.analyze(num_visits=katago_visits, include_ownership=False)
            top = kr.top_move
            vertex = top.move if top is not None else "pass"
            rc = None
            if vertex.lower() != "pass":
                from .tokenizer import gtp_vertex_to_token
                token = gtp_vertex_to_token(vertex)
                rc = token_to_point(token)
                # also record from model's perspective
                history_for_model.append(token)
            else:
                history_for_model.append(PASS_TOKEN)

        # Apply move to true board
        played_color_val = 1 if to_move == "B" else 2
        board, ko = play_stone(board, played_color_val, rc)
        last_move = rc
        katago.play_move(to_move, vertex)
        history_for_katago.append((to_move, vertex))
        passes_in_row = passes_in_row + 1 if rc is None else 0
        to_move = "W" if to_move == "B" else "B"
        ply += 1

    # Ask KataGo to score the final position (analyze with 1 visit just to get
    # rootInfo; ownership-summed score lead approximates final).
    final = katago.analyze(num_visits=64, include_ownership=True)
    # Convention: positive score_lead favors the side to move at the end. We
    # convert to a B-relative score for clarity.
    lead = final.root_score_lead
    side_to_move = final.to_move
    if side_to_move == "W":
        lead = -lead
    if abs(lead) < 0.5:
        winner = "draw"
    else:
        winner = "B" if lead > 0 else "W"

    sgf = _build_sgf(history_for_katago, komi=katago.cfg.komi, result_winner=winner)
    return GameResult(model_color=model_color, winner=winner, plies=ply, illegal_moves=illegal, sgf=sgf)


def _build_sgf(moves: list[tuple[str, str]], komi: float, result_winner: str) -> str:
    """Build a minimal SGF for the played game."""
    letters = "abcdefghi"
    parts = [f"(;GM[1]FF[4]SZ[9]KM[{komi}]"]
    if result_winner != "draw":
        parts.append(f"RE[{result_winner}+]")
    for c, v in moves:
        if v.lower() == "pass":
            parts.append(f";{c}[]")
        else:
            from .tokenizer import gtp_vertex_to_token
            tok = gtp_vertex_to_token(v)
            row, col = token_to_point(tok)  # type: ignore[misc]
            parts.append(f";{c}[{letters[col]}{letters[row]}]")
    parts.append(")")
    return "".join(parts)


def run_match(
    model: GoGPT,
    *,
    katago_cfg: KataGoConfig,
    num_games: int,
    katago_visits: int,
    temperature: float = 0.0,
    device: torch.device | None = None,
    seed: int = 0,
    save_dir: Path | None = None,
) -> MatchSummary:
    device = device or next(model.parameters()).device
    rng = random.Random(seed)
    summary = MatchSummary()
    model.eval()
    with KataGo(katago_cfg) as katago:
        for g in range(num_games):
            model_color = "B" if g % 2 == 0 else "W"
            result = play_one_game(
                model, katago, model_color,
                katago_visits=katago_visits,
                temperature=temperature,
                device=device,
                rng=rng,
            )
            if result.winner == model_color:
                summary.wins += 1
            elif result.winner == "draw":
                summary.draws += 1
            else:
                summary.losses += 1
            summary.illegal_total += result.illegal_moves
            summary.games.append(result)
            log.info(
                "game %d/%d: model=%s winner=%s (illegal=%d, plies=%d) -- wr=%.2f",
                g + 1, num_games, model_color, result.winner,
                result.illegal_moves, result.plies, summary.win_rate,
            )
            if save_dir is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
                (save_dir / f"game_{g:03d}.sgf").write_text(result.sgf)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-config", required=True, help="YAML matching the trained model")
    parser.add_argument("--num-games", type=int, default=20)
    parser.add_argument("--visits", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import yaml
    raw = yaml.safe_load(Path(args.model_config).read_text())
    model_cfg = GoGPTConfig(**raw["model"])
    model = GoGPT(model_cfg)
    state = torch.load(args.checkpoint, map_location="cpu")
    sd = state.get("model", state)
    model.load_state_dict(sd)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    katago_cfg = KataGoConfig(
        binary=find_katago_binary(),
        model=default_model_path(),
        default_visits=args.visits,
    )
    summary = run_match(
        model,
        katago_cfg=katago_cfg,
        num_games=args.num_games,
        katago_visits=args.visits,
        temperature=args.temperature,
        device=device,
        seed=args.seed,
        save_dir=Path(args.save_dir) if args.save_dir else None,
    )
    log.info(
        "match summary: w=%d l=%d d=%d wr=%.3f elo=%+.0f illegal=%d",
        summary.wins, summary.losses, summary.draws,
        summary.win_rate, summary.elo_estimate(), summary.illegal_total,
    )


if __name__ == "__main__":
    main()
