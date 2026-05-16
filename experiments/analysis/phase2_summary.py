"""Aggregate Phase 2 ablation results and apply PROTOCOL §9 decision rules.

Phase 2 runs 250 jobs = 10 cells × 5 seeds × 5 LRs. For each cell we pick the
best LR by median val_loss across seeds, then perform paired statistical
comparisons against the "Full SOTR" cell (A) per PROTOCOL §3 H2 (component
necessity).

Run after the SLURM array drains:

    python -m experiments.analysis.phase2_summary --job-id 40082656

Or auto-detect the most recent array job:

    python -m experiments.analysis.phase2_summary

Inputs:
- ``results/phase2/<run_id>/eval.jsonl`` — per-eval val_loss records
- ``results/phase2/<run_id>/stability_incidents.jsonl`` — fired §8 incidents
- ``results/slurm/ablation-<jobid>_<task>.out`` — stdout containing the
  ``[SOTR] run_id=... cfg=name=phase2_<cell>_seed<n>_lr<lr> ...`` line we
  parse to recover (cell, seed, lr) for each run_id.

Outputs:
- prints a markdown report to stdout
- writes the same report to ``results/phase2/_summary_<jobid>.md`` for archival
- writes ``results/phase2/_per_run_<jobid>.csv`` with one row per (cell, seed, lr)
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable

# Cells from PROTOCOL §9; A is the "full SOTR" reference.
CELLS = (
    "A_sotr_full",
    "B_drop_alpha",
    "C_drop_delta",
    "D_drop_both",
    "E_drop_ns",
    "F_full_ns",
    "G_alpha_schedule",
    "H_delta_schedule",
    "I_muon_plus_cap",
    "J_partial_ns_muon",
)

# H2 family — the three component drops (Holm-Bonferroni correction).
H2_FAMILY = ("B_drop_alpha", "C_drop_delta", "E_drop_ns")

# Pattern matching the train.py [SOTR] header line.
_RE_HEADER = re.compile(
    r"\[SOTR\]\s+run_id=(?P<run_id>\S+)\s+output=(?P<output>\S+)\s+cfg=name=phase2_(?P<cell>[A-J]_\w+?)_seed(?P<seed>\d+)_lr(?P<lr>[\d.]+)"
)


@dataclass(frozen=True)
class RunRecord:
    """One Phase 2 run."""

    run_id: str
    cell: str
    seed: int
    lr: float
    final_val_loss: float | None  # None if eval.jsonl missing / empty
    incident_counts: dict[str, int]
    n_eval_points: int


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def discover_runs(slurm_log_dir: Path, job_id: str | None) -> list[tuple[str, str, int, float]]:
    """Parse SLURM stdout files into (run_id, cell, seed, lr) tuples.

    Returns one tuple per ablation task that produced a [SOTR] header line.
    """
    pattern = f"ablation-{job_id}_*.out" if job_id else "ablation-*.out"
    runs: list[tuple[str, str, int, float]] = []
    for out_path in sorted(slurm_log_dir.glob(pattern)):
        try:
            for line in out_path.read_text(errors="replace").splitlines():
                m = _RE_HEADER.search(line)
                if m:
                    runs.append(
                        (m["run_id"], m["cell"], int(m["seed"]), float(m["lr"]))
                    )
                    break  # one header per run
        except OSError:
            continue
    return runs


def _read_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_run(results_root: Path, run_id: str, cell: str, seed: int, lr: float) -> RunRecord:
    run_dir = results_root / run_id
    final_loss: float | None = None
    n_evals = 0
    for row in _read_jsonl(run_dir / "eval.jsonl"):
        n_evals += 1
        v = row.get("val_loss_nats")
        if v is not None:
            final_loss = float(v)
    counts: dict[str, int] = {}
    for inc in _read_jsonl(run_dir / "stability_incidents.jsonl"):
        kind = str(inc.get("kind", "unknown"))
        counts[kind] = counts.get(kind, 0) + 1
    return RunRecord(
        run_id=run_id,
        cell=cell,
        seed=seed,
        lr=lr,
        final_val_loss=final_loss,
        incident_counts=counts,
        n_eval_points=n_evals,
    )


# ---------------------------------------------------------------------------
# Per-cell aggregation
# ---------------------------------------------------------------------------


def best_lr_per_cell(records: list[RunRecord]) -> dict[str, float]:
    """For each cell, return the LR with the lowest median val_loss across seeds.

    Ties broken by smaller LR (smaller models prefer smaller LRs as tiebreak).
    """
    by_cell_lr: dict[tuple[str, float], list[float]] = {}
    for r in records:
        if r.final_val_loss is None or not math.isfinite(r.final_val_loss):
            continue
        by_cell_lr.setdefault((r.cell, r.lr), []).append(r.final_val_loss)

    best: dict[str, float] = {}
    for cell in CELLS:
        cands = [(lr, losses) for (c, lr), losses in by_cell_lr.items() if c == cell]
        if not cands:
            continue
        # Need at least one finite loss to count.
        scored = [(median(losses), lr) for lr, losses in cands if losses]
        if not scored:
            continue
        scored.sort()  # by median, then lr (Python sorts tuples lexicographically)
        best[cell] = scored[0][1]
    return best


def cell_seed_losses(
    records: list[RunRecord], cell: str, lr: float
) -> dict[int, float]:
    """Final val_loss keyed by seed for one (cell, lr)."""
    out: dict[int, float] = {}
    for r in records:
        if r.cell != cell or r.lr != lr or r.final_val_loss is None:
            continue
        if not math.isfinite(r.final_val_loss):
            continue
        out[r.seed] = r.final_val_loss
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def paired_bootstrap(
    a: list[float],
    b: list[float],
    *,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float, float]:
    """Paired bootstrap of mean(b - a).

    Returns (point_estimate, ci_low, ci_high, two_sided_p).

    The pairing assumes ``a[i]`` and ``b[i]`` share the same seed (the caller
    is responsible for aligning seeds before calling). A negative point estimate
    means b < a (i.e., cell b's loss is lower than cell a's — b is "better").
    """
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    n = len(a)
    if n < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")

    import random

    rng = random.Random(seed)
    diffs = [b[i] - a[i] for i in range(n)]
    point = sum(diffs) / n
    means: list[float] = []
    for _ in range(n_resamples):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo_idx = int(0.025 * n_resamples)
    hi_idx = int(0.975 * n_resamples) - 1
    ci_low, ci_high = means[lo_idx], means[hi_idx]
    n_pos = sum(1 for m in means if m > 0)
    n_neg = sum(1 for m in means if m < 0)
    # Two-sided p ~ 2 × P(resample crosses 0 in the opposite direction of the
    # point estimate). Add-one smoothing avoids p=0 on small bootstraps.
    if point >= 0:
        p = 2.0 * (n_neg + 1) / (n_resamples + 1)
    else:
        p = 2.0 * (n_pos + 1) / (n_resamples + 1)
    return point, ci_low, ci_high, min(1.0, p)


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni step-down. Returns one bool per input (reject H0 yes/no)."""
    indexed = sorted(enumerate(p_values), key=lambda t: t[1])
    k = len(p_values)
    reject = [False] * k
    for rank, (orig_idx, p) in enumerate(indexed):
        threshold = alpha / (k - rank)
        if p <= threshold:
            reject[orig_idx] = True
        else:
            # Stop at first non-rejection (Holm is step-down).
            break
    return reject


# ---------------------------------------------------------------------------
# Decision tree (PROTOCOL §9)
# ---------------------------------------------------------------------------


def decide_narrative(
    comparisons: dict[str, tuple[float, float, float, float, bool]],
) -> list[str]:
    """Apply §9 decision rules. Each item is one bulleted line for the report.

    ``comparisons`` is keyed by cell name (compared against A), value is
    (delta, ci_lo, ci_hi, p, holm_significant).
    """
    notes: list[str] = []

    def sig_better(cell: str) -> bool:
        # "A beats cell X" means delta(X - A) > 0 with significance.
        c = comparisons.get(cell)
        if c is None:
            return False
        delta, _, _, _, hb = c
        return hb and delta > 0

    def equiv(cell: str, eps: float = 0.02) -> bool:
        c = comparisons.get(cell)
        if c is None:
            return False
        delta, lo, hi, _, _ = c
        # CI of (X - A) entirely within ±eps → call them equivalent.
        return abs(delta) < eps and lo > -eps and hi < eps

    if equiv("I_muon_plus_cap"):
        notes.append(
            "**A ≈ I:** Frobenius trust region alone explains the win. "
            "Paper narrative leans toward 'we show per-matrix Δ cap is sufficient; "
            "α-blend and partial NS add little.' Leaner contribution."
        )
    if sig_better("B_drop_alpha"):
        notes.append("**A > B (Holm):** α-blend contributes; keep α as a knob.")
    if sig_better("C_drop_delta"):
        notes.append("**A > C (Holm):** Δ trust region contributes (the novel piece). Keep.")
    if sig_better("E_drop_ns"):
        notes.append("**A > E (Holm):** Newton-Schulz orthogonalization contributes. Keep.")
    if equiv("F_full_ns"):
        notes.append(
            "**A ≈ F:** partial NS (q=2) suffices over full NS (q=5). Drop q from the pitch."
        )
    if equiv("D_drop_both"):
        notes.append(
            "**A ≈ D:** dropping both α-blend and Δ recovers Muon-like behavior. "
            "Combined contributions are not additive — likely a single mechanism dominates."
        )
    if not notes:
        notes.append("No clean §9 outcomes triggered yet. Check per-cell tables below.")
    return notes


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def render_report(
    records: list[RunRecord],
    job_id: str,
) -> str:
    best = best_lr_per_cell(records)
    a_lr = best.get("A_sotr_full")
    if a_lr is None:
        return "# Phase 2 summary\n\nNo runs found for cell A_sotr_full yet. Re-run after array completes."

    a_seeds = cell_seed_losses(records, "A_sotr_full", a_lr)

    # Per-cell summary: count, best LR, median at best LR, n incidents.
    lines = []
    lines.append(f"# Phase 2 summary — job {job_id}")
    lines.append("")
    n_total = sum(1 for r in records if r.final_val_loss is not None)
    n_attempted = len(records)
    lines.append(f"_{n_total}/{n_attempted} runs produced a final val loss (rest crashed or still running)._")
    lines.append("")
    lines.append("## Per-cell best-LR table")
    lines.append("")
    lines.append("| Cell | best LR | median val_loss | range across 5 seeds | total incidents |")
    lines.append("|---|---|---|---|---|")
    for cell in CELLS:
        lr = best.get(cell)
        if lr is None:
            lines.append(f"| {cell} | — | (no data) | — | — |")
            continue
        losses = cell_seed_losses(records, cell, lr)
        if not losses:
            lines.append(f"| {cell} | {lr:g} | (no data) | — | — |")
            continue
        med = median(losses.values())
        lo, hi = min(losses.values()), max(losses.values())
        inc = sum(
            sum(r.incident_counts.values())
            for r in records
            if r.cell == cell and r.lr == lr
        )
        lines.append(
            f"| {cell} | {lr:g} | {med:.4f} | [{lo:.4f}, {hi:.4f}] | {inc} |"
        )
    lines.append("")

    # Pairwise A-vs-X comparisons (paired by seed at each cell's own best LR).
    lines.append("## A-vs-X (paired bootstrap at each cell's best LR)")
    lines.append("")
    lines.append("Δ = mean(X − A). Negative Δ means X has lower loss than A (X is 'better').")
    lines.append("")
    lines.append("| vs. | n seeds | Δ | 95% CI | p | Holm-sig (H2) |")
    lines.append("|---|---|---|---|---|---|")
    comparisons: dict[str, tuple[float, float, float, float, bool]] = {}
    # Collect H2-family p-values first for Holm-Bonferroni.
    h2_pvals: list[tuple[str, float, float, float, float]] = []
    for cell in CELLS:
        if cell == "A_sotr_full":
            continue
        x_lr = best.get(cell)
        if x_lr is None:
            continue
        x_seeds = cell_seed_losses(records, cell, x_lr)
        shared = sorted(set(a_seeds) & set(x_seeds))
        if len(shared) < 2:
            continue
        a_arr = [a_seeds[s] for s in shared]
        x_arr = [x_seeds[s] for s in shared]
        delta, lo, hi, p = paired_bootstrap(a_arr, x_arr)
        if cell in H2_FAMILY:
            h2_pvals.append((cell, delta, lo, hi, p))
        else:
            comparisons[cell] = (delta, lo, hi, p, False)

    # Apply Holm to H2 family.
    h2_rej = holm_bonferroni([t[4] for t in h2_pvals])
    for (cell, delta, lo, hi, p), rej in zip(h2_pvals, h2_rej):
        comparisons[cell] = (delta, lo, hi, p, rej)

    # Render comparison table in cell order.
    for cell in CELLS:
        if cell == "A_sotr_full":
            continue
        c = comparisons.get(cell)
        if c is None:
            lines.append(f"| {cell} | — | — | — | — | — |")
            continue
        delta, lo, hi, p, hb = c
        in_h2 = cell in H2_FAMILY
        hb_str = "✓" if (in_h2 and hb) else ("✗" if in_h2 else "n/a")
        n_seeds_used = len(set(cell_seed_losses(records, cell, best[cell])) & set(a_seeds))
        lines.append(
            f"| {cell} | {n_seeds_used} | {delta:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {p:.4f} | {hb_str} |"
        )
    lines.append("")

    # Decision tree
    lines.append("## §9 decision tree")
    lines.append("")
    for note in decide_narrative(comparisons):
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def write_csv(records: list[RunRecord], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "cell", "seed", "lr", "final_val_loss", "n_eval_points", "n_incidents"])
        for r in records:
            w.writerow(
                [
                    r.run_id,
                    r.cell,
                    r.seed,
                    r.lr,
                    "" if r.final_val_loss is None else f"{r.final_val_loss:.6f}",
                    r.n_eval_points,
                    sum(r.incident_counts.values()),
                ]
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Aggregate Phase 2 ablation runs.")
    ap.add_argument("--job-id", default=None, help="Filter stdout files by SLURM array job ID.")
    ap.add_argument("--results-dir", default="results/phase2", type=Path)
    ap.add_argument("--slurm-log-dir", default="results/slurm", type=Path)
    ap.add_argument(
        "--output-dir",
        default=None,
        type=Path,
        help="Where to write summary .md + CSV. Defaults to --results-dir.",
    )
    args = ap.parse_args(argv)

    out_dir = args.output_dir or args.results_dir
    discovered = discover_runs(args.slurm_log_dir, args.job_id)
    if not discovered:
        print(
            f"No runs found in {args.slurm_log_dir} (job_id={args.job_id}). "
            "Either the array hasn't produced [SOTR] headers yet, or the glob missed.",
            file=sys.stderr,
        )
        return 1

    records = [load_run(args.results_dir, *d) for d in discovered]

    job_id = args.job_id or "auto"
    report = render_report(records, job_id)
    print(report)
    md_path = out_dir / f"_summary_{job_id}.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(report)
    csv_path = out_dir / f"_per_run_{job_id}.csv"
    write_csv(records, csv_path)
    print(f"\nWrote summary → {md_path}", file=sys.stderr)
    print(f"Wrote per-run CSV → {csv_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
