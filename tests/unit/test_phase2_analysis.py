"""Unit tests for experiments.analysis.phase2_summary.

Synthetic data exercises the I/O, aggregation, and statistics paths
without needing real Phase 2 outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.analysis import phase2_summary as ps


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_slurm_out(
    slurm_dir: Path,
    job_id: str,
    task_idx: int,
    run_id: str,
    cell: str,
    seed: int,
    lr: float,
) -> None:
    """Write a minimal SLURM stdout file that the discover_runs regex matches."""
    slurm_dir.mkdir(parents=True, exist_ok=True)
    out = slurm_dir / f"ablation-{job_id}_{task_idx}.out"
    out.write_text(
        "using device: cuda:0\n"
        "Training DataLoader: total number of tokens: 900000000 across 9 files\n"
        f"[SOTR] run_id={run_id}  output=/scratch/.../results/phase2/{run_id}  "
        f"cfg=name=phase2_{cell}_seed{seed}_lr{lr:g} optim=sotr "
        f"lr_h={lr:g} lr_e=0.0036 iters=1500 seed={seed}\n"
        "step:0/1500 val_loss:15.99 train_time:77ms step_avg:nanms\n"
    )


def _write_eval_jsonl(run_dir: Path, val_losses: list[float]) -> None:
    """Write an eval.jsonl with one row per loss value (last row is "final")."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "eval.jsonl").open("w") as f:
        for i, v in enumerate(val_losses):
            f.write(json.dumps({"step": (i + 1) * 100, "val_loss_nats": v}) + "\n")


def _write_incidents_jsonl(run_dir: Path, incidents: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "stability_incidents.jsonl").open("w") as f:
        for inc in incidents:
            f.write(json.dumps(inc) + "\n")


@pytest.fixture
def synthetic_phase2(tmp_path: Path) -> tuple[Path, Path, str]:
    """Build a synthetic results/phase2 + results/slurm tree.

    Cell A_sotr_full beats Cell B_drop_alpha across all seeds (A=3.0, B=3.5).
    All other cells have data only at a subset of LRs to exercise edge cases.
    """
    job_id = "99999999"
    results_root = tmp_path / "results" / "phase2"
    slurm_root = tmp_path / "results" / "slurm"
    task_idx = 0

    # Cell A: 5 seeds × 5 LRs. LR 0.02 is best (median 3.0). Others worse.
    for seed in range(5):
        for lr in (0.005, 0.01, 0.02, 0.04, 0.08):
            run_id = f"runA_s{seed}_lr{lr:g}"
            offset = abs(lr - 0.02) * 5  # LR penalty
            val_loss = 3.0 + offset + seed * 0.01  # tiny per-seed noise
            _write_slurm_out(
                slurm_root, job_id, task_idx, run_id, "A_sotr_full", seed, lr
            )
            _write_eval_jsonl(results_root / run_id, [4.0, 3.5, val_loss])
            _write_incidents_jsonl(results_root / run_id, [])
            task_idx += 1

    # Cell B: same shape but offset +0.5 (clearly worse than A).
    for seed in range(5):
        for lr in (0.005, 0.01, 0.02, 0.04, 0.08):
            run_id = f"runB_s{seed}_lr{lr:g}"
            offset = abs(lr - 0.02) * 5
            val_loss = 3.5 + offset + seed * 0.01
            _write_slurm_out(
                slurm_root, job_id, task_idx, run_id, "B_drop_alpha", seed, lr
            )
            _write_eval_jsonl(results_root / run_id, [4.5, 4.0, val_loss])
            _write_incidents_jsonl(
                results_root / run_id,
                [{"kind": "spike", "step": 42}] if seed == 0 else [],
            )
            task_idx += 1

    return results_root, slurm_root, job_id


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_discover_runs_parses_header(synthetic_phase2):
    _, slurm_root, job_id = synthetic_phase2
    runs = ps.discover_runs(slurm_root, job_id)
    assert len(runs) == 50  # 2 cells × 5 seeds × 5 LRs
    cells = {r[1] for r in runs}
    assert cells == {"A_sotr_full", "B_drop_alpha"}
    # First A run.
    a_runs = [r for r in runs if r[1] == "A_sotr_full"]
    seeds = sorted({r[2] for r in a_runs})
    lrs = sorted({r[3] for r in a_runs})
    assert seeds == [0, 1, 2, 3, 4]
    assert lrs == [0.005, 0.01, 0.02, 0.04, 0.08]


def test_discover_runs_filters_by_job_id(synthetic_phase2):
    _, slurm_root, _ = synthetic_phase2
    assert ps.discover_runs(slurm_root, job_id="not_a_job") == []


def test_load_run_extracts_final_val_loss(synthetic_phase2):
    results_root, _, _ = synthetic_phase2
    r = ps.load_run(results_root, "runA_s0_lr0.02", "A_sotr_full", 0, 0.02)
    assert r.final_val_loss == pytest.approx(3.0)
    assert r.n_eval_points == 3
    assert r.incident_counts == {}


def test_load_run_counts_incidents(synthetic_phase2):
    results_root, _, _ = synthetic_phase2
    r = ps.load_run(results_root, "runB_s0_lr0.02", "B_drop_alpha", 0, 0.02)
    assert r.incident_counts == {"spike": 1}


def test_load_run_handles_missing_files(tmp_path):
    r = ps.load_run(tmp_path, "nonexistent", "A_sotr_full", 0, 0.02)
    assert r.final_val_loss is None
    assert r.incident_counts == {}
    assert r.n_eval_points == 0


def test_best_lr_per_cell(synthetic_phase2):
    results_root, slurm_root, job_id = synthetic_phase2
    discovered = ps.discover_runs(slurm_root, job_id)
    records = [ps.load_run(results_root, *d) for d in discovered]
    best = ps.best_lr_per_cell(records)
    assert best["A_sotr_full"] == pytest.approx(0.02)
    assert best["B_drop_alpha"] == pytest.approx(0.02)


def test_paired_bootstrap_detects_clear_separation():
    # B clearly worse than A — Δ should be positive and CI exclude 0.
    a = [3.0, 3.01, 3.02, 3.03, 3.04]
    b = [3.5, 3.51, 3.52, 3.53, 3.54]
    delta, lo, hi, p = ps.paired_bootstrap(a, b, seed=0)
    assert delta == pytest.approx(0.5, abs=0.01)
    assert lo > 0  # CI excludes 0
    assert p < 0.05  # significant


def test_paired_bootstrap_null_case():
    # a and b are the same distribution shifted by seed only.
    a = [3.0, 3.1, 3.2, 3.3, 3.4]
    b = [3.0, 3.1, 3.2, 3.3, 3.4]  # zero difference exactly
    delta, lo, hi, p = ps.paired_bootstrap(a, b, seed=0, n_resamples=2000)
    assert delta == pytest.approx(0.0)
    assert lo <= 0 <= hi  # CI brackets 0


def test_paired_bootstrap_length_mismatch():
    with pytest.raises(ValueError):
        ps.paired_bootstrap([1.0, 2.0], [1.0, 2.0, 3.0])


def test_paired_bootstrap_too_few_points():
    # n=1 → NaN.
    delta, *_ = ps.paired_bootstrap([3.0], [3.5])
    import math

    assert math.isnan(delta)


def test_holm_bonferroni_all_significant():
    # 3 tiny p-values → all reject.
    assert ps.holm_bonferroni([0.001, 0.002, 0.003], alpha=0.05) == [True, True, True]


def test_holm_bonferroni_none_significant():
    # All large p-values → none reject.
    assert ps.holm_bonferroni([0.5, 0.6, 0.7], alpha=0.05) == [False, False, False]


def test_holm_bonferroni_step_down_logic():
    # p = [0.01, 0.03, 0.04], k=3.
    # Sorted: 0.01 < 0.05/3 ≈ 0.0167 ✓ reject
    #         0.03 < 0.05/2 = 0.025? No → stop.
    # Expect: only the smallest gets rejected.
    assert ps.holm_bonferroni([0.01, 0.03, 0.04], alpha=0.05) == [True, False, False]


def test_holm_bonferroni_preserves_input_order():
    # Largest p first in input — make sure indexing in the output matches input.
    # k=2: smallest (0.001) compared to α/2=0.025 → reject. Then 0.06 vs α/1=0.05
    # → 0.06 > 0.05, stop. Only the smaller p (at input index 1) gets rejected.
    result = ps.holm_bonferroni([0.06, 0.001], alpha=0.05)
    assert result == [False, True]


def test_render_report_includes_decision_tree(synthetic_phase2):
    results_root, slurm_root, job_id = synthetic_phase2
    discovered = ps.discover_runs(slurm_root, job_id)
    records = [ps.load_run(results_root, *d) for d in discovered]
    report = ps.render_report(records, job_id)
    # Expected structural elements.
    assert "Phase 2 summary" in report
    assert "A_sotr_full" in report
    assert "B_drop_alpha" in report
    # B should be flagged as significantly worse than A.
    assert "A > B" in report


def test_write_csv_round_trips(synthetic_phase2, tmp_path):
    results_root, slurm_root, job_id = synthetic_phase2
    discovered = ps.discover_runs(slurm_root, job_id)
    records = [ps.load_run(results_root, *d) for d in discovered]
    csv_path = tmp_path / "_per_run.csv"
    ps.write_csv(records, csv_path)
    text = csv_path.read_text()
    assert "run_id,cell,seed,lr,final_val_loss" in text
    assert "A_sotr_full" in text
    assert "B_drop_alpha" in text
    # 50 data rows + 1 header.
    assert len(text.strip().splitlines()) == 51


def test_main_runs_end_to_end(synthetic_phase2, tmp_path, capsys, monkeypatch):
    results_root, slurm_root, job_id = synthetic_phase2
    rc = ps.main(
        [
            "--job-id",
            job_id,
            "--results-dir",
            str(results_root),
            "--slurm-log-dir",
            str(slurm_root),
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Phase 2 summary" in out
    assert (tmp_path / f"_summary_{job_id}.md").exists()
    assert (tmp_path / f"_per_run_{job_id}.csv").exists()
