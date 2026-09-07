"""Tests for the M1-M3 candidate comparison renderer (Task 5.1)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import summarize_real_candidate_comparison as module  # noqa: E402


def _gate(status: str, observed: float | None, threshold: float | None) -> dict:
    return {"status": status, "observed_value": observed, "threshold": threshold}


def _family_report(
    family: str,
    *,
    mean_rank_ic: float,
    delta: float,
    all_passed: bool,
    mode: str = "full",
) -> dict:
    gates = {
        "G1": _gate("PASS", mean_rank_ic, 0.03),
        "G2": _gate("PASS", 11.0, 2.0),
        "G3": _gate("PASS", 1.0, 0.7),
        "G4": _gate("PASS" if all_passed else "FAIL", delta, 0.005),
        "G5": _gate("PASS", 0.22, 0.5),
        "G6": _gate("PASS", 1.0, 0.95),
    }
    return {
        "family": family,
        "mode": mode,
        "folds": [
            {
                "fold_id": fold,
                "model_id": f"{family.lower()}-{fold.lower()}",
                "best_config": {"hidden_dim": 128},
                "search_results": [
                    {
                        "config": {"hidden_dim": 128},
                        "ensemble_validation_rank_ic": 0.09,
                        "seed_validation_rank_ic": {
                            "20260801": 0.09,
                            "20260802": 0.082,
                            "20260803": 0.088,
                        },
                        "worst_seed": 20260802,
                    }
                ],
            }
            for fold in ("F1", "F2", "F3", "F4")
        ],
        "oos": {
            "valid_day_count": 1455,
            "mean_rank_ic": mean_rank_ic,
            "icir": 0.63,
            "hac_t_statistic": 12.5,
            "baseline_mean_rank_ic": 0.0864,
            "delta_rank_ic": delta,
            "yearly_mean_rank_ic": {
                "2020": 0.069,
                "2021": 0.066,
                "2022": 0.095,
                "2023": 0.085,
                "2024": 0.105,
                "2025": 0.098,
            },
            "gates": gates,
            "all_passed": all_passed,
        },
    }


def _baseline_report() -> dict:
    return {
        "data_versions": {"universe_daily": {"2014-01": "aa", "2026-08": "bb"}},
        "oos": {
            "mean_rank_ic": 0.0864,
            "yearly_mean_rank_ic": {
                "2020": 0.069,
                "2021": 0.066,
                "2022": 0.095,
                "2023": 0.085,
                "2024": 0.105,
                "2025": 0.098,
            },
        },
    }


class TestSummaryTable:
    def test_collects_family_rows(self) -> None:
        rows = module.family_rows(
            {
                "M1": _family_report("M1", mean_rank_ic=0.0871, delta=0.0007, all_passed=False),
                "M2": None,
                "M3": _family_report("M3", mean_rank_ic=0.0921, delta=0.0057, all_passed=True),
            }
        )
        assert [row.family for row in rows] == ["M1", "M2", "M3"]
        assert rows[0].status == "FAILED"
        assert rows[1].status == "NOT_RUN"
        assert rows[2].status == "PASSED"
        assert rows[2].delta_rank_ic == pytest.approx(0.0057)


class TestRender:
    def test_no_candidate_passed_records_baseline_retained(self) -> None:
        markdown = module.render_markdown(
            {
                "M1": _family_report("M1", mean_rank_ic=0.087, delta=0.0006, all_passed=False),
                "M2": None,
                "M3": None,
            },
            baseline_report=_baseline_report(),
        )
        assert "无复杂候选通过，简单基线保留" in markdown
        assert "基线过强" in markdown  # risk-register clause reference
        assert "SHADOW" in markdown  # must not claim shadow eligibility

    def test_passing_candidate_triggers_shadow_checklist(self) -> None:
        markdown = module.render_markdown(
            {
                "M1": None,
                "M2": _family_report("M2", mean_rank_ic=0.093, delta=0.0066, all_passed=True),
                "M3": None,
            },
            baseline_report=_baseline_report(),
        )
        assert "WALK_FORWARD_PASSED" in markdown
        assert "F6.1" in markdown
        assert "60 个有效" in markdown
        assert "不擅自启动" in markdown

    def test_pairwise_deltas_present(self) -> None:
        markdown = module.render_markdown(
            {
                "M1": _family_report("M1", mean_rank_ic=0.0871, delta=0.0007, all_passed=False),
                "M2": _family_report("M2", mean_rank_ic=0.0921, delta=0.0057, all_passed=True),
                "M3": _family_report("M3", mean_rank_ic=0.0905, delta=0.0041, all_passed=False),
            },
            baseline_report=_baseline_report(),
        )
        assert "相对彼此增量" in markdown
        assert "| M1 | M2 |" in markdown or "M1 → M2" in markdown

    def test_worst_seed_and_yearly_distribution_rendered(self) -> None:
        markdown = module.render_markdown(
            {
                "M1": _family_report("M1", mean_rank_ic=0.0871, delta=0.0007, all_passed=False),
                "M2": None,
                "M3": None,
            },
            baseline_report=_baseline_report(),
        )
        assert "20260802" in markdown
        assert "| 2024 |" in markdown

    def test_render_is_deterministic(self) -> None:
        reports = {
            "M1": _family_report("M1", mean_rank_ic=0.0871, delta=0.0007, all_passed=False),
            "M2": None,
            "M3": None,
        }
        baseline = _baseline_report()
        assert module.render_markdown(reports, baseline) == module.render_markdown(
            reports, baseline
        )


class TestCli:
    def test_writes_output_and_reports_verdict(self, tmp_path: Path, capsys) -> None:
        m1 = tmp_path / "m1.json"
        m1.write_text(
            json.dumps(_family_report("M1", mean_rank_ic=0.087, delta=0.0006, all_passed=False)),
            encoding="utf-8",
        )
        baseline = tmp_path / "baseline.json"
        baseline.write_text(json.dumps(_baseline_report()), encoding="utf-8")
        out = tmp_path / "comparison.md"
        sys.argv = [
            "summarize_real_candidate_comparison.py",
            "--m1-report",
            str(m1),
            "--baseline-report",
            str(baseline),
            "--out",
            str(out),
        ]
        module.main()
        text = out.read_text(encoding="utf-8")
        assert text.startswith("# Real-Data Candidate Model Comparison")
        captured = capsys.readouterr()
        assert "无复杂候选通过" in captured.out

    def test_missing_report_paths_are_not_run(self, tmp_path: Path) -> None:
        out = tmp_path / "comparison.md"
        sys.argv = [
            "summarize_real_candidate_comparison.py",
            "--out",
            str(out),
        ]
        module.main()
        text = out.read_text(encoding="utf-8")
        assert text.count("NOT_RUN") >= 3 or "未运行" in text
