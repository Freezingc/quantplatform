"""Tests for the real-data candidate walk-forward verdict renderer."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import summarize_real_candidate_walk_forward as module  # noqa: E402


def _gate(status: str, observed: float | None, threshold: float | None) -> dict:
    return {"status": status, "observed_value": observed, "threshold": threshold}


def _fold_report(
    fold_id: str,
    *,
    ensemble_validation_rank_ic: float = 0.09,
    worst_seed: int = 20260802,
) -> dict:
    return {
        "fold_id": fold_id,
        "dataset_id": f"dataset-{fold_id.lower()}",
        "model_id": f"m1-{fold_id.lower()}",
        "best_config": {"hidden_dim": 128, "num_blocks": 2, "dropout": 0.1},
        "search_results": [
            {
                "config": {"hidden_dim": 64, "num_blocks": 1, "dropout": 0.0},
                "ensemble_validation_rank_ic": ensemble_validation_rank_ic - 0.01,
                "seed_validation_rank_ic": {"20260801": 0.08, "20260802": 0.075, "20260803": 0.079},
                "worst_seed": 20260802,
            },
            {
                "config": {"hidden_dim": 128, "num_blocks": 2, "dropout": 0.1},
                "ensemble_validation_rank_ic": ensemble_validation_rank_ic,
                "seed_validation_rank_ic": {"20260801": 0.09, "20260802": 0.083, "20260803": 0.088},
                "worst_seed": worst_seed,
            },
        ],
        "baseline_selection_id": f"baseline-evidence-{fold_id.lower()}",
        "model_selection_id": f"model-evidence-{fold_id.lower()}",
    }


def _report(*, all_passed: bool = True, folds: str = "all") -> dict:
    fold_ids = list(module.REQUIRED_FOLDS) if folds == "all" else ["F01", "F02"]
    return {
        "family": "M1",
        "mode": "full",
        "folds": [_fold_report(fold_id) for fold_id in fold_ids],
        "oos": {
            "valid_day_count": 1455,
            "mean_rank_ic": 0.0871,
            "icir": 0.64,
            "hac_t_statistic": 12.9,
            "baseline_mean_rank_ic": 0.0864,
            "delta_rank_ic": 0.0007,
            "yearly_mean_rank_ic": {
                "2020": 0.07,
                "2021": 0.066,
                "2022": 0.095,
                "2023": 0.085,
                "2024": 0.105,
                "2025": 0.098,
            },
            "gates": {
                "G1": _gate("PASS", 0.0871, 0.03),
                "G2": _gate("PASS", 12.9, 2.0),
                "G3": _gate("PASS", 1.0, 0.7),
                "G4": _gate("FAIL", 0.0007, 0.005),
                "G5": _gate("PASS", 0.21, 0.5),
                "G6": _gate("PASS", 1.0, 0.95),
            },
            "all_passed": all_passed,
        },
    }


def _baseline_report() -> dict:
    return {
        "selections": [
            {
                "fold_id": fold_id,
                "selected_baseline_id": "B3",
                "validation_scores": [
                    {"baseline_id": "B0", "mean_rank_ic": 0.072, "valid_day_count": 238},
                    {"baseline_id": "B1", "mean_rank_ic": -0.075, "valid_day_count": 238},
                    {"baseline_id": "B2", "mean_rank_ic": 0.094, "valid_day_count": 238},
                    {"baseline_id": "B3", "mean_rank_ic": 0.099, "valid_day_count": 238},
                ],
            }
            for fold_id in module.REQUIRED_FOLDS
        ],
        "data_versions": {
            "universe_daily": {"2014-01": "aa", "2026-08": "bb"},
            "factor_observation_clean": {"2014-01": "cc", "2026-08": "dd"},
            "label_daily": {"2014-01": "ee", "2026-08": "ff"},
        },
    }


class TestGateVerdicts:
    def test_parses_every_gate_in_order(self) -> None:
        verdicts = module.gate_verdicts(_report())
        assert [item.gate for item in verdicts] == list(module.GATE_ORDER)
        assert verdicts[0].status == "PASS"
        assert verdicts[3].status == "FAIL"
        assert verdicts[3].observed_value == pytest.approx(0.0007)

    def test_missing_gate_fails_closed(self) -> None:
        report = _report()
        del report["oos"]["gates"]["G4"]
        verdicts = module.gate_verdicts(report)
        gate = next(item for item in verdicts if item.gate == "G4")
        assert gate.status == "FAIL"
        assert gate.observed_value is None


class TestFoldSummary:
    def test_extracts_best_config_and_worst_seed(self) -> None:
        summary = module.fold_summaries(_report())
        entry = summary["F01"]
        assert entry.model_id == "m1-f01"
        assert entry.best_config == {"hidden_dim": 128, "num_blocks": 2, "dropout": 0.1}
        assert entry.ensemble_validation_rank_ic == pytest.approx(0.09)
        assert entry.worst_seed == 20260802
        assert entry.searched_config_count == 2


class TestVerdict:
    def test_satisfied_requires_gates_folds_and_full_mode(self) -> None:
        passing = _report(all_passed=True)
        passing["oos"]["gates"]["G4"] = _gate("PASS", 0.006, 0.005)
        assert module.candidate_verdict(passing).satisfied

        failed_gates = copy.deepcopy(passing)
        failed_gates["oos"]["all_passed"] = False
        assert not module.candidate_verdict(failed_gates).satisfied

        missing_folds = copy.deepcopy(passing)
        missing_folds["folds"] = missing_folds["folds"][:2]
        assert not module.candidate_verdict(missing_folds).satisfied

        smoke_mode = copy.deepcopy(passing)
        smoke_mode["mode"] = "smoke"
        assert not module.candidate_verdict(smoke_mode).satisfied

    def test_smoke_report_is_rejected_even_with_passing_gates(self) -> None:
        report = _report(all_passed=True)
        report["mode"] = "smoke"
        verdict = module.candidate_verdict(report)
        assert verdict.mode_full is False
        assert verdict.satisfied is False


class TestRender:
    def test_render_is_deterministic(self) -> None:
        report = _report()
        assert module.render_markdown(report) == module.render_markdown(report)

    def test_render_contains_required_sections(self) -> None:
        markdown = module.render_markdown(_report())
        assert "# Real-Data M1 Candidate Walk-Forward Report" in markdown
        assert "WALK_FORWARD_FAILED" in markdown  # G4 fails in the fixture
        assert "| G4 |" in markdown
        assert "m1-f01" in markdown
        assert "20260802" in markdown  # worst seed surfaced

    def test_render_passing_report_announces_passed(self) -> None:
        report = _report()
        report["oos"]["gates"]["G4"] = _gate("PASS", 0.006, 0.005)
        markdown = module.render_markdown(report)
        assert "WALK_FORWARD_PASSED" in markdown

    def test_baseline_comparison_section_optional(self) -> None:
        markdown = module.render_markdown(_report(), baseline_report=_baseline_report())
        assert "Validation-stage comparison against B0-B3" in markdown
        assert "| F01 |" in markdown
        assert "Data versions" in markdown

    def test_missing_fold_rendered_as_missing(self) -> None:
        report = _report()
        report["folds"] = report["folds"][:2]
        markdown = module.render_markdown(report)
        assert "MISSING" in markdown


class TestCli:
    def test_writes_report_and_prints_verdict(self, tmp_path: Path, capsys) -> None:
        report_path = tmp_path / "m1_report.json"
        report_path.write_text(json.dumps(_report()), encoding="utf-8")
        out_path = tmp_path / "verdict.md"
        sys.argv = [
            "summarize_real_candidate_walk_forward.py",
            "--report",
            str(report_path),
            "--out",
            str(out_path),
        ]
        module.main()
        captured = capsys.readouterr()
        assert "WALK_FORWARD_FAILED" in captured.out
        assert out_path.read_text(encoding="utf-8").startswith(
            "# Real-Data M1 Candidate Walk-Forward Report"
        )
