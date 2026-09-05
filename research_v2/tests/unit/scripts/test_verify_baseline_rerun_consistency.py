"""Tests for the baseline re-run consistency verifier."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "verify_baseline_rerun_consistency.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "verify_baseline_rerun_consistency", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_baseline_rerun_consistency"] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def _report(winner: str = "B2", rank_ic: float = 0.09) -> dict:
    return {
        "folds": list(MODULE.EXPECTED_FOLDS),
        "selections": [
            {
                "fold_id": fold,
                "selected_baseline_id": winner,
                "validation_scores": [
                    {
                        "baseline_id": baseline,
                        "mean_rank_ic": rank_ic if baseline == winner else 0.01,
                        "valid_day_count": 240,
                        "invalid_day_count": 0,
                    }
                    for baseline in MODULE.EXPECTED_BASELINES
                ],
                "unselected_reasons": [],
            }
            for fold in MODULE.EXPECTED_FOLDS
        ],
        "oos": {
            "valid_day_count": 1455,
            "mean_rank_ic": 0.0864,
            "icir": 0.6344,
            "hac_t_statistic": 13.09,
            "baseline_mean_rank_ic": 0.0864,
            "delta_rank_ic": 0.0,
            "yearly_mean_rank_ic": {str(year): 0.08 for year in range(2020, 2026)},
            "gates": {
                "G1": {"status": "PASS", "observed_value": 0.0864, "threshold": 0.03},
                "G2": {"status": "PASS", "observed_value": 13.09, "threshold": 2.0},
                "G3": {"status": "PASS", "observed_value": 1.0, "threshold": 0.7},
                "G4": {"status": "NOT_APPLICABLE", "observed_value": None, "threshold": None},
                "G5": {"status": "PASS", "observed_value": 0.2, "threshold": 0.5},
                "G6": {"status": "PASS", "observed_value": 1.0, "threshold": 0.95},
            },
            "all_passed": True,
        },
    }


def test_identical_reports_are_consistent() -> None:
    assert MODULE.compare_reports(_report(), _report()) == []


def test_winner_difference_is_reported() -> None:
    problems = MODULE.compare_reports(_report(winner="B3"), _report())
    assert any("winner differs" in problem for problem in problems)


def test_oos_statistic_difference_is_reported() -> None:
    changed = _report()
    changed["oos"]["mean_rank_ic"] = 0.05
    problems = MODULE.compare_reports(changed, _report())
    assert any("oos.mean_rank_ic differs" in problem for problem in problems)


def test_gate_status_difference_is_reported() -> None:
    changed = _report()
    changed["oos"]["gates"]["G1"]["status"] = "FAIL"
    problems = MODULE.compare_reports(changed, _report())
    assert any("gate statuses differ" in problem for problem in problems)


def test_check_artifacts_requires_all_folds(tmp_path: Path) -> None:
    problems = MODULE.check_artifacts({})
    assert problems == ["report has no baseline_oos_artifacts mapping"]
    partial = {
        "baseline_oos_artifacts": {fold: str(tmp_path / f"{fold}.parquet")
                                    for fold in MODULE.EXPECTED_FOLDS[:-1]}
    }
    problems = MODULE.check_artifacts(partial)
    assert any("F06" in problem for problem in problems)


def test_check_artifacts_validates_columns(tmp_path: Path) -> None:
    paths = {}
    for fold in MODULE.EXPECTED_FOLDS:
        frame = pd.DataFrame(
            {
                "signal_date": [pd.Timestamp("2020-01-02")],
                "ts_code": ["000001.SZ"],
                "score": [0.5],
                "model_id": ["B2"],
                "fold_id": [fold],
                "dataset_id": ["x"],
            }
        )
        target = tmp_path / f"{fold}.parquet"
        frame.to_parquet(target, index=False)
        paths[fold] = str(target)
    assert MODULE.check_artifacts({"baseline_oos_artifacts": paths}) == []
    bad = pd.DataFrame({"signal_date": [1], "ts_code": ["a"]})
    bad_path = tmp_path / "F01.parquet"
    bad.to_parquet(bad_path, index=False)
    paths["F01"] = str(bad_path)
    problems = MODULE.check_artifacts({"baseline_oos_artifacts": paths})
    assert any("artifact columns for F01" in problem for problem in problems)


def test_main_exit_codes(tmp_path: Path) -> None:
    paths = {}
    for fold in MODULE.EXPECTED_FOLDS:
        frame = pd.DataFrame(
            {
                "signal_date": [pd.Timestamp("2020-01-02")],
                "ts_code": ["000001.SZ"],
                "score": [0.5],
                "model_id": ["B2"],
                "fold_id": [fold],
                "dataset_id": ["x"],
            }
        )
        target = tmp_path / f"{fold}.parquet"
        frame.to_parquet(target, index=False)
        paths[fold] = str(target)
    payload = _report()
    payload["baseline_oos_artifacts"] = paths
    report = tmp_path / "new.json"
    reference = tmp_path / "ref.json"
    report.write_text(json.dumps(payload), encoding="utf-8")
    reference.write_text(json.dumps(_report()), encoding="utf-8")
    assert MODULE.main(["--report", str(report), "--reference", str(reference)]) == 0
    changed = _report(winner="B3")
    changed["baseline_oos_artifacts"] = paths
    report.write_text(json.dumps(changed), encoding="utf-8")
    assert MODULE.main(["--report", str(report), "--reference", str(reference)]) == 1
