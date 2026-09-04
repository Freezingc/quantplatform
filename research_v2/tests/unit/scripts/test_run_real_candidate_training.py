"""Contract tests for the real-data candidate training driver (Task 2).

The driver must enforce the E-plan execution order (M1 -> M2 -> M3), refuse
to run without fold-matched baseline artifacts, and load those artifacts
fail-closed. Actual training wiring lands in Tasks 3-4.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[3] / "scripts" / "run_real_candidate_training.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("run_real_candidate_training", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()

PREDICTION_COLUMNS = ["signal_date", "ts_code", "score", "model_id", "fold_id", "dataset_id"]


def _write_fold_artifact(directory: Path, fold_id: str) -> None:
    frame = pd.DataFrame(
        {
            "signal_date": [pd.Timestamp("2020-01-02")],
            "ts_code": ["000001.SZ"],
            "score": [0.5],
            "model_id": ["B2"],
            "fold_id": [fold_id],
            "dataset_id": [f"real-{fold_id.lower()}-v1-test"],
        }
    )
    frame.to_parquet(directory / f"{fold_id}.parquet", index=False)


def test_load_baseline_predictions_reads_all_folds(tmp_path: Path) -> None:
    for fold in MODULE.ORDINARY_FOLDS:
        _write_fold_artifact(tmp_path, fold)
    frames = MODULE.load_baseline_predictions(tmp_path)
    assert sorted(frames) == list(MODULE.ORDINARY_FOLDS)
    assert list(frames["F01"].columns) == PREDICTION_COLUMNS


def test_load_baseline_predictions_requires_every_fold(tmp_path: Path) -> None:
    for fold in MODULE.ORDINARY_FOLDS[:-1]:  # drop F06
        _write_fold_artifact(tmp_path, fold)
    with pytest.raises(ValueError, match="F06"):
        MODULE.load_baseline_predictions(tmp_path)


def test_family_order_guard_allows_m1() -> None:
    MODULE.validate_family_order("M1", prior_families_passed=())


def test_family_order_guard_blocks_m2_without_evidence() -> None:
    with pytest.raises(ValueError, match="M1"):
        MODULE.validate_family_order("M2", prior_families_passed=())


def test_family_order_guard_blocks_m3_without_m2() -> None:
    with pytest.raises(ValueError, match="M2"):
        MODULE.validate_family_order("M3", prior_families_passed=("M1",))


def test_family_order_guard_allows_m3_after_m1_m2() -> None:
    MODULE.validate_family_order("M3", prior_families_passed=("M1", "M2"))


def test_reject_unknown_family() -> None:
    with pytest.raises(ValueError, match="family"):
        MODULE.validate_family_order("M4", prior_families_passed=("M1", "M2"))


def test_parse_args_requires_baseline_artifacts_for_full_runs(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    args = MODULE.parse_args(
        ["--family", "M1", "--baseline-artifacts", str(tmp_path), "--report", str(report)]
    )
    assert args.family == "M1"
    assert args.report == report
    assert not args.smoke


def test_parse_args_smoke_mode_allows_missing_artifacts(tmp_path: Path) -> None:
    args = MODULE.parse_args(
        ["--family", "M1", "--smoke", "--report", str(tmp_path / "smoke.json")]
    )
    assert args.smoke is True
    assert args.baseline_artifacts is None


def test_m1_smoke_pipeline_end_to_end(tmp_path: Path) -> None:
    """Task 3.1: orchestration evidence only - synthetic gates are not model evidence."""
    report = MODULE.train_m1_smoke(tmp_path)
    assert report["family"] == "M1"
    assert report["mode"] == "smoke"
    oos = report["oos"]
    assert oos["valid_day_count"] == 8
    assert set(oos["gates"]) == {"G1", "G2", "G3", "G4", "G5", "G6"}
    assert all(status in {"PASS", "FAIL", "NOT_APPLICABLE"} for status in oos["gates"].values())
    assert oos["baseline_mean_rank_ic"] is not None
    assert set(report["seed_validation_rank_ic"]) == set(MODULE.REQUIRED_SEEDS)
    assert report["worst_seed"] in MODULE.REQUIRED_SEEDS


def test_m2_smoke_pipeline_end_to_end(tmp_path: Path) -> None:
    """Task 4.1: M2 temporal (GRU) orchestration evidence on synthetic frames."""
    report = MODULE.train_m2_smoke(tmp_path)
    assert report["family"] == "M2"
    assert report["mode"] == "smoke"
    oos = report["oos"]
    assert oos["valid_day_count"] == 8
    assert set(oos["gates"]) == {"G1", "G2", "G3", "G4", "G5", "G6"}
    assert oos["baseline_mean_rank_ic"] is not None
    assert set(report["seed_validation_rank_ic"]) == set(MODULE.REQUIRED_SEEDS)
