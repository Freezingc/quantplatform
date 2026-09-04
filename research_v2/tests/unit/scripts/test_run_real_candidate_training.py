"""Contract tests for the real-data candidate training driver (Task 2).

The driver must enforce the E-plan execution order (M1 -> M2 -> M3), refuse
to run without fold-matched baseline artifacts, and load those artifacts
fail-closed. Actual training wiring lands in Tasks 3-4.
"""

from __future__ import annotations

import importlib.util
import sys
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
    sys.modules["run_real_candidate_training"] = module
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


def test_m3_smoke_pipeline_end_to_end(tmp_path: Path) -> None:
    """Task 4.3: M3 set-attention orchestration evidence on synthetic frames."""
    report = MODULE.train_m3_smoke(tmp_path)
    assert report["family"] == "M3"
    assert report["mode"] == "smoke"
    oos = report["oos"]
    assert oos["valid_day_count"] == 8
    assert set(oos["gates"]) == {"G1", "G2", "G3", "G4", "G5", "G6"}
    assert oos["baseline_mean_rank_ic"] is not None
    assert set(report["seed_validation_rank_ic"]) == set(MODULE.REQUIRED_SEEDS)


def _fold_frames() -> dict[str, pd.DataFrame]:
    def _split(day, with_target: bool) -> pd.DataFrame:
        rows = []
        for index in range(32):
            row: dict[str, object] = {
                "signal_date": day,
                "ts_code": f"s{index:02d}",
                "label_valid": True,
                "f00": float(index) * 0.1,
                "f01": float(index) * -0.05,
            }
            if with_target:
                row["raw_return_5d"] = float(index) * 0.01
            rows.append(row)
        return pd.DataFrame(rows)

    return {
        "train": _split(pd.Timestamp("2020-01-02"), with_target=True),
        "validation": _split(pd.Timestamp("2021-01-04"), with_target=True),
        "test": _split(pd.Timestamp("2022-01-04"), with_target=False).assign(
            raw_return_5d=0.0
        ),
    }


def test_publish_fold_dataset_is_content_addressed_and_idempotent(tmp_path: Path) -> None:
    from research_v2.datasets.storage import DatasetArtifactStore

    store = DatasetArtifactStore(tmp_path / "datasets")
    frames = _fold_frames()
    versions = {"label_daily": {"2020-01": "v1"}}
    factor_order = ("f00", "f01")
    first = MODULE.publish_fold_dataset(store, "F01", frames, versions, factor_order)
    second = MODULE.publish_fold_dataset(store, "F01", frames, versions, factor_order)
    assert first == second
    assert first  # non-empty content hash id


def test_reconstruct_baseline_record_round_trip() -> None:
    entry = {
        "fold_id": "F01",
        "selected_baseline_id": "B2",
        "validation_scores": [
            {
                "baseline_id": "B2",
                "mean_rank_ic": 0.02,
                "valid_day_count": 240,
                "invalid_day_count": 1,
            }
        ],
        "unselected_reasons": [["B0", "LOWER_VALIDATION_RANKIC"]],
    }
    record = MODULE.reconstruct_baseline_record(entry)
    assert record.fold_id == "F01"
    assert record.selected_baseline_id == "B2"
    assert record.validation_scores[0].invalid_day_count == 1
    assert record.unselected_reasons == (("B0", "LOWER_VALIDATION_RANKIC"),)


def test_issue_baseline_selection_evidence_signs_and_verifies(tmp_path: Path) -> None:
    from research_v2.datasets.storage import DatasetArtifactStore
    from research_v2.evaluation.selection_evidence import (
        SelectionEvidenceStore,
        SelectionEvidenceVerifier,
    )

    store = DatasetArtifactStore(tmp_path / "datasets")
    frames = _fold_frames()
    versions = {"label_daily": {"2020-01": "v1"}}
    dataset_id = MODULE.publish_fold_dataset(
        store, "F01", frames, versions, ("f00", "f01")
    )
    selection_store = SelectionEvidenceStore(tmp_path / "selections", signing_key=b"unit-key")
    baseline_report = {
        "selections": [
            {
                "fold_id": "F01",
                "selected_baseline_id": "B2",
                "validation_scores": [
                    {
                        "baseline_id": baseline_id,
                        "mean_rank_ic": 0.02 if baseline_id == "B2" else 0.01,
                        "valid_day_count": 240,
                        "invalid_day_count": 2,
                    }
                    for baseline_id in ("B0", "B1", "B2", "B3")
                ],
                "unselected_reasons": [
                    ["B0", "LOWER_VALIDATION_RANKIC"],
                    ["B1", "LOWER_VALIDATION_RANKIC"],
                    ["B3", "LOWER_VALIDATION_RANKIC"],
                ],
            }
        ]
    }
    evidence_ids = MODULE.issue_baseline_selection_evidence(
        baseline_report, {"F01": dataset_id}, selection_store
    )
    assert "F01" in evidence_ids
    verifier = SelectionEvidenceVerifier(tmp_path / "selections", verification_key=b"unit-key")
    loaded = verifier.load_baseline_selection(evidence_ids["F01"])
    assert loaded.fold_id == "F01"
    assert loaded.test_dataset_id == dataset_id
    assert loaded.selected_baseline_id == "B2"


def test_config_grids_respect_e7_budget() -> None:
    """E7: at most 30 configs per family search."""
    assert len(MODULE.m1_config_grid()) == 27 <= 30
    assert len(MODULE.m2_config_grid()) == 24 <= 30
    assert len(MODULE.m3_config_grid()) == 16 <= 30
    m2_families = {config["family"] for config in MODULE.m2_config_grid()}
    assert m2_families == {"GRU", "TCN"}
