"""Tests for the baseline-report candidate-consumption schema gate."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "verify_baseline_report_candidate_schema.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "verify_baseline_report_candidate_schema", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_baseline_report_candidate_schema"] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def _selection(fold_id: str) -> dict:
    return {
        "fold_id": fold_id,
        "selected_baseline_id": "B3",
        "unselected_reasons": [["B0", "LOWER_VALIDATION_RANKIC"]],
        "validation_scores": [
            {
                "baseline_id": baseline,
                "mean_rank_ic": 0.05,
                "valid_day_count": 238,
                "invalid_day_count": 0,
            }
            for baseline in ("B0", "B1", "B2", "B3")
        ],
    }


def _report(tmp_path: Path) -> dict:
    artifacts = {}
    for fold in MODULE.EXPECTED_FOLDS:
        target = tmp_path / f"{fold}.parquet"
        target.write_bytes(b"placeholder")
        artifacts[fold] = str(target)
    return {
        "data_versions": {"universe_daily": {"2014-01": "aa"}},
        "baseline_oos_artifacts": artifacts,
        "oos": {
            "all_passed": True,
            "gates": {gate: {"status": "PASS"} for gate in ("G1", "G2", "G3", "G4", "G5", "G6")},
        },
        "selections": [_selection(fold) for fold in MODULE.EXPECTED_FOLDS],
    }


def test_complete_report_passes(tmp_path: Path) -> None:
    problems = MODULE.check_report(_report(tmp_path))
    assert problems == []


def test_missing_extended_fields_fail(tmp_path: Path) -> None:
    report = _report(tmp_path)
    del report["selections"][0]["unselected_reasons"]
    del report["selections"][1]["validation_scores"][0]["invalid_day_count"]
    problems = MODULE.check_report(report)
    assert any("F1" in item and "unselected_reasons" in item for item in problems)
    assert any("F2" in item and "invalid_day_count" in item for item in problems)


def test_missing_gate_block_fails(tmp_path: Path) -> None:
    report = _report(tmp_path)
    report["oos"]["gates"] = {"G1": {"status": "PASS"}}
    problems = MODULE.check_report(report)
    assert any("G1..G6" in item for item in problems)


def test_missing_artifact_on_disk_fails(tmp_path: Path) -> None:
    report = _report(tmp_path)
    report["baseline_oos_artifacts"]["F3"] = str(tmp_path / "absent.parquet")
    problems = MODULE.check_report(report)
    assert any("F3" in item and "artifact" in item for item in problems)


def test_incomplete_fold_set_fails(tmp_path: Path) -> None:
    report = _report(tmp_path)
    report["selections"] = report["selections"][:3]
    problems = MODULE.check_report(report)
    assert any("F1..F4" in item for item in problems)


def test_cli_exit_codes(tmp_path: Path) -> None:
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_report(tmp_path)), encoding="utf-8")
    assert MODULE.run(["--report", str(good)]) == 0
    broken = _report(tmp_path)
    del broken["selections"][0]["unselected_reasons"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(broken), encoding="utf-8")
    assert MODULE.run(["--report", str(bad)]) == 1
