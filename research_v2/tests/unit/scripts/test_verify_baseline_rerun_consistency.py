"""Tests for the baseline re-run consistency verifier."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

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


# --- the comparison is driven by what the reference carries --------------------------------


def test_a_day_count_difference_is_reported() -> None:
    """The field the coverage problem lives in has to be part of the trust root.

    ``invalid_day_count`` is what Task 3.3 issues and what Task 3.1b's blocked coverage
    question is about. Comparing only ``mean_rank_ic`` means a re-run that scored 228 days
    instead of 240 and still landed within 1e-9 of the archived RankIC would be reported as
    reproducing the evidence -- and the frozen-set report is about to become the reference
    that Task 4.1 points this verifier at.
    """
    changed = _report()
    scores = changed["selections"][0]["validation_scores"]
    scores[0]["valid_day_count"] = 228
    scores[0]["invalid_day_count"] = 12

    problems = MODULE.compare_reports(changed, _report())

    assert any("valid_day_count differs" in problem for problem in problems), problems
    assert any("invalid_day_count differs" in problem for problem in problems), problems


def test_unselected_reasons_are_compared() -> None:
    """A winner that changed is caught; a loser that lost for a different reason was not."""
    changed = _report()
    changed["selections"][0]["unselected_reasons"] = [["B0", "INSUFFICIENT_VALID_DAYS"]]
    reference = _report()
    reference["selections"][0]["unselected_reasons"] = [
        ["B0", "LOWER_VALIDATION_RANKIC"]
    ]

    problems = MODULE.compare_reports(changed, reference)

    assert any("unselected_reasons differ" in problem for problem in problems), problems


def test_a_data_version_difference_is_reported_by_table() -> None:
    """The version mapping is the part of the evidence that says *which data* was scored.

    Task 2.4's verdict rested on a hand-run field-by-field comparison that found
    ``data_versions`` byte-identical; this verifier did not compare it at all, so a re-run
    against different published versions would have been reported CONSISTENT. Naming the
    differing tables rather than dumping the mapping keeps the message readable at 151
    partitions per table.
    """
    reference = _report()
    reference["data_versions"] = {
        "universe_daily": {"2020.01M": "v1"},
        "factor_observation_clean": {"2020.01M": "v1"},
    }
    changed = _report()
    changed["data_versions"] = {
        "universe_daily": {"2020.01M": "v2"},
        "factor_observation_clean": {"2020.01M": "v1"},
    }

    problems = MODULE.compare_reports(changed, reference)

    assert any("data_versions differ" in problem for problem in problems), problems
    assert any("universe_daily" in problem for problem in problems)
    assert not any("factor_observation_clean" in problem for problem in problems), (
        "only the table that actually moved should be named"
    )


def test_a_rerun_missing_versions_the_reference_has_is_reported() -> None:
    reference = _report()
    reference["data_versions"] = {"universe_daily": {"2020.01M": "v1"}}

    problems = MODULE.compare_reports(_report(), reference)

    assert any("data_versions differ" in problem for problem in problems), problems


def test_a_reference_that_predates_the_extended_fields_still_verifies_clean() -> None:
    """The archived criterion-303 reference carries none of these fields, and must stay usable.

    Task 2.4's CONSISTENT verdict was produced against exactly that reference while the re-run
    carried the new fields. Requiring them would turn a proved-consistent comparison into a
    failure, so the comparison is driven by what the *reference* carries: absent there, not
    required here; present there -- as it will be once the frozen-set report becomes the
    reference under Task 4.1 -- verified.
    """
    legacy = _report()
    for selection in legacy["selections"]:
        for score in selection["validation_scores"]:
            del score["invalid_day_count"]
        del selection["unselected_reasons"]

    assert MODULE.compare_reports(_report(), legacy) == []


def test_the_real_criterion_303_pair_is_still_consistent(tmp_path: Path) -> None:
    """End to end against the archived evidence and the real Task 2.4 re-run, if both are present.

    This is the regression that matters: the stricter comparison must not overturn a verdict
    that was already issued and relied on. Skipped when either file is absent, since the
    re-run report lives outside the repository.
    """
    reports = Path(__file__).resolve().parents[4] / "docs" / "reports"
    reference = reports / "2026-09-04-real-walk-forward-criterion-303.json"
    rerun = Path("/tmp/legacy_flagcheck/report.json")
    if not reference.exists() or not rerun.exists():
        pytest.skip("archived reference or Task 2.4 re-run report not present")

    problems = MODULE.compare_reports(
        json.loads(rerun.read_text(encoding="utf-8")),
        json.loads(reference.read_text(encoding="utf-8")),
    )

    assert problems == [], problems
