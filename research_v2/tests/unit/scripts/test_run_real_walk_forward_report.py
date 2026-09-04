"""Regression tests for the walk-forward driver's report construction.

The 2026-09-02 six-fold run crashed at report serialization
(TypeError: 'bool' object is not callable) because GateReport.all_passed is a
property, not a method. These tests exercise build_walk_forward_report with
real evaluation objects so that class of attribute misuse fails fast.
"""

from __future__ import annotations

import pandas as pd
import pytest
import run_real_walk_forward as driver

from research_v2.baselines.base import PredictionFrame
from research_v2.baselines.orchestration import (
    BaselineFoldResult,
    BaselineSelectionRecord,
    BaselineTestEvaluation,
    BaselineValidationScore,
)
from research_v2.evaluation.gates import GateReport, GateResult, GateStatus
from research_v2.evaluation.hac import HACMeanTest
from research_v2.evaluation.model_evaluator import FrozenOosEvaluation
from research_v2.evaluation.rank_ic import RankICSummary
from research_v2.evaluation.walk_forward import BaselineWalkForwardResult


def _gate_report(statuses: dict[str, GateStatus]) -> GateReport:
    return GateReport(
        results={
            gate: GateResult(status=status, observed_value=0.04, threshold=0.03)
            for gate, status in statuses.items()
        }
    )


def _evaluation(gate_report: GateReport) -> FrozenOosEvaluation:
    daily = pd.DataFrame(
        {
            "signal_date": [pd.Timestamp("2020-01-02")],
            "mean_rank_ic": [0.04],
        }
    )
    return FrozenOosEvaluation(
        daily_rank_ic=daily,
        summary=RankICSummary(
            valid_day_count=1,
            mean_rank_ic=0.04,
            rank_ic_std=0.01,
            icir=0.5,
        ),
        hac=HACMeanTest(
            observation_count=1,
            max_lags=5,
            mean=0.04,
            standard_error=0.02,
            t_statistic=2.1,
            p_value=0.03,
        ),
        yearly_mean_rank_ic={2020: 0.04},
        baseline_mean_rank_ic=None,
        delta_rank_ic=None,
        gate_report=gate_report,
        yearly_summary=daily,
    )


def _result(gate_report: GateReport) -> BaselineWalkForwardResult:
    record = BaselineSelectionRecord(
        fold_id="F01",
        selected_baseline_id="B2",
        validation_scores=(
            BaselineValidationScore(
                baseline_id="B2",
                mean_rank_ic=0.022,
                valid_day_count=240,
                invalid_day_count=0,
            ),
        ),
        unselected_reasons=(),
    )
    return BaselineWalkForwardResult(
        fold_results={},
        selection_records=(record,),
        evaluation=_evaluation(gate_report),
    )


def test_build_report_reads_all_passed_property_without_calling() -> None:
    gate_report = _gate_report({"G1": GateStatus.PASS, "G4": GateStatus.NOT_APPLICABLE})
    report = driver.build_walk_forward_report(
        ("F01",), _result(gate_report), {"label_daily": {"2020.01M": "v1"}}
    )
    assert report["oos"]["all_passed"] is True


def test_build_report_flags_failed_gates() -> None:
    gate_report = _gate_report({"G1": GateStatus.PASS, "G2": GateStatus.FAIL})
    report = driver.build_walk_forward_report(("F01",), _result(gate_report), {})
    assert report["oos"]["all_passed"] is False
    assert report["oos"]["gates"]["G2"]["status"] == "FAIL"


def test_build_report_structure() -> None:
    gate_report = _gate_report({"G1": GateStatus.PASS})
    versions = {"universe_daily": {"2020.02M": "v2", "2020.01M": "v1"}}
    report = driver.build_walk_forward_report(("F01", "F02"), _result(gate_report), versions)
    assert report["folds"] == ["F01", "F02"]
    assert report["data_versions"] == {"universe_daily": {"2020.01M": "v1", "2020.02M": "v2"}}
    selection = report["selections"][0]
    assert selection["fold_id"] == "F01"
    assert selection["selected_baseline_id"] == "B2"
    assert selection["validation_scores"][0]["mean_rank_ic"] == 0.022
    assert report["oos"]["yearly_mean_rank_ic"] == {"2020": 0.04}
    assert report["oos"]["hac_t_statistic"] == 2.1


_FOLD_IDS = ("F01", "F02", "F03", "F04", "F05", "F06")


def _fold_result(fold_id: str) -> BaselineFoldResult:
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
    record = BaselineSelectionRecord(
        fold_id=fold_id,
        selected_baseline_id="B2",
        validation_scores=(
            BaselineValidationScore(
                baseline_id="B2", mean_rank_ic=0.02, valid_day_count=1, invalid_day_count=0
            ),
        ),
        unselected_reasons=(),
    )
    evaluation = BaselineTestEvaluation(
        baseline_id="B2",
        predictions=PredictionFrame(frame=frame),
        labels=pd.DataFrame({"signal_date": [pd.Timestamp("2020-01-02")]}),
        daily_rank_ic=pd.DataFrame({"mean_rank_ic": [0.02]}),
        summary=RankICSummary(
            valid_day_count=1, mean_rank_ic=0.02, rank_ic_std=0.01, icir=0.5
        ),
    )
    return BaselineFoldResult(selection_record=record, test_evaluation=evaluation)


def _full_result():
    from research_v2.evaluation.walk_forward import BaselineWalkForwardResult

    return BaselineWalkForwardResult(
        fold_results={fold: _fold_result(fold) for fold in _FOLD_IDS},
        selection_records=(),
        evaluation=_evaluation(
            _gate_report({"G1": GateStatus.PASS, "G4": GateStatus.NOT_APPLICABLE})
        ),
    )


def test_save_baseline_oos_artifacts_writes_per_fold_parquet(tmp_path) -> None:
    paths = driver.save_baseline_oos_artifacts(_full_result(), tmp_path)
    assert sorted(paths) == list(_FOLD_IDS)
    for fold_id, file_path in paths.items():
        loaded = pd.read_parquet(file_path)
        assert list(loaded.columns) == [
            "signal_date",
            "ts_code",
            "score",
            "model_id",
            "fold_id",
            "dataset_id",
        ]
        assert loaded["fold_id"].iloc[0] == fold_id


def test_save_baseline_oos_artifacts_requires_all_folds(tmp_path) -> None:
    from research_v2.evaluation.walk_forward import BaselineWalkForwardResult

    partial = BaselineWalkForwardResult(
        fold_results={"F01": _fold_result("F01")},
        selection_records=(),
        evaluation=_evaluation(
            _gate_report({"G1": GateStatus.PASS, "G4": GateStatus.NOT_APPLICABLE})
        ),
    )
    with pytest.raises(ValueError, match="missing"):
        driver.save_baseline_oos_artifacts(partial, tmp_path)
