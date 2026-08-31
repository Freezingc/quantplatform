from __future__ import annotations

import os
from datetime import date

import pandas as pd
import pytest

from research_v2.inference.daily_pipeline import (
    DailyInferencePipeline,
    DailyPipelineStatus,
)
from research_v2.monitoring.score_health import evaluate_score_health
from research_v2.monitoring.watermarks import WatermarkReport, WatermarkStatus

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("QRV2_INTEGRATION") != "1",
        reason="set QRV2_INTEGRATION=1 to run inference integration tests",
    ),
]

_BASELINE_DATE = date(2026, 8, 28)
_FAILED_DATE = date(2026, 8, 31)
_CODES = tuple(f"00000{index}.SZ" for index in range(1, 13))


def _prediction_frame(signal_date: date) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_date": [signal_date] * len(_CODES),
            "ts_code": _CODES,
            "ensemble_score": [index / len(_CODES) for index in range(len(_CODES))],
        }
    )


def _callbacks(
    signal_date: date,
    calls: list[str],
    visible_predictions: dict[date, pd.DataFrame],
    *,
    failing_step: str | None = None,
) -> tuple[object, object, object, object, object]:
    def build_batch() -> str:
        calls.append("build_batch")
        if failing_step == "build_batch":
            raise ValueError("factor DQ blocker")
        return "batch"

    def load_artifact() -> str:
        calls.append("load_artifact")
        if failing_step == "load_artifact":
            raise RuntimeError("CUDA unavailable")
        return "artifact"

    def predict(batch: object, artifact: object) -> pd.DataFrame:
        calls.append("predict")
        if failing_step == "predict":
            raise RuntimeError("predictor failed")
        return _prediction_frame(signal_date)

    def score_health(prediction: object) -> bool:
        calls.append("score_health")
        if failing_step == "score_health":
            raise RuntimeError("score health failed")
        assert isinstance(prediction, pd.DataFrame)
        report = evaluate_score_health(prediction, expected_key_count=len(_CODES))
        return not report.blocking_reasons

    def publish(prediction: object) -> None:
        calls.append("publish")
        if failing_step == "publish":
            raise RuntimeError("publish interrupted")
        assert isinstance(prediction, pd.DataFrame)
        visible_predictions[signal_date] = prediction.copy(deep=True)

    return build_batch, load_artifact, predict, score_health, publish


def _run_once(
    signal_date: date,
    watermark_report: WatermarkReport,
    visible_predictions: dict[date, pd.DataFrame],
    *,
    failing_step: str | None = None,
) -> tuple[object, list[str]]:
    calls: list[str] = []
    build_batch, load_artifact, predict, score_health, publish = _callbacks(
        signal_date,
        calls,
        visible_predictions,
        failing_step=failing_step,
    )
    result = DailyInferencePipeline().run(
        signal_date,
        watermark_report=watermark_report,
        build_batch=build_batch,
        load_artifact=load_artifact,
        predict=predict,
        score_health=score_health,
        publish=publish,
    )
    return result, calls


def test_daily_shadow_pipeline_publishes_only_a_complete_ready_day() -> None:
    visible_predictions: dict[date, pd.DataFrame] = {}

    result, calls = _run_once(
        _BASELINE_DATE,
        WatermarkReport(WatermarkStatus.READY, ()),
        visible_predictions,
    )

    assert result.status is DailyPipelineStatus.SUCCESS
    assert result.failed_step is None
    assert calls == ["build_batch", "load_artifact", "predict", "score_health", "publish"]
    assert list(visible_predictions) == [_BASELINE_DATE]
    assert len(visible_predictions[_BASELINE_DATE]) == len(_CODES)


def test_daily_shadow_pipeline_noops_non_trading_days_without_callbacks() -> None:
    visible_predictions: dict[date, pd.DataFrame] = {}

    result, calls = _run_once(
        _FAILED_DATE,
        WatermarkReport(WatermarkStatus.NOOP_NON_TRADING_DAY, ()),
        visible_predictions,
    )

    assert result.status is DailyPipelineStatus.NOOP_NON_TRADING_DAY
    assert result.failed_step is None
    assert calls == []
    assert _FAILED_DATE not in visible_predictions


def test_daily_shadow_pipeline_retries_a_late_source_then_publishes_once() -> None:
    visible_predictions: dict[date, pd.DataFrame] = {}
    reports = iter(
        [
            WatermarkReport(WatermarkStatus.BLOCKED, ("STALE:market",)),
            WatermarkReport(WatermarkStatus.READY, ()),
        ]
    )
    calls: list[str] = []
    build_batch, load_artifact, predict, score_health, publish = _callbacks(
        _FAILED_DATE,
        calls,
        visible_predictions,
    )

    retry_result = DailyInferencePipeline().run_with_watermark_retries(
        _FAILED_DATE,
        watermark_report_factory=lambda: next(reports),
        build_batch=build_batch,
        load_artifact=load_artifact,
        predict=predict,
        score_health=score_health,
        publish=publish,
    )

    assert retry_result.final_result.status is DailyPipelineStatus.SUCCESS
    assert len(retry_result.attempts) == 2
    assert retry_result.attempts[0].failed_step == "watermark"
    assert retry_result.attempts[1].failed_step is None
    assert retry_result.attempts[0].run_id != retry_result.attempts[1].run_id
    assert calls == ["build_batch", "load_artifact", "predict", "score_health", "publish"]
    assert list(visible_predictions) == [_FAILED_DATE]


def test_daily_shadow_pipeline_permanent_source_delay_does_not_replace_a_previous_day() -> None:
    visible_predictions = {_BASELINE_DATE: _prediction_frame(_BASELINE_DATE)}
    reports = iter(
        [WatermarkReport(WatermarkStatus.BLOCKED, ("STALE:factor_set",)) for _ in range(4)]
    )
    calls: list[str] = []
    build_batch, load_artifact, predict, score_health, publish = _callbacks(
        _FAILED_DATE,
        calls,
        visible_predictions,
    )

    retry_result = DailyInferencePipeline().run_with_watermark_retries(
        _FAILED_DATE,
        watermark_report_factory=lambda: next(reports),
        build_batch=build_batch,
        load_artifact=load_artifact,
        predict=predict,
        score_health=score_health,
        publish=publish,
    )

    assert retry_result.final_result.status is DailyPipelineStatus.FAILED
    assert len(retry_result.attempts) == 4
    assert all(attempt.failed_step == "watermark" for attempt in retry_result.attempts)
    assert calls == []
    assert _FAILED_DATE not in visible_predictions
    assert list(visible_predictions) == [_BASELINE_DATE]


@pytest.mark.parametrize(
    ("failing_step", "expected_step"),
    [
        ("build_batch", "build_batch"),
        ("load_artifact", "load_artifact"),
        ("predict", "predict"),
        ("score_health", "score_health"),
        ("publish", "publish"),
    ],
)
def test_daily_shadow_pipeline_failures_are_fail_closed(
    failing_step: str,
    expected_step: str,
) -> None:
    visible_predictions: dict[date, pd.DataFrame] = {}

    result, calls = _run_once(
        _FAILED_DATE,
        WatermarkReport(WatermarkStatus.READY, ()),
        visible_predictions,
        failing_step=failing_step,
    )

    assert result.status is DailyPipelineStatus.FAILED
    assert result.failed_step == expected_step
    assert _FAILED_DATE not in visible_predictions
    assert result.step_records[-1].step == "publish"
    assert result.step_records[-1].status != "SUCCESS"
    if failing_step == "build_batch":
        assert calls == ["build_batch"]
    elif failing_step == "load_artifact":
        assert calls == ["build_batch", "load_artifact"]
    elif failing_step == "predict":
        assert calls == ["build_batch", "load_artifact", "predict"]
    elif failing_step == "score_health":
        assert calls == ["build_batch", "load_artifact", "predict", "score_health"]
    else:
        assert calls == ["build_batch", "load_artifact", "predict", "score_health", "publish"]


def test_factor_dq_blocker_stops_before_batch_construction() -> None:
    visible_predictions: dict[date, pd.DataFrame] = {}

    result, calls = _run_once(
        _FAILED_DATE,
        WatermarkReport(WatermarkStatus.BLOCKED, ("BLOCKER:factor_dq",)),
        visible_predictions,
    )

    assert result.status is DailyPipelineStatus.FAILED
    assert result.failed_step == "watermark"
    assert "BLOCKER:factor_dq" in (result.error_message or "")
    assert calls == []
    assert _FAILED_DATE not in visible_predictions
