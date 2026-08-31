from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from research_v2.inference.daily_pipeline import DailyPipelineResult, DailyPipelineStatus
from research_v2.reporting.daily import (
    DailyRunStep,
    DailyRunStepStatus,
    DailyRunWatermark,
    DailyScoreDistribution,
    DailySeedAgreement,
    build_daily_run_report,
    write_daily_run_report,
)


def _build_report(**overrides: object):
    run_id = "00000000-0000-0000-0000-000000000001"
    params: dict[str, object] = {
        "result": DailyPipelineResult(
            date(2026, 8, 31), DailyPipelineStatus.SUCCESS, None, None, run_id
        ),
        "environment": {"region": "cn", "python": "3.11.9"},
        "steps": (
            DailyRunStep("watermark", DailyRunStepStatus.SUCCESS, 0.0, None),
            DailyRunStep("build_batch", DailyRunStepStatus.SUCCESS, 12.5, None),
            DailyRunStep("load_artifact", DailyRunStepStatus.SUCCESS, 1.25, None),
            DailyRunStep("predict", DailyRunStepStatus.SUCCESS, 8.0, None),
            DailyRunStep("score_health", DailyRunStepStatus.SUCCESS, 0.75, None),
            DailyRunStep("publish", DailyRunStepStatus.SUCCESS, 0.5, None),
        ),
        "watermarks": (
            DailyRunWatermark("market", date(2026, 8, 31), "market-20260831-v1", True, False),
            DailyRunWatermark("universe", date(2026, 8, 31), "universe-20260831-v1", True, False),
        ),
        "universe_size": 5000,
        "feature_coverage": {"alpha_1": 1.0, "alpha_2": 0.98},
        "score_distribution": DailyScoreDistribution(
            finite_coverage=1.0,
            min_score=-0.4,
            max_score=0.9,
            mean_score=0.1,
            std_score=0.2,
            unique_score_count=5000,
        ),
        "seed_agreement": DailySeedAgreement(minimum_spearman=0.997),
        "prediction_row_count": 5000,
        "artifact_hashes": {
            "manifest": "a" * 64,
            "model_bundle": "b" * 64,
        },
        "data_hashes": {
            "feature_matrix": "c" * 64,
            "watermark_snapshot": "d" * 64,
        },
        "alerts": ("coverage:warn", "drift:error"),
        "publish_version": "daily-2026-08-31T19:31:00Z",
        "prediction_published": True,
        "latency_slo_seconds": 300.0,
    }
    params.update(overrides)
    return build_daily_run_report(**params)


def test_build_daily_run_report_is_canonical_and_renders_json_and_html() -> None:
    report = _build_report()
    reordered = _build_report(
        environment={"python": "3.11.9", "region": "cn"},
        feature_coverage={"alpha_2": 0.98, "alpha_1": 1.0},
        artifact_hashes={"model_bundle": "b" * 64, "manifest": "a" * 64},
        data_hashes={"watermark_snapshot": "d" * 64, "feature_matrix": "c" * 64},
    )

    assert report.report_id == reordered.report_id
    assert report.to_json() == reordered.to_json()

    payload = json.loads(report.to_json())
    assert payload["report_id"] == report.report_id
    assert payload["signal_date"] == "2026-08-31"
    assert payload["run_id"] == "00000000-0000-0000-0000-000000000001"
    assert payload["pipeline_status"] == "SUCCESS"
    assert payload["prediction_published"] is True
    assert payload["performance_status"] == "LOCKED_NOT_COMPUTED"
    assert payload["total_duration_seconds"] == pytest.approx(23.0)
    assert [step["step_name"] for step in payload["steps"]] == [
        "watermark",
        "build_batch",
        "load_artifact",
        "predict",
        "score_health",
        "publish",
    ]
    assert payload["watermarks"][0]["dependency"] == "market"
    assert payload["score_distribution"]["unique_score_count"] == 5000
    assert "Quant Research V2 daily run" in report.to_html()


def test_build_daily_run_report_rejects_forbidden_tokens_recursively() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        _build_report(
            steps=(
                DailyRunStep("watermark", DailyRunStepStatus.SUCCESS, 0.0, None),
                DailyRunStep("build_batch", DailyRunStepStatus.SUCCESS, 1.0, None),
                DailyRunStep("load_artifact", DailyRunStepStatus.SUCCESS, 1.0, None),
                DailyRunStep("predict", DailyRunStepStatus.SUCCESS, 1.0, None),
                DailyRunStep("score_health", DailyRunStepStatus.SUCCESS, 1.0, None),
                DailyRunStep("publish", DailyRunStepStatus.FAILED, 0.5, "RankIC leak"),
            ),
            result=DailyPipelineResult(
                date(2026, 8, 31),
                DailyPipelineStatus.FAILED,
                "publish",
                "RuntimeError: blocked",
                "00000000-0000-0000-0000-000000000002",
            ),
            run_id="00000000-0000-0000-0000-000000000002",
            publish_version=None,
            prediction_published=False,
        )

    with pytest.raises(ValueError, match="forbidden"):
        _build_report(data_hashes={"daily_return_snapshot": "c" * 64})


def test_build_daily_run_report_validates_values_and_writes_exclusively(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="artifact hash"):
        _build_report(artifact_hashes={"model_bundle": "not-a-hash"})

    with pytest.raises(ValueError, match="feature_coverage"):
        _build_report(feature_coverage={"alpha_1": float("nan")})

    report = _build_report()
    json_path = tmp_path / "reports" / "daily-report.json"
    html_path = tmp_path / "reports" / "daily-report.html"

    write_daily_run_report(report, json_path, format="json")
    write_daily_run_report(report, html_path, format="html")

    assert json_path.read_text(encoding="utf-8") == report.to_json()
    assert "LOCKED_NOT_COMPUTED" in html_path.read_text(encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_daily_run_report(report, json_path, format="json")

    with pytest.raises(ValueError, match="json or html"):
        write_daily_run_report(report, tmp_path / "report.txt", format="txt")


def test_build_daily_run_report_uses_pipeline_run_id_and_allows_empty_failed_outputs() -> None:
    run_id = "00000000-0000-0000-0000-000000000003"
    failed_steps = (
        DailyRunStep("watermark", DailyRunStepStatus.SUCCESS, 0.0, None),
        DailyRunStep("build_batch", DailyRunStepStatus.SUCCESS, 1.0, None),
        DailyRunStep("load_artifact", DailyRunStepStatus.FAILED, 1.0, "GPU unavailable"),
        DailyRunStep("predict", DailyRunStepStatus.SKIPPED, 0.0, None),
        DailyRunStep("score_health", DailyRunStepStatus.SKIPPED, 0.0, None),
        DailyRunStep("publish", DailyRunStepStatus.SKIPPED, 0.0, None),
    )
    report = _build_report(
        result=DailyPipelineResult(
            date(2026, 8, 31),
            DailyPipelineStatus.FAILED,
            "load_artifact",
            "GPU unavailable",
            run_id,
        ),
        steps=failed_steps,
        publish_version=None,
        prediction_published=False,
        universe_size=0,
        prediction_row_count=0,
        feature_coverage={"alpha_1": 0.0},
        score_distribution=DailyScoreDistribution(0.0, 0.0, 0.0, 0.0, 0.0, 0),
    )

    assert report.run_id == run_id


def test_build_daily_run_report_rejects_mismatched_run_id() -> None:
    with pytest.raises(ValueError, match="run_id"):
        _build_report(run_id="00000000-0000-0000-0000-000000000099")


def test_build_daily_run_report_supports_non_trading_day_noop() -> None:
    report = _build_report(
        result=DailyPipelineResult(
            date(2026, 8, 30),
            DailyPipelineStatus.NOOP_NON_TRADING_DAY,
            None,
            None,
            "00000000-0000-0000-0000-000000000004",
        ),
        steps=(
            DailyRunStep("watermark", DailyRunStepStatus.NOOP, 0.0, None),
            DailyRunStep("build_batch", DailyRunStepStatus.NOOP, 0.0, None),
            DailyRunStep("load_artifact", DailyRunStepStatus.NOOP, 0.0, None),
            DailyRunStep("predict", DailyRunStepStatus.NOOP, 0.0, None),
            DailyRunStep("score_health", DailyRunStepStatus.NOOP, 0.0, None),
            DailyRunStep("publish", DailyRunStepStatus.NOOP, 0.0, None),
        ),
        run_id="00000000-0000-0000-0000-000000000004",
        universe_size=0,
        feature_coverage={"alpha_1": 0.0},
        score_distribution=DailyScoreDistribution(0.0, 0.0, 0.0, 0.0, 0.0, 0),
        prediction_row_count=0,
        alerts=(),
        publish_version=None,
        prediction_published=False,
    )

    assert report.pipeline_status is DailyPipelineStatus.NOOP_NON_TRADING_DAY
    assert report.prediction_row_count == 0
