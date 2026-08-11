from __future__ import annotations

from datetime import date

from research_v2.inference.daily_pipeline import DailyInferencePipeline, DailyPipelineStatus
from research_v2.monitoring.watermarks import WatermarkReport, WatermarkStatus


def test_pipeline_runs_ordered_callbacks_only_after_ready_watermarks() -> None:
    steps: list[str] = []
    pipeline = DailyInferencePipeline()

    result = pipeline.run(
        date(2025, 1, 2),
        watermark_report=WatermarkReport(WatermarkStatus.READY, ()),
        build_batch=lambda: steps.append("build") or "batch",
        load_artifact=lambda: steps.append("load") or "artifact",
        predict=lambda batch, artifact: steps.append("predict") or "prediction",
        score_health=lambda prediction: steps.append("health") or True,
        publish=lambda prediction: steps.append("publish"),
    )

    assert result.status is DailyPipelineStatus.SUCCESS
    assert steps == ["build", "load", "predict", "health", "publish"]


def test_pipeline_noops_non_trading_and_blocks_before_build() -> None:
    pipeline = DailyInferencePipeline()
    noop = pipeline.run(
        date(2025, 1, 4),
        watermark_report=WatermarkReport(WatermarkStatus.NOOP_NON_TRADING_DAY, ()),
        build_batch=lambda: (_ for _ in ()).throw(AssertionError("should not build")),
        load_artifact=lambda: None,
        predict=lambda batch, artifact: None,
        score_health=lambda prediction: True,
        publish=lambda prediction: None,
    )
    blocked = pipeline.run(
        date(2025, 1, 2),
        watermark_report=WatermarkReport(WatermarkStatus.BLOCKED, ("STALE:market",)),
        build_batch=lambda: (_ for _ in ()).throw(AssertionError("should not build")),
        load_artifact=lambda: None,
        predict=lambda batch, artifact: None,
        score_health=lambda prediction: True,
        publish=lambda prediction: None,
    )

    assert noop.status is DailyPipelineStatus.NOOP_NON_TRADING_DAY
    assert blocked.status is DailyPipelineStatus.FAILED
