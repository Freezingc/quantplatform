from __future__ import annotations

import uuid
from datetime import date

from research_v2.inference.daily_pipeline import DailyInferencePipeline, DailyPipelineStatus
from research_v2.monitoring.watermarks import WatermarkReport, WatermarkStatus


def _callbacks(calls: list[str]) -> tuple[object, object, object, object, object]:
    def build_batch() -> str:
        calls.append("build_batch")
        return "batch"

    def load_artifact() -> str:
        calls.append("load_artifact")
        return "artifact"

    def predict(batch: object, artifact: object) -> str:
        calls.append("predict")
        return "prediction"

    def score_health(prediction: object) -> bool:
        calls.append("score_health")
        return True

    def publish(prediction: object) -> None:
        calls.append("publish")

    return build_batch, load_artifact, predict, score_health, publish


def test_pipeline_records_terminal_steps_and_accepts_explicit_run_id() -> None:
    calls: list[str] = []
    pipeline = DailyInferencePipeline()
    build_batch, load_artifact, predict, score_health, publish = _callbacks(calls)
    result = pipeline.run(
        date(2025, 1, 2),
        watermark_report=WatermarkReport(WatermarkStatus.READY, ()),
        build_batch=build_batch,
        load_artifact=load_artifact,
        predict=predict,
        score_health=score_health,
        publish=publish,
        run_id="run-001",
    )

    assert result.status is DailyPipelineStatus.SUCCESS
    assert result.failed_step is None
    assert result.error_message is None
    assert result.run_id == "run-001"
    assert [record.step for record in result.step_records] == [
        "watermark",
        "build_batch",
        "load_artifact",
        "predict",
        "score_health",
        "publish",
    ]
    assert [record.status for record in result.step_records] == [
        "SUCCESS",
        "SUCCESS",
        "SUCCESS",
        "SUCCESS",
        "SUCCESS",
        "SUCCESS",
    ]
    assert all(record.duration_seconds >= 0 for record in result.step_records)
    assert calls == ["build_batch", "load_artifact", "predict", "score_health", "publish"]


def test_pipeline_marks_downstream_steps_skipped_after_failure() -> None:
    pipeline = DailyInferencePipeline()
    calls: list[str] = []

    def build_batch() -> str:
        calls.append("build_batch")
        raise ValueError("build batch failed")

    result = pipeline.run(
        date(2025, 1, 2),
        watermark_report=WatermarkReport(WatermarkStatus.READY, ()),
        build_batch=build_batch,
        load_artifact=lambda: calls.append("load_artifact") or "artifact",
        predict=lambda batch, artifact: calls.append("predict") or "prediction",
        score_health=lambda prediction: calls.append("score_health") or True,
        publish=lambda prediction: calls.append("publish"),
        run_id="run-002",
    )

    assert result.status is DailyPipelineStatus.FAILED
    assert result.failed_step == "build_batch"
    assert result.error_message == "ValueError: build batch failed"
    assert [record.step for record in result.step_records] == [
        "watermark",
        "build_batch",
        "load_artifact",
        "predict",
        "score_health",
        "publish",
    ]
    assert [record.status for record in result.step_records] == [
        "SUCCESS",
        "FAILED",
        "SKIPPED_DEPENDENCY_FAILED",
        "SKIPPED_DEPENDENCY_FAILED",
        "SKIPPED_DEPENDENCY_FAILED",
        "SKIPPED_DEPENDENCY_FAILED",
    ]
    assert calls == ["build_batch"]


def test_pipeline_generates_fresh_run_id_for_non_trading_day_without_callbacks() -> None:
    pipeline = DailyInferencePipeline()
    calls: list[str] = []
    result = pipeline.run(
        date(2025, 1, 4),
        watermark_report=WatermarkReport(WatermarkStatus.NOOP_NON_TRADING_DAY, ()),
        build_batch=lambda: calls.append("build_batch"),
        load_artifact=lambda: calls.append("load_artifact"),
        predict=lambda batch, artifact: calls.append("predict"),
        score_health=lambda prediction: calls.append("score_health") or True,
        publish=lambda prediction: calls.append("publish"),
    )

    assert result.status is DailyPipelineStatus.NOOP_NON_TRADING_DAY
    uuid.UUID(result.run_id)
    assert [record.status for record in result.step_records] == [
        "NOOP_NON_TRADING_DAY",
        "NOOP_NON_TRADING_DAY",
        "NOOP_NON_TRADING_DAY",
        "NOOP_NON_TRADING_DAY",
        "NOOP_NON_TRADING_DAY",
        "NOOP_NON_TRADING_DAY",
    ]
    assert calls == []


def test_run_with_watermark_retries_retries_blocked_watermarks_with_fresh_run_ids() -> None:
    pipeline = DailyInferencePipeline()
    reports = iter(
        [
            WatermarkReport(WatermarkStatus.BLOCKED, ("STALE:market",)),
            WatermarkReport(WatermarkStatus.BLOCKED, ("STALE:market",)),
            WatermarkReport(WatermarkStatus.BLOCKED, ("STALE:market",)),
            WatermarkReport(WatermarkStatus.READY, ()),
        ]
    )
    calls: list[str] = []
    build_batch, load_artifact, predict, score_health, publish = _callbacks(calls)
    seen_reports: list[WatermarkReport] = []

    result = pipeline.run_with_watermark_retries(
        date(2025, 1, 2),
        watermark_report_factory=lambda: seen_reports.append(next(reports)) or seen_reports[-1],
        build_batch=build_batch,
        load_artifact=load_artifact,
        predict=predict,
        score_health=score_health,
        publish=publish,
    )

    assert len(result.attempts) == 4
    assert result.final_result.status is DailyPipelineStatus.SUCCESS
    assert [attempt.status for attempt in result.attempts] == [
        DailyPipelineStatus.FAILED,
        DailyPipelineStatus.FAILED,
        DailyPipelineStatus.FAILED,
        DailyPipelineStatus.SUCCESS,
    ]
    assert len({attempt.run_id for attempt in result.attempts}) == 4
    for attempt in result.attempts:
        uuid.UUID(attempt.run_id)
    assert len(seen_reports) == 4
    assert calls == ["build_batch", "load_artifact", "predict", "score_health", "publish"]
    assert [record.status for record in result.attempts[0].step_records] == [
        "FAILED",
        "SKIPPED_DEPENDENCY_FAILED",
        "SKIPPED_DEPENDENCY_FAILED",
        "SKIPPED_DEPENDENCY_FAILED",
        "SKIPPED_DEPENDENCY_FAILED",
        "SKIPPED_DEPENDENCY_FAILED",
    ]


def test_pipeline_marks_a_score_gate_rejection_as_a_failed_step() -> None:
    calls: list[str] = []
    build_batch, load_artifact, predict, _, publish = _callbacks(calls)

    result = DailyInferencePipeline().run(
        date(2025, 1, 2),
        watermark_report=WatermarkReport(WatermarkStatus.READY, ()),
        build_batch=build_batch,
        load_artifact=load_artifact,
        predict=predict,
        score_health=lambda prediction: calls.append("score_health") or False,
        publish=publish,
        run_id="run-003",
    )

    assert result.status is DailyPipelineStatus.FAILED
    assert result.failed_step == "score_health"
    assert result.error_message == "score health gate rejected prediction"
    assert [record.status for record in result.step_records] == [
        "SUCCESS",
        "SUCCESS",
        "SUCCESS",
        "SUCCESS",
        "FAILED",
        "SKIPPED_DEPENDENCY_FAILED",
    ]
    assert calls == ["build_batch", "load_artifact", "predict", "score_health"]


def test_pipeline_checks_end_watermark_before_publish() -> None:
    calls: list[str] = []
    build_batch, load_artifact, predict, score_health, publish = _callbacks(calls)

    result = DailyInferencePipeline().run(
        date(2025, 1, 2),
        watermark_report=WatermarkReport(WatermarkStatus.READY, (), "snapshot-a"),
        build_batch=build_batch,
        load_artifact=load_artifact,
        predict=predict,
        score_health=score_health,
        publish=publish,
        watermark_end_report_factory=lambda: WatermarkReport(
            WatermarkStatus.READY, (), "snapshot-a"
        ),
        run_id="run-end-watermark",
    )

    assert result.status is DailyPipelineStatus.SUCCESS
    assert [record.step for record in result.step_records] == [
        "watermark",
        "build_batch",
        "load_artifact",
        "predict",
        "score_health",
        "watermark_end",
        "publish",
    ]
    assert calls == [
        "build_batch",
        "load_artifact",
        "predict",
        "score_health",
        "publish",
    ]


def test_pipeline_blocks_publish_when_end_watermark_changes() -> None:
    calls: list[str] = []
    build_batch, load_artifact, predict, score_health, publish = _callbacks(calls)

    result = DailyInferencePipeline().run(
        date(2025, 1, 2),
        watermark_report=WatermarkReport(WatermarkStatus.READY, (), "snapshot-a"),
        build_batch=build_batch,
        load_artifact=load_artifact,
        predict=predict,
        score_health=score_health,
        publish=publish,
        watermark_end_report_factory=lambda: WatermarkReport(
            WatermarkStatus.READY, (), "snapshot-b"
        ),
        run_id="run-changed-watermark",
    )

    assert result.status is DailyPipelineStatus.FAILED
    assert result.failed_step == "watermark_end"
    assert result.error_message == "SOURCE_CHANGED_DURING_RUN"
    assert [record.status for record in result.step_records] == [
        "SUCCESS",
        "SUCCESS",
        "SUCCESS",
        "SUCCESS",
        "SUCCESS",
        "FAILED",
        "SKIPPED_DEPENDENCY_FAILED",
    ]
    assert calls == ["build_batch", "load_artifact", "predict", "score_health"]


def test_pipeline_requires_fingerprints_when_end_watermark_is_enabled() -> None:
    build_batch, load_artifact, predict, score_health, publish = _callbacks([])

    result = DailyInferencePipeline().run(
        date(2025, 1, 2),
        watermark_report=WatermarkReport(WatermarkStatus.READY, ()),
        build_batch=build_batch,
        load_artifact=load_artifact,
        predict=predict,
        score_health=score_health,
        publish=publish,
        watermark_end_report_factory=lambda: WatermarkReport(WatermarkStatus.READY, ()),
        run_id="run-missing-watermark-fingerprint",
    )

    assert result.status is DailyPipelineStatus.FAILED
    assert result.failed_step == "watermark_end"
    assert result.error_message == "MISSING_START_FINGERPRINT"
