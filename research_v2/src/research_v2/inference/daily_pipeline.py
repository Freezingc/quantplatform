"""Fail-closed daily inference DAG orchestration without label or performance inputs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

from research_v2.monitoring.watermarks import WatermarkReport, WatermarkStatus


class DailyPipelineStatus(StrEnum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    NOOP_NON_TRADING_DAY = "NOOP_NON_TRADING_DAY"


@dataclass(frozen=True)
class DailyPipelineResult:
    signal_date: date
    status: DailyPipelineStatus
    failed_step: str | None
    error_message: str | None


class DailyInferencePipeline:
    """Run one immutable daily DAG; no callback runs if source watermarks are not ready."""

    def run(
        self,
        signal_date: date,
        *,
        watermark_report: WatermarkReport,
        build_batch: Callable[[], object],
        load_artifact: Callable[[], object],
        predict: Callable[[object, object], object],
        score_health: Callable[[object], bool],
        publish: Callable[[object], None],
    ) -> DailyPipelineResult:
        if watermark_report.status is WatermarkStatus.NOOP_NON_TRADING_DAY:
            return DailyPipelineResult(
                signal_date,
                DailyPipelineStatus.NOOP_NON_TRADING_DAY,
                None,
                None,
            )
        if watermark_report.status is not WatermarkStatus.READY:
            return DailyPipelineResult(
                signal_date,
                DailyPipelineStatus.FAILED,
                "watermark",
                "; ".join(watermark_report.blockers),
            )
        try:
            batch = build_batch()
        except Exception as error:
            return self._failed(signal_date, "build_batch", error)
        try:
            artifact = load_artifact()
        except Exception as error:
            return self._failed(signal_date, "load_artifact", error)
        try:
            prediction = predict(batch, artifact)
        except Exception as error:
            return self._failed(signal_date, "predict", error)
        try:
            healthy = score_health(prediction)
        except Exception as error:
            return self._failed(signal_date, "score_health", error)
        if not healthy:
            return DailyPipelineResult(
                signal_date,
                DailyPipelineStatus.FAILED,
                "score_health",
                "score health gate rejected prediction",
            )
        try:
            publish(prediction)
        except Exception as error:
            return self._failed(signal_date, "publish", error)
        return DailyPipelineResult(signal_date, DailyPipelineStatus.SUCCESS, None, None)

    @staticmethod
    def _failed(signal_date: date, step: str, error: Exception) -> DailyPipelineResult:
        return DailyPipelineResult(
            signal_date,
            DailyPipelineStatus.FAILED,
            step,
            f"{type(error).__name__}: {error}",
        )
