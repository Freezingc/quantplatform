"""Fail-closed daily inference DAG orchestration without label or performance inputs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from time import perf_counter
from uuid import uuid4

from research_v2.monitoring.watermarks import WatermarkReport, WatermarkStatus


class DailyPipelineStatus(StrEnum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    NOOP_NON_TRADING_DAY = "NOOP_NON_TRADING_DAY"


class DailyPipelineStepStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED_DEPENDENCY_FAILED = "SKIPPED_DEPENDENCY_FAILED"
    NOOP_NON_TRADING_DAY = "NOOP_NON_TRADING_DAY"


@dataclass(frozen=True)
class DailyPipelineStepRecord:
    step: str
    status: DailyPipelineStepStatus
    duration_seconds: float
    error_message: str | None = None


@dataclass(frozen=True)
class DailyPipelineResult:
    signal_date: date
    status: DailyPipelineStatus
    failed_step: str | None
    error_message: str | None
    run_id: str | None = None
    step_records: tuple[DailyPipelineStepRecord, ...] = ()


@dataclass(frozen=True)
class DailyPipelineRetryResult:
    attempts: tuple[DailyPipelineResult, ...]
    final_result: DailyPipelineResult


class DailyInferencePipeline:
    """Run one immutable daily DAG with fail-closed source-watermark checks."""

    _STEP_ORDER = (
        "watermark",
        "build_batch",
        "load_artifact",
        "predict",
        "score_health",
        "publish",
    )

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
        watermark_end_report_factory: Callable[[], WatermarkReport] | None = None,
        run_id: str | None = None,
    ) -> DailyPipelineResult:
        return self._run_once(
            signal_date,
            watermark_report=watermark_report,
            build_batch=build_batch,
            load_artifact=load_artifact,
            predict=predict,
            score_health=score_health,
            publish=publish,
            watermark_end_report_factory=watermark_end_report_factory,
            run_id=run_id or str(uuid4()),
        )

    def run_with_watermark_retries(
        self,
        signal_date: date,
        *,
        watermark_report_factory: Callable[[], WatermarkReport],
        build_batch: Callable[[], object],
        load_artifact: Callable[[], object],
        predict: Callable[[object, object], object],
        score_health: Callable[[object], bool],
        publish: Callable[[object], None],
        watermark_end_report_factory: Callable[[], WatermarkReport] | None = None,
    ) -> DailyPipelineRetryResult:
        attempts: list[DailyPipelineResult] = []
        final_result: DailyPipelineResult | None = None
        for attempt_index in range(4):
            watermark_report = watermark_report_factory()
            attempt = self._run_once(
                signal_date,
                watermark_report=watermark_report,
                build_batch=build_batch,
                load_artifact=load_artifact,
                predict=predict,
                score_health=score_health,
                publish=publish,
                watermark_end_report_factory=watermark_end_report_factory,
                run_id=str(uuid4()),
            )
            attempts.append(attempt)
            final_result = attempt
            if attempt.status is DailyPipelineStatus.SUCCESS:
                break
            if attempt.status is DailyPipelineStatus.NOOP_NON_TRADING_DAY:
                break
            if attempt.failed_step != "watermark":
                break
            if attempt_index == 3:
                break
        assert final_result is not None
        return DailyPipelineRetryResult(attempts=tuple(attempts), final_result=final_result)

    def _run_once(
        self,
        signal_date: date,
        *,
        watermark_report: WatermarkReport,
        build_batch: Callable[[], object],
        load_artifact: Callable[[], object],
        predict: Callable[[object, object], object],
        score_health: Callable[[object], bool],
        publish: Callable[[object], None],
        watermark_end_report_factory: Callable[[], WatermarkReport] | None,
        run_id: str,
    ) -> DailyPipelineResult:
        step_order = self._step_order(watermark_end_report_factory is not None)
        if watermark_report.status is WatermarkStatus.NOOP_NON_TRADING_DAY:
            noop_records = (
                self._step_record("watermark", DailyPipelineStepStatus.NOOP_NON_TRADING_DAY, 0.0),
                *self._noop_records(step_order[1:]),
            )
            return DailyPipelineResult(
                signal_date,
                DailyPipelineStatus.NOOP_NON_TRADING_DAY,
                None,
                None,
                run_id,
                noop_records,
            )
        if watermark_report.status is not WatermarkStatus.READY:
            blocked_records = (
                self._step_record(
                    "watermark",
                    DailyPipelineStepStatus.FAILED,
                    0.0,
                    "; ".join(watermark_report.blockers),
                ),
                *self._skipped_records(step_order[1:]),
            )
            return DailyPipelineResult(
                signal_date,
                DailyPipelineStatus.FAILED,
                "watermark",
                "; ".join(watermark_report.blockers),
                run_id,
                blocked_records,
            )
        if watermark_report.start_fingerprint is not None and (
            watermark_end_report_factory is None
        ):
            blocked_records = (
                self._step_record(
                    "watermark",
                    DailyPipelineStepStatus.FAILED,
                    0.0,
                    "MISSING_END_WATERMARK_FACTORY",
                ),
                *self._skipped_records(step_order[1:]),
            )
            return DailyPipelineResult(
                signal_date,
                DailyPipelineStatus.FAILED,
                "watermark",
                "MISSING_END_WATERMARK_FACTORY",
                run_id,
                blocked_records,
            )

        step_records: list[DailyPipelineStepRecord] = [
            self._step_record("watermark", DailyPipelineStepStatus.SUCCESS, 0.0)
        ]

        build_result = self._run_callback_step("build_batch", build_batch)
        step_records.append(build_result.record)
        if build_result.record.status is DailyPipelineStepStatus.FAILED:
            step_records.extend(self._skipped_records(step_order[2:]))
            return self._failed_result(
                signal_date,
                "build_batch",
                build_result.record.error_message,
                run_id,
                step_records,
            )

        load_result = self._run_callback_step("load_artifact", load_artifact)
        step_records.append(load_result.record)
        if load_result.record.status is DailyPipelineStepStatus.FAILED:
            step_records.extend(self._skipped_records(step_order[3:]))
            return self._failed_result(
                signal_date,
                "load_artifact",
                load_result.record.error_message,
                run_id,
                step_records,
            )

        predict_result = self._run_callback_step(
            "predict",
            predict,
            build_result.value,
            load_result.value,
        )
        step_records.append(predict_result.record)
        if predict_result.record.status is DailyPipelineStepStatus.FAILED:
            step_records.extend(self._skipped_records(step_order[4:]))
            return self._failed_result(
                signal_date,
                "predict",
                predict_result.record.error_message,
                run_id,
                step_records,
            )

        score_result = self._run_callback_step("score_health", score_health, predict_result.value)
        step_records.append(score_result.record)
        if score_result.record.status is DailyPipelineStepStatus.FAILED:
            step_records.extend(self._skipped_records(step_order[5:]))
            return self._failed_result(
                signal_date,
                "score_health",
                score_result.record.error_message,
                run_id,
                step_records,
            )
        if not score_result.value:
            step_records[-1] = self._step_record(
                "score_health",
                DailyPipelineStepStatus.FAILED,
                score_result.record.duration_seconds,
                "score health gate rejected prediction",
            )
            step_records.extend(self._skipped_records(step_order[5:]))
            return self._failed_result(
                signal_date,
                "score_health",
                "score health gate rejected prediction",
                run_id,
                step_records,
            )

        if watermark_end_report_factory is not None:
            end_result = self._run_callback_step(
                "watermark_end", watermark_end_report_factory
            )
            step_records.append(end_result.record)
            end_error = self._end_watermark_error(watermark_report, end_result.value)
            if end_result.record.status is DailyPipelineStepStatus.FAILED:
                return self._failed_result(
                    signal_date,
                    "watermark_end",
                    end_result.record.error_message,
                    run_id,
                    [*step_records, *self._skipped_records(step_order[6:])],
                )
            if end_error is not None:
                step_records[-1] = self._step_record(
                    "watermark_end",
                    DailyPipelineStepStatus.FAILED,
                    end_result.record.duration_seconds,
                    end_error,
                )
                return self._failed_result(
                    signal_date,
                    "watermark_end",
                    end_error,
                    run_id,
                    [*step_records, *self._skipped_records(step_order[6:])],
                )

        publish_result = self._run_callback_step("publish", publish, predict_result.value)
        step_records.append(publish_result.record)
        if publish_result.record.status is DailyPipelineStepStatus.FAILED:
            return self._failed_result(
                signal_date,
                "publish",
                publish_result.record.error_message,
                run_id,
                step_records,
            )

        return DailyPipelineResult(
            signal_date,
            DailyPipelineStatus.SUCCESS,
            None,
            None,
            run_id,
            tuple(step_records),
        )

    @classmethod
    def _step_order(cls, end_watermark_enabled: bool) -> tuple[str, ...]:
        if end_watermark_enabled:
            return (*cls._STEP_ORDER[:5], "watermark_end", cls._STEP_ORDER[5])
        return cls._STEP_ORDER

    @staticmethod
    def _end_watermark_error(
        start_report: WatermarkReport,
        end_report: object | None,
    ) -> str | None:
        if not isinstance(end_report, WatermarkReport):
            return "INVALID_END_WATERMARK_REPORT"
        if end_report.blockers:
            return "; ".join(end_report.blockers)
        if end_report.status is not WatermarkStatus.READY:
            return "; ".join(end_report.blockers) or "END_WATERMARK_NOT_READY"
        if start_report.start_fingerprint is None:
            return "MISSING_START_FINGERPRINT"
        if end_report.start_fingerprint is None:
            return "MISSING_END_FINGERPRINT"
        if start_report.start_fingerprint != end_report.start_fingerprint:
            return "SOURCE_CHANGED_DURING_RUN"
        return None

    @dataclass(frozen=True)
    class _StepOutcome:
        record: DailyPipelineStepRecord
        value: object | None

    def _run_callback_step(
        self,
        step: str,
        callback: Callable[..., object],
        *args: object,
    ) -> DailyInferencePipeline._StepOutcome:
        started_at = perf_counter()
        try:
            value = callback(*args)
        except Exception as error:  # noqa: BLE001 - fail closed on ordinary callback errors
            duration = perf_counter() - started_at
            return self._StepOutcome(
                self._step_record(
                    step,
                    DailyPipelineStepStatus.FAILED,
                    duration,
                    self._format_error(error),
                ),
                None,
            )
        duration = perf_counter() - started_at
        return self._StepOutcome(
            self._step_record(step, DailyPipelineStepStatus.SUCCESS, duration),
            value,
        )

    @staticmethod
    def _step_record(
        step: str,
        status: DailyPipelineStepStatus,
        duration_seconds: float,
        error_message: str | None = None,
    ) -> DailyPipelineStepRecord:
        return DailyPipelineStepRecord(
            step=step,
            status=status,
            duration_seconds=duration_seconds,
            error_message=error_message,
        )

    def _skipped_records(self, steps: tuple[str, ...]) -> tuple[DailyPipelineStepRecord, ...]:
        return tuple(
            self._step_record(step, DailyPipelineStepStatus.SKIPPED_DEPENDENCY_FAILED, 0.0)
            for step in steps
        )

    def _noop_records(self, steps: tuple[str, ...]) -> tuple[DailyPipelineStepRecord, ...]:
        return tuple(
            self._step_record(step, DailyPipelineStepStatus.NOOP_NON_TRADING_DAY, 0.0)
            for step in steps
        )

    def _failed_result(
        self,
        signal_date: date,
        step: str,
        error_message: str | None,
        run_id: str,
        step_records: list[DailyPipelineStepRecord],
    ) -> DailyPipelineResult:
        return DailyPipelineResult(
            signal_date,
            DailyPipelineStatus.FAILED,
            step,
            error_message,
            run_id,
            tuple(step_records),
        )

    @staticmethod
    def _format_error(error: Exception) -> str:
        return f"{type(error).__name__}: {error}"
