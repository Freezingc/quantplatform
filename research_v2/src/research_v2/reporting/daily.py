"""Immutable, label-free daily HTML/JSON reports for inference operations."""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from research_v2.inference.daily_pipeline import DailyPipelineResult, DailyPipelineStatus

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ALERT_KEY = re.compile(r"^[a-z][a-z0-9]{0,31}:[a-z][a-z0-9]{0,31}$")
_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_DEPENDENCY_NAME = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_FORBIDDEN_TOKENS = ("label", "pnl", "rankic", "return")
_STEP_ORDER = (
    "watermark",
    "build_batch",
    "load_artifact",
    "predict",
    "score_health",
    "publish",
)


class DailyRunStepStatus(StrEnum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    NOOP = "NOOP"


@dataclass(frozen=True)
class DailyRunStep:
    step_name: str
    status: DailyRunStepStatus
    duration_seconds: float
    error_message: str | None


@dataclass(frozen=True)
class DailyRunWatermark:
    dependency: str
    max_signal_date: date
    data_version: str
    published: bool
    has_blocker: bool


@dataclass(frozen=True)
class DailyScoreDistribution:
    finite_coverage: float
    min_score: float
    max_score: float
    mean_score: float
    std_score: float
    unique_score_count: int


@dataclass(frozen=True)
class DailySeedAgreement:
    minimum_spearman: float


@dataclass(frozen=True)
class DailyRunReport:
    report_id: str
    signal_date: date
    run_id: str
    environment: Mapping[str, str]
    pipeline_status: DailyPipelineStatus
    failed_step: str | None
    error_message: str | None
    steps: tuple[DailyRunStep, ...]
    watermarks: tuple[DailyRunWatermark, ...]
    universe_size: int
    feature_coverage: Mapping[str, float]
    score_distribution: DailyScoreDistribution
    seed_agreement: DailySeedAgreement
    prediction_row_count: int
    artifact_hashes: Mapping[str, str]
    data_hashes: Mapping[str, str]
    alerts: tuple[str, ...]
    publish_version: str | None
    prediction_published: bool
    latency_slo_seconds: float
    performance_status: Literal["LOCKED_NOT_COMPUTED"] = "LOCKED_NOT_COMPUTED"

    @property
    def latency_seconds(self) -> float:
        return sum(step.duration_seconds for step in self.steps)

    @property
    def total_duration_seconds(self) -> float:
        return self.latency_seconds

    def to_json(self) -> str:
        self.verify()
        return _canonical_json({"report_id": self.report_id, **self._payload()}) + "\n"

    def to_html(self) -> str:
        self.verify()
        escaped = html.escape(self.to_json())
        return (
            "<!doctype html>\n"
            "<html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<title>Quant Research V2 daily run</title></head>"
            "<body><main><h1>Quant Research V2 daily run</h1>"
            f"<pre>{escaped}</pre></main></body></html>\n"
        )

    def verify(self) -> None:
        _validate_report(self)
        if (
            hashlib.sha256(_canonical_json(self._payload()).encode("utf-8")).hexdigest()
            != self.report_id
        ):
            raise ValueError("daily report content does not match report_id")

    def _payload(self) -> dict[str, object]:
        return {
            "alerts": list(self.alerts),
            "artifact_hashes": dict(self.artifact_hashes),
            "data_hashes": dict(self.data_hashes),
            "environment": dict(self.environment),
            "error_message": self.error_message,
            "failed_step": self.failed_step,
            "feature_coverage": dict(self.feature_coverage),
            "latency_seconds": self.latency_seconds,
            "latency_slo_seconds": self.latency_slo_seconds,
            "performance_status": self.performance_status,
            "pipeline_status": self.pipeline_status.value,
            "prediction_published": self.prediction_published,
            "prediction_row_count": self.prediction_row_count,
            "publish_version": self.publish_version,
            "run_id": self.run_id,
            "schema_version": 1,
            "score_distribution": {
                "finite_coverage": self.score_distribution.finite_coverage,
                "max_score": self.score_distribution.max_score,
                "mean_score": self.score_distribution.mean_score,
                "min_score": self.score_distribution.min_score,
                "std_score": self.score_distribution.std_score,
                "unique_score_count": self.score_distribution.unique_score_count,
            },
            "seed_agreement": {
                "minimum_spearman": self.seed_agreement.minimum_spearman,
            },
            "signal_date": self.signal_date.isoformat(),
            "steps": [
                {
                    "duration_seconds": step.duration_seconds,
                    "error_message": step.error_message,
                    "status": step.status.value,
                    "step_name": step.step_name,
                }
                for step in self.steps
            ],
            "total_duration_seconds": self.total_duration_seconds,
            "universe_size": self.universe_size,
            "watermarks": [
                {
                    "data_version": watermark.data_version,
                    "dependency": watermark.dependency,
                    "has_blocker": watermark.has_blocker,
                    "max_signal_date": watermark.max_signal_date.isoformat(),
                    "published": watermark.published,
                }
                for watermark in self.watermarks
            ],
        }


def build_daily_run_report(
    *,
    result: DailyPipelineResult,
    environment: Mapping[str, str],
    steps: Sequence[DailyRunStep],
    watermarks: Sequence[DailyRunWatermark],
    universe_size: int,
    feature_coverage: Mapping[str, float],
    score_distribution: DailyScoreDistribution,
    seed_agreement: DailySeedAgreement,
    prediction_row_count: int,
    artifact_hashes: Mapping[str, str],
    data_hashes: Mapping[str, str],
    alerts: Sequence[str],
    publish_version: str | None,
    prediction_published: bool,
    latency_slo_seconds: float,
    run_id: str | None = None,
) -> DailyRunReport:
    resolved_run_id = result.run_id if run_id is None else run_id
    if resolved_run_id is None:
        raise ValueError("daily report run_id is required")
    if result.run_id is not None and result.run_id != resolved_run_id:
        raise ValueError("daily report run_id does not match pipeline result")
    report = DailyRunReport(
        report_id="0" * 64,
        signal_date=result.signal_date,
        run_id=resolved_run_id,
        environment=MappingProxyType(dict(environment)),
        pipeline_status=result.status,
        failed_step=result.failed_step,
        error_message=result.error_message,
        steps=tuple(steps),
        watermarks=tuple(watermarks),
        universe_size=universe_size,
        feature_coverage=MappingProxyType(dict(feature_coverage)),
        score_distribution=score_distribution,
        seed_agreement=seed_agreement,
        prediction_row_count=prediction_row_count,
        artifact_hashes=MappingProxyType(dict(artifact_hashes)),
        data_hashes=MappingProxyType(dict(data_hashes)),
        alerts=tuple(alerts),
        publish_version=publish_version,
        prediction_published=prediction_published,
        latency_slo_seconds=latency_slo_seconds,
    )
    _validate_report(report)
    report_id = hashlib.sha256(_canonical_json(report._payload()).encode("utf-8")).hexdigest()
    return DailyRunReport(
        report_id=report_id,
        signal_date=report.signal_date,
        run_id=report.run_id,
        environment=report.environment,
        pipeline_status=report.pipeline_status,
        failed_step=report.failed_step,
        error_message=report.error_message,
        steps=report.steps,
        watermarks=report.watermarks,
        universe_size=report.universe_size,
        feature_coverage=report.feature_coverage,
        score_distribution=report.score_distribution,
        seed_agreement=report.seed_agreement,
        prediction_row_count=report.prediction_row_count,
        artifact_hashes=report.artifact_hashes,
        data_hashes=report.data_hashes,
        alerts=report.alerts,
        publish_version=report.publish_version,
        prediction_published=report.prediction_published,
        latency_slo_seconds=report.latency_slo_seconds,
    )


def write_daily_run_report(report: DailyRunReport, output: Path, *, format: str) -> None:
    if format not in {"json", "html"}:
        raise ValueError("report format must be json or html")
    report.verify()
    content = report.to_json() if format == "json" else report.to_html()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic_exclusive(output, content)


def _validate_report(report: DailyRunReport) -> None:
    if not _SHA256.fullmatch(report.report_id):
        raise ValueError("daily report_id is invalid")
    if not _is_uuid(report.run_id):
        raise ValueError("daily report run_id is invalid")
    if report.performance_status != "LOCKED_NOT_COMPUTED":
        raise ValueError("daily performance_status is invalid")
    _validate_mapping_strings(report.environment, "environment")
    _validate_steps(report.steps)
    _validate_watermarks(report.watermarks)
    _validate_nonnegative_int(report.universe_size, "universe_size")
    _validate_coverage_mapping(report.feature_coverage, "feature_coverage")
    _validate_score_distribution(report.score_distribution)
    _validate_seed_agreement(report.seed_agreement)
    _validate_nonnegative_int(report.prediction_row_count, "prediction_row_count")
    _validate_hash_mapping(report.artifact_hashes, "artifact hash")
    _validate_hash_mapping(report.data_hashes, "data hash")
    _validate_alerts(report.alerts)
    _validate_status_consistency(report)
    _finite_positive(report.latency_slo_seconds, "latency_slo_seconds")
    if _contains_forbidden_token(report._payload()):
        raise ValueError("daily report contains forbidden token")


def _validate_status_consistency(report: DailyRunReport) -> None:
    expected_statuses: tuple[DailyRunStepStatus, ...]
    failed_steps = [
        step.step_name for step in report.steps if step.status is DailyRunStepStatus.FAILED
    ]
    if len(failed_steps) > 1:
        raise ValueError("daily report has multiple failed steps")
    if report.pipeline_status is DailyPipelineStatus.SUCCESS:
        if report.failed_step is not None or report.error_message is not None:
            raise ValueError("daily report success result is inconsistent")
        if failed_steps:
            raise ValueError("daily report success steps are inconsistent")
        if not report.prediction_published or not report.publish_version:
            raise ValueError("daily report publish state is inconsistent")
        expected_statuses = (DailyRunStepStatus.SUCCESS,) * len(_STEP_ORDER)
    elif report.pipeline_status is DailyPipelineStatus.FAILED:
        if not report.failed_step or not report.error_message:
            raise ValueError("daily report failure result is inconsistent")
        if failed_steps != [report.failed_step]:
            raise ValueError("daily report failed step is inconsistent")
        if report.prediction_published or report.publish_version is not None:
            raise ValueError("daily report failure publish state is inconsistent")
        failed_index = _STEP_ORDER.index(report.failed_step)
        expected_statuses = tuple(
            DailyRunStepStatus.SUCCESS if index < failed_index else
            DailyRunStepStatus.FAILED if index == failed_index else
            DailyRunStepStatus.SKIPPED
            for index in range(len(_STEP_ORDER))
        )
    elif report.pipeline_status is DailyPipelineStatus.NOOP_NON_TRADING_DAY:
        if report.failed_step is not None or report.error_message is not None:
            raise ValueError("daily report noop result is inconsistent")
        if report.prediction_published or report.publish_version is not None:
            raise ValueError("daily report noop publish state is inconsistent")
        expected_statuses = (DailyRunStepStatus.NOOP,) * len(_STEP_ORDER)
    else:
        raise ValueError("daily report pipeline_status is invalid")
    if tuple(step.status for step in report.steps) != expected_statuses:
        raise ValueError("daily report step statuses are inconsistent")


def _validate_steps(steps: Sequence[DailyRunStep]) -> None:
    if tuple(step.step_name for step in steps) != _STEP_ORDER:
        raise ValueError("daily report steps must follow the pipeline order")
    seen: set[str] = set()
    for step in steps:
        if not _NAME.fullmatch(step.step_name):
            raise ValueError("daily report step_name is invalid")
        if step.step_name in seen:
            raise ValueError("daily report step names must be unique")
        seen.add(step.step_name)
        _finite_nonnegative(step.duration_seconds, "step duration")
        if step.status is DailyRunStepStatus.FAILED:
            if not isinstance(step.error_message, str) or not step.error_message:
                raise ValueError("daily report step error is invalid")
        elif step.error_message is not None:
            raise ValueError("daily report step error is invalid")


def _validate_watermarks(watermarks: Sequence[DailyRunWatermark]) -> None:
    if not watermarks:
        raise ValueError("daily report watermarks are invalid")
    seen: set[str] = set()
    for watermark in watermarks:
        if not _DEPENDENCY_NAME.fullmatch(watermark.dependency) or watermark.dependency in seen:
            raise ValueError("daily report watermark dependency is invalid")
        seen.add(watermark.dependency)
        if not isinstance(watermark.max_signal_date, date):
            raise ValueError("daily report watermark max_signal_date is invalid")
        if not isinstance(watermark.data_version, str) or not watermark.data_version:
            raise ValueError("daily report watermark data_version is invalid")
        if not isinstance(watermark.published, bool) or not isinstance(watermark.has_blocker, bool):
            raise ValueError("daily report watermark flags are invalid")


def _validate_mapping_strings(values: Mapping[str, str], key: str) -> None:
    if not values:
        raise ValueError(f"daily report {key} is invalid")
    for name, value in values.items():
        if not _NAME.fullmatch(name) or not isinstance(value, str) or not value:
            raise ValueError(f"daily report {key} is invalid")


def _validate_coverage_mapping(values: Mapping[str, float], key: str) -> None:
    if not values:
        raise ValueError(f"daily report {key} is invalid")
    for name, value in values.items():
        if not _NAME.fullmatch(name):
            raise ValueError(f"daily report {key} is invalid")
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"daily report {key} is invalid")
        if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
            raise ValueError(f"daily report {key} is invalid")


def _validate_score_distribution(distribution: DailyScoreDistribution) -> None:
    _finite_nonnegative(distribution.finite_coverage, "score_distribution")
    if distribution.finite_coverage > 1:
        raise ValueError("daily report score_distribution is invalid")
    _finite(distribution.min_score, "score_distribution")
    _finite(distribution.max_score, "score_distribution")
    _finite(distribution.mean_score, "score_distribution")
    _finite_nonnegative(distribution.std_score, "score_distribution")
    _validate_nonnegative_int(distribution.unique_score_count, "score_distribution")
    if distribution.min_score > distribution.max_score:
        raise ValueError("daily report score_distribution is invalid")
    if not distribution.min_score <= distribution.mean_score <= distribution.max_score:
        raise ValueError("daily report score_distribution is invalid")


def _validate_seed_agreement(agreement: DailySeedAgreement) -> None:
    _finite(agreement.minimum_spearman, "seed_agreement")
    if not -1 <= agreement.minimum_spearman <= 1:
        raise ValueError("daily report seed_agreement is invalid")


def _validate_hash_mapping(values: Mapping[str, str], key: str) -> None:
    if not values:
        raise ValueError(f"daily report {key} is invalid")
    for name, value in values.items():
        if not _NAME.fullmatch(name) or not isinstance(value, str) or not _SHA256.fullmatch(value):
            raise ValueError(f"daily report {key} is invalid")


def _validate_alerts(alerts: Sequence[str]) -> None:
    for alert in alerts:
        if not isinstance(alert, str) or not _ALERT_KEY.fullmatch(alert):
            raise ValueError("daily report alerts are invalid")


def _validate_nonnegative_int(value: object, key: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"daily report {key} is invalid")


def _is_uuid(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return str(uuid.UUID(value)) == value
    except ValueError:
        return False


def _finite_nonnegative(value: object, key: str) -> None:
    numeric = _finite_number(value, key)
    if numeric < 0:
        raise ValueError(f"daily report {key} is invalid")


def _finite_positive(value: object, key: str) -> None:
    numeric = _finite_number(value, key)
    if numeric <= 0:
        raise ValueError(f"daily report {key} is invalid")


def _finite(value: object, key: str) -> None:
    _finite_number(value, key)


def _finite_number(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"daily report {key} is invalid")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"daily report {key} is invalid")
    return numeric


def _contains_forbidden_token(value: object) -> bool:
    if isinstance(value, str):
        return any(token in value.lower() for token in _FORBIDDEN_TOKENS)
    if isinstance(value, Mapping):
        return any(
            any(token in str(key).lower() for token in _FORBIDDEN_TOKENS)
            or _contains_forbidden_token(nested)
            for key, nested in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return any(_contains_forbidden_token(nested) for nested in value)
    return False


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _write_atomic_exclusive(path: Path, content: str) -> None:
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    temporary.unlink()


__all__ = [
    "DailyRunReport",
    "DailyRunStep",
    "DailyRunStepStatus",
    "DailyRunWatermark",
    "DailyScoreDistribution",
    "DailySeedAgreement",
    "build_daily_run_report",
    "write_daily_run_report",
]
