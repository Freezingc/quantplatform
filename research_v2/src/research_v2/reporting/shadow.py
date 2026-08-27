"""Immutable, label-free daily reports for SHADOW operation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

_MODEL_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_INFERENCE_MANIFEST_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_EVENT_ID = re.compile(r"^[0-9a-f]{64}$")
_ALERT_KEY = re.compile(r"^[a-z][a-z0-9]{0,31}:[a-z][a-z0-9]{0,31}$")
_METRIC_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_FORBIDDEN_TOKENS = ("label", "pnl", "rankic", "return")


@dataclass(frozen=True)
class ShadowDailyReport:
    report_id: str
    payload: Mapping[str, object]

    def to_json(self) -> str:
        return _canonical_json({"report_id": self.report_id, **self.payload}) + "\n"

    def verify(self) -> None:
        if not _SHA256.fullmatch(self.report_id):
            raise ValueError("shadow report_id is invalid")
        _validate_payload(self.payload)
        if (
            hashlib.sha256(_canonical_json(self.payload).encode("utf-8")).hexdigest()
            != self.report_id
        ):
            raise ValueError("shadow report content does not match report_id")


def build_shadow_daily_report(
    *,
    signal_date: date,
    model_id: str,
    model_version: int,
    pipeline_status: str,
    prediction_published: bool,
    inference_manifest_id: str,
    artifact_bundle_hash: str,
    source_watermark_id: str,
    latency_seconds: float,
    universe_size: int,
    feature_coverage: Mapping[str, float],
    score_health: Mapping[str, float | int],
    seed_agreement: Mapping[str, float],
    drift_alert_keys: Sequence[str],
    event_ids: Sequence[str],
    latency_slo_seconds: float = 300.0,
    psi_error_features: Sequence[str] = (),
) -> ShadowDailyReport:
    """Record operational health without accepting labels, returns, PnL, or RankIC."""
    payload: dict[str, object] = {
        "artifact_bundle_hash": artifact_bundle_hash,
        "drift_alert_keys": list(drift_alert_keys),
        "event_ids": list(event_ids),
        "feature_coverage": dict(feature_coverage),
        "inference_manifest_id": inference_manifest_id,
        "latency_seconds": latency_seconds,
        "latency_slo_seconds": latency_slo_seconds,
        "model_id": model_id,
        "model_version": model_version,
        "performance_status": "LOCKED_NOT_COMPUTED",
        "pipeline_status": pipeline_status,
        "prediction_published": prediction_published,
        "psi_error_features": list(psi_error_features),
        "schema_version": 1,
        "score_health": dict(score_health),
        "seed_agreement": dict(seed_agreement),
        "signal_date": signal_date.isoformat(),
        "source_watermark_id": source_watermark_id,
        "universe_size": universe_size,
    }
    _validate_payload(payload)
    report_id = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return ShadowDailyReport(report_id, payload)


def _validate_payload(payload: Mapping[str, object]) -> None:
    expected = {
        "artifact_bundle_hash",
        "drift_alert_keys",
        "event_ids",
        "feature_coverage",
        "inference_manifest_id",
        "latency_seconds",
        "latency_slo_seconds",
        "model_id",
        "model_version",
        "performance_status",
        "pipeline_status",
        "prediction_published",
        "psi_error_features",
        "schema_version",
        "score_health",
        "seed_agreement",
        "signal_date",
        "source_watermark_id",
        "universe_size",
    }
    if set(payload) != expected or _contains_forbidden_token(payload):
        raise ValueError("shadow report payload is invalid")
    if not _MODEL_ID.fullmatch(_string(payload, "model_id")):
        raise ValueError("shadow report model_id is invalid")
    if not _INFERENCE_MANIFEST_ID.fullmatch(_string(payload, "inference_manifest_id")):
        raise ValueError("shadow report inference_manifest_id is invalid")
    for key in ("artifact_bundle_hash", "source_watermark_id"):
        if not _SHA256.fullmatch(_string(payload, key)):
            raise ValueError(f"shadow report {key} is invalid")
    try:
        date.fromisoformat(_string(payload, "signal_date"))
    except ValueError as error:
        raise ValueError("shadow report signal_date is invalid") from error
    if (
        payload.get("schema_version") != 1
        or payload.get("performance_status") != "LOCKED_NOT_COMPUTED"
    ):
        raise ValueError("shadow report schema or performance status is invalid")
    if payload.get("pipeline_status") not in {"SUCCESS", "FAILED", "NOOP_NON_TRADING_DAY"}:
        raise ValueError("shadow report pipeline_status is invalid")
    if not isinstance(payload.get("prediction_published"), bool):
        raise ValueError("shadow report prediction_published is invalid")
    if payload["prediction_published"] is not (payload["pipeline_status"] == "SUCCESS"):
        raise ValueError("shadow report pipeline status conflicts with publication state")
    for key in ("model_version", "universe_size"):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"shadow report {key} is invalid")
    _finite_nonnegative(payload.get("latency_seconds"), "latency_seconds")
    _finite_positive(payload.get("latency_slo_seconds"), "latency_slo_seconds")
    _coverage_mapping(payload.get("feature_coverage"), "feature_coverage")
    _finite_mapping(payload.get("score_health"), "score_health")
    _finite_mapping(payload.get("seed_agreement"), "seed_agreement")
    alert_keys = payload.get("drift_alert_keys")
    psi_error_features = payload.get("psi_error_features")
    event_ids = payload.get("event_ids")
    if (
        not isinstance(alert_keys, list)
        or not all(isinstance(value, str) and _ALERT_KEY.fullmatch(value) for value in alert_keys)
        or not isinstance(psi_error_features, list)
        or not all(
            isinstance(value, str) and _METRIC_NAME.fullmatch(value)
            for value in psi_error_features
        )
        or not isinstance(event_ids, list)
        or not all(isinstance(value, str) and _EVENT_ID.fullmatch(value) for value in event_ids)
    ):
        raise ValueError("shadow report event or alert references are invalid")


def _string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"shadow report {key} is invalid")
    return value


def _finite_nonnegative(value: object, key: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ValueError(f"shadow report {key} is invalid")


def _finite_positive(value: object, key: str) -> None:
    _finite_nonnegative(value, key)
    if value == 0:
        raise ValueError(f"shadow report {key} is invalid")


def _coverage_mapping(value: object, key: str) -> None:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"shadow report {key} is invalid")
    for name, coverage in value.items():
        if (
            not isinstance(name, str)
            or not name
            or isinstance(coverage, bool)
            or not isinstance(coverage, int | float)
        ):
            raise ValueError(f"shadow report {key} is invalid")
        if not math.isfinite(float(coverage)) or not 0 <= coverage <= 1:
            raise ValueError(f"shadow report {key} is invalid")


def _finite_mapping(value: object, key: str) -> None:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"shadow report {key} is invalid")
    for name, metric in value.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"shadow report {key} is invalid")
        _finite_nonnegative(metric, key)


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
