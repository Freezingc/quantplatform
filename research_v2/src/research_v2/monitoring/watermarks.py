"""Fail-closed dependency-watermark gate for label-free daily inference."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Set
from dataclasses import dataclass
from datetime import date
from enum import StrEnum


class WatermarkStatus(StrEnum):
    READY = "READY"
    BLOCKED = "BLOCKED"
    NOOP_NON_TRADING_DAY = "NOOP_NON_TRADING_DAY"


@dataclass(frozen=True)
class InferenceWatermark:
    dependency: str
    max_signal_date: date
    data_version: str
    published: bool
    has_blocker: bool


@dataclass(frozen=True)
class WatermarkReport:
    status: WatermarkStatus
    blockers: tuple[str, ...]


def check_inference_watermarks(
    signal_date: date,
    *,
    is_trading_day: bool,
    required_dependencies: Set[str],
    watermarks: Mapping[str, InferenceWatermark],
    end_watermarks: Mapping[str, InferenceWatermark] | None = None,
) -> WatermarkReport:
    """Allow a trading-day inference batch only when every non-label dependency is current."""
    if any("label" in dependency.lower() for dependency in required_dependencies):
        raise ValueError("inference dependencies must not include labels")
    if not is_trading_day:
        return WatermarkReport(WatermarkStatus.NOOP_NON_TRADING_DAY, ())
    blockers: list[str] = []
    for dependency in sorted(required_dependencies):
        watermark = watermarks.get(dependency)
        if watermark is None:
            blockers.append(f"MISSING:{dependency}")
            continue
        if watermark.dependency != dependency:
            blockers.append(f"IDENTITY_MISMATCH:{dependency}")
        if not watermark.published:
            blockers.append(f"UNPUBLISHED:{dependency}")
        if not watermark.data_version:
            blockers.append(f"MISSING_VERSION:{dependency}")
        if watermark.max_signal_date < signal_date:
            blockers.append(f"STALE:{dependency}")
        if watermark.has_blocker:
            blockers.append(f"DQ_BLOCKER:{dependency}")
    if end_watermarks is not None and _scoped_fingerprint(
        required_dependencies,
        watermarks,
    ) != _scoped_fingerprint(required_dependencies, end_watermarks):
        blockers.append("SOURCE_CHANGED_DURING_RUN")
    return WatermarkReport(
        status=WatermarkStatus.BLOCKED if blockers else WatermarkStatus.READY,
        blockers=tuple(blockers),
    )


def watermark_fingerprint(watermarks: Mapping[str, InferenceWatermark]) -> str:
    """Hash the ordered dependency identity and state for an immutable run snapshot."""
    if not isinstance(watermarks, Mapping):
        raise ValueError("watermarks must be a mapping")
    payload = []
    for dependency in sorted(watermarks):
        watermark = watermarks[dependency]
        if not isinstance(dependency, str) or not isinstance(watermark, InferenceWatermark):
            raise ValueError("watermark snapshot contains an invalid record")
        payload.append(
            {
                "data_version": watermark.data_version,
                "dependency": dependency,
                "has_blocker": watermark.has_blocker,
                "max_signal_date": watermark.max_signal_date.isoformat(),
                "published": watermark.published,
                "record_dependency": watermark.dependency,
            }
        )
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _scoped_fingerprint(
    required_dependencies: Set[str],
    watermarks: Mapping[str, InferenceWatermark],
) -> str:
    return watermark_fingerprint(
        {
            dependency: watermarks[dependency]
            for dependency in required_dependencies
            if dependency in watermarks
        }
    )
