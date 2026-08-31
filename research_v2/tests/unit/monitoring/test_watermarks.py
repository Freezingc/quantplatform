from __future__ import annotations

from datetime import date

from research_v2.monitoring.watermarks import (
    InferenceWatermark,
    WatermarkStatus,
    check_inference_watermarks,
    watermark_fingerprint,
)


def test_watermark_gate_requires_published_coverage_without_labels() -> None:
    signal_date = date(2025, 1, 2)
    dependencies = {"calendar", "market", "security", "tradability", "universe", "factor:f1"}
    records = {
        dependency: InferenceWatermark(
            dependency=dependency,
            max_signal_date=signal_date,
            data_version="v1",
            published=True,
            has_blocker=False,
        )
        for dependency in dependencies
    }

    ready = check_inference_watermarks(
        signal_date,
        is_trading_day=True,
        required_dependencies=dependencies,
        watermarks=records,
    )
    assert ready.status is WatermarkStatus.READY
    assert not ready.blockers
    records["market"] = InferenceWatermark(
        dependency="market",
        max_signal_date=date(2025, 1, 1),
        data_version="v1",
        published=True,
        has_blocker=False,
    )
    blocked = check_inference_watermarks(
        signal_date,
        is_trading_day=True,
        required_dependencies=dependencies,
        watermarks=records,
    )
    assert blocked.status is WatermarkStatus.BLOCKED
    assert "STALE:market" in blocked.blockers
    noop = check_inference_watermarks(
        signal_date,
        is_trading_day=False,
        required_dependencies=dependencies,
        watermarks=records,
    )
    assert noop.status is WatermarkStatus.NOOP_NON_TRADING_DAY


def test_watermark_gate_rejects_sources_changed_during_run() -> None:
    signal_date = date(2025, 1, 2)
    dependencies = {"market", "factor:f1"}
    records = {
        dependency: InferenceWatermark(
            dependency=dependency,
            max_signal_date=signal_date,
            data_version="v1",
            published=True,
            has_blocker=False,
        )
        for dependency in dependencies
    }
    changed = {
        **records,
        "market": InferenceWatermark(
            dependency="market",
            max_signal_date=signal_date,
            data_version="v2",
            published=True,
            has_blocker=False,
        ),
    }

    report = check_inference_watermarks(
        signal_date,
        is_trading_day=True,
        required_dependencies=dependencies,
        watermarks=records,
        end_watermarks=changed,
    )

    assert report.status is WatermarkStatus.BLOCKED
    assert report.blockers == ("SOURCE_CHANGED_DURING_RUN",)
    assert report.start_fingerprint == watermark_fingerprint(records)
    assert report.end_fingerprint == watermark_fingerprint(changed)
    assert watermark_fingerprint(records) != watermark_fingerprint(changed)


def test_watermark_fingerprint_is_order_independent() -> None:
    first = {
        "b": InferenceWatermark("b", date(2025, 1, 2), "v1", True, False),
        "a": InferenceWatermark("a", date(2025, 1, 2), "v1", True, False),
    }
    second = dict(reversed(tuple(first.items())))
    assert watermark_fingerprint(first) == watermark_fingerprint(second)


def test_watermark_gate_detects_a_dependency_removed_during_run() -> None:
    signal_date = date(2025, 1, 2)
    records = {
        "market": InferenceWatermark("market", signal_date, "v1", True, False),
    }

    report = check_inference_watermarks(
        signal_date,
        is_trading_day=True,
        required_dependencies=set(records),
        watermarks=records,
        end_watermarks={},
    )

    assert report.blockers == ("SOURCE_CHANGED_DURING_RUN",)
