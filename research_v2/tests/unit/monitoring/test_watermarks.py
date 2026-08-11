from __future__ import annotations

from datetime import date

from research_v2.monitoring.watermarks import (
    InferenceWatermark,
    WatermarkStatus,
    check_inference_watermarks,
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
