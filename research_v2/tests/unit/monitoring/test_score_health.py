from __future__ import annotations

import pandas as pd

from research_v2.monitoring.score_health import evaluate_score_health


def test_score_health_blocks_duplicate_low_coverage_and_score_collapse() -> None:
    healthy = pd.DataFrame(
        {
            "signal_date": ["2025-01-02"] * 10,
            "ts_code": [f"S{index:03d}" for index in range(10)],
            "ensemble_score": [index / 10 for index in range(10)],
        }
    )

    report = evaluate_score_health(healthy, expected_key_count=10)
    assert not report.blocking_reasons
    duplicate = pd.concat([healthy, healthy.iloc[[0]]], ignore_index=True)
    assert "DUPLICATE_BUSINESS_KEY" in evaluate_score_health(
        duplicate,
        expected_key_count=10,
    ).blocking_reasons
    collapsed = healthy.assign(ensemble_score=0.5)
    assert "INSUFFICIENT_UNIQUE_SCORES" in evaluate_score_health(
        collapsed,
        expected_key_count=10,
    ).blocking_reasons
