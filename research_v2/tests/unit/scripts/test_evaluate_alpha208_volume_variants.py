from __future__ import annotations

import evaluate_alpha208_volume_variants as evaluation
import pandas as pd
import pytest


def test_compute_variants_keeps_price_term_and_returns_all_scales() -> None:
    dates = pd.date_range("2025-01-01", periods=61)
    market = pd.DataFrame(
        {
            "trade_date": dates.repeat(2),
            "ts_code": ["000001.SZ", "600000.SH"] * len(dates),
            "close": [10.0 + i for i in range(len(dates))] * 2,
            "vol": [100.0 + i for i in range(len(dates))] * 2,
        }
    )

    variants = evaluation.compute_variants(market)

    assert set(variants) == set(evaluation.VARIANTS)
    assert all(frame.shape == (61, 2) for frame in variants.values())
    assert all(frame.iloc[:59].isna().all().all() for frame in variants.values())


def test_empty_monthly_summary_has_stable_columns() -> None:
    result = evaluation.summarize_monthly(pd.DataFrame(columns=["signal_date", "rank_ic"]))

    assert "signed_mean_rank_ic" in result.columns
    assert result.empty


def test_parser_rejects_locked_test() -> None:
    with pytest.raises(SystemExit):
        evaluation.parse_args(
            ["--start", "2026-01-01", "--end", "2026-01-02", "--output", "/tmp/x"]
        )


def test_warmup_covers_the_longest_price_window() -> None:
    assert evaluation.WARMUP_CALENDAR_DAYS >= 120
