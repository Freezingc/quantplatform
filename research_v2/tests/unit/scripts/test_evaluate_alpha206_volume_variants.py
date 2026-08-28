from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import evaluate_alpha206_volume_variants as evaluation


def test_compute_variants_has_expected_warmup() -> None:
    dates = pd.date_range("2025-01-01", periods=61)
    market = pd.DataFrame(
        {
            "trade_date": dates.repeat(2),
            "ts_code": ["000001.SZ", "600000.SH"] * len(dates),
            "vol": np.tile(np.arange(1.0, 62.0), 2),
        }
    )
    result = evaluation.compute_variants(market)
    assert tuple(result) == evaluation.VARIANTS
    assert result["raw_sma_abs"].iloc[:59].isna().all().all()
    assert result["raw_sma_abs"].iloc[59:].notna().all().all()


def test_parser_rejects_locked_test() -> None:
    with pytest.raises(SystemExit):
        evaluation.parse_args(["--start", "2026-01-01", "--end", "2026-01-02", "--output", "/tmp/x"])


def test_summarize_variant_serializes_validation_only_window() -> None:
    daily = pd.DataFrame(
        {
            "signal_date": pd.date_range("2025-01-02", periods=3, freq="B"),
            "rank_ic": [-0.03, -0.05, -0.07],
            "sample_count": [100, 100, 100],
        }
    )

    summary = evaluation.summarize_variant(daily, "median_abs")

    assert summary["annual"] == {}
    assert summary["strict_annual_icir_pass_years"] == 0
    json.dumps(summary, allow_nan=False)
