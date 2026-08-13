from __future__ import annotations

import audit_yeslab_adjusted_price_consistency as audit
import pandas as pd
import pytest


def test_internal_adjusted_price_check_uses_explicit_tolerance() -> None:
    frame = pd.DataFrame(
        {
            "high": [10.0, 11.0],
            "low": [9.0, 9.5],
            "adj_factor": [1.2, 1.1],
            "high_hfq": [12.0, 12.100004],
            "low_hfq": [10.8, 10.45],
        }
    )

    result = audit.summarize_internal_consistency(frame, tolerance=1e-5)

    assert result["rows"] == 2
    assert result["high_mismatch_rows"] == 0
    assert result["low_mismatch_rows"] == 0
    assert result["max_high_abs_diff"] == pytest.approx(4e-6)


def test_internal_adjusted_price_check_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        audit.summarize_internal_consistency(pd.DataFrame({"high": [1.0]}))


def test_compare_market_fields_classifies_raw_and_adjustment_differences() -> None:
    source = pd.DataFrame(
        {
            "ts_code": ["600000.SH", "600001.SH"],
            "trade_date": pd.to_datetime(["2025-01-02", "2025-01-02"]),
            "high": [10.0, 20.0],
            "low": [9.0, 19.0],
            "adj_factor": [1.2, 1.0],
            "high_hfq": [12.0, 20.0],
            "low_hfq": [10.8, 19.0],
        }
    )
    market = pd.DataFrame(
        {
            "ts_code": ["600000.SH", "600001.SH"],
            "trade_date": pd.to_datetime(["2025-01-02", "2025-01-02"]),
            "high": [10.01, 20.0],
            "low": [9.0, 19.0],
            "adj_factor": [1.2, 1.01],
        }
    )

    result = audit.compare_market_fields(source, market)

    assert result["overlap_rows"] == 2
    assert result["raw_high_mismatch_rows"] == 1
    assert result["adj_factor_mismatch_rows"] == 1
    assert result["derived_high_mismatch_rows"] == 2


def test_source_only_parser_does_not_require_market_cache() -> None:
    args = audit.parse_args(
        [
            "--start",
            "2020-01-02",
            "--end",
            "2020-01-31",
            "--source-only",
            "--output",
            "/tmp/audit.json",
        ]
    )

    assert args.source_only is True
