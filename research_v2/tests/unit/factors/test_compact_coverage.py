from __future__ import annotations

from datetime import date

import pandas as pd

from research_v2.factors.compact_coverage import (
    summarize_compact_coverage,
    summarize_compact_coverage_counts,
)


def _universe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_date": [date(2020, 12, 1)] * 4,
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"],
            "is_member": [True] * 4,
        }
    )


def _labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_date": [date(2020, 12, 1)] * 4,
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"],
            "label_valid": [True] * 4,
        }
    )


def test_compact_coverage_deduplicates_source_and_uses_mature_denominator() -> None:
    source = pd.DataFrame(
        {
            "trade_date": [date(2020, 12, 1)] * 6,
            "ts_code": [
                "000001.SZ",
                "000001.SZ",
                "000002.SZ",
                "000003.SZ",
                "000005.SZ",
                "000006.SZ",
            ],
            "value": [1.0, 1.0, 2.0, float("nan"), 3.0, 4.0],
        }
    )

    result = summarize_compact_coverage(
        _universe(),
        _labels(),
        source,
        factor_id="yeslab_1",
        factor_name="F",
        logical_partition="2020-12",
    )

    assert result["source_rows"] == 6
    assert result["source_exact_duplicate_rows"] == 1
    assert result["compact_rows"] == 4
    assert result["compact_finite_rows"] == 2
    assert result["compact_finite_ratio"] == 0.5
    assert result["decision_status"] == "COMPACT_DQ_FAIL"


def test_compact_coverage_blocks_distinct_source_values_for_one_key() -> None:
    source = pd.DataFrame(
        {
            "trade_date": [date(2020, 12, 1), date(2020, 12, 1)],
            "ts_code": ["000001.SZ", "000001.SZ"],
            "value": [1.0, 2.0],
        }
    )

    result = summarize_compact_coverage(
        _universe(),
        _labels(),
        source,
        factor_id="yeslab_1",
        factor_name="F",
        logical_partition="2020-12",
    )

    assert result["source_conflict_keys"] == 1
    assert result["decision_status"] == "COMPACT_DQ_BLOCKED"


def test_compact_coverage_counts_uses_zero_for_missing_days() -> None:
    finite_keys = pd.DataFrame(
        {
            "signal_date": [date(2020, 12, 1)],
            "finite_keys": [4],
        }
    )
    result = summarize_compact_coverage_counts(
        _universe(),
        _labels(),
        finite_keys,
        factor_id="yeslab_1",
        factor_name="F",
        logical_partition="2020-12",
    )

    assert result["compact_rows"] == 4
    assert result["compact_finite_rows"] == 4
    assert result["finite_key_overflow_days"] == 0
    assert result["decision_status"] == "COMPACT_DQ_PASS_PENDING_SIMILARITY"
