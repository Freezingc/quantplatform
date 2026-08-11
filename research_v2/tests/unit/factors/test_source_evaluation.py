from __future__ import annotations

from datetime import date

import pandas as pd

from research_v2.factors.source_evaluation import (
    evaluate_materialized_factor_frames,
    evaluate_source_factor_range,
)


def test_materialized_factor_evaluation_returns_coverage_and_rank_ic() -> None:
    clean = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2)] * 3 + [date(2025, 1, 3)] * 3,
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"] * 2,
            "factor_id": ["yeslab_1"] * 6,
            "value": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
        }
    )
    universe = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2)] * 3 + [date(2025, 1, 3)] * 3,
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"] * 2,
            "is_member": [True] * 6,
        }
    )
    labels = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2)] * 3 + [date(2025, 1, 3)] * 3,
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"] * 2,
            "raw_return": [0.1, 0.2, 0.3, 0.3, 0.2, 0.1],
            "label_valid": [True] * 6,
        }
    )

    result = evaluate_materialized_factor_frames(
        [clean],
        universe=universe,
        labels=labels,
        logical_partition="2025-01",
        minimum_cross_section=2,
    )

    assert result.logical_partition == "2025-01"
    assert result.clean_row_count == 6
    assert result.coverage.to_dict("records") == [
        {
            "factor_id": "yeslab_1",
            "finite_count": 6,
            "mature_row_count": 6,
            "finite_ratio": 1.0,
        }
    ]
    assert len(result.daily_rank_ic) == 2
    assert result.factor_summary.loc[0, "factor_name"] == "yeslab_1"
    assert result.factor_summary.loc[0, "mean_rank_ic"] == 1.0


def test_materialized_factor_evaluation_counts_missing_values_against_mature_universe() -> None:
    clean = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2)],
            "ts_code": ["000001.SZ"],
            "factor_id": ["yeslab_1"],
            "value": [1.0],
        }
    )
    universe = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2)] * 2,
            "ts_code": ["000001.SZ", "000002.SZ"],
            "is_member": [True, True],
        }
    )
    labels = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2)] * 2,
            "ts_code": ["000001.SZ", "000002.SZ"],
            "raw_return": [0.1, 0.2],
            "label_valid": [True, False],
        }
    )

    result = evaluate_materialized_factor_frames(
        [clean],
        universe=universe,
        labels=labels,
        logical_partition="2025-01",
        minimum_cross_section=2,
    )

    assert result.coverage.loc[0, "mature_row_count"] == 1
    assert result.coverage.loc[0, "finite_ratio"] == 1.0
    assert result.daily_rank_ic.empty


def test_source_range_evaluation_reads_one_fake_source_and_returns_each_month() -> None:
    source = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ"] * 2,
            "trade_date": [date(2025, 1, 2)] * 2 + [date(2025, 2, 3)] * 2,
            "factor_name": ["alpha", "beta"] * 2,
            "value": [1.0, 2.0, 2.0, 1.0],
        }
    )

    class SourceSession:
        def query(self, script: str) -> object:
            assert "date('2025.01.01')" in script
            assert "date('2025.02.28')" in script
            return source.copy()

    metadata = pd.DataFrame(
        {
            "factor_id": [1, 2],
            "factor_name": ["alpha", "beta"],
            "creator": ["research", "research"],
            "update_frequency": ["daily", "daily"],
            "store_factor": ["factor_data1", "factor_data1"],
        }
    )
    universe = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2)] * 2,
            "ts_code": ["000001.SZ", "000002.SZ"],
            "is_member": [True, True],
        }
    )
    labels = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2)] * 2,
            "ts_code": ["000001.SZ", "000002.SZ"],
            "raw_return": [0.1, 0.2],
            "label_valid": [True, True],
        }
    )

    evaluations = evaluate_source_factor_range(
        SourceSession(),
        metadata,
        factor_ids=("yeslab_1", "yeslab_2"),
        start_date=date(2025, 1, 1),
        end_date=date(2025, 2, 28),
        month_inputs={"2025-01": (universe, labels), "2025-02": (universe, labels)},
        minimum_cross_section=2,
    )

    assert [evaluation.logical_partition for evaluation in evaluations] == ["2025-01", "2025-02"]
    assert all(evaluation.clean_row_count == 2 for evaluation in evaluations)
