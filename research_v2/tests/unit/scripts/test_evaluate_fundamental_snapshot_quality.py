from __future__ import annotations

from datetime import date

import evaluate_fundamental_snapshot_quality as evaluation
import materialize_fundamental_pit_snapshot as materialization
import pandas as pd
import pytest


def test_parser_rejects_locked_2026_range() -> None:
    with pytest.raises(SystemExit):
        evaluation.parse_args(
            ["--start", "2026-01-01", "--end", "2026-01-02", "--output", "/tmp/out"]
        )


def test_summary_uses_mature_universe_and_daily_rank_correlation() -> None:
    snapshot = pd.DataFrame(
        {
            "signal_date": [date(2020, 1, 2)] * 3,
            "ts_code": ["000001.SZ", "000002.SZ", "600000.SH"],
            "metric_id": ["roa"] * 3,
            "value": [3.0, 1.0, 2.0],
        }
    )
    universe = pd.DataFrame(
        {
            "signal_date": [date(2020, 1, 2)] * 4,
            "ts_code": ["000001.SZ", "000002.SZ", "600000.SH", "830001.BJ"],
            "is_member": [True, True, True, True],
        }
    )
    labels = pd.DataFrame(
        {
            "signal_date": [date(2020, 1, 2)] * 4,
            "ts_code": ["000001.SZ", "000002.SZ", "600000.SH", "830001.BJ"],
            "raw_return": [0.3, 0.1, 0.2, 0.4],
            "label_valid": [True, True, True, False],
        }
    )

    coverage, daily, summary = evaluation.summarize_snapshot_quality(
        snapshot, universe, labels, minimum_cross_section=3
    )

    assert coverage.to_dict("records") == [
        {"metric_id": "roa", "finite_count": 3, "mature_row_count": 3, "finite_ratio": 1.0}
    ]
    assert daily[["rank_ic", "sample_count"]].to_dict("records") == [
        {"rank_ic": 1.0, "sample_count": 3}
    ]
    assert summary.loc[0, "mean_rank_ic"] == 1.0


def test_summary_rejects_duplicate_snapshot_keys() -> None:
    snapshot = pd.DataFrame(
        {
            "signal_date": [date(2020, 1, 2), date(2020, 1, 2)],
            "ts_code": ["000001.SZ", "000001.SZ"],
            "metric_id": ["roa", "roa"],
            "value": [1.0, 1.0],
        }
    )
    universe = pd.DataFrame(
        {"signal_date": [date(2020, 1, 2)], "ts_code": ["000001.SZ"], "is_member": [True]}
    )
    labels = pd.DataFrame(
        {
            "signal_date": [date(2020, 1, 2)],
            "ts_code": ["000001.SZ"],
            "raw_return": [0.1],
            "label_valid": [True],
        }
    )

    with pytest.raises(ValueError, match="duplicate"):
        evaluation.summarize_snapshot_quality(snapshot, universe, labels)


def test_yoy_derivative_uses_exact_prior_report_period_available_before_current_event() -> None:
    events = pd.DataFrame(
        {
            "metric_id": ["roa", "roa", "roa"],
            "ts_code": ["000001.SZ"] * 3,
            "report_period": [date(2023, 12, 31), date(2024, 12, 31), date(2024, 12, 31)],
            "ann_date": [date(2024, 3, 28), date(2025, 3, 28), date(2025, 4, 10)],
            "source_revision_id": ["initial", "initial", "revision"],
            "raw_value": [1.0, 3.0, 4.0],
        }
    )

    result = materialization.derive_year_over_year_events(
        events, source_metric="roa", output_metric="road", operation="difference"
    )

    assert result.to_dict("records") == [
        {
            "metric_id": "road",
            "ts_code": "000001.SZ",
            "report_period": date(2024, 12, 31),
            "ann_date": date(2025, 3, 28),
            "source_revision_id": "current:initial|prior:initial",
            "raw_value": 2.0,
        },
        {
            "metric_id": "road",
            "ts_code": "000001.SZ",
            "report_period": date(2024, 12, 31),
            "ann_date": date(2025, 4, 10),
            "source_revision_id": "current:revision|prior:initial",
            "raw_value": 3.0,
        },
    ]


def test_quarter_derivative_uses_prior_quarter_cumulative_value() -> None:
    events = pd.DataFrame(
        {
            "metric_id": ["net_income_cumulative"] * 3,
            "ts_code": ["000001.SZ"] * 3,
            "report_period": [date(2024, 3, 31), date(2024, 6, 30), date(2024, 9, 30)],
            "ann_date": [date(2024, 4, 30), date(2024, 8, 30), date(2024, 10, 30)],
            "source_revision_id": ["q1", "q2", "q3"],
            "raw_value": [10.0, 25.0, 42.0],
        }
    )

    result = materialization.derive_quarter_over_quarter_events(
        events, source_metric="net_income_cumulative", output_metric="np_qoq"
    )

    assert result[["report_period", "raw_value"]].to_dict("records") == [
        {"report_period": date(2024, 6, 30), "raw_value": 0.5},
        {"report_period": date(2024, 9, 30), "raw_value": 2.0 / 15.0},
    ]


def test_publish_parser_rejects_nonpositive_chunk_size() -> None:
    with pytest.raises(SystemExit):
        materialization.parse_args(["--publish-chunk-rows", "0"])


def test_publish_chunks_are_bounded_and_partitioned_by_metric_month() -> None:
    frame = pd.DataFrame(
        {
            "signal_date": [date(2021, 1, 4), date(2021, 1, 5), date(2021, 2, 1), date(2021, 1, 6)],
            "ts_code": ["000001.SZ", "000002.SZ", "000001.SZ", "000003.SZ"],
            "metric_id": ["roa", "roa", "roa", "roe"],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )

    chunks = materialization._iter_publish_chunks(frame, chunk_rows=1)

    assert [len(chunk) for chunk in chunks] == [1, 1, 1, 1]
    assert [
        (chunk.iloc[0]["metric_id"], chunk.iloc[0]["signal_date"].month) for chunk in chunks
    ] == [
        ("roa", 1),
        ("roa", 1),
        ("roa", 2),
        ("roe", 1),
    ]
