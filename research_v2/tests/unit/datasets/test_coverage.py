"""Tests for the assembled-frame cross-section measurement.

The fixtures encode the two measured failure modes: the frozen 17 losing whole crash
days, and the accepted legacy 14 losing 2017-10-09 entirely to one absent
(factor, date) cell. Both were invisible to G1-G6, which read TEST evidence only.
"""

from __future__ import annotations

import pandas as pd
import pytest

from research_v2.datasets.coverage import (
    WidthSummary,
    daily_complete_widths,
    factor_columns,
    summarize_frame_coverage,
    summarize_widths,
)


def _frame(widths: dict[str, int], factors: tuple[str, ...] = ("f00", "f01")) -> pd.DataFrame:
    """A frame whose per-day complete-row counts are exactly ``widths``.

    Every day also carries one row with a NaN factor, so the day is present in the
    calendar while its complete count stays at the requested width. A width of 0
    therefore means the day exists and loses every row, which is the case that a
    groupby on the post-dropna frame would silently drop.
    """
    rows: list[dict[str, object]] = []
    for day, complete in widths.items():
        for index in range(complete):
            row: dict[str, object] = {
                "signal_date": pd.Timestamp(day),
                "ts_code": f"{index:06d}.SZ",
            }
            row.update({factor: float(index) for factor in factors})
            rows.append(row)
        incomplete: dict[str, object] = {
            "signal_date": pd.Timestamp(day),
            "ts_code": "999999.SZ",
            factors[0]: float("nan"),
        }
        incomplete.update({factor: 1.0 for factor in factors[1:]})
        rows.append(incomplete)
    return pd.DataFrame(rows)


def test_widths_apply_exactly_the_runner_dropna() -> None:
    frame = _frame({"2020-01-02": 2, "2020-01-03": 1})

    widths = daily_complete_widths(frame, ("f00", "f01"))

    assert widths == {pd.Timestamp("2020-01-02"): 2, pd.Timestamp("2020-01-03"): 1}


def test_a_day_that_loses_every_row_is_reported_as_zero_not_omitted() -> None:
    frame = _frame({"2015-08-24": 0, "2015-08-25": 2})

    widths = daily_complete_widths(frame, ("f00", "f01"))

    assert widths[pd.Timestamp("2015-08-24")] == 0
    assert widths[pd.Timestamp("2015-08-25")] == 2


def test_empty_frame_yields_no_widths() -> None:
    frame = pd.DataFrame({"signal_date": pd.Series(dtype="datetime64[ns]"), "f00": []})

    assert daily_complete_widths(frame, ("f00",)) == {}


def test_summary_counts_rows_days_and_the_two_kinds_of_lost_day() -> None:
    frame = _frame({"2017-10-09": 0, "2017-10-10": 40, "2017-10-11": 20})

    summary = summarize_frame_coverage(frame, ("f00", "f01"))

    assert summary == WidthSummary(
        rows=63,             # (0+1) + (40+1) + (20+1)
        complete_rows=60,
        days=3,
        days_below_floor=2,  # the empty day and the 20-wide day, both under the floor
        empty_days=1,
        median_width=20,
        min_width=0,
    )
    assert summary.survival_ratio == pytest.approx(60 / 63)


def test_summary_derives_the_feature_columns_when_not_supplied() -> None:
    frame = _frame({"2020-01-02": 5})

    explicit = summarize_frame_coverage(frame, ("f00", "f01"))
    derived = summarize_frame_coverage(frame)

    assert derived == explicit


def test_summary_honours_a_custom_floor() -> None:
    frame = _frame({"2020-01-02": 40, "2020-01-03": 20})

    strict = summarize_frame_coverage(frame, ("f00", "f01"), minimum_cross_section=30)
    lenient = summarize_frame_coverage(frame, ("f00", "f01"), minimum_cross_section=10)

    assert strict.days_below_floor == 1
    assert lenient.days_below_floor == 0


def test_summary_rejects_an_unusable_floor() -> None:
    frame = _frame({"2020-01-02": 5})

    with pytest.raises(ValueError, match="at least 2"):
        summarize_frame_coverage(frame, ("f00", "f01"), minimum_cross_section=1)


def test_to_json_carries_the_survival_ratio_the_dataclass_cannot_store() -> None:
    summary = WidthSummary(
        rows=100, complete_rows=25, days=10, days_below_floor=0,
        empty_days=0, median_width=10, min_width=5,
    )

    payload = summary.to_json()

    assert payload["survival_ratio"] == pytest.approx(0.25)
    assert payload["rows"] == 100
    assert payload["min_width"] == 5


def test_factor_columns_selects_only_fnn_and_rejects_a_frame_without_them() -> None:
    columns = ["signal_date", "ts_code", "f00", "f01", "f10", "raw_return_5d", "label_valid"]

    assert factor_columns(columns) == ("f00", "f01", "f10")
    with pytest.raises(ValueError, match="no f"):
        factor_columns(["signal_date", "ts_code", "raw_return_5d"])


def test_summarize_widths_agrees_with_summarize_frame_coverage() -> None:
    """The split exists to avoid a second dropna, so the two paths must not diverge."""
    frame = _frame({"2017-10-09": 0, "2017-10-10": 40, "2017-10-11": 20})
    widths = daily_complete_widths(frame, ("f00", "f01"))

    from_widths = summarize_widths(widths, rows=len(frame))
    from_frame = summarize_frame_coverage(frame, ("f00", "f01"))

    assert from_widths == from_frame


def test_summarize_widths_rejects_an_unusable_floor() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        summarize_widths({pd.Timestamp("2020-01-02"): 5}, rows=5, minimum_cross_section=1)
