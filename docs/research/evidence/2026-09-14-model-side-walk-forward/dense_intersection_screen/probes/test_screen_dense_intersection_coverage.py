"""Tests for the dense-intersection coverage screen: the instrument the 24 days should have had.

Every coverage gate in use reads one factor at a time (median and p05, at 0.70/0.30 or 0.80/0.70),
while the score-health floor at evaluation/gates.py:35 is applied per day to the INTERSECTION of the
dense factors against valid-label rows. The frozen v2 set passes all nine dense factors on both
instruments -- yeslab_471 at p05 0.9091 -- and still lost 24 OOS days, because 471's own fifth
percentile sits 0.0002 above the panel's measured 2023 intersection median of 0.9093 and the floor
is 0.90. This screen measures the quantity the floor actually constrains, from the published tables
alone, so the day set can be known before a run spends nine days producing it.

The floor is parsed out of the evaluator's own source rather than retyped here: a screen whose
threshold can drift from the gate it predicts is worse than no screen, because it still looks
authoritative.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

# tests/conftest.py puts research_v2/scripts on sys.path before test modules are imported.
import walk_forward_common as COMMON  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO / "scripts" / "screen_dense_intersection_coverage.py"
GATES_PATH = REPO / "src" / "research_v2" / "evaluation" / "gates.py"


def _load():
    spec = importlib.util.spec_from_file_location("screen_dense_intersection_coverage", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["screen_dense_intersection_coverage"] = module
    spec.loader.exec_module(module)
    return module


M = _load()

_FACTOR_PIN = "f" * 64
_LABEL_PIN = "1" * 64
_STALE = "s" * 64


def _days(start: date, count: int) -> list[date]:
    return [start + timedelta(days=offset) for offset in range(count)]


# ---------------------------------------------------------------------------------------------
# the floor must come from the evaluator
# ---------------------------------------------------------------------------------------------


def test_floor_is_parsed_out_of_the_evaluator_source() -> None:
    """The screen predicts a gate, so it must read that gate's threshold, not restate it."""
    parsed = M.parse_floor_from_gates(GATES_PATH.read_text(encoding="utf-8"))
    assert parsed == 0.90
    assert parsed == M.DEFAULT_FLOOR, (
        "the evaluator's score-health floor moved and this screen still predicts the old one; "
        "update DEFAULT_FLOOR only after re-validating the predicted day set"
    )


def test_floor_parser_rejects_a_source_it_cannot_read() -> None:
    with pytest.raises(M.FloorNotFoundError):
        M.parse_floor_from_gates("return coverage <= 1.0 and self.unique_score_count >= 10")


# ---------------------------------------------------------------------------------------------
# the pure measurement
# ---------------------------------------------------------------------------------------------


def test_coverage_is_intersection_over_valid_label_rows() -> None:
    days = _days(date(2023, 7, 3), 3)
    intersection = {days[0]: 3259, days[1]: 3262, days[2]: 3230}
    denominator = {days[0]: 3617, days[1]: 3624, days[2]: 3627}
    measured = M.daily_coverage(intersection, denominator, floor=0.90)
    assert [round(entry.coverage, 4) for entry in measured] == [0.9010, 0.9001, 0.8905]
    assert [entry.below_floor for entry in measured] == [False, False, True]


def test_the_measured_worst_days_reproduce_the_published_ones() -> None:
    """Two days whose coverage is already recorded elsewhere in the evidence pack.

    2023-07-05 was dropped and 2023-07-04 survived with the thinnest margin of the year (0.9001).
    If this screen cannot separate those two from published tables alone it predicts nothing.
    """
    intersection = {date(2023, 7, 4): 3262, date(2023, 7, 5): 3230}
    denominator = {date(2023, 7, 4): 3624, date(2023, 7, 5): 3627}
    measured = {
        entry.day: entry
        for entry in M.daily_coverage(intersection, denominator, floor=0.90)
    }
    assert measured[date(2023, 7, 4)].below_floor is False
    assert round(measured[date(2023, 7, 4)].coverage, 4) == 0.9001
    assert measured[date(2023, 7, 5)].below_floor is True
    assert round(measured[date(2023, 7, 5)].coverage, 4) == 0.8905


def test_a_day_the_labels_do_not_cover_is_not_counted_against_the_factors() -> None:
    """Same rule the evaluator and the per-factor screen both use: no denominator, no verdict."""
    days = _days(date(2023, 7, 3), 2)
    measured = M.daily_coverage({days[0]: 100}, {days[0]: 100, days[1]: 0}, floor=0.90)
    assert [entry.day for entry in measured] == [days[0]]


def test_a_day_with_no_intersection_rows_at_all_is_below_the_floor() -> None:
    days = _days(date(2023, 7, 3), 1)
    measured = M.daily_coverage({}, {days[0]: 3617}, floor=0.90)
    assert len(measured) == 1
    assert measured[0].intersection_rows == 0
    assert measured[0].coverage == 0.0
    assert measured[0].below_floor is True


def test_summary_is_per_year_because_that_is_how_the_reports_read_it() -> None:
    intersection = {
        date(2022, 7, 29): 2900,
        date(2023, 7, 5): 3230,
        date(2023, 7, 6): 3300,
        date(2024, 1, 2): 3700,
    }
    denominator = {
        date(2022, 7, 29): 3355,
        date(2023, 7, 5): 3627,
        date(2023, 7, 6): 3630,
        date(2024, 1, 2): 3800,
    }
    measured = M.daily_coverage(intersection, denominator, floor=0.90)
    summary = M.summarize(measured, floor=0.90)
    assert summary["total_days"] == 4
    assert summary["days_below_floor"] == 2
    assert summary["by_year"]["2022"]["days_below_floor"] == 1
    assert summary["by_year"]["2023"]["days_below_floor"] == 1
    assert summary["by_year"]["2023"]["days"] == 2
    assert summary["by_year"]["2024"]["days_below_floor"] == 0
    assert sorted(summary["days_below_floor_list"]) == ["2022-07-29", "2023-07-05"]


def test_comparison_against_a_known_day_set_reports_both_differences() -> None:
    """A prediction nobody can check against the truth is not a prediction."""
    known = ["2022-07-29", "2023-07-05", "2023-07-06"]
    predicted = ["2022-07-29", "2023-07-05", "2023-07-07"]
    comparison = M.compare_with_known(predicted, known)
    assert comparison["matched"] == ["2022-07-29", "2023-07-05"]
    assert comparison["only_predicted"] == ["2023-07-07"]
    assert comparison["only_known"] == ["2023-07-06"]
    assert comparison["identical"] is False
    assert M.compare_with_known(known, known)["identical"] is True


# ---------------------------------------------------------------------------------------------
# the queries: what they must ask for, and what they must refuse to do
# ---------------------------------------------------------------------------------------------


def _patch_db(monkeypatch, *, intersection: list[dict], labels: list[dict],
              factor_pins: dict[str, str] | None = None,
              label_pins: dict[str, str] | None = None) -> list[str]:
    captured: list[str] = []

    def fake_query(holder: object, sql: str):
        captured.append(sql)
        if "ej(" in sql:
            return True, pd.DataFrame(intersection)
        if "label_daily" in sql:
            return True, pd.DataFrame(labels)
        if "factor_daily" in sql:
            return True, pd.DataFrame(intersection)
        raise AssertionError(f"unexpected query: {sql}")

    def fake_current_versions(holder: object, object_name: str) -> dict[str, str]:
        if object_name == "factor_daily":
            return {"2023-07": _FACTOR_PIN} if factor_pins is None else dict(factor_pins)
        return {"2023-07": _LABEL_PIN} if label_pins is None else dict(label_pins)

    monkeypatch.setattr(COMMON, "query", fake_query)
    monkeypatch.setattr(COMMON, "current_versions", fake_current_versions)
    return captured


def test_intersection_query_deduplicates_then_requires_every_factor(monkeypatch) -> None:
    """The intersection is a per-(day, security) count over distinct factors, not a row count.

    Without the inner grouping a duplicated business key would let eight factors read as nine, and
    without the ``having count(*) = k`` clause the query returns the union instead of the
    intersection -- the union is exactly what a per-factor screen already reports.
    """
    captured = _patch_db(monkeypatch, intersection=[], labels=[])
    M.collect({}, date(2023, 7, 1), date(2023, 7, 31), ("a", "b", "c"))
    sql = next(statement for statement in captured if "ej(" in statement)
    assert "group by signal_date, ts_code, factor_id" in sql
    assert "having count(*) = 3" in sql
    assert _FACTOR_PIN in sql, "the intersection must be read at the published data_version"


def test_the_intersection_is_restricted_to_securities_with_a_valid_label(monkeypatch) -> None:
    """A security holding every factor but no valid label is not a row of the panel.

    Counting it inflates coverage, and the inflation is exactly what hid three of the 24 measured
    days: on 2023-04-28 the unrestricted count is 3,237 (coverage 0.9070, predicted to survive)
    while the restricted count is 3,212 (0.89997, dropped by the run). All three misses were in the
    unsafe direction, so the join is part of the contract, not an optimisation.
    """
    captured = _patch_db(monkeypatch, intersection=[], labels=[])
    M.collect({}, date(2023, 7, 1), date(2023, 7, 31), ("a",))
    sql = next(statement for statement in captured if "ej(" in statement)
    assert "label_valid=1" in sql
    assert "ej(lab, complete" in sql
    assert _LABEL_PIN in sql


def test_denominator_is_valid_label_rows_at_the_published_version(monkeypatch) -> None:
    """The evaluator's coverage denominator is valid-label rows, not universe members.

    Measured on 2023-07-05: label_daily's label_valid=1 count at the published version is 3,627,
    which is the panel's valid-label row count for that day exactly. Universe membership is a
    different, larger number and would understate coverage.
    """
    captured = _patch_db(monkeypatch, intersection=[], labels=[])
    M.collect({}, date(2023, 7, 1), date(2023, 7, 31), ("a",))
    sql = next(statement for statement in captured if "label_daily" in statement)
    assert "label_valid=1" in sql
    assert _LABEL_PIN in sql


def test_rows_under_an_unpublished_version_fail_closed(monkeypatch) -> None:
    """A partition with no published version cannot be measured, and must not be skipped."""
    _patch_db(
        monkeypatch,
        intersection=[{"signal_date": pd.Timestamp("2023-07-05"), "n": 10}],
        labels=[{"signal_date": pd.Timestamp("2023-07-05"), "n": 100}],
        factor_pins={},
        label_pins={},
    )
    with pytest.raises(M.UnpublishedPartitionError) as excinfo:
        M.collect({}, date(2023, 7, 1), date(2023, 7, 31), ("a",))
    assert "2023-07" in str(excinfo.value)


def test_collect_returns_the_per_day_counts_the_pure_layer_needs(monkeypatch) -> None:
    _patch_db(
        monkeypatch,
        intersection=[
            {"signal_date": pd.Timestamp("2023-07-05"), "n": 3230},
            {"signal_date": pd.Timestamp("2023-07-06"), "n": 3300},
        ],
        labels=[
            {"signal_date": pd.Timestamp("2023-07-05"), "n": 3627},
            {"signal_date": pd.Timestamp("2023-07-06"), "n": 3630},
        ],
    )
    intersection, denominator, diagnostics = M.collect(
        {}, date(2023, 7, 1), date(2023, 7, 31), ("a",)
    )
    assert intersection == {date(2023, 7, 5): 3230, date(2023, 7, 6): 3300}
    assert denominator == {date(2023, 7, 5): 3627, date(2023, 7, 6): 3630}
    assert diagnostics["factor_count"] == 1
    assert diagnostics["partitions_resolved"] >= 1
