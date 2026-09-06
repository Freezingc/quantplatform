"""Tests for the per-day usable-share measurement that separates two explanations.

The measurement exists to settle one argument: a dense control factor reads ``median_cov``
1.0018 at the coverage gate and 79.0% publishable in a pooled census of the same window, and
the two candidate explanations -- NULLs concentrated on whole days, or a denominator
difference -- have opposite consequences for whether that factor is a control at all. So the
thing that has to be right is the rule that reads the numbers, and that rule is what most of
these tests pin.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date
from typing import Any

import measure_daily_usable_share as daily
import pandas as pd
import pytest

from research_v2.db.session import ReadOnlyViolation, YesLabReadOnlySession

_AGGREGATE_HEAD = re.compile(r"^(count|sum|max|min|avg|median|percentile|var|stddev)\(")


def _metrics(**overrides: float) -> dict[str, float]:
    """A uniformly degraded factor: no empty days, and median equal to pooled."""
    values: dict[str, float] = {
        "day_count": 1216,
        "days_zero_usable": 0,
        "days_below_half": 0,
        "min_daily_share": 0.78,
        "median_daily_share": 0.80,
        "usable_row_count": 5_700_000,
        "row_count": 7_125_000,
        "pooled_share": 0.80,
    }
    values.update(overrides)
    return values


class _RecordingClient:
    def __init__(self, results: Sequence[Any]) -> None:
        self.scripts: list[str] = []
        self._results = list(results)

    def run(self, script: str) -> object:
        self.scripts.append(script)
        if not self._results:
            raise AssertionError(f"more queries were issued than supplied: {script}")
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def test_uniform_degradation_is_called_a_denominator_artifact() -> None:
    verdict = daily.classify(_metrics())

    assert verdict.startswith("UNIFORM_ACROSS_DAYS")
    assert "denominator gap" in verdict


def test_chronic_sparsity_that_is_even_across_days_is_not_called_a_hole() -> None:
    """The case that broke the first version of the rule, taken from real output.

    GP_003 over 2020-2024 has 879 of 1216 days below half usable and not one empty day, and
    its median (0.361) and pooled share (0.380) agree. A rule keyed on the day counts called
    that WHOLE_DAY_HOLES and went on to say "a median over days reads better than the window",
    which was false for this factor: the two agree, so its sparsity is chronic and even. The
    discriminator has to be the median-versus-pooled gap, with the day counts reported as
    detail that qualifies the verdict rather than deciding it.
    """
    verdict = daily.classify(
        _metrics(
            median_daily_share=0.361,
            pooled_share=0.380,
            min_daily_share=0.021,
            days_below_half=879,
        )
    )

    assert verdict.startswith("UNIFORM_ACROSS_DAYS"), verdict
    assert "879 of 1216 days below half" in verdict, "the counts still have to be visible"


def test_empty_days_show_up_as_skew_because_they_move_the_median_off_the_window() -> None:
    """The shape Task 3.1b recorded: eleven bad days in five years, and a gate blind to them."""
    verdict = daily.classify(
        _metrics(days_zero_usable=11, median_daily_share=0.995, pooled_share=0.79)
    )

    assert verdict.startswith("DAY_LEVEL_SKEW")
    assert "11 empty days" in verdict


def test_a_minority_of_bad_days_is_skew_even_when_none_of_them_is_empty() -> None:
    verdict = daily.classify(_metrics(days_below_half=40, median_daily_share=0.95))

    assert verdict.startswith("DAY_LEVEL_SKEW")
    assert "40 of 1216 days below half" in verdict


def test_the_verdict_carries_the_worst_day_so_skew_cannot_hide_how_bad_it_gets() -> None:
    """Skew and uniform are the two verdicts; how bad the worst day is belongs in both.

    Without it, "uniform across days" reads as "fine", when a factor can be uniformly thin
    every single day -- which is what the six sparse frozen factors are.
    """
    uniform = daily.classify(_metrics(min_daily_share=0.31))
    skewed = daily.classify(_metrics(median_daily_share=0.95, min_daily_share=0.02))

    assert "worst day 0.310" in uniform
    assert "worst day 0.020" in skewed


def test_a_gate_number_is_appended_as_a_ratio_rather_than_compared_by_a_threshold() -> None:
    """The gate's own median is the thing to compare against, so the rule must not invent one.

    Whether a census median of 0.807 contradicts a gate median_cov of 1.0018 is not decidable
    by a threshold inside this function; the ratio is what a reader needs, because a constant
    ratio across factors with very different coverage is a denominator effect and a
    factor-specific ratio is not.
    """
    verdict = daily.classify(_metrics(), gate_median_cov=1.0018)

    assert "gate median_cov 1.0018" in verdict
    assert "this median / gate = 0.799" in verdict
    assert "this median / gate" not in daily.classify(_metrics()), (
        "with no gate number supplied there is nothing to ratio against, and inventing one "
        "would be the rule deciding the question the measurement was taken to ask"
    )


def test_the_uniform_branch_has_a_tolerance_and_it_is_not_the_whole_gap() -> None:
    """Two points either side of the threshold, so the threshold is pinned and not incidental."""
    assert daily.classify(_metrics(median_daily_share=0.815)).startswith("UNIFORM_ACROSS_DAYS")
    assert daily.classify(_metrics(median_daily_share=0.83)).startswith("DAY_LEVEL_SKEW")


def test_pooled_share_is_derived_not_queried() -> None:
    measured = daily.read_metrics(pd.DataFrame([{name: 1.0 for name in daily.METRICS}]))

    assert measured["pooled_share"] == 1.0
    assert set(daily.METRICS) <= set(measured)
    assert "pooled_share" not in daily.METRICS, (
        "it is a ratio of two queried totals, so the database must not be asked for it"
    )


def test_the_client_may_answer_with_a_frame_a_mapping_or_a_column_list() -> None:
    row = {name: float(index + 1) for index, name in enumerate(daily.METRICS)}
    frame = pd.DataFrame([row])
    columns = [pd.Series([row[name]]) for name in daily.METRICS]

    assert daily.read_metrics(frame) == daily.read_metrics(dict(row))
    assert daily.read_metrics(frame) == daily.read_metrics(columns)


def test_every_reported_metric_is_an_aggregate_so_nothing_can_return_rows() -> None:
    offenders = {
        name: expression
        for name, expression in daily._METRIC_EXPRESSIONS.items()
        if not _AGGREGATE_HEAD.match(expression)
    }

    assert offenders == {}
    assert tuple(daily._METRIC_EXPRESSIONS) == daily.METRICS


def test_the_query_passes_the_read_only_gate_and_ends_in_one_projection() -> None:
    client = _RecordingClient([pd.DataFrame([{name: 1.0 for name in daily.METRICS}])])
    source = YesLabReadOnlySession(client)

    measured, scripts = daily.measure_factor(
        source,
        table="factor_data10",
        factor_name="羊群效应_CSAD波动率21",
        start=date(2020, 1, 1),
        end=date(2024, 12, 31),
    )

    assert client.scripts == scripts
    assert measured["pooled_share"] == 1.0
    script = scripts[0]
    assert script.count("\nselect ") == 1
    assert "date('2020.01.01') and date('2024.12.31')" in script, (
        "an unbounded run would mix the screening window with years nobody is arguing about"
    )
    assert "羊群效应_CSAD波动率21" in script


def test_a_mutating_script_would_be_refused_by_the_same_gate() -> None:
    """Control: passing the gate means something only because failing it is possible."""
    client = _RecordingClient([])
    source = YesLabReadOnlySession(client)

    with pytest.raises(ReadOnlyViolation):
        source.query(
            "raw = select value from loadTable('dfs://YesLab','factor_data10')\n"
            "tableinsert(loadTable('dfs://YesLab','factor_data10'), raw)"
        )

    assert client.scripts == []


@pytest.mark.parametrize(
    ("table", "factor_name"),
    [
        ("factor_data10'", "GP_077"),
        ("factor_data10; drop table x", "GP_077"),
        ("factor_data10", "GP_077'; drop table x"),
        ("factor_data10", "GP_077\nselect 1"),
    ],
)
def test_an_unsafe_table_or_factor_name_is_refused_before_any_query_exists(
    table: str, factor_name: str
) -> None:
    client = _RecordingClient([])
    source = YesLabReadOnlySession(client)

    with pytest.raises(ValueError, match="unsafe"):
        daily.measure_factor(
            source,
            table=table,
            factor_name=factor_name,
            start=date(2020, 1, 1),
            end=date(2024, 12, 31),
        )

    assert client.scripts == []
