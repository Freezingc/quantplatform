"""Tests for the read-only duplicate-key census that C3.4's fixture is built from.

The census is the provenance of a regression fixture, so two of its promises matter more
than its numbers and are pinned here rather than left to the docstring: it must not be
able to return source rows, and its own internal-consistency check must actually be able
to fail. A measurement whose self-check cannot fail is not a check, and a fixture built
on it would carry the error into the test suite looking like evidence.

The numbers themselves are validated against the live source by the script's
``--scope self-test`` and ``--scope crossval``, whose results are archived with the
fixture; nothing here reaches a database.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import census_duplicate_key_shapes as census
import pandas as pd
import pytest

from research_v2.db.session import ReadOnlyViolation, YesLabReadOnlySession

_AGGREGATE_HEAD = re.compile(r"^(count|sum|max|min|avg|median|percentile|var|stddev)\(")
_YEAR_IN_SCRIPT = re.compile(r"date\('(\d{4})\.01\.01'\)")


def _metrics(**overrides: int) -> dict[str, int]:
    """A consistent census of eight keys: two unique, six duplicated, one of them a conflict.

    unique_usable 1 + unique_unusable 1 + duplicated_identical 2 + duplicated_single_nonnull 1
    + duplicated_conflict 1 + duplicated_no_usable 2 == key_count 8, which is the same
    partition the script's own synthetic self-test is hand-computed to.
    """
    values: dict[str, int] = {name: 0 for name in census.METRICS}
    values.update(
        {
            "key_count": 8,
            "row_count": 15,
            "duplicated_key_count": 6,
            "max_multiplicity": 3,
            "conflict_key_count": 1,
            "conflict_key_count_naive": 2,
            "no_usable_value_key_count": 3,
            "duplicated_no_usable_value_key_count": 2,
            "unique_key_count": 2,
            "usable_row_count": 9,
            "unique_usable_key_count": 1,
            "unique_unusable_key_count": 1,
            "duplicated_identical_key_count": 2,
            "duplicated_single_nonnull_key_count": 1,
            "duplicated_conflict_key_count": 1,
        }
    )
    values.update(overrides)
    return values


class _RecordingClient:
    """Stands in for the DolphinDB client, recording every script it is handed."""

    def __init__(self, results: Sequence[Any]) -> None:
        self.scripts: list[str] = []
        self._results = list(results)

    def run(self, script: str) -> object:
        self.scripts.append(script)
        if not self._results:
            raise AssertionError(f"the census issued more queries than the test supplied: {script}")
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class _YearAwareClient:
    """Answers a per-year census query from a mapping, and can fail the whole-range one."""

    def __init__(self, per_year: dict[int, dict[str, int]], *, fail_whole_range: bool) -> None:
        self.scripts: list[str] = []
        self._per_year = per_year
        self._fail_whole_range = fail_whole_range

    def run(self, script: str) -> object:
        self.scripts.append(script)
        match = _YEAR_IN_SCRIPT.search(script)
        if match is None:
            if self._fail_whole_range:
                raise RuntimeError("Server response: 'Out of memory. RefId:S00000'")
            return pd.DataFrame([_metrics()])
        year = int(match.group(1))
        empty = {name: 0 for name in census.METRICS}
        return pd.DataFrame([self._per_year.get(year, empty)])


def test_every_reported_metric_is_an_aggregate_so_the_census_cannot_return_rows() -> None:
    """The whole design of this script is that a 27M-row factor costs one row of traffic.

    Checked over the metric expressions rather than over the script text, because the text
    legitimately mentions ``value`` while building the per-key aggregate. A metric that is
    not an aggregate -- ``"example": "rv"`` -- would turn the final projection into a row
    extract, and the docstring's promise would be the only thing standing in its way.
    """
    offenders = {
        name: expression
        for name, expression in census._METRIC_EXPRESSIONS.items()
        if not _AGGREGATE_HEAD.match(expression)
    }

    assert offenders == {}
    assert tuple(census._METRIC_EXPRESSIONS) == census.METRICS
    assert len(census.METRICS) == 15, (
        "a new metric has to be hand-computed into SELF_TEST_EXPECTED and into the "
        "partition the fixture is built from, so adding one is a deliberate act"
    )


def test_the_census_script_passes_the_read_only_gate_and_projects_one_row() -> None:
    """The project's own gate is the authority on whether a script may touch the source."""
    client = _RecordingClient([pd.DataFrame([_metrics()])])
    source = YesLabReadOnlySession(client)
    script = census.build_script(
        "loadTable('dfs://YesLab','factor_data8') where factor_name = 'GP_006'"
    )

    measured = census.read_metrics(source.query(script))

    assert client.scripts == [script]
    assert measured == _metrics()
    assert script.count("\nselect ") == 1, "one projection, at the end, over the per-key aggregate"


def test_a_mutating_census_script_would_be_refused_by_the_same_gate() -> None:
    """Control for the test above: passing the gate means something only if it can fail."""
    client = _RecordingClient([])
    source = YesLabReadOnlySession(client)

    with pytest.raises(ReadOnlyViolation):
        source.query(
            "raw = select value from loadTable('dfs://YesLab','factor_data8')\n"
            "tableinsert(loadTable('dfs://YesLab','factor_data8'), raw)"
        )

    assert client.scripts == []


def test_a_consistent_census_reports_no_partition_violation() -> None:
    assert census.check_partition(_metrics()) == ()


@pytest.mark.parametrize(
    ("broken", "fragment"),
    [
        # Each case isolates exactly one violation. Bumping a cell alone would also break
        # the six-cell total, and a checker that reports the total every time looks like
        # one that reports the specific problem, so the totals are bumped with it.
        ({"key_count": 9}, "not to key_count 9"),
        ({"unique_usable_key_count": 2, "key_count": 9}, "the two unique cells sum to 3"),
        (
            {"duplicated_identical_key_count": 3, "key_count": 9},
            "the four duplicated cells sum to 7",
        ),
        ({"conflict_key_count": 2}, "the two conflict counts must agree"),
    ],
)
def test_each_way_the_partition_can_fail_to_add_up_is_named(
    broken: dict[str, int], fragment: str
) -> None:
    """Every one of the four checks is shown to fire on its own."""
    violations = census.check_partition(_metrics(**broken))

    assert len(violations) == 1, violations
    assert fragment in violations[0]


def test_the_client_may_answer_with_a_frame_a_mapping_or_a_bare_column_list() -> None:
    """The DolphinDB client's return shape depends on the query, and all three occur here.

    A ``select`` over an aggregate of an in-memory vector comes back as a list of
    one-element arrays rather than a frame -- observed while probing the NULL semantics
    this census depends on -- so accepting only a DataFrame would fail on the self-test.
    """
    expected = _metrics()
    as_frame = pd.DataFrame([expected])
    as_columns = [pd.Series([expected[name]]) for name in census.METRICS]

    assert census.read_metrics(as_frame) == expected
    assert census.read_metrics(dict(expected)) == expected
    assert census.read_metrics(as_columns) == expected


def test_a_wrongly_sized_column_list_is_refused_rather_than_zipped_into_nonsense() -> None:
    with pytest.raises(RuntimeError, match="returned 3 values, want 15"):
        census.read_metrics([1, 2, 3])


def test_a_server_refusal_falls_back_to_one_query_per_year() -> None:
    """The batched query exhausted the server's memory once; one factor per year did not.

    Years partition the (ts_code, trade_date) key space, so every count adds -- except the
    maximum multiplicity, which has to be maximised. Getting that wrong would understate
    the worst duplicate group in the source, and the fixture's multiplicity comes from it.
    """
    per_year = {
        2020: _metrics(key_count=5, row_count=9, duplicated_key_count=3, max_multiplicity=3),
        2021: _metrics(key_count=3, row_count=6, duplicated_key_count=3, max_multiplicity=2),
    }
    client = _YearAwareClient(per_year, fail_whole_range=True)
    source = YesLabReadOnlySession(client)

    measured, evidence = census.census_factor(
        source, table="factor_data8", factor_name="GP_006", fallback_years=(2020, 2021)
    )

    assert evidence["chunked_by_year"] is True
    assert "whole_range_error" in evidence
    assert len(evidence["queries"]) == 3, "one refused whole-range query plus one per year"
    assert measured["key_count"] == 8
    assert measured["row_count"] == 15
    assert measured["duplicated_key_count"] == 6
    assert measured["max_multiplicity"] == 3, "maximised across years, not summed"


def test_the_chunked_fallback_still_checks_its_own_partition() -> None:
    """Summing years can produce a census no single year had, so the check runs again."""
    per_year = {
        2020: _metrics(key_count=5),
        2021: _metrics(key_count=3),
    }
    client = _YearAwareClient(per_year, fail_whole_range=True)
    source = YesLabReadOnlySession(client)

    _, evidence = census.census_factor(
        source, table="factor_data8", factor_name="GP_006", fallback_years=(2020, 2021)
    )

    assert evidence["partition_violations"], "eight keys doubled by summing cannot still add up"


@pytest.mark.parametrize(
    ("table", "factor_name"),
    [
        ("factor_data8'", "GP_006"),
        ("factor_data8; drop table x", "GP_006"),
        ("factor_data8", "GP_006'; drop table x"),
        ("factor_data8", "GP_006\nselect 1"),
    ],
)
def test_an_unsafe_table_or_factor_name_is_refused_before_any_query_exists(
    table: str, factor_name: str
) -> None:
    client = _RecordingClient([])
    source = YesLabReadOnlySession(client)

    with pytest.raises(ValueError, match="unsafe"):
        census.census_factor(source, table=table, factor_name=factor_name)

    assert client.scripts == [], "the guard has to run before the script is built, not after"
