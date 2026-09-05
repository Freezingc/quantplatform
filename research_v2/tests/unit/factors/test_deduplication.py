from __future__ import annotations

import math
from time import perf_counter

import numpy as np
import pandas as pd

from research_v2.factors.deduplication import deduplicate_factor_observations


def _row(value: float, source_row_id: str) -> dict[str, object]:
    return {
        "observation_date": "2025-01-02",
        "ts_code": "000001.SZ",
        "factor_id": "yeslab_1",
        "factor_version": 1,
        "raw_value": value,
        "source_row_id": source_row_id,
    }


def test_deduplication_collapses_identical_and_preserves_single_nonnull_value() -> None:
    observations = pd.DataFrame(
        [
            _row(1.0, "a"),
            _row(1.0, "b"),
            {**_row(2.0, "c"), "ts_code": "000002.SZ"},
            {**_row(math.nan, "d"), "ts_code": "000002.SZ"},
        ]
    )

    result = deduplicate_factor_observations(observations)

    assert result.quarantine.empty
    assert result.clean["dedup_resolution"].tolist() == ["IDENTICAL_COLLAPSE", "SINGLE_NONNULL"]
    assert result.clean["raw_value"].tolist() == [1.0, 2.0]


def test_deduplication_quarantines_conflicting_finite_values_without_choosing_one() -> None:
    observations = pd.DataFrame([_row(1.0, "a"), _row(2.0, "b")])

    result = deduplicate_factor_observations(observations)

    assert result.clean.empty
    assert len(result.quarantine) == 2
    assert set(result.quarantine["reason"]) == {"CONFLICT_QUARANTINE"}
    assert result.statistics.conflict_groups == 1


def test_deduplication_handles_large_unique_cross_section_without_group_loops() -> None:
    count = 10_000
    observations = pd.DataFrame(
        {
            "observation_date": ["2025-01-02"] * count,
            "ts_code": [f"{index:06d}.SZ" for index in range(count)],
            "factor_id": ["yeslab_1"] * count,
            "factor_version": [1] * count,
            "raw_value": [1.0] * count,
            "source_row_id": [f"source-{index}" for index in range(count)],
        }
    )
    started_at = perf_counter()

    result = deduplicate_factor_observations(observations)

    assert len(result.clean) == count
    assert perf_counter() - started_at < 1.0


# --- C3.1: comparison is exact, never a tolerance -------------------------------
#
# The plan requires canonical binary/decimal comparison and forbids a loose isclose,
# because a tolerance would collapse two genuinely different observations and publish
# one of them -- silently resolving a real conflict into an arbitrary choice of whichever
# row sorted first. The implementation compares with nunique on the raw floats, which is
# exact, but nothing pinned that, so adding a tolerance later would have passed every
# existing test.


def test_values_one_ulp_apart_are_a_conflict_not_an_identical_collapse() -> None:
    low = 1.0
    high = math.nextafter(low, math.inf)
    assert high != low, "the fixture must actually differ"

    result = deduplicate_factor_observations(
        pd.DataFrame([_row(low, "a"), _row(high, "b")])
    )

    assert result.clean.empty, "a one-ULP difference was published as if unanimous"
    assert result.statistics.conflict_groups == 1
    assert result.statistics.identical_groups == 0
    assert set(result.quarantine["reason"]) == {"CONFLICT_QUARANTINE"}


def test_a_difference_inside_isclose_default_tolerance_is_still_a_conflict() -> None:
    """1.0 against 1.000001 is well inside numpy's default rtol of 1e-5.

    numpy rather than ``math.isclose``, whose default rel_tol is 1e-09 and would not
    forgive this pair: the risk being pinned is a vectorised dedup reaching for
    ``np.isclose``, which is what a group comparison over a float column would use.
    """
    assert bool(np.isclose(1.0, 1.000001)), "fixture premise: np.isclose would forgive this"
    assert not math.isclose(1.0, 1.000001), "and math.isclose would not, at its 1e-09 default"

    result = deduplicate_factor_observations(
        pd.DataFrame([_row(1.0, "a"), _row(1.000001, "b")])
    )

    assert result.clean.empty
    assert result.statistics.conflict_groups == 1


def test_the_published_value_is_bit_identical_to_the_observed_one() -> None:
    """C3.1 also requires preserving raw precision: no rounding on the way out.

    0.1 + 0.2 is 0.30000000000000004, so comparing by hex rather than by equality catches
    any normalisation that a decimal-looking assertion would wave through.
    """
    value = 0.1 + 0.2
    assert value != 0.3

    result = deduplicate_factor_observations(
        pd.DataFrame([_row(value, "a"), _row(value, "b")])
    )

    assert result.statistics.identical_groups == 1
    published = float(result.clean["raw_value"].iloc[0])
    assert published.hex() == value.hex(), "the published value was not bit-preserved"


def test_quarantine_evidence_keeps_both_conflicting_values_at_full_precision() -> None:
    """The quarantine is the audit trail, so it keeps every original candidate row.

    C3.2 requires the quarantine to hold all original candidates rather than a chosen one.
    At this layer that means both rows survive with their values bit-intact, so an
    investigator can see the actual discrepancy; the aggregated ``distinct_values_json``
    column is added later by the materializer, and its full-precision preservation is
    pinned by the factor golden sample's conflict case.
    """
    low = 1.0
    high = math.nextafter(low, math.inf)

    result = deduplicate_factor_observations(
        pd.DataFrame([_row(low, "a"), _row(high, "b")])
    )

    assert len(result.quarantine) == 2, "a conflicting candidate was dropped from the audit trail"
    recorded = [float(value).hex() for value in result.quarantine["raw_value"]]
    assert sorted(recorded) == sorted([low.hex(), high.hex()])
    assert set(result.quarantine["source_row_id"]) == {"a", "b"}
