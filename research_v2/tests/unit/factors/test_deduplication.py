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


# --- C3.2's statistics contract, and two C3.1 clauses where the code and the spec differ --------


def _keyed(code: str, value: float, source_row_id: str, **extra: object) -> dict[str, object]:
    return {
        "observation_date": "2025-01-02",
        "ts_code": code,
        "factor_id": "yeslab_1",
        "factor_version": 1,
        "raw_value": value,
        "source_row_id": source_row_id,
        **extra,
    }


def test_statistics_count_groups_not_rows_and_cover_all_four_classes() -> None:
    """C3.2's ``statistics`` is a published number, so all four counts need pinning.

    These are not internal bookkeeping: ``conflict_groups`` is a stored column in the persisted
    quality snapshot, and the other three are the only summary of how much of the source was
    collapsed, kept, quarantined or unusable. ``invalid_groups`` and ``single_nonnull_groups``
    were asserted nowhere in the suite before this, so either could have been silently
    miscounted -- and a miscount reads as "less conflict in the source than there was".

    One group of each class, plus one unique finite row and one unique non-finite row, so the
    per-group (not per-row) counting and the unique-row contribution are both visible:

    ====== ================================================== ==========================
    key    rows                                               expected class
    ====== ================================================== ==========================
    A      1.0, 1.0                                           identical collapse
    B      2.0, NaN                                           single non-null
    C      3.0, 4.0                                           conflict
    D      NaN, NaN                                           duplicate group, no value
    E      5.0 (single row)                                   unique, clean
    F      NaN (single row)                                   unique, invalid
    ====== ================================================== ==========================

    ``invalid_groups`` counts D (a duplicate group with no finite value) plus F (a unique
    non-finite row), which is why it is 2 rather than 1: the two arrive by different code paths
    and only a fixture containing both can pin the sum.
    """
    observations = pd.DataFrame(
        [
            _keyed("000001.SZ", 1.0, "a"),
            _keyed("000001.SZ", 1.0, "b"),
            _keyed("000002.SZ", 2.0, "c"),
            _keyed("000002.SZ", math.nan, "d"),
            _keyed("000003.SZ", 3.0, "e"),
            _keyed("000003.SZ", 4.0, "f"),
            _keyed("000004.SZ", math.nan, "g"),
            _keyed("000004.SZ", math.nan, "h"),
            _keyed("000005.SZ", 5.0, "i"),
            _keyed("000006.SZ", math.nan, "j"),
        ]
    )

    result = deduplicate_factor_observations(observations)

    assert result.statistics.identical_groups == 1
    assert result.statistics.single_nonnull_groups == 1
    assert result.statistics.conflict_groups == 1
    assert result.statistics.invalid_groups == 2

    assert len(result.clean) == 3, "one surviving row each for A, B and E"
    assert result.clean["ts_code"].tolist() == ["000001.SZ", "000002.SZ", "000005.SZ"]
    assert result.clean["dedup_resolution"].tolist() == [
        "IDENTICAL_COLLAPSE",
        "SINGLE_NONNULL",
        "UNIQUE",
    ]
    assert result.clean["raw_value"].tolist() == [1.0, 2.0, 5.0]

    assert len(result.quarantine) == 5, "F alone, plus both rows of C and both rows of D"
    assert set(result.quarantine["ts_code"]) == {"000003.SZ", "000004.SZ", "000006.SZ"}
    assert set(result.quarantine["reason"]) == {"CONFLICT_QUARANTINE", "NO_FINITE_VALUE"}
    assert set(result.quarantine["source_row_id"]) == {"e", "f", "g", "h", "j"}


def test_a_conflict_never_leaves_a_partial_group_behind_in_clean() -> None:
    """C3.1: a conflicted key contributes zero clean rows -- not one, not the survivor of a tie."""
    observations = pd.DataFrame(
        [
            _keyed("000003.SZ", 3.0, "e"),
            _keyed("000003.SZ", 4.0, "f"),
            _keyed("000003.SZ", 4.0, "g"),
            _keyed("000005.SZ", 5.0, "i"),
        ]
    )

    result = deduplicate_factor_observations(observations)

    assert not (result.clean["ts_code"] == "000003.SZ").any()
    assert len(result.quarantine.loc[result.quarantine["ts_code"] == "000003.SZ"]) == 3
    assert result.clean["ts_code"].tolist() == ["000005.SZ"]


def test_identical_collapse_compares_the_value_only_not_every_non_audit_field() -> None:
    """Current behaviour, and a divergence from C3.1 that must not be "fixed" casually.

    C3.1 defines IDENTICAL_COLLAPSE as *all* non-audit fields being identical. The implementation
    compares ``raw_value`` alone, so two rows agreeing on the value but disagreeing on a
    point-in-time field such as ``source_available_date`` are collapsed -- and which row survives
    is decided by the lowest ``source_row_id``, a provenance field with no semantic meaning.
    Silently choosing an availability date is the part that matters, because that date is what
    decides when a value becomes usable downstream.

    Recorded rather than changed. ``deduplicate_factor_observations`` sits on the live
    materialization path (``materializer.py`` calls it), so tightening the comparison would
    reclassify such rows into quarantine and change materialized factor data -- which is exactly
    what must not happen while a bit-for-bit legacy reproduction is in flight.
    """
    observations = pd.DataFrame(
        [
            _keyed("000001.SZ", 1.0, "b", source_available_date="2025-01-03"),
            _keyed("000001.SZ", 1.0, "a", source_available_date="2025-01-02"),
        ]
    )

    result = deduplicate_factor_observations(observations)

    assert result.quarantine.empty, "the differing availability date is not treated as a conflict"
    assert result.statistics.identical_groups == 1
    assert len(result.clean) == 1
    assert result.clean["dedup_resolution"].iloc[0] == "IDENTICAL_COLLAPSE"
    assert result.clean["source_row_id"].iloc[0] == "a"
    assert result.clean["source_available_date"].iloc[0] == "2025-01-02", (
        "the survivor is picked by lowest source_row_id, so the earlier availability date wins "
        "here by accident of provenance ordering rather than by any rule about availability"
    )


def test_two_equal_finite_values_and_a_null_are_labelled_single_nonnull() -> None:
    """Second divergence from C3.1, also recorded rather than changed.

    C3.1 defines SINGLE_NONNULL as *exactly one* finite non-null value with the rest null. The
    implementation labels a group SINGLE_NONNULL whenever it is not a conflict and not every row
    is finite, so a group holding two *equal* finite values plus a null lands there too. The
    published value is unaffected (both finite values are the same, and a differing pair would be
    a conflict), so this is a labelling imprecision rather than a wrong number -- but the label is
    what a reviewer reads to understand why a row survived, and it is persisted.
    """
    observations = pd.DataFrame(
        [
            _keyed("000001.SZ", 1.0, "a"),
            _keyed("000001.SZ", 1.0, "b"),
            _keyed("000001.SZ", math.nan, "c"),
        ]
    )

    result = deduplicate_factor_observations(observations)

    assert result.statistics.single_nonnull_groups == 1
    assert result.statistics.identical_groups == 0, (
        "not counted as an identical collapse, because one row is null"
    )
    assert result.statistics.conflict_groups == 0
    assert len(result.clean) == 1
    assert result.clean["dedup_resolution"].iloc[0] == "SINGLE_NONNULL"
    assert float(result.clean["raw_value"].iloc[0]) == 1.0
