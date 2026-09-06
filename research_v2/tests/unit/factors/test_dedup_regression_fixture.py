"""C3.4: the deduplicator against the duplicate-key shapes the source was measured to have.

The unit tests cover the three resolutions on hand-built frames and the property tests
cover order and chunk invariance on generated ones. Neither can answer the question C3.4
actually asks, which is what happens on the shapes YesLab really produces -- and the
repository held no such measurement, which is why C3.4 stayed unticked rather than being
filled with an invented fixture that looked sourced.

It is now measured, and the measurement changed the picture. Neither GP_017 nor GP_077
contains a single duplicated key whose rows are all equal: the shape the deduplicator's
own docstring calls normal for YesLab -- "an identical duplicate row for every
security/date key" -- has a measured count of zero in both factors. What the source
actually holds is one value plus one or two NULLs, and for GP_017 also three million keys
where the duplicates genuinely disagree. So these tests assert the class counts the census
measured, not the ones a plausible-looking fixture would have had.

Nothing is copied from the source. Business keys and values are synthetic; what is carried
over is the distribution, and the fixture records the sha256 of the committed census
artifact it was scaled from, which the first test below checks against that artifact.
"""

from __future__ import annotations

import hashlib
import json
from itertools import accumulate
from pathlib import Path
from typing import Any

import build_dedup_regression_fixture as builder
import census_duplicate_key_shapes as census
import pandas as pd
import pytest

from research_v2.factors.deduplication import deduplicate_factor_observations

_HERE = Path(__file__).resolve().parent
_FIXTURE_PATH = _HERE / "data" / "dedup_regression_gp017_gp077_v1.json"
_CENSUS_PATH = _HERE.parents[2] / "artifacts" / "diagnostics" / "duplicate_key_shape_census_v3.json"
BUSINESS_KEYS = ["observation_date", "ts_code", "factor_id", "factor_version"]
COLLAPSED = ("duplicated_identical", "duplicated_single_nonnull")


@pytest.fixture(scope="module")
def fixture_document() -> dict[str, Any]:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def census_document() -> dict[str, Any]:
    return json.loads(_CENSUS_PATH.read_text(encoding="utf-8"))


def _frames(fixture_document: dict[str, Any]) -> dict[str, Any]:
    """One expanded frame per factor, plus the sample it came from."""
    return {
        label: {
            "sample": sample,
            "frame": builder.expand_rows(sample["groups"], factor_id=label),
        }
        for label, sample in fixture_document["factors"].items()
    }


def _keys(frame: Any) -> set[tuple[Any, ...]]:
    return {tuple(row) for row in frame.loc[:, BUSINESS_KEYS].itertuples(index=False, name=None)}


def _measured(**overrides: int) -> dict[str, int]:
    """A small GP_077-shaped census: seven unique keys, ninety-three duplicated by two."""
    values: dict[str, int] = {name: 0 for name in census.METRICS}
    values.update(
        {
            "key_count": 100,
            "row_count": 193,
            "duplicated_key_count": 93,
            "max_multiplicity": 2,
            "unique_key_count": 7,
            "usable_row_count": 75,
            "unique_usable_key_count": 5,
            "unique_unusable_key_count": 2,
            "duplicated_single_nonnull_key_count": 70,
            "duplicated_no_usable_value_key_count": 23,
            "single_nonnull_usable_row_count": 70,
        }
    )
    values.update(overrides)
    return values


def test_the_fixture_is_scaled_from_the_census_committed_beside_it(
    fixture_document: dict[str, Any], census_document: dict[str, Any]
) -> None:
    """The anti-fabrication link: the recorded hash has to match the artifact in the tree.

    C3.4 was left unticked rather than filled with a made-up fixture, so the thing that has
    to hold is that these numbers came from a measurement that is itself committed. Editing
    either file without the other fails here.
    """
    provenance = fixture_document["provenance"]

    assert provenance["census_script"].endswith("census_duplicate_key_shapes.py")
    assert hashlib.sha256(_CENSUS_PATH.read_bytes()).hexdigest() == provenance["census_sha256"]
    assert census_document["self_test"]["status"] == "PASS"
    assert census_document["crossval"]["status"] == "PASS"
    for label, sample in fixture_document["factors"].items():
        measured = census_document["crossval"]["factors"][label]
        assert sample["measured"] == measured["measured"]
        assert sample["source_factor_id"] == measured["factor_id"]
        assert provenance["source_tables"][label] == measured["source_table"]


def test_the_allocation_is_largest_remainder_rounding_recomputed_here(
    fixture_document: dict[str, Any],
) -> None:
    """Recomputed independently rather than by calling the generator that produced it.

    Asking the builder whether it built what it built would pass for any rule at all. This
    reimplements the rounding in four lines so the fixture's shape is checkable against the
    measurement it cites.
    """
    for label, sample in fixture_document["factors"].items():
        total = sample["sample_keys"]
        keys = sample["measured"]["key_count"]
        exact = {
            name: sample["measured"][f"{name}_key_count"] * total / keys for name in builder.CLASSES
        }
        expected = {name: int(value // 1) for name, value in exact.items()}
        ranked = sorted(builder.CLASSES, key=lambda name: (-(exact[name] - expected[name]), name))
        for name in ranked[: total - sum(expected.values())]:
            expected[name] += 1

        assert sample["allocated"] == dict(sorted(expected.items())), label
        assert sum(sample["allocated"].values()) == total
        assert builder.histogram(sample["groups"]) == sample["allocated"], (
            "the groups written to disk are not the classes the allocation asked for"
        )


def test_the_sampled_conflict_share_is_within_one_key_of_the_measured_share(
    fixture_document: dict[str, Any],
) -> None:
    for label, sample in fixture_document["factors"].items():
        measured = sample["measured"]
        wanted = measured["duplicated_conflict_key_count"] / measured["key_count"]
        got = sample["allocated"]["duplicated_conflict"] / sample["sample_keys"]

        assert abs(got - wanted) <= 1 / sample["sample_keys"], label


def test_expansion_gives_one_business_key_per_group_at_the_measured_multiplicity(
    fixture_document: dict[str, Any],
) -> None:
    """Pins the expansion itself, so a shared key between two groups cannot hide a merge."""
    for label, entry in _frames(fixture_document).items():
        sample, frame = entry["sample"], entry["frame"]
        sizes = frame.groupby(BUSINESS_KEYS, sort=False).size()

        assert len(sizes) == len(sample["groups"]), label
        assert list(sizes) == [len(group["values"]) for group in sample["groups"]]
        duplicated = [
            len(group["values"])
            for group in sample["groups"]
            if str(group["class"]).startswith("duplicated")
        ]
        assert set(duplicated) == {sample["duplicated_multiplicity"]}


def test_the_deduplicator_counts_every_class_the_census_says_is_there(
    fixture_document: dict[str, Any],
) -> None:
    """The six measured classes map one to one onto the four published counters.

    ``invalid_groups`` is the sum of two different code paths -- a unique row with no usable
    value, and a duplicated group with none -- which is why both are present in the sample.
    """
    for label, entry in _frames(fixture_document).items():
        allocated = entry["sample"]["allocated"]
        statistics = deduplicate_factor_observations(entry["frame"]).statistics

        assert statistics.conflict_groups == allocated["duplicated_conflict"], label
        assert statistics.identical_groups == allocated["duplicated_identical"], label
        assert statistics.single_nonnull_groups == allocated["duplicated_single_nonnull"], label
        assert statistics.invalid_groups == (
            allocated["unique_unusable"] + allocated["duplicated_no_usable_value"]
        ), label


def test_a_conflicting_key_contributes_no_row_to_clean_and_keeps_all_of_its_rows_in_quarantine(
    fixture_document: dict[str, Any],
) -> None:
    """C3.4's second half, on the factor that was measured to have three million of them."""
    entry = _frames(fixture_document)["GP_017"]
    sample, frame = entry["sample"], entry["frame"]
    assert sample["allocated"]["duplicated_conflict"] > 0, (
        "GP_017 was measured at 33.7% conflicting keys, so a sample without one is not a "
        "sample of GP_017"
    )

    offsets = list(accumulate([0] + [len(group["values"]) for group in sample["groups"]]))
    conflicting = [
        tuple(frame.loc[:, BUSINESS_KEYS].iloc[offsets[index]])
        for index, group in enumerate(sample["groups"])
        if group["class"] == "duplicated_conflict"
    ]

    result = deduplicate_factor_observations(frame)
    published = pd.MultiIndex.from_frame(result.clean.loc[:, BUSINESS_KEYS])
    assert not published.isin(conflicting).any(), "a conflicting key reached clean"

    quarantined = result.quarantine
    held = pd.MultiIndex.from_frame(quarantined.loc[:, BUSINESS_KEYS])
    rows = quarantined.loc[held.isin(conflicting)]
    assert set(rows["reason"]) == {"CONFLICT_QUARANTINE"}
    assert len(rows) == len(conflicting) * sample["duplicated_multiplicity"], (
        "a conflict is quarantined whole; leaving a tie-break survivor behind is the failure "
        "this exists to prevent"
    )
    assert len(_keys(rows)) == len(conflicting)


def test_clean_is_unique_on_the_business_key_and_disjoint_from_quarantine(
    fixture_document: dict[str, Any],
) -> None:
    for label, entry in _frames(fixture_document).items():
        result = deduplicate_factor_observations(entry["frame"])

        assert len(_keys(result.clean)) == len(result.clean), label
        assert _keys(result.clean).isdisjoint(_keys(result.quarantine)), label


def test_no_published_value_was_absent_from_the_input(fixture_document: dict[str, Any]) -> None:
    for label, entry in _frames(fixture_document).items():
        result = deduplicate_factor_observations(entry["frame"])
        offered = {
            float(value).hex() for value in entry["frame"]["raw_value"] if pd.notna(value)
        }

        assert {float(value).hex() for value in result.clean["raw_value"]} <= offered, label


def test_the_rows_that_disappear_are_exactly_the_collapsed_duplicates(
    fixture_document: dict[str, Any],
) -> None:
    """Accounting, so "collapsed" cannot quietly become "dropped something else too".

    A collapsed group publishes one row and quarantines none, so its other rows leave the
    pipeline entirely. That is intended, and it is the only intended loss: every other input
    row has to be in clean or in quarantine.
    """
    for label, entry in _frames(fixture_document).items():
        sample, frame = entry["sample"], entry["frame"]
        result = deduplicate_factor_observations(frame)
        expected_loss = sum(
            len(group["values"]) - 1
            for group in sample["groups"]
            if group["class"] in COLLAPSED
        )

        assert len(frame) - len(result.clean) - len(result.quarantine) == expected_loss, label


def test_gp077_absorbs_ninety_three_percent_duplication_without_a_single_conflict(
    fixture_document: dict[str, Any],
) -> None:
    """The property that makes a heavily duplicated source usable at all.

    GP_077 duplicates 92.7% of its keys and not one of them disagrees with itself, so the
    whole factor collapses to one row per key with at least one usable value. Its naive
    conflict count is 6,764,199 -- exactly its single-non-null count -- because ``nunique``
    counts the NULL in a value-plus-NULL pair as a second value. That equality is the
    measurement that shows the naive predicate's entire error is this shape.
    """
    sample = fixture_document["factors"]["GP_077"]
    measured = sample["measured"]
    assert measured["duplicated_conflict_key_count"] == 0
    assert measured["conflict_key_count_naive"] == measured["duplicated_single_nonnull_key_count"]
    assert measured["duplicated_identical_key_count"] == 0

    result = deduplicate_factor_observations(
        builder.expand_rows(sample["groups"], factor_id="GP_077")
    )
    allocated = sample["allocated"]

    assert result.statistics.conflict_groups == 0
    assert len(result.clean) == (
        allocated["unique_usable"]
        + allocated["duplicated_identical"]
        + allocated["duplicated_single_nonnull"]
    ), "exactly one published row per key that has any usable value"


def test_gp017_has_no_identical_duplicate_at_all_which_contradicts_the_docstring(
    fixture_document: dict[str, Any],
) -> None:
    """Pinned because it is the opposite of what the production code says to expect.

    ``deduplicate_factor_observations``' docstring calls an identical duplicate row for
    every security/date key the normal YesLab case. For GP_017 the measured count of
    duplicated keys whose rows are all equal is zero out of 8,951,808, and the same is true
    of GP_077. IDENTICAL_COLLAPSE does fire on other factors -- the frozen censuses show
    seven where it is the only shape -- so this is not a claim that the branch is dead, only
    that these two factors are not what the docstring describes.
    """
    sample = fixture_document["factors"]["GP_017"]
    measured = sample["measured"]

    assert measured["duplicated_identical_key_count"] == 0
    assert measured["duplicated_key_count"] == 8951808
    assert measured["duplicated_single_nonnull_key_count"] > 0
    assert measured["duplicated_conflict_key_count"] > 0

    result = deduplicate_factor_observations(
        builder.expand_rows(sample["groups"], factor_id="GP_017")
    )

    assert result.statistics.identical_groups == 0
    assert (
        result.statistics.single_nonnull_groups
        == sample["allocated"]["duplicated_single_nonnull"]
    )


def test_a_multiplicity_that_is_not_uniform_is_refused_rather_than_averaged() -> None:
    """The generator will not invent a shape the census did not measure.

    Averaging 2.5 rows per duplicated key into "2" or "3" would produce a fixture that looks
    measured and is not, which is the exact defect C3.4 was left unticked to avoid.
    """
    with pytest.raises(ValueError, match="multiplicity histogram"):
        builder.duplicated_multiplicity(_measured(row_count=194))


def test_a_uniform_multiplicity_that_disagrees_with_the_maximum_is_refused() -> None:
    """Two independent readings of the same thing have to agree, or one of them is wrong."""
    with pytest.raises(ValueError, match="max_multiplicity"):
        builder.duplicated_multiplicity(_measured(max_multiplicity=3))


def test_a_census_with_no_duplicated_keys_cannot_be_sampled() -> None:
    with pytest.raises(ValueError, match="no duplicated keys"):
        builder.duplicated_multiplicity(_measured(duplicated_key_count=0, row_count=7))


def test_the_allocation_does_not_depend_on_the_order_it_is_handed() -> None:
    ratios = builder.class_ratios(_measured())

    forward = builder.allocate(ratios, 37)
    backward = builder.allocate(dict(reversed(list(ratios.items()))), 37)

    assert forward == backward
    assert sum(forward.values()) == 37
