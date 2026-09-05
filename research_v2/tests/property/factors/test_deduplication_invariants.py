"""Property evidence that factor deduplication is order-, index- and chunk-invariant (C3.3).

``deduplicate_factor_observations`` is the first thing that touches raw YesLab factor rows,
and every downstream artifact inherits whatever it decides: ``factor_observation_clean``,
the point-in-time dataset, the B0-B3 baselines, and therefore the trust root. The unit
test covers the three resolution classes on hand-built frames, but the property that makes
it safe to run ahead of a fifteen-hour walk-forward is stronger and was unproven -- the
result must not depend on the order rows arrive in, on the DataFrame's index labels, or on
how an extract happens to be batched, and it must never publish a value that was not in the
input.

That matters because YesLab extracts legitimately arrive in different orders (per column,
per month, per factor group), so an order dependence would not be a reproducible failure;
it would be a silent, load-dependent change in the published factor values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from research_v2.factors.deduplication import deduplicate_factor_observations

BUSINESS_KEYS = ["observation_date", "ts_code", "factor_id", "factor_version"]
DATES = [pd.Timestamp("2024-01-02").date(), pd.Timestamp("2024-01-03").date()]
CODES = ["000001.SZ", "600000.SH"]
FACTOR_IDS = ["yeslab_387", "yeslab_484"]
VERSIONS = [1, 2]


@st.composite
def observation_frames(draw, min_rows: int = 1, max_rows: int = 24) -> pd.DataFrame:
    """A long-form extract over 16 business keys, dense enough to collide often.

    Values include NaN and +/-inf on purpose: the function must treat a non-finite value
    as absent rather than as a candidate, and quarantine a group that has no finite value
    at all. ``source_row_id`` is unique per row, as it is in a real extract, because that
    is what breaks ties deterministically.
    """
    row_count = draw(st.integers(min_value=min_rows, max_value=max_rows))
    rows: list[dict[str, object]] = []
    for index in range(row_count):
        rows.append(
            {
                "observation_date": draw(st.sampled_from(DATES)),
                "ts_code": draw(st.sampled_from(CODES)),
                "factor_id": draw(st.sampled_from(FACTOR_IDS)),
                "factor_version": draw(st.sampled_from(VERSIONS)),
                "source_row_id": f"r{index:04d}",
                "raw_value": draw(
                    st.one_of(
                        st.floats(
                            min_value=-1e6, max_value=1e6,
                            allow_nan=False, allow_infinity=False,
                        ),
                        st.just(float("nan")),
                        st.just(float("inf")),
                        st.just(float("-inf")),
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def _key_tuples(frame: pd.DataFrame) -> list[tuple[object, ...]]:
    return [tuple(key) for key in frame.loc[:, BUSINESS_KEYS].to_numpy().tolist()]


def _finite_inputs(frame: pd.DataFrame) -> dict[tuple[object, ...], list[float]]:
    """Every finite input value, keyed by business key, in arrival order."""
    values = pd.to_numeric(frame["raw_value"], errors="coerce").to_numpy(dtype=float)
    grouped: dict[tuple[object, ...], list[float]] = {}
    for key, value in zip(_key_tuples(frame), values, strict=True):
        if np.isfinite(value):
            grouped.setdefault(key, []).append(float(value))
    return grouped


@given(frame=observation_frames(), seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=60, deadline=None)
def test_row_order_does_not_change_the_resolution(frame: pd.DataFrame, seed: int) -> None:
    baseline = deduplicate_factor_observations(frame)

    shuffled = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    permuted = deduplicate_factor_observations(shuffled)

    pd.testing.assert_frame_equal(baseline.clean, permuted.clean)
    pd.testing.assert_frame_equal(baseline.quarantine, permuted.quarantine)
    assert baseline.statistics == permuted.statistics


@given(frame=observation_frames())
@settings(max_examples=60, deadline=None)
def test_index_labels_do_not_change_the_resolution(frame: pd.DataFrame) -> None:
    """A chunked extract reassembled under arbitrary index labels must resolve identically."""
    baseline = deduplicate_factor_observations(frame)

    relabeled = frame.copy()
    relabeled.index = pd.Index([f"chunk-{index}" for index in range(len(relabeled))])
    result = deduplicate_factor_observations(relabeled)

    pd.testing.assert_frame_equal(baseline.clean, result.clean)
    pd.testing.assert_frame_equal(baseline.quarantine, result.quarantine)


@given(frame=observation_frames())
@settings(max_examples=60, deadline=None)
def test_splitting_into_chunks_and_recombining_resolves_identically(frame: pd.DataFrame) -> None:
    """Concatenating the same rows in two chunks is one call over the same multiset."""
    if len(frame) < 2:
        return
    split = len(frame) // 2
    baseline = deduplicate_factor_observations(frame)

    recombined = pd.concat(
        [frame.iloc[split:], frame.iloc[:split]], ignore_index=True
    )
    result = deduplicate_factor_observations(recombined)

    pd.testing.assert_frame_equal(baseline.clean, result.clean)
    pd.testing.assert_frame_equal(baseline.quarantine, result.quarantine)


@given(frame=observation_frames())
@settings(max_examples=60, deadline=None)
def test_adding_an_identical_row_does_not_change_the_published_values(
    frame: pd.DataFrame,
) -> None:
    """An exact duplicate collapses; it must not change what is published for any key."""
    baseline = deduplicate_factor_observations(frame)

    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    result = deduplicate_factor_observations(duplicated)

    before = dict(zip(_key_tuples(baseline.clean), baseline.clean["raw_value"], strict=True))
    after = dict(zip(_key_tuples(result.clean), result.clean["raw_value"], strict=True))
    assert set(after) == set(before)
    for key, value in before.items():
        assert float(after[key]) == pytest.approx(float(value))


@given(frame=observation_frames(min_rows=2))
@settings(max_examples=60, deadline=None)
def test_a_conflicting_value_removes_the_whole_group_from_clean(frame: pd.DataFrame) -> None:
    """Two distinct finite values for one key must quarantine the group, never pick one."""
    baseline = deduplicate_factor_observations(frame)
    if baseline.clean.empty:
        return
    target = baseline.clean.iloc[0]
    key = tuple(target[name] for name in BUSINESS_KEYS)
    conflicting = pd.DataFrame(
        [
            {
                **{name: target[name] for name in BUSINESS_KEYS},
                "source_row_id": "conflict",
                "raw_value": float(target["raw_value"]) + 1.0,
            }
        ]
    )

    result = deduplicate_factor_observations(
        pd.concat([frame, conflicting], ignore_index=True)
    )

    assert key not in set(_key_tuples(result.clean)), "a conflicted key was published"
    assert key in set(_key_tuples(result.quarantine)), "a conflicted key was silently dropped"
    assert result.statistics.conflict_groups >= 1


@given(frame=observation_frames())
@settings(max_examples=60, deadline=None)
def test_the_published_value_is_the_smallest_source_row_id_of_its_group(
    frame: pd.DataFrame,
) -> None:
    """Pin *which* row wins, not merely that the choice is stable.

    Invariance alone would still be satisfied by an implementation that consistently
    picked the largest source row id, or the last arrival after an internal sort. Naming
    the rule means a refactor cannot silently change published factor values while keeping
    every invariance test green.
    """
    result = deduplicate_factor_observations(frame)
    if result.clean.empty:
        return
    finite = frame.copy()
    finite["raw_value"] = pd.to_numeric(finite["raw_value"], errors="coerce")
    finite = finite.loc[np.isfinite(finite["raw_value"].to_numpy(dtype=float))]
    expected = finite.sort_values(
        [*BUSINESS_KEYS, "source_row_id"], kind="mergesort"
    ).drop_duplicates(BUSINESS_KEYS, keep="first")
    expected_values = dict(
        zip(_key_tuples(expected), expected["raw_value"].astype(float), strict=True)
    )

    for key, value in zip(_key_tuples(result.clean), result.clean["raw_value"], strict=True):
        assert float(value) == pytest.approx(expected_values[key])


@given(frame=observation_frames())
@settings(max_examples=60, deadline=None)
def test_no_published_value_was_absent_from_the_input(frame: pd.DataFrame) -> None:
    """The safety property behind all of the above: dedup selects, it never synthesises."""
    result = deduplicate_factor_observations(frame)
    available = _finite_inputs(frame)

    for key, value in zip(_key_tuples(result.clean), result.clean["raw_value"], strict=True):
        assert float(value) in available.get(key, []), f"invented a value for {key}"
