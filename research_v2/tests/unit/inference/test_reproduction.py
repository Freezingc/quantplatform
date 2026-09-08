"""Tests for the prediction-reproduction verifier.

The property that matters most is non-vacuity: a verifier that always passes is worse than none,
because it turns "never checked" into "checked and matched". So the sensitivity cases are pinned
alongside the happy path, including exactness -- there is no float tolerance to hide behind.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from research_v2.inference.reproduction import (
    FoldReproduction,
    ReproductionError,
    _compare_scores,
    _normalized,
)

KEYS = ["signal_date", "ts_code"]
DAYS = [date(2025, 1, 2), date(2025, 1, 3)]


def _scores(values: list[float]) -> pd.DataFrame:
    rows = [
        {"signal_date": day, "ts_code": f"S{i:03d}", "score": value}
        for day in DAYS
        for i, value in enumerate(values[:3])
    ]
    return pd.DataFrame(rows)


def _compare(left: pd.DataFrame, right: pd.DataFrame) -> FoldReproduction:
    return _compare_scores(
        left, right, fold_id="F1", model_artifact_id="m0r-f1", family="LGBM_RANKREG"
    )


def test_identical_scores_reproduce_exactly() -> None:
    frame = _scores([0.25, 0.5, 0.75])
    result = _compare(frame, frame.copy())
    assert result.identical is True
    assert result.max_absolute_difference == 0.0
    assert result.row_count == 6
    assert result.fold_id == "F1"
    assert result.model_artifact_id == "m0r-f1"
    assert result.family == "LGBM_RANKREG"


def test_a_single_changed_value_is_refused() -> None:
    """The core sensitivity check: one row out of six must fail, not be averaged away."""
    stored = _scores([0.25, 0.5, 0.75])
    stored.loc[3, "score"] = 0.99
    with pytest.raises(ReproductionError, match="max absolute difference"):
        _compare(_scores([0.25, 0.5, 0.75]), stored)


def test_the_comparison_has_no_float_tolerance() -> None:
    """A one-ulp difference is still a difference; an epsilon here would hide real divergence."""
    stored = _scores([0.25, 0.5, 0.75])
    stored.loc[0, "score"] = stored.loc[0, "score"] + 1e-16
    with pytest.raises(ReproductionError, match="differ from what bundle") as info:
        _compare(_scores([0.25, 0.5, 0.75]), stored)
    assert "e-16" in str(info.value), "a one-ulp difference must be reported, not tolerated"


def test_a_missing_stored_row_is_refused_with_counts() -> None:
    stored = _scores([0.25, 0.5, 0.75]).iloc[:-1]
    with pytest.raises(ReproductionError, match="row sets differ"):
        _compare(_scores([0.25, 0.5, 0.75]), stored)


def test_an_extra_stored_row_is_refused_with_counts() -> None:
    recomputed = _scores([0.25, 0.5, 0.75]).iloc[:-1]
    with pytest.raises(ReproductionError, match="row sets differ"):
        _compare(recomputed, _scores([0.25, 0.5, 0.75]))


def test_a_permuted_row_order_still_reproduces() -> None:
    """Comparison is on business keys, so row order must not matter."""
    frame = _scores([0.25, 0.5, 0.75])
    shuffled = frame.sample(frac=1.0, random_state=3).reset_index(drop=True)
    assert _compare(frame, shuffled).identical is True


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        pytest.param(lambda f: f.drop(columns=["score"]), "missing columns", id="no_score"),
        pytest.param(lambda f: f.drop(columns=["ts_code"]), "missing columns", id="no_key"),
        pytest.param(
            lambda f: pd.concat([f, f.head(1)], ignore_index=True),
            "duplicate business keys",
            id="duplicates",
        ),
        pytest.param(
            lambda f: f.assign(score=["x"] * len(f)), "non-numeric", id="non_numeric"
        ),
    ],
)
def test_normalized_refuses_a_malformed_score_frame(mutate, match: str) -> None:
    with pytest.raises(ReproductionError, match=match):
        _normalized(mutate(_scores([0.25, 0.5, 0.75])), "recomputed")


def test_normalized_aligns_key_types_so_the_join_is_real() -> None:
    """Timestamps and date objects must meet as one type, or the join silently matches nothing."""
    frame = _scores([0.25, 0.5, 0.75])
    as_timestamps = frame.assign(signal_date=pd.to_datetime(frame["signal_date"]))
    left = _normalized(frame, "recomputed")
    right = _normalized(as_timestamps, "stored")
    assert _compare(left, right).row_count == 6
