from __future__ import annotations

import math
from datetime import date

import pandas as pd
import pytest

from research_v2.evaluation.rank_ic import (
    compute_daily_rank_ic,
    summarize_rank_ic,
    summarize_rank_ic_by_year,
)


def test_rank_ic_is_computed_cross_sectionally_per_signal_date() -> None:
    predictions = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2)] * 3 + [date(2025, 1, 3)] * 3,
            "ts_code": ["a", "b", "c"] * 2,
            "score": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        }
    )
    labels = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2)] * 3 + [date(2025, 1, 3)] * 3,
            "ts_code": ["a", "b", "c"] * 2,
            "raw_return_5d": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
            "label_valid": [True] * 6,
        }
    )

    daily = compute_daily_rank_ic(predictions, labels, minimum_cross_section=3)
    summary = summarize_rank_ic(daily)

    assert list(daily.columns) == ["signal_date", "rank_ic", "sample_count"]
    assert isinstance(daily.loc[0, "signal_date"], date)
    assert daily["rank_ic"].tolist() == pytest.approx([1.0, -1.0])
    assert summary.mean_rank_ic == pytest.approx(0.0)
    assert summary.icir == pytest.approx(0.0)


def test_rank_ic_excludes_invalid_labels_and_undersized_cross_sections() -> None:
    predictions = pd.DataFrame(
        {"signal_date": [date(2025, 1, 2)] * 2, "ts_code": ["a", "b"], "score": [1.0, 2.0]}
    )
    labels = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2)] * 2,
            "ts_code": ["a", "b"],
            "raw_return_5d": [1.0, 2.0],
            "label_valid": [True, False],
        }
    )

    daily = compute_daily_rank_ic(predictions, labels, minimum_cross_section=3)

    assert daily.empty


def test_rank_ic_normalizes_intraday_timestamps_before_matching_business_keys() -> None:
    predictions = pd.DataFrame(
        {
            "signal_date": [pd.Timestamp("2025-01-02 09:30")] * 3,
            "ts_code": ["a", "b", "c"],
            "score": [1.0, 2.0, 3.0],
        }
    )
    labels = pd.DataFrame(
        {
            "signal_date": [pd.Timestamp("2025-01-02 15:00")] * 3,
            "ts_code": ["a", "b", "c"],
            "raw_return_5d": [1.0, 2.0, 3.0],
            "label_valid": [True] * 3,
        }
    )

    daily = compute_daily_rank_ic(predictions, labels, minimum_cross_section=3)

    assert daily["rank_ic"].tolist() == pytest.approx([1.0])
    assert daily.loc[0, "signal_date"] == date(2025, 1, 2)


def test_rank_ic_reports_score_health_and_excludes_collapsed_days() -> None:
    predictions = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2)] * 30 + [date(2025, 1, 3)] * 30,
            "ts_code": [f"S{index:03d}" for index in range(30)] * 2,
            "score": list(range(30)) + [1.0] * 30,
        }
    )
    labels = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2)] * 30 + [date(2025, 1, 3)] * 30,
            "ts_code": [f"S{index:03d}" for index in range(30)] * 2,
            "raw_return_5d": list(range(30)) * 2,
            "label_valid": [True] * 60,
        }
    )

    daily = compute_daily_rank_ic(
        predictions,
        labels,
        minimum_cross_section=30,
        include_invalid=True,
        require_score_health=True,
    )
    summary = summarize_rank_ic(daily)
    yearly = summarize_rank_ic_by_year(daily)

    assert daily["coverage"].tolist() == pytest.approx([1.0, 1.0])
    assert daily["prediction_unique_count"].tolist() == [30, 1]
    assert daily.loc[0, "prediction_std"] > 1e-6
    assert daily.loc[0, "rank_ic_valid"]
    assert not daily.loc[1, "rank_ic_valid"]
    assert pd.isna(daily.loc[1, "rank_ic"])
    assert summary.valid_day_count == 1
    assert summary.invalid_day_count == 1
    assert yearly.loc[0, "year"] == 2025
    assert yearly.loc[0, "valid_day_count"] == 1
    assert yearly.loc[0, "invalid_day_count"] == 1
    assert yearly.loc[0, "mean_rank_ic"] == pytest.approx(1.0)
    assert pd.isna(yearly.loc[0, "rank_ic_std"])
    assert pd.isna(yearly.loc[0, "icir"])
    assert yearly.loc[0, "positive_day_share"] == pytest.approx(1.0)
    assert yearly.loc[0, "mean_sample_count"] == pytest.approx(30.0)
    assert yearly.loc[0, "sample_count_p05"] == pytest.approx(30.0)
    assert yearly.loc[0, "sample_count_median"] == pytest.approx(30.0)
    assert yearly.loc[0, "sample_count_p95"] == pytest.approx(30.0)


def test_rank_ic_excludes_near_collapsed_scores_from_metric_evidence() -> None:
    predictions = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2)] * 30,
            "ts_code": [f"S{index:03d}" for index in range(30)],
            "score": [float(index % 9) for index in range(30)],
        }
    )
    labels = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2)] * 30,
            "ts_code": [f"S{index:03d}" for index in range(30)],
            "raw_return_5d": list(range(30)),
            "label_valid": [True] * 30,
        }
    )

    daily = compute_daily_rank_ic(
        predictions,
        labels,
        minimum_cross_section=30,
        include_invalid=True,
    )

    assert daily.loc[0, "prediction_unique_count"] == 9
    assert not daily.loc[0, "rank_ic_valid"]
    assert pd.isna(daily.loc[0, "rank_ic"])


# --- the arithmetic everything else is computed from -------------------------
#
# gates.py decides, model_evaluator.py assembles the evidence, and every number both
# of them use is computed here: the daily cross-sectional Spearman RankIC behind G1 and
# G3, the series the HAC test turns into G2, and the coverage / unique-score /
# dispersion triple behind G6. Each test below corresponds to a mutation of this file
# that survived a full measurement round, or that was killed for a reason which was not
# the property under test, and each asserts an observed difference.


def test_coverage_counts_a_label_without_a_prediction_as_unscored() -> None:
    """An unscored label is missing coverage, not a row that never existed.

    Probed in both directions. Joining labels to predictions with how="inner" instead of
    how="left" drops the three unscored labels outright, so the day reports coverage 1.0
    and valid_label_count 27 instead of 0.9 and 30 -- a day that scored 90% of its
    labelled universe reports as perfectly covered, and coverage is one of the three
    per-day inputs to G6. Dividing by eligible_count rather than valid_label_count is the
    same mistake the other way: it charges the ten rows whose labels were never valid
    against the model's coverage.
    """
    day = date(2025, 1, 2)
    codes = [f"S{index:03d}" for index in range(40)]
    labels = pd.DataFrame(
        {
            "signal_date": [day] * 40,
            "ts_code": codes,
            # Thirty genuinely labelled rows, then ten whose label was never valid.
            "raw_return_5d": [float(index) for index in range(30)] + [float("nan")] * 10,
            "label_valid": [True] * 30 + [False] * 10,
        }
    )
    predictions = pd.DataFrame(
        {
            "signal_date": [day] * 27,
            "ts_code": codes[:27],  # three valid labels have no prediction at all
            "score": [float(index) for index in range(27)],
        }
    )

    daily = compute_daily_rank_ic(
        predictions,
        labels,
        minimum_cross_section=27,
        include_invalid=True,
        require_score_health=False,
    )

    assert daily.loc[0, "eligible_count"] == 40
    assert daily.loc[0, "valid_label_count"] == 30
    assert daily.loc[0, "sample_count"] == 27
    assert daily.loc[0, "coverage"] == pytest.approx(27 / 30)
    # Guard the guard: a real RankIC came out of those 27 stocks.
    assert bool(daily.loc[0, "rank_ic_valid"])


def test_rank_ic_averages_tied_ranks_on_both_sides_not_just_the_score_side() -> None:
    """Both tie-breaks are average; only the score side was pinned before this test.

    This is the day-level twin of the tie-break chain 2 pinned in predictions_to_frame,
    and it was half-pinned the same way chain 2's eval() asymmetry was: breaking SCORE
    ties by row position is caught by a golden factor test, breaking LABEL ties by row
    position was caught by nothing.

    With thirty stocks whose values rise strictly except for one tied pair, averaged ties
    give a RankIC just under 1.0; breaking ties by position makes the two rank vectors
    identical and reports exactly 1.0 -- a perfect correlation manufactured out of row
    order. Both sides are run, because each fixture is blind to the other's mutation.
    """
    day = date(2025, 1, 2)
    codes = [f"S{index:03d}" for index in range(30)]
    strict = [float(index) for index in range(30)]
    tied = [1.0, 1.0] + [float(index) for index in range(2, 30)]

    for name, scores, returns in (("score tie", tied, strict), ("label tie", strict, tied)):
        predictions = pd.DataFrame({"signal_date": [day] * 30, "ts_code": codes, "score": scores})
        labels = pd.DataFrame(
            {
                "signal_date": [day] * 30,
                "ts_code": codes,
                "raw_return_5d": returns,
                "label_valid": [True] * 30,
            }
        )

        daily = compute_daily_rank_ic(
            predictions, labels, minimum_cross_section=30, include_invalid=True
        )

        # Guard the guard: exactly one tied pair on exactly one side, so the two
        # tie-break methods must diverge -- 29 distinct values where tied, 30 where strict.
        assert sorted((len(set(scores)), len(set(returns)))) == [29, 30], name
        assert bool(daily.loc[0, "rank_ic_valid"]), name
        assert daily.loc[0, "rank_ic"] > 0.99, name
        assert daily.loc[0, "rank_ic"] < 1.0, name


def test_icir_is_undefined_rather_than_a_division_by_zero_when_every_day_agrees() -> None:
    """Zero RankIC dispersion must give NaN, and this asserts that instead of crashing.

    Dropping the ``std > 0`` condition is already caught -- by twelve tests failing with
    ``ZeroDivisionError``, because single-valid-day summaries are common in the suite and
    two identical RankIC values give std 0.0. That is a crash, not an assertion about
    ICIR, so it pins "does not divide by zero" and not "reports ICIR as undefined"; the
    same distinction 续五十二 found in chain 2, where a mutant died on a KeyError before
    the semantics under test were reached. With the guard relaxed to allow a zero
    denominator without crashing, ICIR would become ``inf`` -- a value that survives
    json.dumps, prints as a number in a report, and compares greater than every finite
    threshold it is ever held against.
    """
    frames = []
    for day in (date(2025, 1, 2), date(2025, 1, 3)):
        codes = ["a", "b", "c", "d"]
        frames.append(
            compute_daily_rank_ic(
                pd.DataFrame(
                    {"signal_date": [day] * 4, "ts_code": codes, "score": [1.0, 2.0, 3.0, 4.0]}
                ),
                pd.DataFrame(
                    {
                        "signal_date": [day] * 4,
                        "ts_code": codes,
                        "raw_return_5d": [1.0, 2.0, 3.0, 4.0],
                        "label_valid": [True] * 4,
                    }
                ),
                minimum_cross_section=4,
                require_score_health=False,
            )
        )

    summary = summarize_rank_ic(pd.concat(frames, ignore_index=True))

    assert summary.valid_day_count == 2
    assert summary.mean_rank_ic == pytest.approx(1.0)
    # Guard the guard: the dispersion really is zero, which is the case under test.
    assert summary.rank_ic_std == pytest.approx(0.0)
    assert math.isnan(summary.icir)


def test_a_non_finite_rank_ic_never_enters_the_mean_even_if_it_is_marked_valid() -> None:
    """The valid-day mask is the flag AND finiteness, and the second half is load-bearing.

    ``compute_daily_rank_ic`` never emits rank_ic_valid=True beside a non-finite rank_ic,
    so this is unreachable on the production path -- but ``summarize_rank_ic`` is public
    and accepts any daily frame. Probed: with the finiteness test dropped, a hand-built
    frame carrying inf marked valid reports valid_day_count 2 and mean_rank_ic inf instead
    of 1 and 0.10. G1 would still refuse it, because gates.py requires a finite mean and
    that requirement is pinned -- but the inf reaches the run report and ICIR first.
    """
    daily = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2), date(2025, 1, 3)],
            "rank_ic": [float("inf"), 0.10],
            "sample_count": [30, 30],
            "rank_ic_valid": [True, True],
        }
    )

    summary = summarize_rank_ic(daily)

    assert summary.valid_day_count == 1
    assert summary.invalid_day_count == 1
    assert summary.mean_rank_ic == pytest.approx(0.10)
    assert math.isnan(summary.icir)  # one valid day, so the dispersion is undefined


def test_a_day_with_exactly_zero_rank_ic_is_not_a_positive_day() -> None:
    """Zero is not positive at day level, exactly as G3 treats a zero year.

    Four stocks whose label ranks are [2, 4, 1, 3] against score ranks [1, 2, 3, 4] are
    orthogonal, so the day's RankIC is exactly 0.0 and positive_day_share must be 0.0.
    Counting zero as positive reports 1.0: a day on which the signal predicted nothing
    reads as a day on which it worked. This is the same comparator question chain 3 pinned
    for G3's yearly ratio, one level down.
    """
    day = date(2025, 1, 2)
    codes = ["a", "b", "c", "d"]
    predictions = pd.DataFrame(
        {"signal_date": [day] * 4, "ts_code": codes, "score": [1.0, 2.0, 3.0, 4.0]}
    )
    labels = pd.DataFrame(
        {
            "signal_date": [day] * 4,
            "ts_code": codes,
            "raw_return_5d": [2.0, 4.0, 1.0, 3.0],
            "label_valid": [True] * 4,
        }
    )

    summary = summarize_rank_ic(
        compute_daily_rank_ic(
            predictions, labels, minimum_cross_section=4, require_score_health=False
        )
    )

    assert summary.valid_day_count == 1
    # Guard the guard: the day really is exactly zero, not merely small.
    assert summary.mean_rank_ic == pytest.approx(0.0, abs=1e-12)
    assert summary.positive_day_share == pytest.approx(0.0)


def test_the_yearly_summary_is_ordered_by_year_whatever_order_the_days_arrive() -> None:
    """The docstring promises a deterministic by-year summary, which means sorted.

    Probed: ``groupby(..., sort=False)`` emits years in first-seen order. Frames coming
    out of ``compute_daily_rank_ic`` are already date-sorted, so on that path the
    difference is invisible -- but ``summarize_rank_ic_by_year`` is public, accepts any
    daily frame, and its rows land in run reports, where a reordered block reads as a
    changed result.
    """
    daily = pd.DataFrame(
        {
            "signal_date": [date(2021, 1, 4), date(2020, 1, 2), date(2021, 1, 5), date(2020, 1, 3)],
            "rank_ic": [0.1, 0.2, 0.3, 0.4],
            "sample_count": [30, 30, 30, 30],
        }
    )

    yearly = summarize_rank_ic_by_year(daily)

    assert yearly["year"].tolist() == [2020, 2021]
    assert yearly.loc[0, "mean_rank_ic"] == pytest.approx(0.3)
    assert yearly.loc[1, "mean_rank_ic"] == pytest.approx(0.2)


def test_a_cross_section_floor_below_two_is_refused() -> None:
    """A one-stock cross-section has no rank correlation, so the floor is a contract.

    Probed: with the guard removed a floor of 1 is accepted, and a floor of 0 or -5
    silently means "no floor at all". The single-stock case happens to be harmless
    numerically -- one observation correlates to NaN and is reported as an invalid day --
    but the caller's mistake stops being named, and nothing else in the function checks
    the argument.
    """
    day = date(2025, 1, 2)
    predictions = pd.DataFrame({"signal_date": [day], "ts_code": ["a"], "score": [1.0]})
    labels = pd.DataFrame(
        {"signal_date": [day], "ts_code": ["a"], "raw_return_5d": [1.0], "label_valid": [True]}
    )

    for floor in (1, 0, -5):
        with pytest.raises(ValueError, match="at least 2"):
            compute_daily_rank_ic(predictions, labels, minimum_cross_section=floor)
