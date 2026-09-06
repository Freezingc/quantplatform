from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from research_v2.evaluation.hac import hac_mean_test


def test_hac_mean_test_uses_configured_overlap_lag() -> None:
    result = hac_mean_test(pd.Series([0.01, 0.02, 0.03, 0.02, 0.01, 0.02, 0.03]), max_lags=5)

    assert result.observation_count == 7
    assert result.max_lags == 5
    assert result.mean > 0
    assert result.standard_error > 0


# --- what makes this HAC rather than an ordinary mean test -------------------
#
# The labels are overlapping five-day forward returns, so consecutive daily RankIC
# observations are autocorrelated by construction and a classical standard error
# understates the uncertainty. G2 ("HAC t >= 2.000") is the only gate that is a
# statistical inference, and this file is all of the correction behind it. Measured on
# the AR(1) fixture below: HAC with five lags widens the standard error to 2.11 times
# the iid formula and 2.13 times HAC truncated to zero lags, and |t| falls from 4.13 to
# 1.96 -- the same series clears G2's bar or does not depending on nothing but this
# covariance choice.


def _ar1_series(rho: float = 0.8, count: int = 60, seed: int = 7) -> pd.Series:
    """A deterministic, strongly positively autocorrelated, homoskedastic series."""
    rng = np.random.default_rng(seed)
    values = np.empty(count)
    values[0] = 0.02
    for index in range(1, count):
        values[index] = rho * values[index - 1] + rng.standard_normal() * 0.01
    return pd.Series(values)


def _iid_standard_error(series: pd.Series) -> float:
    return float(series.std(ddof=1) / math.sqrt(len(series)))


def test_hac_widens_the_standard_error_of_an_autocorrelated_series() -> None:
    """The covariance really is HAC, and by enough to change the verdict.

    This property was already pinned before this test existed -- but only by a golden
    metric value in tests/golden/test_dataset_and_metrics.py, which caught a classical
    covariance by reporting 0.3334 instead of the stored 0.1304. Regenerating that golden
    file is a routine step after any deliberate metric change, and doing so would have
    removed the only pin on G2's inference without telling anyone. So this asserts the
    inequality rather than a stored number, with margins wide enough to survive a
    statsmodels upgrade: HAC must beat both the iid formula and a zero-lag truncation by
    more than half again, and |t| must fall to well under the iid value.
    """
    series = _ar1_series()
    iid_t = float(series.mean()) / _iid_standard_error(series)

    corrected = hac_mean_test(series, max_lags=5)
    uncorrected = hac_mean_test(series, max_lags=0)

    assert corrected.standard_error > 1.5 * _iid_standard_error(series)
    assert corrected.standard_error > 1.5 * uncorrected.standard_error
    assert abs(corrected.t_statistic) < 0.6 * abs(iid_t)
    # Guard the guard: the iid comparison really is a significant result, so the
    # correction is what removes the significance rather than noise either way.
    assert abs(iid_t) > 3.5


def test_the_reported_t_is_the_reported_mean_over_the_reported_standard_error() -> None:
    """G2 consumes t while reports display the mean and its error; they must be one test.

    Also pins that an intercept-only fit reports the sample mean, which is the property a
    rank-deficient design matrix breaks: with a second identical column the fitted
    coefficient halves, and the golden metric test catches that only by value.
    """
    series = _ar1_series()

    result = hac_mean_test(series, max_lags=5)

    assert result.mean == pytest.approx(float(series.mean()))
    assert result.t_statistic == pytest.approx(result.mean / result.standard_error)
    assert result.observation_count == len(series)
    assert result.max_lags == 5


def test_hac_refuses_a_series_no_longer_than_its_own_lag_truncation() -> None:
    """The guard's comparator is inclusive, and a negative truncation is refused too.

    Measured, and the guard is not a formality: with six observations and max_lags=5 the
    fit succeeds but the standard error is so small that t comes back as 3.5e15, and with
    seven identical values t comes back as inf. A one-character slip from ``<=`` to ``<``
    is what stands between that and G2.
    """
    values = pd.Series([0.01 * (index + 1) for index in range(9)])

    for length, lags in ((5, 5), (4, 5), (9, -1)):
        with pytest.raises(ValueError, match="more observations than max_lags"):
            hac_mean_test(values.iloc[:length], max_lags=lags)

    # Control: one observation more than the truncation is accepted, so the guard is a
    # boundary and not a blanket refusal.
    assert hac_mean_test(values.iloc[:6], max_lags=5).observation_count == 6


def test_non_finite_entries_are_dropped_before_the_fit_and_out_of_the_count() -> None:
    """A NaN or an infinity is not an observation, and the reported count must agree.

    Measured: nine entries of which one is NaN and two are infinite leave six, and the
    fitted mean is the mean of those six. Reporting the pre-cleaning length instead, or
    letting an infinity reach the fit, puts a sample size and a mean into the G2 evidence
    that describe a series nobody actually had.
    """
    dirty = pd.Series(
        [0.01, float("nan"), 0.02, float("inf"), 0.03, -float("inf"), 0.02, 0.01, 0.04]
    )
    finite = pd.Series([0.01, 0.02, 0.03, 0.02, 0.01, 0.04])

    result = hac_mean_test(dirty, max_lags=2)

    assert result.observation_count == 6
    assert result.mean == pytest.approx(float(finite.mean()))
    assert math.isfinite(result.t_statistic)


def test_the_default_lag_truncation_is_five() -> None:
    """Five is the label window's overlap, and the default is nearly dead code today.

    All four current call sites pass max_lags explicitly, which is exactly the condition
    under which a silent change from five to four costs nothing in the tests and
    everything in the meaning: the truncation is supposed to match the five-day overlap
    that makes consecutive observations autocorrelated.
    """
    series = _ar1_series()

    defaulted = hac_mean_test(series)

    assert defaulted.max_lags == 5
    assert defaulted.standard_error == pytest.approx(
        hac_mean_test(series, max_lags=5).standard_error
    )
