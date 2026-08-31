from __future__ import annotations

from datetime import date

import pytest

from research_v2.datasets.builder import LockedTestAccess
from research_v2.datasets.splits import LockedPeriodError, SplitResolver, annual_folds
from research_v2.pit.calendar import TradingCalendar


@pytest.fixture(autouse=True)
def _locked_access_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QRV2_LOCKED_TEST_ACCESS_HMAC_KEY", "split-test-access-key")


def _locked_access() -> LockedTestAccess:
    return LockedTestAccess._issue_authorized(
        locked_year=2026,
        authorization_id="approved",
        signing_key=b"split-test-access-key",
    )


def _calendar() -> TradingCalendar:
    return TradingCalendar.from_dates(
        [date(year, 1, day) for year in range(2014, 2027) for day in range(2, 10)]
    )


def test_outer_fold_years_are_exact_and_ordered() -> None:
    fold = next(fold for fold in annual_folds() if fold.fold_id == "F01")

    assert fold.train_years == (2014, 2015, 2016, 2017, 2018)
    assert fold.validation_years == (2019,)
    assert fold.test_years == (2020,)


def test_resolver_purges_last_six_signal_dates_from_train_and_validation() -> None:
    resolver = SplitResolver(_calendar())

    train = resolver.resolve("F01", "train")
    validation = resolver.resolve("F01", "validation")
    test = resolver.resolve("F01", "test")

    assert len(train.signal_dates) == 5 * 8 - 6
    assert len(validation.signal_dates) == 2
    assert len(test.signal_dates) == 8
    assert train.signal_dates[-1] == date(2018, 1, 3)
    assert validation.signal_dates[-1] == date(2019, 1, 3)


def test_final_test_is_locked_from_normal_resolver() -> None:
    with pytest.raises(LockedPeriodError, match="FINAL"):
        SplitResolver(_calendar()).resolve("FINAL", "test")


def test_locked_final_test_requires_explicit_access_capability() -> None:
    resolver = SplitResolver(_calendar())

    segment = resolver.resolve_locked_test(
        "FINAL", _locked_access()
    )

    assert segment.locked
    assert segment.role == "test"


def test_inner_folds_expand_only_inside_outer_train_years() -> None:
    inner_folds = SplitResolver(_calendar()).inner_folds("F01")

    assert len(inner_folds) == 3
    assert {day.year for day in inner_folds[0].train.signal_dates} == {2014, 2015}
    assert {day.year for day in inner_folds[0].validation.signal_dates} == {2016}
    assert {day.year for day in inner_folds[-1].train.signal_dates} == {2014, 2015, 2016, 2017}
    assert {day.year for day in inner_folds[-1].validation.signal_dates} == {2018}
