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
    folds = {fold.fold_id: fold for fold in annual_folds()}
    assert tuple(folds) == ("F1", "F2", "F3", "F4", "FINAL")

    fold = folds["F1"]

    assert fold.train_years == (2017, 2018, 2019, 2020)
    assert fold.validation_years == (2021,)
    assert fold.test_years == (2022,)


def test_v2_fold_structure_matches_the_frozen_v2_contract() -> None:
    """Every ordinary fold: 4y train, 1y validation, 1y test, test years 2022-2025."""
    expected = {
        "F1": ((2017, 2018, 2019, 2020), (2021,), (2022,)),
        "F2": ((2018, 2019, 2020, 2021), (2022,), (2023,)),
        "F3": ((2019, 2020, 2021, 2022), (2023,), (2024,)),
        "F4": ((2020, 2021, 2022, 2023), (2024,), (2025,)),
    }
    folds = {fold.fold_id: fold for fold in annual_folds()}
    for fold_id, (train, validation, test) in expected.items():
        assert folds[fold_id].train_years == train
        assert folds[fold_id].validation_years == validation
        assert folds[fold_id].test_years == test
    # The locked FINAL production fold is unchanged: 5y train 2020-2024, locked 2026 test.
    assert folds["FINAL"].train_years == (2020, 2021, 2022, 2023, 2024)
    assert folds["FINAL"].validation_years == (2025,)
    assert folds["FINAL"].test_years == (2026,)
    assert folds["FINAL"].locked


def test_resolver_purges_last_six_signal_dates_from_train_and_validation() -> None:
    resolver = SplitResolver(_calendar())

    train = resolver.resolve("F1", "train")
    validation = resolver.resolve("F1", "validation")
    test = resolver.resolve("F1", "test")

    assert len(train.signal_dates) == 4 * 8 - 6
    assert len(validation.signal_dates) == 2
    assert len(test.signal_dates) == 8
    assert train.signal_dates[-1] == date(2020, 1, 3)
    assert validation.signal_dates[-1] == date(2021, 1, 3)


def test_zero_purge_is_rejected_instead_of_silently_emptying_segments() -> None:
    with pytest.raises(ValueError, match="purge_trading_days"):
        SplitResolver(_calendar(), purge_trading_days=0)


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
    inner_folds = SplitResolver(_calendar()).inner_folds("F1")

    assert len(inner_folds) == 2
    assert {day.year for day in inner_folds[0].train.signal_dates} == {2017, 2018}
    assert {day.year for day in inner_folds[0].validation.signal_dates} == {2019}
    assert {day.year for day in inner_folds[-1].train.signal_dates} == {2017, 2018, 2019}
    assert {day.year for day in inner_folds[-1].validation.signal_dates} == {2020}
