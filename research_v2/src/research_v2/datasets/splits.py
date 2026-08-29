"""Purged annual 5y/1y/1y walk-forward split resolution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

from research_v2.datasets.builder import LockedTestAccess
from research_v2.pit.calendar import TradingCalendar

SplitRole = Literal["train", "validation", "test"]


class LockedPeriodError(ValueError):
    """Raised when normal code attempts to read a locked test segment."""


@dataclass(frozen=True)
class AnnualFold:
    fold_id: str
    train_years: tuple[int, ...]
    validation_years: tuple[int, ...]
    test_years: tuple[int, ...]
    locked: bool = False


@dataclass(frozen=True)
class ResolvedSegment:
    fold_id: str
    role: SplitRole
    signal_dates: tuple[date, ...]
    purge_trading_days: int
    locked: bool


@dataclass(frozen=True)
class InnerFold:
    inner_fold_id: str
    train: ResolvedSegment
    validation: ResolvedSegment


def annual_folds() -> tuple[AnnualFold, ...]:
    """Return the approved 2014–2026 outer folds in chronological order."""
    folds: list[AnnualFold] = []
    for index, train_start_year in enumerate(range(2014, 2020), start=1):
        validation_year = train_start_year + 5
        folds.append(
            AnnualFold(
                fold_id=f"F{index:02d}",
                train_years=tuple(range(train_start_year, train_start_year + 5)),
                validation_years=(validation_year,),
                test_years=(validation_year + 1,),
            )
        )
    folds.append(
        AnnualFold(
            fold_id="FINAL",
            train_years=(2020, 2021, 2022, 2023, 2024),
            validation_years=(2025,),
            test_years=(2026,),
            locked=True,
        )
    )
    return tuple(folds)


class SplitResolver:
    """Resolve only normal-access segments on a supplied immutable trading calendar."""

    def __init__(self, calendar: TradingCalendar, *, purge_trading_days: int = 6) -> None:
        if purge_trading_days < 0:
            raise ValueError("purge_trading_days cannot be negative")
        self._calendar = calendar
        self._purge_trading_days = purge_trading_days
        self._folds = {fold.fold_id: fold for fold in annual_folds()}

    def resolve(self, fold_id: str, role: SplitRole) -> ResolvedSegment:
        """Resolve segment signal dates and apply purge without exposing locked FINAL test."""
        if fold_id not in self._folds:
            raise KeyError(f"unknown fold: {fold_id}")
        fold = self._folds[fold_id]
        if role == "test" and fold.locked:
            raise LockedPeriodError(f"{fold_id} test dates are locked")
        years = {
            "train": fold.train_years,
            "validation": fold.validation_years,
            "test": fold.test_years,
        }[role]
        return self._resolve_years(fold_id, role, years, locked=fold.locked and role == "test")

    def resolve_locked_test(
        self, fold_id: str, access: LockedTestAccess
    ) -> ResolvedSegment:
        """Resolve FINAL/TEST dates only after an explicit locked-test capability."""
        access.verify()
        if fold_id != "FINAL":
            raise LockedPeriodError("locked access is only valid for FINAL test")
        fold = self._folds[fold_id]
        if not fold.locked or access.locked_year != fold.test_years[0]:
            raise LockedPeriodError("locked access does not match the FINAL test year")
        return self._resolve_years(fold_id, "test", fold.test_years, locked=True)

    def inner_folds(self, fold_id: str) -> tuple[InnerFold, ...]:
        """Return the three expanding inner folds contained entirely in outer training years."""
        if fold_id not in self._folds:
            raise KeyError(f"unknown fold: {fold_id}")
        outer = self._folds[fold_id]
        if len(outer.train_years) != 5:
            raise ValueError("inner folds require exactly five outer training years")
        inner: list[InnerFold] = []
        for index, validation_index in enumerate((2, 3, 4), start=1):
            train_years = outer.train_years[:validation_index]
            validation_years = (outer.train_years[validation_index],)
            inner.append(
                InnerFold(
                    inner_fold_id=f"{fold_id}.I{index}",
                    train=self._resolve_years(fold_id, "train", train_years, locked=False),
                    validation=self._resolve_years(
                        fold_id,
                        "validation",
                        validation_years,
                        locked=False,
                    ),
                )
            )
        return tuple(inner)

    def _resolve_years(
        self,
        fold_id: str,
        role: SplitRole,
        years: tuple[int, ...],
        *,
        locked: bool,
    ) -> ResolvedSegment:
        dates = tuple(day for day in self._calendar.dates if day.year in years)
        if not dates:
            raise ValueError(f"calendar has no trading dates for {fold_id}/{role}")
        if role in {"train", "validation"}:
            if len(dates) <= self._purge_trading_days:
                raise ValueError(f"insufficient dates to purge {fold_id}/{role}")
            dates = dates[: -self._purge_trading_days]
        return ResolvedSegment(
            fold_id=fold_id,
            role=role,
            signal_dates=dates,
            purge_trading_days=self._purge_trading_days if role != "test" else 0,
            locked=locked,
        )
