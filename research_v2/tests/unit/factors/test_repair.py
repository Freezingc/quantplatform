from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from research_v2.factors.repair import (
    FactorRepairError,
    RepairStatus,
    apply_repair_overlay,
    audit_factor_source,
    compute_gp022_candidate,
    compute_gp064_candidate,
    fill_missing_market_rows,
    patch_factor_source,
)


def _market_frame(days: int = 70, *, code: str = "000001.SZ") -> pd.DataFrame:
    dates = [date(2025, 1, 1) + timedelta(days=index) for index in range(days)]
    return pd.DataFrame(
        {
            "trade_date": dates,
            "ts_code": code,
            "close": np.linspace(10.0, 20.0, days),
            "amount": np.linspace(100.0, 200.0, days),
            "pct_chg": np.linspace(-2.0, 2.0, days),
        }
    )


def test_fill_missing_market_rows_preserves_primary_values_and_adds_only_new_keys() -> None:
    primary = pd.DataFrame(
        {
            "trade_date": [date(2026, 5, 8)],
            "ts_code": ["000001.SZ"],
            "close": [10.0],
            "amount": [100.0],
            "pct_chg": [1.0],
        }
    )
    secondary = pd.DataFrame(
        {
            "trade_date": [date(2026, 5, 8), date(2026, 5, 8)],
            "ts_code": ["000001.SZ", "000002.SZ"],
            "close": [11.0, 20.0],
            "amount": [110.0, 200.0],
            "pct_chg": [2.0, 3.0],
        }
    )

    repaired = fill_missing_market_rows(primary, secondary)

    assert len(repaired) == 2
    assert repaired.loc[repaired["ts_code"] == "000001.SZ", "close"].iloc[0] == 10.0
    assert repaired.loc[repaired["ts_code"] == "000002.SZ", "close"].iloc[0] == 20.0


def test_fill_missing_market_rows_rejects_conflicting_primary_duplicates() -> None:
    primary = pd.DataFrame(
        {
            "trade_date": [date(2026, 5, 8), date(2026, 5, 8)],
            "ts_code": ["000001.SZ", "000001.SZ"],
            "close": [10.0, 11.0],
            "amount": [100.0, 100.0],
            "pct_chg": [1.0, 1.0],
        }
    )
    secondary = primary.iloc[:0].copy()

    with pytest.raises(FactorRepairError, match="conflicting duplicate"):
        fill_missing_market_rows(primary, secondary)


def test_gp022_candidate_has_the_expected_sixty_day_warmup() -> None:
    market = _market_frame()

    candidate = compute_gp022_candidate(market)

    assert candidate.columns.tolist() == ["trade_date", "ts_code", "candidate_value"]
    assert candidate["candidate_value"].iloc[:59].isna().all()
    assert pd.notna(candidate["candidate_value"].iloc[59])


def test_gp064_candidate_is_finite_after_warmup() -> None:
    market = _market_frame(140)

    candidate = compute_gp064_candidate(market)

    assert len(candidate) == len(market)
    assert candidate["candidate_value"].iloc[:69].isna().all()
    assert np.isfinite(candidate["candidate_value"].iloc[-1])


def test_patch_factor_source_fills_only_nonfinite_source_values() -> None:
    source = pd.DataFrame(
        {
            "trade_date": [date(2026, 5, 8)] * 3,
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "factor_name": ["GP_022"] * 3,
            "value": [1.0, np.nan, np.inf],
        }
    )
    candidate = pd.DataFrame(
        {
            "trade_date": [date(2026, 5, 8)] * 3,
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "candidate_value": [9.0, 2.0, 3.0],
        }
    )

    repaired = patch_factor_source(source, candidate, factor_name="GP_022")

    assert repaired["value"].tolist() == [1.0, 2.0, 3.0]
    assert repaired.columns.tolist() == source.columns.tolist()


def test_patch_factor_source_rejects_duplicate_candidate_keys() -> None:
    source = pd.DataFrame(
        {
            "trade_date": [date(2026, 5, 8)],
            "ts_code": ["000001.SZ"],
            "factor_name": ["GP_022"],
            "value": [np.nan],
        }
    )
    candidate = pd.DataFrame(
        {
            "trade_date": [date(2026, 5, 8), date(2026, 5, 8)],
            "ts_code": ["000001.SZ", "000001.SZ"],
            "candidate_value": [1.0, 2.0],
        }
    )

    with pytest.raises(FactorRepairError, match="duplicate candidate keys"):
        patch_factor_source(source, candidate, factor_name="GP_022")


def test_audit_factor_source_counts_exact_duplicates_and_finite_conflicts() -> None:
    source = pd.DataFrame(
        {
            "trade_date": [date(2026, 5, 8)] * 5,
            "ts_code": [
                "000001.SZ",
                "000001.SZ",
                "000002.SZ",
                "000002.SZ",
                "000003.SZ",
            ],
            "factor_name": ["GP_022"] * 5,
            "value": [1.0, 1.0, 2.0, 3.0, np.nan],
        }
    )

    audit = audit_factor_source(source)

    assert audit.source_rows == 5
    assert audit.exact_duplicate_groups == 1
    assert audit.exact_duplicate_rows == 1
    assert audit.conflict_groups == 1
    assert audit.finite_rows == 4
    assert audit.nonfinite_rows == 1


def test_apply_repair_overlay_blocks_conflicting_source_values() -> None:
    source = pd.DataFrame(
        {
            "trade_date": [date(2026, 5, 8), date(2026, 5, 8)],
            "ts_code": ["000001.SZ", "000001.SZ"],
            "factor_name": ["GP_022", "GP_022"],
            "value": [1.0, 2.0],
        }
    )

    decision = apply_repair_overlay(source, None, factor_name="GP_022")

    assert decision.status is RepairStatus.BLOCKED
    assert decision.clean.empty
    assert decision.quarantine["reason"].tolist() == [
        "CONFLICT_QUARANTINE",
        "CONFLICT_QUARANTINE",
    ]


def test_apply_repair_overlay_never_overwrites_finite_source_values() -> None:
    source = pd.DataFrame(
        {
            "trade_date": [date(2026, 5, 8), date(2026, 5, 8)],
            "ts_code": ["000001.SZ", "000002.SZ"],
            "factor_name": ["GP_022", "GP_022"],
            "value": [1.0, np.nan],
        }
    )
    candidate = pd.DataFrame(
        {
            "trade_date": [date(2026, 5, 8), date(2026, 5, 8)],
            "ts_code": ["000001.SZ", "000002.SZ"],
            "candidate_value": [1.0, 2.0],
        }
    )

    decision = apply_repair_overlay(source, candidate, factor_name="GP_022")

    assert decision.status is RepairStatus.PENDING_RECHECK
    assert decision.filled_rows == 1
    assert decision.clean["value"].tolist() == [1.0, 2.0]
    assert decision.quarantine.empty


def test_apply_repair_overlay_quarantines_candidate_after_signal_date() -> None:
    source = pd.DataFrame(
        {
            "trade_date": [date(2026, 5, 8)],
            "ts_code": ["000001.SZ"],
            "factor_name": ["GP_022"],
            "value": [np.nan],
        }
    )
    candidate = pd.DataFrame(
        {
            "trade_date": [date(2026, 5, 9)],
            "ts_code": ["000001.SZ"],
            "candidate_value": [2.0],
        }
    )

    decision = apply_repair_overlay(
        source,
        candidate,
        factor_name="GP_022",
        signal_date=date(2026, 5, 8),
    )

    assert decision.status is RepairStatus.BLOCKED
    assert decision.filled_rows == 0
    assert decision.quarantine["reason"].tolist() == ["PIT_DATE_VIOLATION"]
