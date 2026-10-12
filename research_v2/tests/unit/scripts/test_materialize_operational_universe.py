"""Unit tests for the operational universe builder's pure helpers.

The whole point of this script is that a rebuild can be checked against the published universe
before it is trusted past the published range, so the validation arithmetic and its gate are the
parts that must not be wrong: an agreement metric that silently saturates would let a mis-specified
rebuild through, and a gate that does not name the measured number gives no way to triage it.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

import materialize_operational_universe as operational_universe


def test_derive_suspensions_keeps_only_suspension_rows_and_parses_dates() -> None:
    raw = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "trade_date": ["20260803", "20260803", "20260804"],
            "suspend_type": ["S", "R", "S"],
            "suspend_timing": [None, None, None],
        }
    )
    derived = operational_universe.derive_suspensions(raw)
    assert derived.columns.tolist() == operational_universe.SUSPENSION_COLUMNS
    assert derived["ts_code"].tolist() == ["000001.SZ", "000003.SZ"]
    assert derived["trade_date"].tolist() == [date(2026, 8, 3), date(2026, 8, 4)]
    assert derived["suspend_type"].tolist() == ["S", "S"]


def test_derive_suspensions_dedupes_a_repeated_business_key() -> None:
    raw = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "trade_date": ["20260803", "20260803"],
            "suspend_type": ["S", "S"],
        }
    )
    assert len(operational_universe.derive_suspensions(raw)) == 1


def _members(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_date": [date.fromisoformat(row[0]) for row in rows],
            "ts_code": [row[1] for row in rows],
            "is_member": True,
        }
    )


def test_validate_against_official_reports_exact_agreement() -> None:
    official = _members([("2026-07-30", "a"), ("2026-07-30", "b")])
    summary = operational_universe.validate_against_official(official, official.copy())
    assert summary["dates"] == 1
    assert summary["exact_dates"] == 1
    assert summary["min_member_agreement"] == 1.0
    assert summary["mean_member_agreement"] == 1.0
    assert summary["member_count_max_abs_diff"] == 0


def test_validate_against_official_uses_jaccard_on_the_member_sets() -> None:
    official = _members([("2026-07-30", "a"), ("2026-07-30", "b")])
    derived = _members([("2026-07-30", "a"), ("2026-07-30", "c")])
    summary = operational_universe.validate_against_official(derived, official)
    # {a, b} vs {a, c}: intersection 1, union 3.
    assert summary["min_member_agreement"] == pytest.approx(1 / 3)
    assert summary["exact_dates"] == 0
    assert summary["member_count_max_abs_diff"] == 0


def test_require_member_agreement_names_the_measured_value() -> None:
    summary = {"min_member_agreement": 0.91, "dates": 20}
    with pytest.raises(ValueError, match="0.91"):
        operational_universe.require_member_agreement(summary, minimum=0.99)
    operational_universe.require_member_agreement(summary, minimum=0.90)


def _scd(rows: list[tuple[str, str, str, str | None, bool, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": [row[0] for row in rows],
            "board": [row[1] for row in rows],
            "valid_from": [row[2] for row in rows],
            "valid_to": [row[3] for row in rows],
            "is_st": [row[4] for row in rows],
            "published_at": [row[5] for row in rows],
        }
    )


def test_collapse_security_master_keeps_superseded_intervals() -> None:
    """A later interval start must not erase the one still in force before it.

    Dropping to the newest row per board is what made the July rebuild exclude 600768.SH and
    688309.SH as OUTSIDE_BOARD while the published universe still carried them.
    """
    frame = _scd(
        [
            ("A.SH", "SH_MAIN", "2013-01-01", None, False, "2026-01-01"),
            ("A.SH", "SH_MAIN", "2026-07-24", None, False, "2026-08-01"),
        ]
    )
    collapsed = operational_universe.collapse_security_master(frame)
    assert collapsed["valid_from"].tolist() == ["2013-01-01", "2026-07-24"]


def test_collapse_security_master_prefers_the_newest_publication_for_one_interval() -> None:
    frame = _scd(
        [
            ("A.SH", "SH_MAIN", "2013-01-01", None, False, "2026-01-01"),
            ("A.SH", "SH_MAIN", "2013-01-01", None, True, "2026-08-01"),
        ]
    )
    collapsed = operational_universe.collapse_security_master(frame)
    assert len(collapsed) == 1
    assert bool(collapsed["is_st"].iloc[0]) is True
