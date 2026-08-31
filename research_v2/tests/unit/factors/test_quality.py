from __future__ import annotations

from datetime import date

import pandas as pd

from research_v2.factors.quality import QualityStatus, assess_factor_quality


def _universe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2)] * 3 + [date(2025, 1, 3)] * 3,
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"] * 2,
        }
    )


def test_quality_uses_universe_members_as_coverage_denominator_and_expires_stale_values() -> None:
    observations = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2), date(2025, 1, 2), date(2025, 1, 3)],
            "ts_code": ["000001.SZ", "000002.SZ", "000001.SZ"],
            "value": [1.0, 2.0, 3.0],
            "stale_days": [0, 0, 181],
        }
    )

    snapshot = assess_factor_quality(
        _universe(),
        observations,
        factor_id="yeslab_1",
        factor_version=1,
    )

    assert snapshot.metrics["universe_member_count"] == 6
    assert snapshot.metrics["finite_value_count"] == 2
    assert snapshot.metrics["coverage"] == 2 / 6
    assert snapshot.status is QualityStatus.ELIGIBILITY_FAIL


def test_quality_blocks_duplicate_business_keys_and_nonfinite_clean_values() -> None:
    observations = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2), date(2025, 1, 2)],
            "ts_code": ["000001.SZ", "000001.SZ"],
            "value": [1.0, 2.0],
            "stale_days": [0, 0],
        }
    )

    snapshot = assess_factor_quality(
        _universe(),
        observations,
        factor_id="yeslab_1",
        factor_version=1,
    )

    assert snapshot.status is QualityStatus.BLOCKER
    assert "DUPLICATE_BUSINESS_KEY" in snapshot.blockers


def test_quality_returns_blocker_snapshot_for_duplicate_universe_keys_instead_of_crashing() -> None:
    duplicated_universe = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2), date(2025, 1, 2)],
            "ts_code": ["000001.SZ", "000001.SZ"],
        }
    )
    observations = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2)],
            "ts_code": ["000001.SZ"],
            "value": [1.0],
            "stale_days": [0],
        }
    )

    snapshot = assess_factor_quality(
        duplicated_universe,
        observations,
        factor_id="yeslab_1",
        factor_version=1,
    )

    assert snapshot.status is QualityStatus.BLOCKER
    assert "DUPLICATE_UNIVERSE_KEY" in snapshot.blockers
