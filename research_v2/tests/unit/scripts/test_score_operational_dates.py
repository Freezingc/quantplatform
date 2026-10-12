"""Unit tests for the operational scoring script's pure assembly helpers.

The load-bearing decision here is the feature-to-factor mapping: ``f00..f14`` are positional, so
any reordering of the frozen set silently scores each factor through the wrong weight column while
every shape check still passes. The combine step has the mirror-image hazard: two seeds or two
models emit their rows in score order, so averaging positionally blends different stocks.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
import score_operational_dates as scoring
import walk_forward_factor_daily as wffd


def test_feature_factor_pairs_follow_the_training_contract_order() -> None:
    pairs = scoring.feature_factor_pairs()
    assert [name for name, _ in pairs] == [f"f{index:02d}" for index in range(15)]
    assert tuple(factor for _, factor in pairs) == wffd.FROZEN_FACTOR_SET_V2
    # f00 must be the first frozen factor, not merely *a* frozen factor.
    assert pairs[0] == ("f00", "yeslab_206")
    assert pairs[-1] == ("f14", "yeslab_96")


def test_feature_factor_pairs_covers_every_frozen_v2_factor_once() -> None:
    factors = [factor for _, factor in scoring.feature_factor_pairs()]
    assert len(factors) == len(set(factors)) == 15


def _ranks(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_date": [date.fromisoformat(row[0]) for row in rows],
            "ts_code": [row[1] for row in rows],
            "percentile_rank": [row[2] for row in rows],
        }
    )


def test_combine_model_ranks_aligns_on_business_keys_not_position() -> None:
    first = _ranks([("2026-01-05", "a", 0.1), ("2026-01-05", "b", 0.9)])
    second = _ranks([("2026-01-05", "b", 0.8), ("2026-01-05", "a", 0.2)])
    combined = scoring.combine_model_ranks({"m0r": first, "m1": second})
    assert combined.columns.tolist() == [
        "signal_date",
        "ts_code",
        "m0r_percentile",
        "m1_percentile",
        "ensemble_percentile",
    ]
    by_code = combined.set_index("ts_code")
    assert by_code.loc["a", "m0r_percentile"] == pytest.approx(0.1)
    assert by_code.loc["a", "m1_percentile"] == pytest.approx(0.2)
    assert by_code.loc["a", "ensemble_percentile"] == pytest.approx(0.15)
    assert by_code.loc["b", "ensemble_percentile"] == pytest.approx(0.85)


def test_combine_model_ranks_refuses_a_key_present_in_only_one_model() -> None:
    """A silent inner join would hide a universe mismatch between two scored models."""
    first = _ranks([("2026-01-05", "a", 0.1), ("2026-01-05", "b", 0.9)])
    second = _ranks([("2026-01-05", "a", 0.2), ("2026-01-05", "c", 0.3)])
    with pytest.raises(ValueError, match="different business keys"):
        scoring.combine_model_ranks({"m0r": first, "m1": second})


def test_combine_model_ranks_refuses_duplicate_business_keys() -> None:
    duplicated = _ranks([("2026-01-05", "a", 0.1), ("2026-01-05", "a", 0.2)])
    with pytest.raises(ValueError, match="duplicate"):
        scoring.combine_model_ranks({"m0r": duplicated})


def test_combine_model_ranks_requires_at_least_one_model() -> None:
    with pytest.raises(ValueError, match="at least one model"):
        scoring.combine_model_ranks({})


def test_missing_factor_ids_reports_absent_ids() -> None:
    frame = pd.DataFrame({"factor_id": ["yeslab_96", "yeslab_206"]})
    assert scoring.missing_factor_ids(frame, ("yeslab_96", "yeslab_425")) == ("yeslab_425",)
    assert scoring.missing_factor_ids(frame, ("yeslab_96", "yeslab_206")) == ()
