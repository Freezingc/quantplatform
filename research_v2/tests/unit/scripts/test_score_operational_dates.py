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
import torch
import walk_forward_factor_daily as wffd

from research_v2.datasets.builder import FactorInput
from research_v2.datasets.preprocessing import FoldPreprocessor
from research_v2.inference.builder import build_label_free_inference_batch


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


def test_partition_by_date_keeps_every_row_exactly_once() -> None:
    frame = pd.DataFrame(
        {
            "signal_date": [date(2026, 1, 5), date(2026, 1, 5), date(2026, 1, 6)],
            "ts_code": ["a", "b", "a"],
            "value": [1.0, 2.0, 3.0],
        }
    )
    parts = scoring.partition_by_date(frame)
    assert set(parts) == {date(2026, 1, 5), date(2026, 1, 6)}
    assert sum(len(part) for part in parts.values()) == len(frame)
    assert sorted(parts[date(2026, 1, 5)]["ts_code"]) == ["a", "b"]


def test_sparse_dates_flags_a_date_far_below_the_median() -> None:
    """A published universe can be partial on one date without any error; scoring it silently
    produces ranks over a tenth of the cross-section."""
    counts = {
        date(2026, 5, 6): 3879,
        date(2026, 5, 7): 3880,
        date(2026, 5, 8): 946,
        date(2026, 5, 11): 3879,
    }
    assert scoring.sparse_dates(counts, floor_share=0.5) == (date(2026, 5, 8),)
    assert scoring.sparse_dates(counts, floor_share=0.2) == ()


def test_sparse_dates_is_empty_for_a_uniform_window() -> None:
    counts = {date(2026, 1, 5): 3923, date(2026, 1, 6): 3924}
    assert scoring.sparse_dates(counts, floor_share=0.5) == ()


def _inference_fixture() -> tuple[pd.DataFrame, pd.DataFrame, tuple[FactorInput, ...]]:
    days = (date(2026, 1, 5), date(2026, 1, 6))
    universe = pd.DataFrame(
        {
            "signal_date": [day for day in days for _ in ("a", "b")],
            "ts_code": ["a", "b"] * len(days),
            "is_member": [True] * (2 * len(days)),
        }
    )
    rows = [
        {
            "signal_date": day,
            "ts_code": code,
            "factor_id": factor_id,
            "factor_version": 1,
            "value": value,
            "available_date": day,
        }
        for day in days
        for code, first, second in (("a", 1.0, 4.0), ("b", 2.0, 5.0))
        for factor_id, value in (("factor_a", first), ("factor_b", second))
    ]
    members = (FactorInput("factor_a", 1, "f00"), FactorInput("factor_b", 1, "f01"))
    return universe, pd.DataFrame(rows), members


def test_date_scoped_factor_rows_match_the_full_frame_batch() -> None:
    """Scoping the long frame to one date must not change a single batch element.

    The builder re-copies and re-parses whatever frame it is handed, so scoring 139 dates through
    one 6.4M-row frame costs 139 full copies. Handing it only the date's rows is the fix; this pins
    that the two produce the same batch.
    """
    universe, factors, members = _inference_fixture()
    preprocessor = FoldPreprocessor(factor_order=("f00", "f01"))
    preprocessor.fit(
        pd.DataFrame({"f00": [1.0, 2.0, 3.0], "f01": [4.0, 5.0, 6.0]}), role="TRAIN"
    )
    day = date(2026, 1, 5)
    full = build_label_free_inference_batch(
        universe, factors, factor_members=members, signal_date=day, preprocessor=preprocessor
    )
    scoped = build_label_free_inference_batch(
        universe,
        scoring.partition_by_date(factors)[day],
        factor_members=members,
        signal_date=day,
        preprocessor=preprocessor,
    )
    assert full.ts_codes == scoped.ts_codes
    assert full.signal_dates == scoped.signal_dates
    assert torch.equal(full.features, scoped.features)
    assert torch.equal(full.feature_missing, scoped.feature_missing)
    assert torch.equal(full.stock_mask, scoped.stock_mask)
