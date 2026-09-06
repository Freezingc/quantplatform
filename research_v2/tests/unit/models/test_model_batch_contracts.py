from __future__ import annotations

from datetime import date

import pytest
import torch

from research_v2.models.base import FeatureBatch, ModelSignature, predictions_to_frame
from research_v2.training.batches import TrainingBatch


def test_feature_batch_excludes_padding_from_keyed_predictions() -> None:
    batch = FeatureBatch(
        signal_dates=(date(2025, 1, 2),),
        ts_codes=(("a", "b", ""),),
        features=torch.zeros((1, 3, 2)),
        feature_missing=torch.zeros((1, 3, 2), dtype=torch.bool),
        stock_mask=torch.tensor([[True, True, False]]),
    )
    frame = predictions_to_frame(batch, torch.tensor([[0.2, 0.1, 999.0]]))

    assert frame["ts_code"].tolist() == ["a", "b"]
    assert frame["raw_score"].tolist() == pytest.approx([0.2, 0.1])
    assert frame["percentile_rank"].tolist() == [1.0, 0.5]
    assert ModelSignature("M1", 2, "abc").model_family == "M1"


def test_predictions_to_frame_keeps_tied_scores_at_their_average_rank() -> None:
    """Ties average; they must not be broken by input row position.

    ``static_batches`` ranks its training targets with ``method="average"`` for the
    reason its own comment gives: the percentile rank is later compared against
    Spearman ranks in evaluation. Breaking ties by position instead would put the
    input row order -- within one date that is ts_code order -- into the score that
    selects the winning config, so two stocks the model cannot tell apart would be
    ranked differently because of their security codes. Real models do emit exact
    ties: a collapsed head or dead ReLU units score many stocks identically in
    float32, and padding slots all score exactly zero.
    """
    batch = FeatureBatch(
        signal_dates=(date(2025, 1, 2),),
        ts_codes=(("a", "b", "c", "d"),),
        features=torch.zeros((1, 4, 2)),
        feature_missing=torch.zeros((1, 4, 2), dtype=torch.bool),
        stock_mask=torch.ones((1, 4), dtype=torch.bool),
    )

    # b and c are indistinguishable to the model; a and d are not.
    frame = predictions_to_frame(batch, torch.tensor([[1.0, 2.0, 2.0, 3.0]]))

    # Guard the guard: if this fixture stops producing a tie the test proves nothing.
    assert bool(frame["raw_score"].duplicated().any())
    ranks = frame.set_index("ts_code")["percentile_rank"].to_dict()
    assert ranks["b"] == ranks["c"] == pytest.approx(0.625)  # average of 2/4 and 3/4
    assert ranks["a"] == pytest.approx(0.25)
    assert ranks["d"] == pytest.approx(1.0)


def test_predictions_to_frame_refuses_scores_whose_shape_disagrees_with_the_mask() -> None:
    """The shape guard is fail-closed, and removing it fails silently, not loudly.

    Probed rather than assumed. With the guard deleted, a scores tensor carrying an
    extra date row is accepted: ``date_index`` 0 reads whatever the model put in row
    0, so the frame comes back holding the wrong date's scores attributed to this
    date's securities, with no exception anywhere downstream. A wider stock axis is
    accepted the same way. Ranking the wrong numbers is worse than raising, because
    the output still looks exactly like a valid percentile frame.
    """
    batch = FeatureBatch(
        signal_dates=(date(2025, 1, 2),),
        ts_codes=(("a", "b", ""),),
        features=torch.zeros((1, 3, 2)),
        feature_missing=torch.zeros((1, 3, 2), dtype=torch.bool),
        stock_mask=torch.tensor([[True, True, False]]),
    )

    # Two date rows against a one-date batch, and four stock columns against a
    # three-wide mask. Both are shape disagreements the guard must refuse.
    with pytest.raises(ValueError, match="shape"):
        predictions_to_frame(batch, torch.tensor([[9.0, 8.0, 7.0], [0.2, 0.1, 0.0]]))
    with pytest.raises(ValueError, match="shape"):
        predictions_to_frame(batch, torch.tensor([[0.2, 0.1, 0.0, 999.0]]))


def test_predictions_to_frame_orders_each_date_by_descending_score() -> None:
    """Row order is the premise of the seed-averaging pin, so it must be pinned too.

    No consumer observes this order today: ``_mean_seed_percentile_scores``,
    ``combine_seed_percentile_ranks`` and ``compute_daily_rank_ic`` all merge on
    business keys, and ``FrozenPredictionStore._normalize_scores`` re-sorts before the
    artifact is hashed. It is load-bearing anyway. Two seeds emit their rows in
    different orders *only* because each frame is sorted by score, and that difference
    is what makes positional seed averaging blend different stocks -- the defect the
    driver's key-alignment regression test exists to catch.

    Probed: deleting this sort alone survives the whole suite, but deleting it
    together with a faithful positional-averaging mutant survives too, which means the
    key-alignment test would have become a tautology. An unpinned premise is a pin
    somebody can silently withdraw.
    """
    batch = FeatureBatch(
        signal_dates=(date(2025, 1, 2), date(2025, 1, 3)),
        ts_codes=(("a", "b"), ("c", "d")),
        features=torch.zeros((2, 2, 2)),
        feature_missing=torch.zeros((2, 2, 2), dtype=torch.bool),
        stock_mask=torch.ones((2, 2), dtype=torch.bool),
    )

    frame = predictions_to_frame(batch, torch.tensor([[0.1, 0.9], [0.7, 0.2]]))

    # Highest score first within each date; ties keep ts_code order (see the tie test).
    assert frame["ts_code"].tolist() == ["b", "a", "c", "d"]
    assert frame["raw_score"].tolist() == pytest.approx([0.9, 0.1, 0.7, 0.2])


def test_batch_contract_rejects_shape_key_and_non_finite_violations() -> None:
    with pytest.raises(ValueError, match="ts_codes"):
        FeatureBatch(
            signal_dates=(date(2025, 1, 2),),
            ts_codes=(("a",),),
            features=torch.zeros((1, 2, 2)),
            feature_missing=torch.zeros((1, 2, 2), dtype=torch.bool),
            stock_mask=torch.ones((1, 2), dtype=torch.bool),
        )
    batch = FeatureBatch(
        signal_dates=(date(2025, 1, 2),),
        ts_codes=(("a", ""),),
        features=torch.zeros((1, 2, 2)),
        feature_missing=torch.zeros((1, 2, 2), dtype=torch.bool),
        stock_mask=torch.tensor([[True, False]]),
    )
    with pytest.raises(ValueError, match="non-finite"):
        predictions_to_frame(batch, torch.tensor([[float("nan"), 0.0]]))
    with pytest.raises(ValueError, match="target_raw"):
        TrainingBatch(
            inputs=batch,
            target_raw=torch.zeros((1, 1)),
            target_rank=torch.zeros((1, 2)),
        )
