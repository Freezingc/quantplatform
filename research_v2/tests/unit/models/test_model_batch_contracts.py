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
