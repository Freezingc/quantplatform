from __future__ import annotations

from datetime import date

import pytest
import torch

from research_v2.models.base import FeatureBatch
from research_v2.training.batches import TrainingBatch


def test_training_batch_requires_padding_targets_to_be_nan() -> None:
    inputs = FeatureBatch(
        signal_dates=(date(2025, 1, 2),),
        ts_codes=(("a", ""),),
        features=torch.zeros((1, 2, 1)),
        feature_missing=torch.zeros((1, 2, 1), dtype=torch.bool),
        stock_mask=torch.tensor([[True, False]]),
    )
    with pytest.raises(ValueError, match="padding"):
        TrainingBatch(
            inputs=inputs,
            target_raw=torch.tensor([[0.1, 0.0]]),
            target_rank=torch.tensor([[0.5, float("nan")]]),
        )
