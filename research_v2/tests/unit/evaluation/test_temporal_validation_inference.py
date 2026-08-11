from __future__ import annotations

from datetime import date

import torch
from torch import nn

from research_v2.models.base import TemporalFeatureBatch
from research_v2.training.batches import TemporalTrainingBatch
from research_v2.training.validation import evaluate_validation_batches


class LastTemporalFactor(nn.Module):
    def forward(self, batch: TemporalFeatureBatch) -> torch.Tensor:
        return batch.features[:, :, -1, 0]


def test_validation_evaluator_accepts_temporal_supervised_batches() -> None:
    signal_date = date(2025, 1, 30)
    values = torch.arange(30, dtype=torch.float32)
    features = torch.zeros((1, 30, 20, 15))
    features[0, :, -1, 0] = values
    batch = TemporalTrainingBatch(
        inputs=TemporalFeatureBatch(
            signal_dates=(signal_date,),
            ts_codes=(tuple(f"S{index:03d}" for index in range(30)),),
            features=features,
            feature_missing=torch.zeros_like(features, dtype=torch.bool),
            stock_mask=torch.ones((1, 30), dtype=torch.bool),
            history_mask=torch.ones((1, 30, 20), dtype=torch.bool),
        ),
        target_raw=values.unsqueeze(0),
        target_rank=values.unsqueeze(0),
    )

    result = evaluate_validation_batches(LastTemporalFactor(), {signal_date: batch})

    assert result.summary.mean_rank_ic == 1.0
