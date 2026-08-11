from __future__ import annotations

from datetime import date

import torch
from torch import nn

from research_v2.models.base import FeatureBatch
from research_v2.training.batches import TrainingBatch
from research_v2.training.validation import evaluate_validation_batches


class FirstFactorModel(nn.Module):
    def forward(self, batch: FeatureBatch) -> torch.Tensor:
        return batch.features[:, :, 0]


def test_validation_evaluation_uses_keyed_scores_and_restores_mode() -> None:
    signal_date = date(2025, 1, 2)
    values = torch.arange(30, dtype=torch.float32).reshape(1, 30, 1)
    batch = TrainingBatch(
        inputs=FeatureBatch(
            signal_dates=(signal_date,),
            ts_codes=(tuple(f"S{index:03d}" for index in range(30)),),
            features=values,
            feature_missing=torch.zeros_like(values, dtype=torch.bool),
            stock_mask=torch.ones((1, 30), dtype=torch.bool),
        ),
        target_raw=values[:, :, 0],
        target_rank=values[:, :, 0],
    )
    model = FirstFactorModel().train()

    result = evaluate_validation_batches(model, {signal_date: batch})

    assert result.summary.mean_rank_ic == 1.0
    assert result.summary.valid_day_count == 1
    assert model.training
