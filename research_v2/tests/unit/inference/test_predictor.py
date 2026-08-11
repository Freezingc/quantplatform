from __future__ import annotations

from datetime import date

import pytest
import torch
from torch import nn

from research_v2.inference.predictor import predict_three_seed_ensemble
from research_v2.models.base import FeatureBatch
from research_v2.training.ensemble import REQUIRED_SEEDS


class FactorModel(nn.Module):
    def __init__(self, sign: float) -> None:
        super().__init__()
        self._sign = sign

    def forward(self, batch: FeatureBatch) -> torch.Tensor:
        return batch.features[:, :, 0] * self._sign


def test_predictor_builds_keyed_equal_weight_three_seed_rank_ensemble() -> None:
    batch = FeatureBatch(
        signal_dates=(date(2025, 1, 2),),
        ts_codes=(("a", "b", "c"),),
        features=torch.tensor([[[1.0], [2.0], [3.0]]]),
        feature_missing=torch.zeros((1, 3, 1), dtype=torch.bool),
        stock_mask=torch.ones((1, 3), dtype=torch.bool),
    )
    prediction = predict_three_seed_ensemble(
        batch,
        {
            REQUIRED_SEEDS[0]: FactorModel(1.0),
            REQUIRED_SEEDS[1]: FactorModel(1.0),
            REQUIRED_SEEDS[2]: FactorModel(-1.0),
        },
    )

    assert prediction.ensemble["ts_code"].tolist() == ["a", "b", "c"]
    assert prediction.ensemble["ensemble_score"].tolist() == pytest.approx([5 / 9, 2 / 3, 7 / 9])
    assert set(prediction.seed_percentile_ranks) == set(REQUIRED_SEEDS)
