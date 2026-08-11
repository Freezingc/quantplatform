from __future__ import annotations

from datetime import date

import pytest
import torch

from research_v2.models.base import FeatureBatch
from research_v2.models.residual_mlp import ResidualMLP, ResidualMLPConfig


def _batch() -> FeatureBatch:
    return FeatureBatch(
        signal_dates=(date(2025, 1, 2), date(2025, 1, 3)),
        ts_codes=(("a", "b", ""), ("c", "", "")),
        features=torch.randn((2, 3, 15), generator=torch.Generator().manual_seed(4)),
        feature_missing=torch.zeros((2, 3, 15), dtype=torch.bool),
        stock_mask=torch.tensor([[True, True, False], [True, False, False]]),
    )


def test_residual_mlp_has_masked_shape_and_finite_gradients() -> None:
    model = ResidualMLP(ResidualMLPConfig(raw_factor_count=15, hidden_dim=32, num_blocks=2))
    batch = _batch()

    scores = model(batch)
    scores.sum().backward()

    assert scores.shape == (2, 3)
    assert torch.equal(scores[~batch.stock_mask], torch.zeros(3))
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert model.artifact_signature().raw_factor_count == 15


def test_residual_mlp_eval_is_deterministic_and_rejects_wrong_factor_count() -> None:
    model = ResidualMLP(
        ResidualMLPConfig(raw_factor_count=15, hidden_dim=32, num_blocks=1, dropout=0.2)
    ).eval()
    batch = _batch()

    assert torch.equal(model(batch), model(batch))
    wrong = FeatureBatch(
        signal_dates=(date(2025, 1, 2),),
        ts_codes=(("a",),),
        features=torch.zeros((1, 1, 16)),
        feature_missing=torch.zeros((1, 1, 16), dtype=torch.bool),
        stock_mask=torch.ones((1, 1), dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="factor"):
        model(wrong)


def test_residual_mlp_accepts_a_six_factor_frozen_set() -> None:
    model = ResidualMLP(ResidualMLPConfig(raw_factor_count=6, hidden_dim=8, num_blocks=1))

    assert model.artifact_signature().raw_factor_count == 6
