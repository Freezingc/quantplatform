from __future__ import annotations

from datetime import date

import pytest
import torch

from research_v2.models.base import FeatureBatch
from research_v2.models.residual_mlp import ResidualMLP, ResidualMLPConfig
from research_v2.training.batches import TrainingBatch
from research_v2.training.torch_loop import TorchDateTrainingLoop


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


def test_residual_mlp_is_stock_permutation_equivariant() -> None:
    model = ResidualMLP(
        ResidualMLPConfig(raw_factor_count=15, hidden_dim=32, num_blocks=2, dropout=0.0)
    ).eval()
    stock_count = 5
    batch = FeatureBatch(
        signal_dates=(date(2024, 1, 2),),
        ts_codes=(tuple(f"s{index}" for index in range(stock_count)),),
        features=torch.randn((1, stock_count, 15), generator=torch.Generator().manual_seed(6)),
        feature_missing=torch.zeros((1, stock_count, 15), dtype=torch.bool),
        stock_mask=torch.ones((1, stock_count), dtype=torch.bool),
    )
    permutation = [3, 0, 4, 1, 2]
    permuted = FeatureBatch(
        signal_dates=batch.signal_dates,
        ts_codes=(tuple(batch.ts_codes[0][index] for index in permutation),),
        features=batch.features[:, permutation],
        feature_missing=batch.feature_missing[:, permutation],
        stock_mask=batch.stock_mask[:, permutation],
    )

    with torch.no_grad():
        original = model(batch)
        reordered = model(permuted)

    assert torch.allclose(reordered[0], original[0, permutation], atol=1e-6)


def test_residual_mlp_overfits_a_tiny_linear_signal() -> None:
    factor_count = 15
    stock_count = 40
    dates = (date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4))
    batches: dict[date, TrainingBatch] = {}
    for day_index, signal_date in enumerate(dates):
        generator = torch.Generator().manual_seed(200 + day_index)
        signal = torch.randn(stock_count, generator=generator)
        noise = torch.randn((stock_count, factor_count - 1), generator=generator)
        features = torch.cat((signal.unsqueeze(1), noise), dim=1).unsqueeze(0)
        target_raw = signal.clone()
        ranks = target_raw.argsort().argsort().to(dtype=torch.float)
        inputs = FeatureBatch(
            signal_dates=(signal_date,),
            ts_codes=(tuple(f"s{index:02d}" for index in range(stock_count)),),
            features=features,
            feature_missing=torch.zeros_like(features, dtype=torch.bool),
            stock_mask=torch.ones((1, stock_count), dtype=torch.bool),
        )
        batches[signal_date] = TrainingBatch(
            inputs=inputs,
            target_raw=target_raw.unsqueeze(0),
            target_rank=(ranks / (stock_count - 1)).unsqueeze(0),
        )

    torch.manual_seed(17)
    model = ResidualMLP(
        ResidualMLPConfig(
            raw_factor_count=factor_count, hidden_dim=32, num_blocks=2, dropout=0.0
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    loop = TorchDateTrainingLoop(model, optimizer, batches, seed=11, gradient_clip_norm=1.0)
    initial_loss = loop.train_epoch(0)
    final_loss = initial_loss
    for epoch in range(1, 35):
        final_loss = loop.train_epoch(epoch)

    assert final_loss < initial_loss * 0.75

    model.eval()
    correlations: list[float] = []
    for batch in batches.values():
        with torch.no_grad():
            scores = model(batch.inputs)[0]
        centered_scores = scores - scores.mean()
        centered_ranks = batch.target_rank[0] - batch.target_rank[0].mean()
        correlations.append(
            float(
                (centered_scores * centered_ranks).sum()
                / (centered_scores.norm() * centered_ranks.norm())
            )
        )
    assert sum(correlations) / len(correlations) > 0.5
