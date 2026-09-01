from __future__ import annotations

from datetime import date

import torch

from research_v2.models.base import TemporalFeatureBatch
from research_v2.models.temporal import (
    TemporalGRU,
    TemporalGRUConfig,
    TemporalTCN,
    TemporalTCNConfig,
)
from research_v2.training.batches import TemporalTrainingBatch
from research_v2.training.torch_loop import TorchDateTrainingLoop


def _batch() -> TemporalFeatureBatch:
    features = torch.randn((1, 2, 20, 15), generator=torch.Generator().manual_seed(3))
    return TemporalFeatureBatch(
        signal_dates=(date(2025, 1, 30),),
        ts_codes=(("a", ""),),
        features=features,
        feature_missing=torch.zeros_like(features, dtype=torch.bool),
        stock_mask=torch.tensor([[True, False]]),
        history_mask=torch.tensor([[[True] * 15 + [False] * 5, [False] * 20]]),
    )


def test_gru_scores_only_valid_stocks_and_ignores_masked_history() -> None:
    model = TemporalGRU(TemporalGRUConfig(raw_factor_count=15, hidden_dim=16, dropout=0.0)).eval()
    batch = _batch()
    changed = TemporalFeatureBatch(
        signal_dates=batch.signal_dates,
        ts_codes=batch.ts_codes,
        features=batch.features.clone(),
        feature_missing=batch.feature_missing,
        stock_mask=batch.stock_mask,
        history_mask=batch.history_mask,
    )
    changed.features[0, 0, 15:] = 999.0

    first = model(batch)
    second = model(changed)

    assert first.shape == (1, 2)
    assert first[0, 1] == 0.0
    assert torch.equal(first, second)
    first.sum().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_causal_tcn_ignores_masked_history_and_padding() -> None:
    model = TemporalTCN(TemporalTCNConfig(raw_factor_count=15, hidden_dim=16, dropout=0.0)).eval()
    batch = _batch()
    changed_features = batch.features.clone()
    changed_features[0, 0, 15:] = -999.0
    changed = TemporalFeatureBatch(
        signal_dates=batch.signal_dates,
        ts_codes=batch.ts_codes,
        features=changed_features,
        feature_missing=batch.feature_missing,
        stock_mask=batch.stock_mask,
        history_mask=batch.history_mask,
    )

    assert torch.equal(model(batch), model(changed))
    assert model(batch)[0, 1] == 0.0


def test_temporal_models_accept_a_six_factor_frozen_set() -> None:
    gru = TemporalGRU(TemporalGRUConfig(raw_factor_count=6, hidden_dim=8, dropout=0.0))
    tcn = TemporalTCN(TemporalTCNConfig(raw_factor_count=6, hidden_dim=8, dropout=0.0))

    assert gru.artifact_signature().raw_factor_count == 6
    assert tcn.artifact_signature().raw_factor_count == 6


def _three_stock_batch() -> TemporalFeatureBatch:
    features = torch.randn((1, 4, 20, 6), generator=torch.Generator().manual_seed(11))
    return TemporalFeatureBatch(
        signal_dates=(date(2025, 1, 30),),
        ts_codes=(("a", "b", "c", ""),),
        features=features,
        feature_missing=torch.zeros_like(features, dtype=torch.bool),
        stock_mask=torch.tensor([[True, True, True, False]]),
        history_mask=torch.tensor(
            [[[True] * 20, [True] * 20, [True] * 20, [False] * 20]]
        ),
    )


def test_temporal_models_are_stock_permutation_equivariant() -> None:
    gru = TemporalGRU(TemporalGRUConfig(raw_factor_count=6, hidden_dim=8, dropout=0.0)).eval()
    tcn = TemporalTCN(TemporalTCNConfig(raw_factor_count=6, hidden_dim=8, dropout=0.0)).eval()
    batch = _three_stock_batch()
    permutation = [2, 0, 1]
    permuted = TemporalFeatureBatch(
        signal_dates=batch.signal_dates,
        ts_codes=(tuple(batch.ts_codes[0][index] for index in permutation + [3]),),
        features=batch.features[:, permutation + [3]],
        feature_missing=batch.feature_missing[:, permutation + [3]],
        stock_mask=batch.stock_mask[:, permutation + [3]],
        history_mask=batch.history_mask[:, permutation + [3]],
    )

    for model in (gru, tcn):
        original = model(batch)
        reordered = model(permuted)
        assert torch.allclose(reordered[0, :3], original[0, permutation], atol=1e-6)
        assert reordered[0, 3] == 0.0


def test_temporal_models_react_to_time_order_but_not_masked_history() -> None:
    gru = TemporalGRU(TemporalGRUConfig(raw_factor_count=6, hidden_dim=8, dropout=0.0)).eval()
    tcn = TemporalTCN(TemporalTCNConfig(raw_factor_count=6, hidden_dim=8, dropout=0.0)).eval()
    batch = _three_stock_batch()
    reversed_batch = TemporalFeatureBatch(
        signal_dates=batch.signal_dates,
        ts_codes=batch.ts_codes,
        features=batch.features.flip(dims=(2,)),
        feature_missing=batch.feature_missing,
        stock_mask=batch.stock_mask,
        history_mask=batch.history_mask,
    )
    masked_history_batch = TemporalFeatureBatch(
        signal_dates=batch.signal_dates,
        ts_codes=batch.ts_codes,
        features=batch.features.clone(),
        feature_missing=batch.feature_missing,
        stock_mask=batch.stock_mask,
        history_mask=batch.history_mask,
    )
    masked_history_batch.features[0, 3] = 123.0

    for model in (gru, tcn):
        original = model(batch)
        reversed_scores = model(reversed_batch)
        assert not torch.allclose(original, reversed_scores, atol=1e-4)
        assert torch.equal(model(masked_history_batch), original)


def test_temporal_gru_overfits_a_tiny_history_signal() -> None:
    factor_count = 6
    stock_count = 40
    history_length = 20
    dates = (date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4))
    batches: dict[date, TemporalTrainingBatch] = {}
    for day_index, signal_date in enumerate(dates):
        generator = torch.Generator().manual_seed(300 + day_index)
        alpha = torch.randn(stock_count, generator=generator)
        noise = torch.randn((stock_count, history_length, factor_count), generator=generator)
        features = noise * 0.05
        features[:, :, 0] = alpha.unsqueeze(1) + noise[:, :, 0] * 0.05
        target_raw = alpha.clone()
        ranks = target_raw.argsort().argsort().to(dtype=torch.float)
        inputs = TemporalFeatureBatch(
            signal_dates=(signal_date,),
            ts_codes=(tuple(f"s{index:02d}" for index in range(stock_count)),),
            features=features.unsqueeze(0),
            feature_missing=torch.zeros(
                (1, stock_count, history_length, factor_count), dtype=torch.bool
            ),
            stock_mask=torch.ones((1, stock_count), dtype=torch.bool),
            history_mask=torch.ones((1, stock_count, history_length), dtype=torch.bool),
        )
        batches[signal_date] = TemporalTrainingBatch(
            inputs=inputs,
            target_raw=target_raw.unsqueeze(0),
            target_rank=(ranks / (stock_count - 1)).unsqueeze(0),
        )

    torch.manual_seed(19)
    model = TemporalGRU(
        TemporalGRUConfig(raw_factor_count=factor_count, hidden_dim=32, dropout=0.0)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    loop = TorchDateTrainingLoop(model, optimizer, batches, seed=13, gradient_clip_norm=1.0)
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
