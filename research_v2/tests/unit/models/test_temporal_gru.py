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
