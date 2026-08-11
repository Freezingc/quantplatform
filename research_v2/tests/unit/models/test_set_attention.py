from __future__ import annotations

from datetime import date

import torch

from research_v2.models.base import FeatureBatch
from research_v2.models.set_attention import SetAttention, SetAttentionConfig


def _batch() -> FeatureBatch:
    return FeatureBatch(
        signal_dates=(date(2025, 1, 2),),
        ts_codes=(("a", "b", "c", "d", "", ""),),
        features=torch.randn((1, 6, 15), generator=torch.Generator().manual_seed(5)),
        feature_missing=torch.zeros((1, 6, 15), dtype=torch.bool),
        stock_mask=torch.tensor([[True, True, True, True, False, False]]),
    )


def test_set_attention_is_permutation_equivariant_and_padding_isolated() -> None:
    model = SetAttention(
        SetAttentionConfig(
            raw_factor_count=15,
            hidden_dim=16,
            num_heads=2,
            inducing_points=4,
            num_blocks=1,
        )
    ).eval()
    batch = _batch()
    scores = model(batch)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = FeatureBatch(
        signal_dates=batch.signal_dates,
        ts_codes=(tuple(batch.ts_codes[0][index] for index in permutation.tolist()) + ("", ""),),
        features=torch.cat((batch.features[:, permutation], batch.features[:, 4:]), dim=1),
        feature_missing=torch.cat(
            (batch.feature_missing[:, permutation], batch.feature_missing[:, 4:]),
            dim=1,
        ),
        stock_mask=batch.stock_mask,
    )
    changed_padding = FeatureBatch(
        signal_dates=batch.signal_dates,
        ts_codes=batch.ts_codes,
        features=batch.features.clone(),
        feature_missing=batch.feature_missing,
        stock_mask=batch.stock_mask,
    )
    changed_padding.features[:, 4:] = 999.0

    assert torch.allclose(model(permuted)[0, :4], scores[0, permutation], atol=1e-6)
    assert torch.equal(model(changed_padding), scores)
    assert torch.equal(scores[0, 4:], torch.zeros(2))


def test_set_attention_accepts_a_six_factor_frozen_set() -> None:
    model = SetAttention(
        SetAttentionConfig(raw_factor_count=6, hidden_dim=8, num_heads=2, inducing_points=2)
    )

    assert model.artifact_signature().raw_factor_count == 6
