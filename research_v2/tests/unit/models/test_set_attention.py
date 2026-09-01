from __future__ import annotations

from datetime import date

import pytest
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


def test_set_attention_signature_binds_dropout_for_artifact_reconstruction() -> None:
    common = {
        "raw_factor_count": 6,
        "hidden_dim": 8,
        "num_heads": 2,
        "inducing_points": 2,
        "num_blocks": 1,
    }
    without_dropout = SetAttention(SetAttentionConfig(**common, dropout=0.0))
    with_dropout = SetAttention(SetAttentionConfig(**common, dropout=0.2))

    assert (
        without_dropout.artifact_signature().config_hash
        != with_dropout.artifact_signature().config_hash
    )


def test_set_attention_isolates_dates_within_one_packed_batch() -> None:
    model = SetAttention(
        SetAttentionConfig(
            raw_factor_count=15, hidden_dim=16, num_heads=2, inducing_points=4, num_blocks=1
        )
    ).eval()
    features = torch.randn((2, 3, 15), generator=torch.Generator().manual_seed(8))
    batch = FeatureBatch(
        signal_dates=(date(2025, 1, 2), date(2025, 1, 3)),
        ts_codes=(("a", "b", ""), ("c", "d", "e")),
        features=features,
        feature_missing=torch.zeros_like(features, dtype=torch.bool),
        stock_mask=torch.tensor([[True, True, False], [True, True, True]]),
    )
    single_first = FeatureBatch(
        signal_dates=(date(2025, 1, 2),),
        ts_codes=(("a", "b", ""),),
        features=features[:1],
        feature_missing=batch.feature_missing[:1],
        stock_mask=batch.stock_mask[:1],
    )
    single_second = FeatureBatch(
        signal_dates=(date(2025, 1, 3),),
        ts_codes=(("c", "d", "e"),),
        features=features[1:],
        feature_missing=batch.feature_missing[1:],
        stock_mask=batch.stock_mask[1:],
    )

    packed = model(batch)
    assert torch.allclose(packed[0], model(single_first)[0], atol=1e-6)
    assert torch.allclose(packed[1], model(single_second)[0], atol=1e-6)


def test_set_attention_extreme_cross_section_cpu_equivalent() -> None:
    model = SetAttention(
        SetAttentionConfig(
            raw_factor_count=60,
            hidden_dim=128,
            num_heads=4,
            inducing_points=64,
            num_blocks=2,
            dropout=0.0,
        )
    )
    stock_count = 256
    batch = FeatureBatch(
        signal_dates=(date(2025, 1, 2),),
        ts_codes=(tuple(f"s{index:04d}" for index in range(stock_count)),),
        features=torch.randn(
            (1, stock_count, 60), generator=torch.Generator().manual_seed(9)
        ),
        feature_missing=torch.zeros((1, stock_count, 60), dtype=torch.bool),
        stock_mask=torch.ones((1, stock_count), dtype=torch.bool),
    )

    scores = model(batch)
    loss = scores[batch.stock_mask].pow(2).mean()
    loss.backward()  # type: ignore[no-untyped-call]

    assert scores.shape == (1, stock_count)
    assert bool(torch.isfinite(scores).all().item())
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
        for parameter in model.parameters()
    )


def test_set_attention_cuda_capacity_stays_under_twenty_gib() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA capacity budget test requires a GPU")
    model = SetAttention(
        SetAttentionConfig(
            raw_factor_count=60,
            hidden_dim=128,
            num_heads=4,
            inducing_points=64,
            num_blocks=2,
            dropout=0.0,
        )
    ).cuda()
    stock_count = 6000
    batch = FeatureBatch(
        signal_dates=(date(2025, 1, 2),),
        ts_codes=(tuple(f"s{index:04d}" for index in range(stock_count)),),
        features=torch.randn(
            (1, stock_count, 60), generator=torch.Generator().manual_seed(9)
        ).cuda(),
        feature_missing=torch.zeros(
            (1, stock_count, 60), dtype=torch.bool, device="cuda"
        ),
        stock_mask=torch.ones((1, stock_count), dtype=torch.bool, device="cuda"),
    )

    torch.cuda.reset_peak_memory_stats()
    scores = model(batch)
    scores[batch.stock_mask].pow(2).mean().backward()

    peak_gib = torch.cuda.max_memory_allocated() / float(1024**3)
    assert peak_gib < 20.0
