from __future__ import annotations

from datetime import date

import torch

from research_v2.models.base import FeatureBatch
from research_v2.models.residual_mlp import ResidualMLP, ResidualMLPConfig
from research_v2.training.batches import TrainingBatch
from research_v2.training.torch_loop import TorchDateTrainingLoop


def test_torch_loop_updates_parameters_by_complete_date() -> None:
    signal_date = date(2025, 1, 2)
    factor_count = 15
    count = 30
    features = torch.randn((1, count, factor_count), generator=torch.Generator().manual_seed(1))
    inputs = FeatureBatch(
        signal_dates=(signal_date,),
        ts_codes=(tuple(f"S{index:03d}" for index in range(count)),),
        features=features,
        feature_missing=torch.zeros_like(features, dtype=torch.bool),
        stock_mask=torch.ones((1, count), dtype=torch.bool),
    )
    target = features[:, :, 0]
    batch = TrainingBatch(inputs=inputs, target_raw=target, target_rank=target)
    model = ResidualMLP(
        ResidualMLPConfig(raw_factor_count=factor_count, hidden_dim=16, num_blocks=1)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loop = TorchDateTrainingLoop(
        model,
        optimizer,
        {signal_date: batch},
        seed=20260801,
        gradient_clip_norm=1.0,
    )
    before = [parameter.detach().clone() for parameter in model.parameters()]

    loss = loop.train_epoch(0)

    assert loss >= 0
    assert loop.last_gradient_norm is not None
    assert any(
        not torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=True)
    )
