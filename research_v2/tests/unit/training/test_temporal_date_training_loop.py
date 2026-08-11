from __future__ import annotations

from datetime import date

import torch

from research_v2.models.base import TemporalFeatureBatch
from research_v2.models.temporal import TemporalGRU, TemporalGRUConfig
from research_v2.training.batches import TemporalTrainingBatch
from research_v2.training.torch_loop import TorchDateTrainingLoop


def test_date_training_loop_accepts_temporal_supervised_batches() -> None:
    signal_date = date(2025, 1, 30)
    features = torch.randn((1, 30, 20, 15), generator=torch.Generator().manual_seed(4))
    inputs = TemporalFeatureBatch(
        signal_dates=(signal_date,),
        ts_codes=(tuple(f"S{index:03d}" for index in range(30)),),
        features=features,
        feature_missing=torch.zeros_like(features, dtype=torch.bool),
        stock_mask=torch.ones((1, 30), dtype=torch.bool),
        history_mask=torch.ones((1, 30, 20), dtype=torch.bool),
    )
    target = features[:, :, -1, 0]
    batch = TemporalTrainingBatch(inputs=inputs, target_raw=target, target_rank=target)
    model = TemporalGRU(TemporalGRUConfig(raw_factor_count=15, hidden_dim=16, dropout=0.0))
    loop = TorchDateTrainingLoop(
        model,
        torch.optim.AdamW(model.parameters(), lr=0.01),
        {signal_date: batch},
        seed=20260801,
        gradient_clip_norm=1.0,
    )

    assert loop.train_epoch(0) >= 0
