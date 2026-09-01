"""Label-free fixed-three-seed inference and percentile-rank ensemble prediction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import pandas as pd
import torch
from torch import nn

from research_v2.models.base import PredictionInputBatch, predictions_to_frame
from research_v2.training.ensemble import REQUIRED_SEEDS, combine_seed_percentile_ranks


@dataclass(frozen=True)
class InferencePrediction:
    ensemble: pd.DataFrame
    seed_percentile_ranks: Mapping[int, pd.DataFrame]


def predict_three_seed_ensemble(
    batch: PredictionInputBatch,
    models: Mapping[int, nn.Module],
) -> InferencePrediction:
    """Generate all three seed ranks and combine them equally without any label access."""
    if set(models) != set(REQUIRED_SEEDS):
        raise ValueError("inference requires all three fixed seed models")
    seed_frames: dict[int, pd.DataFrame] = {}
    for seed in REQUIRED_SEEDS:
        model = models[seed]
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                scores = cast(torch.Tensor, model(batch))
                keyed = predictions_to_frame(batch, scores)
        finally:
            model.train(was_training)
        seed_frames[seed] = keyed.loc[:, ["signal_date", "ts_code", "percentile_rank"]]
    return InferencePrediction(
        ensemble=combine_seed_percentile_ranks(seed_frames),
        seed_percentile_ranks=seed_frames,
    )
