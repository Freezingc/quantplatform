"""Validation-only keyed inference and RankIC evaluation for training callbacks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import date
from typing import cast

import pandas as pd
import torch
from torch import nn

from research_v2.evaluation.rank_ic import RankICSummary, compute_daily_rank_ic, summarize_rank_ic
from research_v2.models.base import TemporalFeatureBatch, predictions_to_frame
from research_v2.training.batches import SupervisedBatch, TemporalTrainingBatch, TrainingBatch


@dataclass(frozen=True)
class ValidationEvaluation:
    daily_rank_ic: pd.DataFrame
    summary: RankICSummary


def evaluate_validation_batches(
    model: nn.Module,
    batches: Mapping[date, SupervisedBatch],
    *,
    minimum_cross_section: int = 30,
) -> ValidationEvaluation:
    """Run inference only over validation batches and summarize date-equal RankIC."""
    if not batches:
        raise ValueError("validation evaluation requires at least one batch")
    if minimum_cross_section < 2:
        raise ValueError("minimum_cross_section must be at least two")
    prediction_frames: list[pd.DataFrame] = []
    label_rows: list[dict[str, object]] = []
    was_training = model.training
    model.eval()
    device = _model_device(model)
    try:
        with torch.no_grad():
            for signal_date in sorted(batches):
                batch = batches[signal_date]
                if batch.inputs.signal_dates != (signal_date,):
                    raise ValueError(
                        "each validation batch must contain exactly its mapping signal date"
                    )
                moved = _batch_to_device(batch, device)
                scores = cast(torch.Tensor, model(moved.inputs))
                prediction_frames.append(predictions_to_frame(moved.inputs, scores))
                label_rows.extend(_label_rows(moved))
    finally:
        model.train(was_training)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    labels = pd.DataFrame(label_rows)
    daily = compute_daily_rank_ic(
        predictions,
        labels,
        minimum_cross_section=minimum_cross_section,
        score_column="raw_score",
    )
    return ValidationEvaluation(daily_rank_ic=daily, summary=summarize_rank_ic(daily))


def _label_rows(batch: SupervisedBatch) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for date_index, signal_date in enumerate(batch.inputs.signal_dates):
        positions = (
            torch.nonzero(batch.inputs.stock_mask[date_index], as_tuple=False).flatten().tolist()
        )
        values = batch.target_raw[date_index].detach().cpu()
        for position in positions:
            rows.append(
                {
                    "signal_date": signal_date,
                    "ts_code": batch.inputs.ts_codes[date_index][position],
                    "raw_return_5d": float(values[position]),
                    "label_valid": True,
                }
            )
    return rows


def _model_device(model: nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    return parameter.device if parameter is not None else torch.device("cpu")


def _batch_to_device(batch: SupervisedBatch, device: torch.device) -> SupervisedBatch:
    inputs = batch.inputs
    if isinstance(inputs, TemporalFeatureBatch):
        temporal_inputs = replace(
            inputs,
            features=inputs.features.to(device),
            feature_missing=inputs.feature_missing.to(device),
            stock_mask=inputs.stock_mask.to(device),
            history_mask=inputs.history_mask.to(device),
        )
        return TemporalTrainingBatch(
            inputs=temporal_inputs,
            target_raw=batch.target_raw.to(device),
            target_rank=batch.target_rank.to(device),
        )
    static_inputs = replace(
        inputs,
        features=inputs.features.to(device),
        feature_missing=inputs.feature_missing.to(device),
        stock_mask=inputs.stock_mask.to(device),
    )
    return TrainingBatch(
        inputs=static_inputs,
        target_raw=batch.target_raw.to(device),
        target_rank=batch.target_rank.to(device),
    )
