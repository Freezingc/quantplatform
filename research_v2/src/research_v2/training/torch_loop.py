"""Concrete date-at-a-time PyTorch optimization loop for keyed TrainingBatch data."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from datetime import date
from typing import cast

import torch
from torch import nn

from research_v2.models.base import TemporalFeatureBatch
from research_v2.training.batches import SupervisedBatch, TemporalTrainingBatch, TrainingBatch
from research_v2.training.losses import DEFAULT_LOSS_WEIGHTS, LossWeights, cross_sectional_loss
from research_v2.training.sampler import DateBalancedSampler


class TorchDateTrainingLoop:
    """Optimize one complete cross-section at a time so every date has equal weight."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batches: Mapping[date, SupervisedBatch],
        *,
        seed: int,
        gradient_clip_norm: float,
        loss_weights: LossWeights = DEFAULT_LOSS_WEIGHTS,
    ) -> None:
        if not batches:
            raise ValueError("training loop requires at least one date batch")
        if gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        for signal_date, batch in batches.items():
            if batch.inputs.signal_dates != (signal_date,):
                raise ValueError("each training batch must contain exactly its mapping signal date")
        self._model = model
        self._optimizer = optimizer
        self._batches = dict(batches)
        self._sampler = DateBalancedSampler(tuple(batches), seed=seed)
        self._seed = seed
        self._gradient_clip_norm = gradient_clip_norm
        self._loss_weights = loss_weights
        self._last_gradient_norm: float | None = None

    @property
    def last_gradient_norm(self) -> float | None:
        return self._last_gradient_norm

    def train_epoch(self, epoch: int) -> float:
        """Run one optimizer update per complete date and return mean date-equal loss."""
        self._model.train()
        losses: list[float] = []
        device = _model_device(self._model)
        for signal_date in self._sampler.for_epoch(epoch):
            batch = _batch_to_device(self._batches[signal_date], device)
            self._optimizer.zero_grad(set_to_none=True)
            scores = cast(torch.Tensor, self._model(batch.inputs))
            breakdown = cross_sectional_loss(
                scores,
                batch,
                run_seed=self._seed,
                weights=self._loss_weights,
            )
            breakdown.total.backward()  # type: ignore[no-untyped-call]
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self._model.parameters(),
                self._gradient_clip_norm,
            )
            self._last_gradient_norm = float(gradient_norm.detach().cpu())
            if not math.isfinite(self._last_gradient_norm):
                raise ValueError("non-finite gradient norm")
            self._optimizer.step()
            losses.append(float(breakdown.total.detach().cpu()))
        return sum(losses) / len(losses)


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
