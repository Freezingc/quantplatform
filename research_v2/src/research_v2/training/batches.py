"""Supervised batch contract layered on top of keyed feature batches."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from research_v2.models.base import FeatureBatch, TemporalFeatureBatch

InputBatch = FeatureBatch | TemporalFeatureBatch


@dataclass(frozen=True)
class TrainingBatch:
    inputs: FeatureBatch
    target_raw: torch.Tensor
    target_rank: torch.Tensor

    def __post_init__(self) -> None:
        _validate_targets(self.inputs, self.target_raw, self.target_rank)


@dataclass(frozen=True)
class TemporalTrainingBatch:
    inputs: TemporalFeatureBatch
    target_raw: torch.Tensor
    target_rank: torch.Tensor

    def __post_init__(self) -> None:
        _validate_targets(self.inputs, self.target_raw, self.target_rank)


SupervisedBatch = TrainingBatch | TemporalTrainingBatch


def _validate_targets(
    inputs: InputBatch,
    target_raw: torch.Tensor,
    target_rank: torch.Tensor,
) -> None:
    expected_shape = inputs.stock_mask.shape
    if target_raw.shape != expected_shape:
        raise ValueError("target_raw must match [date, stock] input shape")
    if target_rank.shape != expected_shape:
        raise ValueError("target_rank must match [date, stock] input shape")
    valid = inputs.stock_mask
    padding = ~valid
    if not bool(torch.isfinite(target_raw[valid]).all().item()):
        raise ValueError("valid target_raw values must be finite")
    if not bool(torch.isfinite(target_rank[valid]).all().item()):
        raise ValueError("valid target_rank values must be finite")
    if bool(torch.isfinite(target_raw[padding]).any().item()) or bool(
        torch.isfinite(target_rank[padding]).any().item()
    ):
        raise ValueError("padding targets must be non-finite")
