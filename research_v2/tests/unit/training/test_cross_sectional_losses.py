from __future__ import annotations

from datetime import date

import pytest
import torch

from research_v2.models.base import FeatureBatch
from research_v2.training.batches import TrainingBatch
from research_v2.training.losses import LossWeights, cross_sectional_loss


def _batch(target: torch.Tensor) -> TrainingBatch:
    count = target.shape[1]
    inputs = FeatureBatch(
        signal_dates=(date(2025, 1, 2),),
        ts_codes=(tuple(f"S{index:03d}" for index in range(count)),),
        features=torch.zeros((1, count, 1)),
        feature_missing=torch.zeros((1, count, 1), dtype=torch.bool),
        stock_mask=torch.ones((1, count), dtype=torch.bool),
    )
    return TrainingBatch(inputs=inputs, target_raw=target, target_rank=target)


def test_ic_loss_distinguishes_same_and_reverse_cross_sectional_order() -> None:
    target = torch.linspace(0.0, 1.0, 30).unsqueeze(0)
    batch = _batch(target)

    aligned = cross_sectional_loss(target, batch, run_seed=7)
    reversed_scores = cross_sectional_loss(-target, batch, run_seed=7)

    assert aligned.ic_loss.item() == pytest.approx(0.0, abs=1e-6)
    assert reversed_scores.ic_loss.item() == pytest.approx(2.0, abs=1e-6)
    assert aligned.total.requires_grad is False

    differentiable_scores = target.clone().requires_grad_()
    differentiable = cross_sectional_loss(differentiable_scores, batch, run_seed=7)
    differentiable.total.backward()
    assert differentiable_scores.grad is not None
    assert torch.isfinite(differentiable_scores.grad).all()


def test_loss_ignores_padding_and_is_stock_permutation_invariant() -> None:
    values = torch.linspace(0.0, 1.0, 30)
    batch = _batch(values.unsqueeze(0))
    scores = values.unsqueeze(0)
    first = cross_sectional_loss(scores, batch, run_seed=99)
    permutation = torch.randperm(30, generator=torch.Generator().manual_seed(3))
    permuted_batch = _batch(values[permutation].unsqueeze(0))
    second = cross_sectional_loss(scores[:, permutation], permuted_batch, run_seed=99)

    assert first.total.item() == pytest.approx(second.total.item(), abs=1e-6)
    assert LossWeights().ic == pytest.approx(0.6)


def test_loss_rejects_when_no_date_has_a_valid_cross_section() -> None:
    target = torch.linspace(0.0, 1.0, 29).unsqueeze(0)
    with pytest.raises(ValueError, match="no valid"):
        cross_sectional_loss(target, _batch(target), run_seed=1)
