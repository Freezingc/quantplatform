"""Date-equal cross-sectional ranking losses for quant model candidates."""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from datetime import date
from typing import cast

import torch
from torch.nn import functional as functional

from research_v2.training.batches import SupervisedBatch


@dataclass(frozen=True)
class LossWeights:
    ic: float = 0.60
    pair: float = 0.30
    huber: float = 0.10

    def __post_init__(self) -> None:
        if min(self.ic, self.pair, self.huber) < 0:
            raise ValueError("loss weights cannot be negative")
        if abs(self.ic + self.pair + self.huber - 1.0) > 1e-12:
            raise ValueError("loss weights must sum to one")


@dataclass(frozen=True)
class LossBreakdown:
    total: torch.Tensor
    ic_loss: torch.Tensor
    pair_loss: torch.Tensor
    huber_loss: torch.Tensor
    valid_date_count: int


DEFAULT_LOSS_WEIGHTS = LossWeights()


def cross_sectional_loss(
    scores: torch.Tensor,
    batch: SupervisedBatch,
    *,
    run_seed: int,
    weights: LossWeights = DEFAULT_LOSS_WEIGHTS,
    minimum_valid_stocks: int = 30,
    maximum_pairs_per_date: int = 4096,
    huber_beta: float = 0.01,
) -> LossBreakdown:
    """Compute IC/pair/Huber loss by date, then average dates equally.

    All random-looking pair choices derive only from the registered run seed, signal
    date, and security keys; batching order and CUDA scheduling cannot change them.
    """
    if scores.shape != batch.inputs.stock_mask.shape:
        raise ValueError("scores must match the batch [date, stock] shape")
    if minimum_valid_stocks < 2 or maximum_pairs_per_date < 1 or huber_beta <= 0:
        raise ValueError("loss configuration thresholds are invalid")
    ic_losses: list[torch.Tensor] = []
    pair_losses: list[torch.Tensor] = []
    huber_losses: list[torch.Tensor] = []
    for date_index, signal_date in enumerate(batch.inputs.signal_dates):
        valid_positions = (
            torch.nonzero(batch.inputs.stock_mask[date_index], as_tuple=False).flatten()
        )
        if len(valid_positions) < minimum_valid_stocks:
            continue
        daily_scores = scores[date_index, valid_positions]
        target_rank = batch.target_rank[date_index, valid_positions]
        target_raw = batch.target_raw[date_index, valid_positions]
        if not bool(torch.isfinite(daily_scores).all().item()):
            raise ValueError("scores contain non-finite values for valid stocks")
        correlation = _pearson_correlation(daily_scores, target_rank)
        if correlation is None:
            continue
        ic_losses.append(1.0 - correlation)
        codes = tuple(
            batch.inputs.ts_codes[date_index][int(position)] for position in valid_positions
        )
        pair_losses.append(
            _pairwise_loss(
                daily_scores,
                target_raw,
                signal_date,
                codes,
                run_seed=run_seed,
                maximum_pairs=maximum_pairs_per_date,
            )
        )
        centered_target = target_raw - target_raw.mean()
        huber_losses.append(
            functional.smooth_l1_loss(daily_scores, centered_target, beta=huber_beta)
        )
    if not ic_losses:
        raise ValueError("no valid date cross-section is available for loss calculation")
    ic_loss = torch.stack(ic_losses).mean()
    pair_loss = torch.stack(pair_losses).mean()
    huber_loss = torch.stack(huber_losses).mean()
    total = weights.ic * ic_loss + weights.pair * pair_loss + weights.huber * huber_loss
    return LossBreakdown(total, ic_loss, pair_loss, huber_loss, len(ic_losses))


def _pearson_correlation(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor | None:
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = torch.linalg.vector_norm(left_centered) * torch.linalg.vector_norm(right_centered)
    if float(denominator.detach().cpu()) == 0.0:
        return None
    return cast(torch.Tensor, (left_centered * right_centered).sum() / denominator)


def _pairwise_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    signal_date: date,
    codes: tuple[str, ...],
    *,
    run_seed: int,
    maximum_pairs: int,
) -> torch.Tensor:
    pairs = _stable_pairs(codes, signal_date, run_seed, maximum_pairs)
    differences = torch.stack([torch.abs(targets[left] - targets[right]) for left, right in pairs])
    threshold = torch.quantile(differences, 0.5)
    informative = [
        pair for pair, difference in zip(pairs, differences, strict=True) if difference >= threshold
    ]
    if not informative:
        return scores.sum() * 0.0
    pair_terms = []
    for left, right in informative:
        direction = torch.sign(targets[left] - targets[right])
        pair_terms.append(functional.softplus(-direction * (scores[left] - scores[right])))
    return torch.stack(pair_terms).mean()


def _stable_pairs(
    codes: tuple[str, ...],
    signal_date: date,
    run_seed: int,
    maximum_pairs: int,
) -> list[tuple[int, int]]:
    total_pair_count = len(codes) * (len(codes) - 1) // 2
    if total_pair_count <= maximum_pairs:
        pairs = list(itertools.combinations(range(len(codes)), 2))
    else:
        pairs = []
        seen: set[tuple[int, int]] = set()
        counter = 0
        while len(pairs) < maximum_pairs * 2 and len(seen) < total_pair_count:
            digest = hashlib.sha256(
                f"{run_seed}|{signal_date.isoformat()}|{counter}".encode()
            ).digest()
            left = int.from_bytes(digest[:8], "big") % len(codes)
            right = int.from_bytes(digest[8:16], "big") % len(codes)
            counter += 1
            if left == right:
                continue
            pair = (min(left, right), max(left, right))
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
    return sorted(
        pairs,
        key=lambda pair: hashlib.sha256(
            f"{run_seed}|{signal_date.isoformat()}|{codes[pair[0]]}|{codes[pair[1]]}".encode()
        ).hexdigest(),
    )[:maximum_pairs]
