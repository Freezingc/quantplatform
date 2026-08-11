"""Business-keyed feature batches and model prediction normalization."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol, cast

import pandas as pd
import torch


@dataclass(frozen=True)
class ModelSignature:
    model_family: str
    raw_factor_count: int
    config_hash: str

    def __post_init__(self) -> None:
        if not self.model_family or self.raw_factor_count < 1 or not self.config_hash:
            raise ValueError("model signature fields cannot be empty")


@dataclass(frozen=True)
class FeatureBatch:
    """A padded, date-isolated batch with all valid positions carrying real security keys."""

    signal_dates: tuple[date, ...]
    ts_codes: tuple[tuple[str, ...], ...]
    features: torch.Tensor
    feature_missing: torch.Tensor
    stock_mask: torch.Tensor

    def __post_init__(self) -> None:
        if self.features.ndim != 3:
            raise ValueError("features must have shape [date, stock, factor]")
        if self.feature_missing.shape != self.features.shape:
            raise ValueError("feature_missing must match features shape")
        if self.stock_mask.shape != self.features.shape[:2]:
            raise ValueError("stock_mask must have shape [date, stock]")
        if self.feature_missing.dtype is not torch.bool or self.stock_mask.dtype is not torch.bool:
            raise ValueError("feature_missing and stock_mask must be boolean tensors")
        date_count, stock_count, factor_count = self.features.shape
        if not self.signal_dates or factor_count < 1 or len(self.signal_dates) != date_count:
            raise ValueError("signal_dates and feature dimensions are inconsistent")
        has_wrong_code_shape = len(self.ts_codes) != date_count or any(
            len(codes) != stock_count for codes in self.ts_codes
        )
        if has_wrong_code_shape:
            raise ValueError("ts_codes must match [date, stock] dimensions")
        if not bool(torch.isfinite(self.features).all().item()):
            raise ValueError("features must be finite after preprocessing")
        for date_index, codes in enumerate(self.ts_codes):
            positions = self.stock_mask[date_index].tolist()
            valid_codes = [
                code for code, valid in zip(codes, positions, strict=True) if valid
            ]
            padding_codes = [
                code for code, valid in zip(codes, positions, strict=True) if not valid
            ]
            if any(not code for code in valid_codes) or len(valid_codes) != len(set(valid_codes)):
                raise ValueError("valid stocks require unique non-empty ts_codes")
            if any(code for code in padding_codes):
                raise ValueError("padding positions require empty ts_codes")


@dataclass(frozen=True)
class TemporalFeatureBatch:
    """A left-aligned causal history window for each stock at each signal date."""

    signal_dates: tuple[date, ...]
    ts_codes: tuple[tuple[str, ...], ...]
    features: torch.Tensor
    feature_missing: torch.Tensor
    stock_mask: torch.Tensor
    history_mask: torch.Tensor

    def __post_init__(self) -> None:
        if self.features.ndim != 4:
            raise ValueError("temporal features must have shape [date, stock, time, factor]")
        if self.feature_missing.shape != self.features.shape:
            raise ValueError("temporal feature_missing must match features shape")
        date_count, stock_count, history_length, factor_count = self.features.shape
        if self.stock_mask.shape != (date_count, stock_count):
            raise ValueError("temporal stock_mask must have shape [date, stock]")
        if self.history_mask.shape != (date_count, stock_count, history_length):
            raise ValueError("history_mask must have shape [date, stock, time]")
        if self.feature_missing.dtype is not torch.bool:
            raise ValueError("temporal feature_missing must be boolean")
        if self.stock_mask.dtype is not torch.bool or self.history_mask.dtype is not torch.bool:
            raise ValueError("temporal stock_mask and history_mask must be boolean")
        if not self.signal_dates or factor_count < 1 or len(self.signal_dates) != date_count:
            raise ValueError("temporal signal_dates and feature dimensions are inconsistent")
        has_wrong_code_shape = len(self.ts_codes) != date_count or any(
            len(codes) != stock_count for codes in self.ts_codes
        )
        if has_wrong_code_shape:
            raise ValueError("temporal ts_codes must match [date, stock] dimensions")
        if not bool(torch.isfinite(self.features).all().item()):
            raise ValueError("temporal features must be finite after preprocessing")
        for date_index, codes in enumerate(self.ts_codes):
            stock_positions = self.stock_mask[date_index].tolist()
            valid_codes = [
                code for code, valid in zip(codes, stock_positions, strict=True) if valid
            ]
            if any(not code for code in valid_codes) or len(valid_codes) != len(set(valid_codes)):
                raise ValueError("valid temporal stocks require unique non-empty ts_codes")
            for stock_index, is_stock in enumerate(stock_positions):
                history = self.history_mask[date_index, stock_index].tolist()
                if not is_stock and any(history):
                    raise ValueError("temporal padding stocks cannot have history")
                if is_stock and not any(history):
                    raise ValueError("valid temporal stocks require at least one history step")
                is_non_contiguous = any(
                    history[index] for index in range(1, len(history)) if not history[index - 1]
                )
                if is_non_contiguous:
                    raise ValueError("temporal history must be left-aligned and contiguous")
            if any(code for code, valid in zip(codes, stock_positions, strict=True) if not valid):
                raise ValueError("temporal padding positions require empty ts_codes")


PredictionInputBatch = FeatureBatch | TemporalFeatureBatch


class QuantModel(Protocol):
    """A score-producing model that never owns training data or selects positions."""

    model_family: str

    def __call__(self, batch: FeatureBatch) -> torch.Tensor: ...

    def artifact_signature(self) -> ModelSignature: ...


def predictions_to_frame(batch: PredictionInputBatch, scores: torch.Tensor) -> pd.DataFrame:
    """Drop padding and return stable business-keyed scores plus within-date percentile ranks."""
    if scores.shape != batch.stock_mask.shape:
        raise ValueError("scores must have shape [date, stock]")
    rows: list[dict[str, object]] = []
    for date_index, signal_date in enumerate(batch.signal_dates):
        valid_positions = (
            torch.nonzero(batch.stock_mask[date_index], as_tuple=False).flatten().tolist()
        )
        valid_scores = scores[date_index, valid_positions]
        if not bool(torch.isfinite(valid_scores).all().item()):
            raise ValueError("model emitted non-finite scores for valid stocks")
        daily = pd.DataFrame(
            {
                "ts_code": [batch.ts_codes[date_index][position] for position in valid_positions],
                "raw_score": valid_scores.detach().cpu().tolist(),
            }
        )
        daily["percentile_rank"] = daily["raw_score"].rank(method="average", pct=True)
        daily = daily.sort_values(
            ["raw_score", "ts_code"],
            ascending=[False, True],
            kind="mergesort",
        )
        for row in daily.to_dict(orient="records"):
            rows.append({"signal_date": signal_date, **cast(dict[str, object], row)})
    return pd.DataFrame(rows, columns=["signal_date", "ts_code", "raw_score", "percentile_rank"])
