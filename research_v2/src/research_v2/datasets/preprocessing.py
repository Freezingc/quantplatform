"""Fold-scoped robust preprocessing with train-only fitted state."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

FitRole = Literal["TRAIN", "VALIDATION", "TEST"]


@dataclass(frozen=True)
class FactorPreprocessingState:
    lower: float
    upper: float
    median: float
    iqr: float
    constant: bool


@dataclass(frozen=True)
class PreprocessedBatch:
    values: pd.DataFrame
    missing_mask: pd.DataFrame


class FoldPreprocessor:
    """Fit robust factor transforms only once on one outer or inner TRAIN frame."""

    def __init__(
        self,
        factor_order: tuple[str, ...],
        *,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
        minimum_iqr: float = 1e-8,
        transformed_clip: float = 10.0,
    ) -> None:
        if not factor_order or len(set(factor_order)) != len(factor_order):
            raise ValueError("factor_order must be non-empty and unique")
        if not 0 <= lower_quantile < upper_quantile <= 1:
            raise ValueError("winsor quantiles must satisfy 0 <= lower < upper <= 1")
        self._factor_order = factor_order
        self._lower_quantile = lower_quantile
        self._upper_quantile = upper_quantile
        self._minimum_iqr = minimum_iqr
        self._transformed_clip = transformed_clip
        self._state: dict[str, FactorPreprocessingState] | None = None

    @property
    def state_hash(self) -> str:
        if self._state is None:
            raise ValueError("preprocessor has not been fitted")
        payload = {
            "factor_order": self._factor_order,
            "settings": {
                "lower_quantile": self._lower_quantile,
                "upper_quantile": self._upper_quantile,
                "minimum_iqr": self._minimum_iqr,
                "transformed_clip": self._transformed_clip,
            },
            "state": {
                factor: {
                    "lower": values.lower,
                    "upper": values.upper,
                    "median": values.median,
                    "iqr": values.iqr,
                    "constant": values.constant,
                }
                for factor, values in self._state.items()
            },
        }
        return hashlib.sha256(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()

    def fit(self, frame: pd.DataFrame, *, role: FitRole) -> None:
        """Fit once from TRAIN observations only."""
        if role != "TRAIN":
            raise ValueError("preprocessor fit is permitted only for TRAIN")
        if self._state is not None:
            raise ValueError("preprocessor may be fitted only once")
        self._validate_schema(frame)
        numeric = frame.loc[:, self._factor_order].replace([np.inf, -np.inf], np.nan)
        state: dict[str, FactorPreprocessingState] = {}
        for factor in self._factor_order:
            values = pd.to_numeric(numeric[factor], errors="coerce").dropna()
            if values.empty:
                raise ValueError(f"factor is all missing in TRAIN: {factor}")
            lower = float(values.quantile(self._lower_quantile))
            upper = float(values.quantile(self._upper_quantile))
            winsorized = values.clip(lower=lower, upper=upper)
            median = float(winsorized.median())
            iqr = float(winsorized.quantile(0.75) - winsorized.quantile(0.25))
            state[factor] = FactorPreprocessingState(
                lower=lower,
                upper=upper,
                median=median,
                iqr=iqr,
                constant=abs(iqr) < self._minimum_iqr,
            )
        self._state = state

    def transform(self, frame: pd.DataFrame) -> PreprocessedBatch:
        """Transform any role using frozen train-derived bounds, scale, and missing masks."""
        if self._state is None:
            raise ValueError("preprocessor has not been fitted")
        self._validate_schema(frame)
        values = pd.DataFrame(index=frame.index)
        masks = pd.DataFrame(index=frame.index)
        for factor in self._factor_order:
            raw = pd.to_numeric(frame[factor], errors="coerce").replace([np.inf, -np.inf], np.nan)
            masks[factor] = raw.isna()
            state = self._state[factor]
            scaled = (raw.clip(lower=state.lower, upper=state.upper) - state.median) / max(
                state.iqr,
                self._minimum_iqr,
            )
            values[factor] = scaled.clip(
                lower=-self._transformed_clip,
                upper=self._transformed_clip,
            ).fillna(0.0)
        return PreprocessedBatch(values=values, missing_mask=masks.astype(bool))

    def _validate_schema(self, frame: pd.DataFrame) -> None:
        if tuple(frame.columns) != self._factor_order:
            raise ValueError("factor frame columns must exactly match fitted factor order")
