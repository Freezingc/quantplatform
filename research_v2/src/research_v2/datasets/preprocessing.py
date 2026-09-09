"""Fold-scoped robust preprocessing with train-only fitted state."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
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


_STATE_FIELDS = frozenset({"lower", "upper", "median", "iqr", "constant"})
_SETTING_FIELDS = frozenset(
    {"lower_quantile", "upper_quantile", "minimum_iqr", "transformed_clip"}
)


def _digest_state_payload(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def _restored_factor_order(value: object) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(name, str) and name for name in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError("preprocessing state factor_order must be unique non-empty names")
    return tuple(value)


def _restored_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"preprocessing state {name} must be a number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"preprocessing state {name} must be finite")
    return number


def _restored_settings(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != _SETTING_FIELDS:
        raise ValueError("preprocessing state settings are invalid")
    settings = {name: _restored_float(value[name], name) for name in sorted(_SETTING_FIELDS)}
    if not 0 <= settings["lower_quantile"] < settings["upper_quantile"] <= 1:
        raise ValueError("winsor quantiles must satisfy 0 <= lower < upper <= 1")
    return settings


def _restored_state(
    value: object, factor_order: tuple[str, ...]
) -> dict[str, FactorPreprocessingState]:
    if not isinstance(value, Mapping) or set(value) != set(factor_order):
        raise ValueError("preprocessing state must cover exactly the fitted factors")
    state: dict[str, FactorPreprocessingState] = {}
    for factor in factor_order:
        entry = value[factor]
        if not isinstance(entry, Mapping) or set(entry) != _STATE_FIELDS:
            raise ValueError(f"preprocessing state for {factor} is invalid")
        constant = entry["constant"]
        if not isinstance(constant, bool):
            raise ValueError(f"preprocessing state constant flag for {factor} must be a bool")
        state[factor] = FactorPreprocessingState(
            lower=_restored_float(entry["lower"], f"{factor}.lower"),
            upper=_restored_float(entry["upper"], f"{factor}.upper"),
            median=_restored_float(entry["median"], f"{factor}.median"),
            iqr=_restored_float(entry["iqr"], f"{factor}.iqr"),
            constant=constant,
        )
    return state


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

    def state_payload(self) -> dict[str, object]:
        """The fitted state, in exactly the canonical form ``state_hash`` digests.

        This is what a bundle must persist. Recording only the hash -- which is what the
        candidate driver did -- leaves a model that cannot be served: transforming a new frame
        needs the fitted bounds, and a hash cannot be inverted into them. The two are published
        together so the hash binds the payload rather than merely naming it.
        """
        if self._state is None:
            raise ValueError("preprocessor has not been fitted")
        return {
            "factor_order": list(self._factor_order),
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

    @property
    def fitted_factor_order(self) -> tuple[str, ...]:
        """The factor order this instance was fitted (or restored) with."""
        return self._factor_order

    @property
    def state_hash(self) -> str:
        """SHA-256 over the canonical state payload; the bundle records it beside the payload."""
        return _digest_state_payload(self.state_payload())

    @classmethod
    def restore(
        cls, payload: Mapping[str, object], *, expected_state_hash: str
    ) -> FoldPreprocessor:
        """Rebuild a fitted preprocessor from persisted state, bound to its recorded hash.

        The hash is recomputed over the payload the REBUILT object produces, not over the bytes
        handed in, so a payload that parses but canonicalises differently cannot pass. Restoring
        is what makes a persisted torch bundle servable: without it a consumer holds weights and
        a hash, and no way to put a new frame into the space the weights were trained in.
        """
        if not isinstance(expected_state_hash, str) or not expected_state_hash:
            raise ValueError("restoring a preprocessor requires the recorded state hash")
        if set(payload) != {"factor_order", "settings", "state"}:
            raise ValueError("preprocessing state payload fields are invalid")
        factor_order = _restored_factor_order(payload["factor_order"])
        settings = _restored_settings(payload["settings"])
        state = _restored_state(payload["state"], factor_order)
        restored = cls(factor_order=factor_order, **settings)
        restored._state = state
        if restored.state_hash != expected_state_hash:
            raise ValueError(
                "persisted preprocessing state does not match its recorded state hash"
            )
        return restored

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
