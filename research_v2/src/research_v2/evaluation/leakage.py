"""Train-scope leakage canaries for fold-scoped research components."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from research_v2.datasets.preprocessing import FoldPreprocessor
from research_v2.datasets.selection import FactorSelectionResult


@dataclass(frozen=True)
class TrainScopeFingerprint:
    """Stable digest of state that must be independent of validation and test rows."""

    preprocessor_hash: str
    factor_selection_hash: str | None = None
    baseline_fit_hash: str | None = None

    @property
    def digest(self) -> str:
        return _stable_hash(
            {
                "baseline_fit_hash": self.baseline_fit_hash,
                "factor_selection_hash": self.factor_selection_hash,
                "preprocessor_hash": self.preprocessor_hash,
            }
        )


def fingerprint_train_scope(
    *,
    train_features: pd.DataFrame,
    factor_order: tuple[str, ...],
    preprocessor_factory: Callable[[tuple[str, ...]], FoldPreprocessor] = FoldPreprocessor,
    factor_selection: FactorSelectionResult | None = None,
    baseline_fit_hash: str | None = None,
) -> TrainScopeFingerprint:
    """Fingerprint only explicitly Train-derived state.

    A baseline fit digest can be supplied by a caller that hashes model parameters
    fitted on Train. Validation-based model selection is intentionally not accepted
    here because it is a separate, later decision.
    """
    preprocessor = preprocessor_factory(factor_order)
    preprocessor.fit(train_features, role="TRAIN")
    selection_hash = (
        _stable_hash(factor_selection.candidates.to_dict(orient="records"))
        if factor_selection is not None
        else None
    )
    return TrainScopeFingerprint(
        preprocessor_hash=preprocessor.state_hash,
        factor_selection_hash=selection_hash,
        baseline_fit_hash=baseline_fit_hash,
    )


def assert_train_scope_isolation(
    original: TrainScopeFingerprint,
    contaminated: TrainScopeFingerprint,
) -> None:
    """Raise when future-data contamination changes Train-derived state."""
    if original != contaminated:
        raise AssertionError("future observations changed Train-derived research state")


def _stable_hash(value: object) -> str:
    payload = json.dumps(
        _json_value(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_value(value: object) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return _json_value(value.item())
    return value
