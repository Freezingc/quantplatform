"""Hard model-input admission for frozen factor sets."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from research_v2.factors.factor_sets import FactorSet, FactorSetStatus

AdmissionRole = Literal["TRAIN", "VALIDATION", "TEST"]
DEFAULT_EXCLUDED_FACTOR_IDS = frozenset({"yeslab_0", "yeslab_124"})


class FactorAdmissionError(ValueError):
    """Raised when a feature frame cannot satisfy a frozen factor contract."""


@dataclass(frozen=True)
class FactorAdmissionContract:
    """Immutable factor identities, order, direction, and preprocessing linkage."""

    factor_set_id: str
    selection_scope_id: str
    factor_ids: tuple[str, ...]
    factor_versions: tuple[int, ...]
    factor_names: tuple[str, ...]
    frozen_directions: tuple[int, ...]
    rules_hash: str
    preprocessor_hash: str
    contract_hash: str

    @classmethod
    def from_factor_set(
        cls,
        factor_set: FactorSet,
        *,
        feature_names_by_factor_id: Mapping[str, str],
        rules_hash: str,
        preprocessor_hash: str,
        excluded_factor_ids: frozenset[str] = DEFAULT_EXCLUDED_FACTOR_IDS,
    ) -> FactorAdmissionContract:
        if factor_set.status is not FactorSetStatus.FROZEN:
            raise FactorAdmissionError("factor set must be FROZEN before admission")
        members = tuple(sorted(factor_set.members, key=lambda member: member.feature_order))
        if not members:
            raise FactorAdmissionError("factor set must contain at least one member")
        ids = tuple(member.factor_id for member in members)
        excluded = sorted(set(ids).intersection(excluded_factor_ids))
        if excluded:
            raise FactorAdmissionError(f"factor set contains excluded factor IDs: {excluded}")
        if any(member.feature_order != index for index, member in enumerate(members, start=1)):
            raise FactorAdmissionError("factor set feature_order must be contiguous")
        missing_names = sorted(set(ids).difference(feature_names_by_factor_id))
        if missing_names:
            raise FactorAdmissionError(f"missing feature names for factor IDs: {missing_names}")
        names = tuple(feature_names_by_factor_id[factor_id] for factor_id in ids)
        if any(not name.strip() for name in names) or len(set(names)) != len(names):
            raise FactorAdmissionError("factor feature names must be non-empty and unique")
        if not _is_sha256(rules_hash) or not _is_sha256(preprocessor_hash):
            raise FactorAdmissionError(
                "rules_hash and preprocessor_hash must be SHA-256 hex digests"
            )
        versions = tuple(member.factor_version for member in members)
        directions = tuple(member.frozen_direction for member in members)
        payload = {
            "factor_set_id": factor_set.factor_set_id,
            "selection_scope_id": factor_set.selection_scope_id,
            "factor_ids": ids,
            "factor_versions": versions,
            "factor_names": names,
            "frozen_directions": directions,
            "rules_hash": rules_hash,
            "preprocessor_hash": preprocessor_hash,
        }
        contract_hash = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return cls(
            factor_set_id=factor_set.factor_set_id,
            selection_scope_id=factor_set.selection_scope_id,
            factor_ids=ids,
            factor_versions=versions,
            factor_names=names,
            frozen_directions=directions,
            rules_hash=rules_hash,
            preprocessor_hash=preprocessor_hash,
            contract_hash=contract_hash,
        )


@dataclass(frozen=True)
class AdmissionResult:
    """Evidence produced after a frame passes the hard feature gate."""

    factor_set_id: str
    contract_hash: str
    role: AdmissionRole
    row_count: int
    factor_names: tuple[str, ...]
    finite_ratio_by_factor: tuple[tuple[str, float], ...]


def admit_feature_frame(
    contract: FactorAdmissionContract,
    frame: pd.DataFrame,
    *,
    role: AdmissionRole,
    minimum_finite_ratio: float = 0.70,
) -> AdmissionResult:
    """Reject any frame that does not exactly match the frozen factor contract."""
    if not 0 < minimum_finite_ratio <= 1:
        raise FactorAdmissionError("minimum_finite_ratio must be in (0, 1]")
    expected_columns = ("signal_date", "ts_code", *contract.factor_names)
    if tuple(frame.columns) != expected_columns:
        raise FactorAdmissionError(
            "feature columns must exactly match signal_date, ts_code, and frozen factor order"
        )
    if frame.empty:
        raise FactorAdmissionError("feature frame cannot be empty")
    if frame.duplicated(["signal_date", "ts_code"]).any():
        raise FactorAdmissionError("feature frame contains duplicate business keys")
    ratios: list[tuple[str, float]] = []
    for factor_name in contract.factor_names:
        values = pd.to_numeric(frame[factor_name], errors="coerce").to_numpy(dtype=float)
        finite_ratio = float(np.isfinite(values).mean()) if len(values) else 0.0
        if not math.isfinite(finite_ratio) or finite_ratio < minimum_finite_ratio:
            raise FactorAdmissionError(
                f"factor finite ratio below threshold: {factor_name} ({finite_ratio:.6f})"
            )
        ratios.append((factor_name, finite_ratio))
    return AdmissionResult(
        factor_set_id=contract.factor_set_id,
        contract_hash=contract.contract_hash,
        role=role,
        row_count=len(frame),
        factor_names=contract.factor_names,
        finite_ratio_by_factor=tuple(ratios),
    )


def _is_sha256(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )
