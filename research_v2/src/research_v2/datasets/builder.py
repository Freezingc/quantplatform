"""Point-in-time joins from published universe, factors, and mature labels."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from datetime import date
from typing import Literal

import pandas as pd

from research_v2.factors.admission import (
    FactorAdmissionContract,
    FactorAdmissionError,
    admit_feature_frame,
)

SplitRole = Literal["TRAIN", "VALIDATION", "TEST"]
_VALID_SPLIT_ROLES = frozenset({"TRAIN", "VALIDATION", "TEST"})
_LOCKED_ACCESS_KEY_ENV = "QRV2_LOCKED_TEST_ACCESS_HMAC_KEY"


class LockedDatasetError(PermissionError):
    """Raised before any input access when code tries to build the locked FINAL test."""


@dataclass(frozen=True)
class LockedTestAccess:
    """Explicit capability required to consume the one locked FINAL test."""

    locked_year: int
    authorization_id: str
    signature: str = ""

    def __post_init__(self) -> None:
        if self.locked_year != 2026:
            raise LockedDatasetError("locked test access must target 2026")
        if not self.authorization_id.strip():
            raise LockedDatasetError("locked test access requires an authorization_id")

    @classmethod
    def issue(
        cls,
        *,
        locked_year: int,
        authorization_id: str,
        signing_key: bytes,
    ) -> LockedTestAccess:
        """Create an authority-signed access record that data boundaries can verify."""
        unsigned = cls(locked_year=locked_year, authorization_id=authorization_id)
        if not isinstance(signing_key, bytes) or not signing_key:
            raise LockedDatasetError("locked test access signing_key must be non-empty bytes")
        return cls(
            locked_year=locked_year,
            authorization_id=authorization_id,
            signature=_locked_access_signature(signing_key, unsigned.payload()),
        )

    def payload(self) -> dict[str, object]:
        return {
            "authorization_id": self.authorization_id,
            "locked_year": self.locked_year,
        }

    def verify(self) -> None:
        """Fail closed unless the access was signed by the locked-test authority key."""
        secret = os.environ.get(_LOCKED_ACCESS_KEY_ENV, "")
        if not secret:
            raise LockedDatasetError(f"locked test access requires {_LOCKED_ACCESS_KEY_ENV}")
        expected = _locked_access_signature(secret.encode("utf-8"), self.payload())
        if not self.signature or not hmac.compare_digest(self.signature, expected):
            raise LockedDatasetError("locked test access signature verification failed")


@dataclass(frozen=True)
class DatasetBuildRequest:
    fold_id: str
    split_role: SplitRole
    locked_access: LockedTestAccess | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.split_role, str) or self.split_role not in _VALID_SPLIT_ROLES:
            raise ValueError("split_role must be one of TRAIN, VALIDATION, TEST")
        if not self.fold_id:
            raise ValueError("fold_id cannot be empty")
        is_locked_test = self.fold_id == "FINAL" and self.split_role == "TEST"
        if is_locked_test and self.locked_access is None:
            raise LockedDatasetError("FINAL test dataset is locked and cannot be built here")
        if not is_locked_test and self.locked_access is not None:
            raise LockedDatasetError("locked test access is only valid for FINAL TEST")
        if self.locked_access is not None:
            self.locked_access.verify()


@dataclass(frozen=True)
class FactorInput:
    factor_id: str
    factor_version: int
    feature_name: str

    def __post_init__(self) -> None:
        if not self.factor_id or self.factor_version < 1 or not self.feature_name:
            raise ValueError("factor input needs an id, positive version, and feature name")


def build_point_in_time_dataset(
    universe: pd.DataFrame,
    factors: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    factor_members: tuple[FactorInput, ...],
    segment_end: date,
    request: DatasetBuildRequest,
    admission_contract: FactorAdmissionContract | None = None,
    minimum_finite_ratio: float = 0.70,
) -> pd.DataFrame:
    """Build a stable feature frame anchored only on the eligible daily universe.

    `DatasetBuildRequest` is deliberately constructed before this function receives data in
    production callers; the explicit guard below makes the FINAL test protection robust even
    if a caller bypasses the dataclass constructor.
    """
    if (
        request.fold_id == "FINAL"
        and request.split_role == "TEST"
        and request.locked_access is None
    ):
        raise LockedDatasetError("FINAL test dataset is locked and cannot be built here")
    if _is_authorized_locked_test(request):
        assert request.locked_access is not None
        request.locked_access.verify()
        _reject_inputs_outside_locked_year(
            ("universe", universe),
            ("factors", factors),
            ("labels", labels),
        )
    else:
        _reject_locked_or_future_inputs(
            ("universe", universe),
            ("factors", factors),
            ("labels", labels),
        )
    if not factor_members:
        raise ValueError("factor_members cannot be empty")
    feature_names = [member.feature_name for member in factor_members]
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("factor feature names must be unique")
    identities = [(member.factor_id, member.factor_version) for member in factor_members]
    if len(set(identities)) != len(identities):
        raise ValueError("factor members cannot repeat an id and version")
    if admission_contract is not None:
        expected_identities = tuple(
            zip(
                admission_contract.factor_ids,
                admission_contract.factor_versions,
                strict=True,
            )
        )
        if tuple(identities) != expected_identities:
            raise FactorAdmissionError(
                "factor members must exactly match frozen factor IDs, versions, and order"
            )
    _require_columns(universe, {"signal_date", "ts_code"}, "universe")
    _require_columns(
        factors,
        {"signal_date", "ts_code", "factor_id", "factor_version", "value", "available_date"},
        "factors",
    )
    normalized_label_input = _normalize_label_return_column(labels)
    _require_columns(
        normalized_label_input,
        {"signal_date", "ts_code", "raw_return_5d", "label_valid", "exit_date"},
        "labels",
    )
    keys = ["signal_date", "ts_code"]
    if universe.duplicated(keys).any():
        raise ValueError("universe contains duplicate business keys")
    if normalized_label_input.duplicated(keys).any():
        raise ValueError("labels contains duplicate business keys")

    result = _normalize_dates(universe.loc[:, keys].copy())
    normalized_factors = _normalize_dates(factors.copy())
    normalized_labels = _normalize_dates(normalized_label_input.copy())
    normalized_labels["exit_date"] = pd.to_datetime(
        normalized_labels["exit_date"], errors="raise"
    ).dt.date
    if normalized_labels["exit_date"].isna().any():
        raise ValueError("labels contain null exit_date")
    if (normalized_labels["exit_date"] > segment_end).any():
        raise ValueError("labels contain an exit_date after the segment end")

    for member in factor_members:
        member_rows = normalized_factors.loc[
            (normalized_factors["factor_id"].astype(str) == member.factor_id)
            & (normalized_factors["factor_version"].astype(int) == member.factor_version),
            keys + ["value", "available_date"],
        ].copy()
        if member_rows.duplicated(keys).any():
            raise ValueError(f"factor has duplicate business keys: {member.factor_id}")
        member_rows["available_date"] = pd.to_datetime(
            member_rows["available_date"], errors="raise"
        ).dt.date
        if member_rows["available_date"].isna().any():
            raise ValueError(f"factor has null available_date: {member.factor_id}")
        if (member_rows["available_date"] > member_rows["signal_date"]).any():
            raise ValueError(f"factor is not available by signal date: {member.factor_id}")
        result = result.merge(
            member_rows.loc[:, keys + ["value"]].rename(columns={"value": member.feature_name}),
            on=keys,
            how="left",
            validate="one_to_one",
        )

    result = result.merge(
        normalized_labels.loc[:, keys + ["raw_return_5d", "label_valid", "exit_date"]],
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    result = result.sort_values(keys, kind="mergesort", ignore_index=True)
    if admission_contract is not None:
        admitted_frame = result.loc[:, keys + feature_names]
        admit_feature_frame(
            admission_contract,
            admitted_frame,
            role=request.split_role,
            minimum_finite_ratio=minimum_finite_ratio,
        )
    return result


def _normalize_dates(frame: pd.DataFrame) -> pd.DataFrame:
    frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="raise").dt.date
    return frame


def _normalize_label_return_column(labels: pd.DataFrame) -> pd.DataFrame:
    has_internal_name = "raw_return_5d" in labels.columns
    has_v2_name = "raw_return" in labels.columns
    if has_internal_name and has_v2_name:
        raise ValueError("labels cannot contain both raw_return and raw_return_5d")
    if has_v2_name:
        return labels.rename(columns={"raw_return": "raw_return_5d"})
    return labels


def _is_authorized_locked_test(request: DatasetBuildRequest) -> bool:
    return (
        request.fold_id == "FINAL"
        and request.split_role == "TEST"
        and request.locked_access is not None
    )


def _locked_access_signature(signing_key: bytes, payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hmac.new(signing_key, encoded.encode("utf-8"), hashlib.sha256).hexdigest()


def _reject_locked_or_future_inputs(
    *named_frames: tuple[str, pd.DataFrame],
) -> None:
    """Keep ordinary dataset construction outside FINAL/2026 and later data."""
    for name, frame in named_frames:
        if "signal_date" not in frame.columns:
            continue
        dates = pd.to_datetime(frame["signal_date"], errors="raise")
        if (dates.dt.year >= 2026).any():
            raise LockedDatasetError(
                f"ordinary dataset construction cannot consume 2026 FINAL or future dates: {name}"
            )


def _reject_inputs_outside_locked_year(
    *named_frames: tuple[str, pd.DataFrame],
) -> None:
    """Keep an authorized FINAL request limited to its declared 2026 segment."""
    for name, frame in named_frames:
        if "signal_date" not in frame.columns:
            continue
        dates = pd.to_datetime(frame["signal_date"], errors="raise")
        if (dates.dt.year != 2026).any():
            raise LockedDatasetError(
                f"authorized FINAL test construction only accepts 2026 dates: {name}"
            )


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"{name} is missing columns: {missing}")
