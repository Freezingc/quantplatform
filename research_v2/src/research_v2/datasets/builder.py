"""Point-in-time joins from published universe, factors, and mature labels."""

from __future__ import annotations

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


class LockedDatasetError(PermissionError):
    """Raised before any input access when code tries to build the locked FINAL test."""


@dataclass(frozen=True)
class LockedTestAccess:
    """Explicit capability required to consume the one locked FINAL test."""

    locked_year: int
    authorization_id: str

    def __post_init__(self) -> None:
        if self.locked_year != 2026:
            raise LockedDatasetError("locked test access must target 2026")
        if not self.authorization_id.strip():
            raise LockedDatasetError("locked test access requires an authorization_id")


@dataclass(frozen=True)
class DatasetBuildRequest:
    fold_id: str
    split_role: SplitRole
    locked_access: LockedTestAccess | None = None

    def __post_init__(self) -> None:
        if not self.fold_id:
            raise ValueError("fold_id cannot be empty")
        is_locked_test = self.fold_id == "FINAL" and self.split_role == "TEST"
        if is_locked_test and self.locked_access is None:
            raise LockedDatasetError("FINAL test dataset is locked and cannot be built here")
        if not is_locked_test and self.locked_access is not None:
            raise LockedDatasetError("locked test access is only valid for FINAL TEST")


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


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"{name} is missing columns: {missing}")
