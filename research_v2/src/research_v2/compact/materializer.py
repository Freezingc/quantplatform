"""Materialize current published rows into lineage-light compact frames."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

import numpy as np
import pandas as pd

from research_v2.compact.contracts import (
    CompactManifestStatus,
    LatestManifestEntry,
    compute_batch_id,
)

_BUSINESS_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "tradability_daily": (
        "trade_date",
        "ts_code",
        "is_st",
        "is_suspended",
        "is_listed",
        "listing_age_trading_days",
        "valid_days_20",
        "adv20",
        "is_eligible_pre_liquidity",
        "reason_bitmap",
        "rule_version",
    ),
    "universe_daily": (
        "signal_date",
        "universe_id",
        "universe_version",
        "ts_code",
        "is_member",
        "exclusion_reason",
        "reason_bitmap",
        "adv20_rank",
        "membership_rank",
    ),
    "label_daily": (
        "signal_date",
        "ts_code",
        "label_id",
        "label_version",
        "entry_date",
        "exit_date",
        "label_available_date",
        "raw_return",
        "rank_pct",
        "winsorized_return",
        "zscore_return",
        "neutral_residual",
        "label_valid",
        "invalid_reason",
        "entry_realized_tradable",
    ),
    "factor_observation_clean": (
        "signal_date",
        "ts_code",
        "factor_id",
        "factor_version",
        "value",
        "observation_date",
        "available_date",
        "report_period",
        "stale_days",
        "dedup_resolution",
        "source_revision_id",
    ),
}


def compact_business_columns(object_name: str) -> tuple[str, ...]:
    """Return the allowlisted business columns for one source object."""
    try:
        return _BUSINESS_COLUMNS[object_name]
    except KeyError as error:
        raise ValueError(f"unsupported compact object: {object_name}") from error


def build_compact_materialization(
    source: pd.DataFrame,
    *,
    object_name: str,
    source_data_version: str,
    logical_partition: str,
    business_columns: tuple[str, ...],
    business_keys: tuple[str, ...],
    partition_date_column: str,
    published_at: datetime | None = None,
) -> CompactMaterialization:
    """Build a compact batch from exactly one published source version."""
    allowed_columns = compact_business_columns(object_name)
    if business_columns != allowed_columns:
        raise ValueError(f"business columns do not match allowlist for {object_name}")
    if not source_data_version.strip() or not logical_partition.strip():
        raise ValueError("source version and logical partition cannot be empty")
    if not business_keys or not set(business_keys).issubset(business_columns):
        raise ValueError("business keys must be included in business columns")
    if partition_date_column not in business_columns:
        raise ValueError("partition date column must be a business column")
    required = set(business_columns) | {"data_version"}
    if missing := sorted(required.difference(source.columns)):
        raise ValueError(f"source is missing columns: {missing}")
    selected = source.loc[source["data_version"].astype(str).eq(source_data_version)].copy()
    if selected.empty:
        raise ValueError("source version has no rows")
    if selected.duplicated(list(business_keys)).any():
        raise ValueError("source contains duplicate business keys")
    dates = pd.to_datetime(selected[partition_date_column], errors="raise")
    if set(dates.dt.strftime("%Y-%m")) != {logical_partition}:
        raise ValueError("source rows do not match logical partition")
    _reject_nonfinite_values(selected, business_columns)
    compact = selected.loc[:, list(business_columns)].copy()
    batch_id = compute_batch_id(object_name, logical_partition, source_data_version)
    compact.insert(0, "batch_id", batch_id)
    business_hash = _business_hash(selected, business_columns, business_keys)
    timestamp = published_at or datetime.now(UTC)
    manifest = LatestManifestEntry(
        object_name=f"{object_name}_latest",
        logical_partition=logical_partition,
        batch_id=batch_id,
        source_object=object_name,
        source_data_version=source_data_version,
        published_at=timestamp,
        row_count=len(compact),
        business_hash=business_hash,
        status=CompactManifestStatus.STAGED,
    )
    return CompactMaterialization(frame=compact, manifest=manifest)


@dataclass(frozen=True)
class CompactMaterialization:
    """A compact frame and the manifest entry that describes it."""

    frame: pd.DataFrame
    manifest: LatestManifestEntry


def _business_hash(
    source: pd.DataFrame,
    columns: tuple[str, ...],
    business_keys: tuple[str, ...],
) -> str:
    ordered = source.loc[:, list(columns)].sort_values(
        list(business_keys), kind="mergesort", ignore_index=True
    )
    hashed = pd.util.hash_pandas_object(ordered, index=False).to_numpy(dtype="uint64")
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _reject_nonfinite_values(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for column in columns:
        values = frame[column]
        if not (pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values)):
            continue
        numeric = pd.to_numeric(values, errors="coerce")
        if np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError(f"business column contains non-finite values: {column}")
