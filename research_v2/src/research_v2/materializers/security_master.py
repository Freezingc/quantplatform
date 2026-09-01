"""Build auditable security-master SCD batches from readonly YesLab extracts."""

from __future__ import annotations

import hashlib
import json
from datetime import date

import pandas as pd

from research_v2.contracts.source import SourceBatch
from research_v2.db.publish import StagedBatch
from research_v2.pit.security_master import build_security_master_scd
from research_v2.runs.context import RunContext

_RUN_LINEAGE_COLUMNS = frozenset({"run_id", "data_version", "created_at"})


def _watermark_id(*batches: SourceBatch) -> str:
    payload = "|".join(sorted(batch.watermark_id for batch in batches))
    return hashlib.sha256(payload.encode()).hexdigest()


def _record_hash(row: pd.Series) -> str:
    payload = {
        column: (value.isoformat() if isinstance(value, (date, pd.Timestamp)) else value)
        for column, value in row.items()
        if column not in _RUN_LINEAGE_COLUMNS
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True).encode()
    ).hexdigest()


def _merge_historical_snapshot(
    primary: SourceBatch,
    historical: SourceBatch | None,
    derived: SourceBatch | None,
) -> tuple[pd.DataFrame, tuple[SourceBatch, ...]]:
    snapshots = (
        (historical, "tushare_stock_basic", "historical security snapshot"),
        (derived, "derived_market_security_profile", "derived security profile"),
    )
    merged = primary.frame.copy(deep=True)
    source_batches: list[SourceBatch] = [primary]
    for candidate, expected_table, label in snapshots:
        if candidate is None:
            continue
        if candidate.table_name != expected_table:
            raise ValueError(f"{label} must be a {expected_table} batch")
        missing_columns = sorted(set(primary.frame.columns).difference(candidate.frame.columns))
        if missing_columns:
            raise ValueError(f"{label} is missing columns: {missing_columns}")
        if candidate.frame["ts_code"].duplicated().any():
            raise ValueError(f"{label} contains duplicate ts_code values")
        existing_codes = set(merged["ts_code"].astype(str))
        additions = candidate.frame.loc[
            ~candidate.frame["ts_code"].astype(str).isin(existing_codes), primary.frame.columns
        ]
        merged = pd.concat([merged, additions], ignore_index=True)
        source_batches.append(candidate)
    return merged, tuple(source_batches)


def build_security_master_batch(
    snapshot: SourceBatch,
    name_history: SourceBatch,
    context: RunContext,
    *,
    as_of_date: date,
    coverage_start: date = date(2013, 1, 1),
    historical_snapshot: SourceBatch | None = None,
    derived_snapshot: SourceBatch | None = None,
) -> StagedBatch:
    """Build a publish-ready SCD batch with data lineage bound to both source extracts."""
    if snapshot.table_name != "stock_info_table" or name_history.table_name != "stock_used_name":
        raise ValueError("security master requires stock_info_table and stock_used_name batches")
    securities, source_batches = _merge_historical_snapshot(
        snapshot,
        historical_snapshot,
        derived_snapshot,
    )
    scd = build_security_master_scd(securities, name_history.frame)
    scd = scd.loc[
        (scd["valid_from"] <= as_of_date)
        & (scd["valid_to"].isna() | (scd["valid_to"] > coverage_start))
    ].copy()
    scd.loc[scd["valid_from"] < coverage_start, "valid_from"] = coverage_start
    source_watermark_id = _watermark_id(*source_batches, name_history)
    materialized = scd.copy(deep=True)
    materialized["run_id"] = context.run_id
    materialized["data_version"] = context.data_version
    materialized["created_at"] = pd.Timestamp.now(tz="UTC").tz_localize(None)
    materialized["source_watermark_id"] = source_watermark_id
    materialized["record_hash"] = materialized.apply(_record_hash, axis=1)
    return StagedBatch.create(
        object_name="security_master_scd",
        logical_partition=as_of_date.isoformat(),
        data_version=context.data_version,
        frame=materialized,
        business_keys=("ts_code", "valid_from"),
    )
