"""Build publish-ready technical factor clean and quarantine batches."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from typing import Literal

import pandas as pd

from research_v2.contracts.factor import FactorDefinition
from research_v2.db.publish import StagedBatch
from research_v2.factors.availability import materialize_technical_availability
from research_v2.factors.deduplication import deduplicate_factor_observations
from research_v2.factors.source_reader import normalize_technical_factor_source
from research_v2.runs.context import RunContext


@dataclass(frozen=True)
class FactorMaterializationBatches:
    clean_batch: StagedBatch
    quarantine_batch: StagedBatch | None


def combine_factor_materialization_batches(
    batches: list[FactorMaterializationBatches],
) -> FactorMaterializationBatches:
    """Combine one factor-partition group before advancing its shared pointer."""
    if not batches:
        raise ValueError("at least one factor materialization batch is required")
    clean_batches = [batch.clean_batch for batch in batches]
    first_clean = clean_batches[0]
    if any(
        batch.data_version != first_clean.data_version
        or batch.logical_partition != first_clean.logical_partition
        for batch in clean_batches[1:]
    ):
        raise ValueError("combined factors must share data_version and logical_partition")
    clean = pd.concat([batch.frame for batch in clean_batches], ignore_index=True)
    clean_batch = StagedBatch.create(
        object_name="factor_observation_clean",
        logical_partition=first_clean.logical_partition,
        data_version=first_clean.data_version,
        frame=clean,
        business_keys=("signal_date", "ts_code", "factor_id", "factor_version"),
        partition_date_column="signal_date",
    )
    quarantines = [batch.quarantine_batch for batch in batches if batch.quarantine_batch]
    if not quarantines:
        return FactorMaterializationBatches(clean_batch=clean_batch, quarantine_batch=None)
    first_quarantine = quarantines[0]
    quarantine = pd.concat([batch.frame for batch in quarantines], ignore_index=True)
    quarantine_batch = StagedBatch.create(
        object_name="factor_quarantine",
        logical_partition=first_quarantine.logical_partition,
        data_version=first_clean.data_version,
        frame=quarantine,
        business_keys=("observation_date", "ts_code", "factor_id", "factor_version", "reason"),
        partition_date_column="observation_date",
    )
    return FactorMaterializationBatches(clean_batch=clean_batch, quarantine_batch=quarantine_batch)


def _record_hash(row: pd.Series) -> str:
    payload = {
        key: (value.isoformat() if isinstance(value, (date, pd.Timestamp)) else value)
        for key, value in row.items()
    }
    return hashlib.sha256(
        json.dumps(payload, default=str, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def _logical_partition(frame: pd.DataFrame, date_column: str) -> str:
    return pd.Timestamp(frame[date_column].min()).strftime("%Y-%m")


def build_technical_factor_batches(
    source: pd.DataFrame,
    definition: FactorDefinition,
    *,
    source_factor_name: str,
    source_watermark_id: str,
    context: RunContext,
    record_hash_strategy: Literal["canonical", "fast"] = "canonical",
    include_lineage: bool = True,
) -> FactorMaterializationBatches:
    """Normalize, deduplicate, PIT-align and audit one registered technical factor."""
    normalized = normalize_technical_factor_source(
        source,
        definition,
        source_factor_name=source_factor_name,
        source_watermark_id=source_watermark_id,
        include_source_row_id=include_lineage,
    )
    deduped = deduplicate_factor_observations(normalized)
    if deduped.clean.empty:
        raise ValueError("technical factor has no clean observations to publish")
    pit = materialize_technical_availability(deduped.clean)
    clean = pd.DataFrame(
        {
            "signal_date": pit["signal_date"],
            "ts_code": pit["ts_code"],
            "factor_id": pit["factor_id"],
            "factor_version": pit["factor_version"],
            "value": pit["raw_value"],
            "observation_date": pit["observation_date"],
            "available_date": pit["available_date"],
            "report_period": pd.NaT,
            "stale_days": pit["stale_days"],
            "dedup_resolution": pit["dedup_resolution"],
            "source_revision_id": pit["source_row_id"],
        }
    )
    if include_lineage:
        clean["run_id"] = context.run_id
        clean["data_version"] = context.data_version
        clean["created_at"] = pd.Timestamp.now(tz="UTC").tz_localize(None)
        clean["source_watermark_id"] = source_watermark_id
        clean["record_hash"] = _record_hashes(clean, strategy=record_hash_strategy)
    clean_batch = StagedBatch.create(
        object_name="factor_observation_clean",
        logical_partition=_logical_partition(clean, "signal_date"),
        data_version=context.data_version,
        frame=clean,
        business_keys=("signal_date", "ts_code", "factor_id", "factor_version"),
        partition_date_column="signal_date",
    )
    if deduped.quarantine.empty:
        return FactorMaterializationBatches(clean_batch=clean_batch, quarantine_batch=None)
    grouped = deduped.quarantine.groupby(
        ["observation_date", "ts_code", "factor_id", "factor_version", "reason"],
        sort=True,
        dropna=False,
    )
    quarantine_rows: list[dict[str, object]] = []
    for key, group in grouped:
        observation_date, ts_code, factor_id, factor_version, reason = key
        finite_values = sorted(
            {
                float(value)
                for value in pd.to_numeric(group["raw_value"], errors="coerce")
                if pd.notna(value)
            }
        )
        quarantine_rows.append(
            {
                "observation_date": observation_date,
                "ts_code": ts_code,
                "factor_id": factor_id,
                "factor_version": factor_version,
                "reason": reason,
                "source_row_ids_json": json.dumps(sorted(group["source_row_id"].astype(str))),
                "distinct_values_json": json.dumps(finite_values),
            }
        )
    quarantine = pd.DataFrame(quarantine_rows)
    if include_lineage:
        quarantine["run_id"] = context.run_id
        quarantine["data_version"] = context.data_version
        quarantine["created_at"] = pd.Timestamp.now(tz="UTC").tz_localize(None)
        quarantine["source_watermark_id"] = source_watermark_id
        quarantine["record_hash"] = _record_hashes(quarantine, strategy=record_hash_strategy)
    quarantine_batch = StagedBatch.create(
        object_name="factor_quarantine",
        logical_partition=_logical_partition(quarantine, "observation_date"),
        data_version=context.data_version,
        frame=quarantine,
        business_keys=("observation_date", "ts_code", "factor_id", "factor_version", "reason"),
        partition_date_column="observation_date",
    )
    return FactorMaterializationBatches(
        clean_batch=clean_batch,
        quarantine_batch=quarantine_batch,
    )


def _record_hashes(
    frame: pd.DataFrame,
    *,
    strategy: Literal["canonical", "fast"],
) -> pd.Series:
    if strategy == "canonical":
        return frame.apply(_record_hash, axis=1)
    if strategy != "fast":
        raise ValueError(f"unsupported record_hash_strategy: {strategy}")
    stable_values = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype="uint64")
    return pd.Series(
        (
            hashlib.sha256(int(value).to_bytes(8, byteorder="little", signed=False)).hexdigest()
            for value in stable_values
        ),
        index=frame.index,
    )
