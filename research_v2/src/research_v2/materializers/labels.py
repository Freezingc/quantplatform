"""Build publish-ready primary forward-open labels from cleaned V2 market data."""

from __future__ import annotations

import hashlib
import json
from datetime import date

import pandas as pd

from research_v2.contracts.source import SourceBatch
from research_v2.db.publish import StagedBatch
from research_v2.labels.forward_return import LockedLabelAccess, build_forward_open_labels
from research_v2.pit.calendar import TradingCalendar
from research_v2.runs.context import RunContext


def build_label_batch(
    members: pd.DataFrame,
    market: SourceBatch,
    calendar: TradingCalendar,
    context: RunContext,
    *,
    label_id: str = "forward_open_5d",
    label_version: int = 1,
    members_watermark_id: str | None = None,
    locked_access: LockedLabelAccess | None = None,
) -> StagedBatch:
    """Map the locked primary raw 5-day label into the V2 label_daily schema."""
    if market.table_name != "market_daily_clean":
        raise ValueError("labels require a market_daily_clean source batch")
    if not label_id or label_version < 1:
        raise ValueError("label_id and positive label_version are required")
    labels = build_forward_open_labels(
        members,
        market.frame,
        calendar,
        locked_access=locked_access,
    )
    if labels.empty:
        raise ValueError("label materialization requires at least one universe member")
    materialized = pd.DataFrame(
        {
            "signal_date": labels["signal_date"],
            "ts_code": labels["ts_code"],
            "label_id": label_id,
            "label_version": label_version,
            "entry_date": labels["entry_date"],
            "exit_date": labels["exit_date"],
            "label_available_date": labels["label_available_date"],
            "raw_return": labels["raw_return_5d"],
            "rank_pct": float("nan"),
            "winsorized_return": float("nan"),
            "zscore_return": float("nan"),
            "neutral_residual": float("nan"),
            "label_valid": labels["label_valid"],
            "invalid_reason": labels["invalid_reason"],
            "entry_realized_tradable": labels["entry_realized_tradable"],
        }
    )
    materialized["run_id"] = context.run_id
    materialized["data_version"] = context.data_version
    materialized["created_at"] = pd.Timestamp.now(tz="UTC").tz_localize(None)
    lineage_values = [market.watermark_id, members_watermark_id]
    if locked_access is not None:
        lineage_values.append(f"locked_access:{locked_access.authorization_id}")
    lineage = "|".join(value for value in lineage_values if value)
    materialized["source_watermark_id"] = hashlib.sha256(lineage.encode()).hexdigest()
    materialized["record_hash"] = materialized.apply(_record_hash, axis=1)
    logical_partition = pd.Timestamp(materialized["signal_date"].min()).strftime("%Y-%m")
    return StagedBatch.create(
        object_name="label_daily",
        logical_partition=logical_partition,
        data_version=context.data_version,
        frame=materialized,
        business_keys=("signal_date", "ts_code", "label_id", "label_version"),
        partition_date_column="signal_date",
    )


def _record_hash(row: pd.Series) -> str:
    payload = {
        column: (value.isoformat() if isinstance(value, (date, pd.Timestamp)) else value)
        for column, value in row.items()
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True).encode()
    ).hexdigest()
