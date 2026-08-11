"""Audit a bounded raw-source month for structurally eligible factors.

The audit is intentionally a sample-window checkpoint (2024-01 by default):
it verifies source presence, finite-value ratio, date bounds, and duplicate
keys before the full five-year PIT materialization. Duplicate rows are not
automatically rejected because YesLab revisions are expected to be resolved by
the existing deterministic deduplication step; the snapshot records the exact
duplicate count and keeps the full-range PIT check pending.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import uuid
from datetime import date

import dolphindb as ddb
import numpy as np
import pandas as pd
from record_factor_quality_snapshot import publish_snapshot

from research_v2.db.session import V2WriteSession, YesLabReadOnlySession

SCOPE_ID = "SCREEN_TRAIN2024_01_RAW_SOURCE_AUDIT_V1"
RULES_VERSION = 1
STRUCTURAL_SCOPE_ID = "SCREEN_TRAIN2020_2024_STRUCTURAL_COVERAGE_V2"
_TABLE_PATTERN = re.compile(r"factor_data(?:[1-9]|10|11)")


def classify_raw_metrics(
    metrics: dict[str, object], *, minimum_finite_ratio: float = 0.70
) -> dict[str, object]:
    """Classify one raw-source aggregate and return auditable reason codes."""
    rows = int(metrics.get("rows", 0) or 0)
    finite = int(metrics.get("finite", 0) or 0)
    duplicate_rows = int(metrics.get("duplicate_rows", 0) or 0)
    ratio = finite / rows if rows else 0.0
    if rows == 0:
        status = "RAW_SAMPLE_AUDIT_FAIL"
        reasons = ["NO_RAW_SOURCE_ROWS"]
    elif ratio < minimum_finite_ratio:
        status = "RAW_SAMPLE_AUDIT_FAIL"
        reasons = ["LOW_FINITE_VALUE_RATIO"]
    else:
        status = "RAW_SAMPLE_AUDIT_PASS_PENDING_FULL_RANGE_PIT"
        reasons = ["RAW_SOURCE_PRESENT", "FINITE_RATIO_GATE_PASS", "FULL_RANGE_PIT_PENDING"]
        if duplicate_rows:
            reasons.append("SOURCE_REVISIONS_REQUIRE_DEDUP")
    return {
        "rows": rows,
        "finite": finite,
        "finite_ratio": ratio,
        "duplicate_rows": duplicate_rows,
        "duplicate_row_ratio": duplicate_rows / rows if rows else 0.0,
        "decision_status": status,
        "reason_codes": reasons,
    }


def summarize_raw_aggregates(
    rows: pd.DataFrame,
    securities: pd.DataFrame,
    *,
    expected_dates: pd.Series | pd.DatetimeIndex,
    expected_factor_ids: tuple[str, ...],
    minimum_day_ratio: float = 0.80,
    minimum_finite_ratio: float = 0.70,
) -> pd.DataFrame:
    """Combine server-side daily aggregates into full-range raw-source metrics."""
    required_rows = {"factor_name", "trade_date", "rows", "finite"}
    required_securities = {"factor_name", "trade_date", "securities"}
    if missing := sorted(required_rows.difference(rows.columns)):
        raise ValueError(f"raw row aggregates are missing columns: {missing}")
    if missing := sorted(required_securities.difference(securities.columns)):
        raise ValueError(f"raw distinct aggregates are missing columns: {missing}")
    if len(set(expected_factor_ids)) != len(expected_factor_ids):
        raise ValueError("expected_factor_ids must be unique")
    normalized_expected = pd.to_datetime(expected_dates, errors="raise")
    if isinstance(normalized_expected, pd.Series):
        normalized_expected = normalized_expected.dt.date
    else:
        normalized_expected = normalized_expected.date
    expected = pd.Series(normalized_expected).drop_duplicates()
    if expected.empty:
        raise ValueError("expected_dates cannot be empty")
    expected_set = set(expected.tolist())
    expected_months = expected.map(lambda value: value.strftime("%Y-%m")).value_counts()
    row_frame = rows.loc[:, ["factor_name", "trade_date", "rows", "finite"]].copy()
    row_frame["factor_name"] = row_frame["factor_name"].astype(str)
    row_frame["trade_date"] = pd.to_datetime(row_frame["trade_date"], errors="raise").dt.date
    row_frame["rows"] = pd.to_numeric(row_frame["rows"], errors="coerce").fillna(0)
    row_frame["finite"] = pd.to_numeric(row_frame["finite"], errors="coerce").fillna(0)
    row_frame = row_frame.groupby(["factor_name", "trade_date"], as_index=False).agg(
        rows=("rows", "sum"), finite=("finite", "sum")
    )
    security_frame = securities.loc[:, ["factor_name", "trade_date", "securities"]].copy()
    security_frame["factor_name"] = security_frame["factor_name"].astype(str)
    security_frame["trade_date"] = pd.to_datetime(
        security_frame["trade_date"], errors="raise"
    ).dt.date
    security_frame["securities"] = pd.to_numeric(
        security_frame["securities"], errors="coerce"
    ).fillna(0)
    security_frame = security_frame.groupby(["factor_name", "trade_date"], as_index=False).agg(
        securities=("securities", "max")
    )
    combined = row_frame.merge(
        security_frame, on=["factor_name", "trade_date"], how="left", validate="one_to_one"
    )
    combined["securities"] = combined["securities"].fillna(0)
    output: list[dict[str, object]] = []
    for factor_name in expected_factor_ids:
        factor = combined.loc[combined["factor_name"] == factor_name].copy()
        factor = factor.loc[factor["trade_date"].isin(expected_set)]
        raw_rows = int(factor["rows"].sum())
        finite = int(factor["finite"].sum())
        row_values = factor["rows"].to_numpy(dtype=float)
        security_values = factor["securities"].to_numpy(dtype=float)
        duplicate_rows = int(np.maximum(row_values - security_values, 0).sum())
        observed_dates = set(factor.loc[factor["rows"] > 0, "trade_date"])
        observed_months = pd.Series(sorted(observed_dates)).map(
            lambda value: value.strftime("%Y-%m")
        ).value_counts()
        monthly_ratios = [
            float(observed_months.get(month, 0) / count)
            for month, count in expected_months.items()
        ]
        date_ratio = len(observed_dates) / len(expected_set)
        finite_ratio = finite / raw_rows if raw_rows else 0.0
        missing_month_count = int(
            sum(1 for month in expected_months.index if observed_months.get(month, 0) == 0)
        )
        if raw_rows == 0:
            status = "RAW_FULL_RANGE_FAIL"
            reasons = ["NO_RAW_SOURCE_ROWS"]
        elif finite_ratio < minimum_finite_ratio:
            status = "RAW_FULL_RANGE_FAIL"
            reasons = ["LOW_FINITE_VALUE_RATIO"]
        elif date_ratio < minimum_day_ratio or missing_month_count:
            status = "RAW_FULL_RANGE_FAIL"
            reasons = ["LOW_RAW_DATE_COVERAGE"]
            if missing_month_count:
                reasons.append("MISSING_RAW_MONTH")
        else:
            status = "RAW_FULL_RANGE_PASS_PENDING_PIT_CONFLICT"
            reasons = ["RAW_FULL_RANGE_COVERAGE_PASS", "PIT_CONFLICT_CHECK_PENDING"]
        output.append(
            {
                "factor_name": factor_name,
                "expected_day_count": len(expected_set),
                "observed_day_count": len(observed_dates),
                "raw_date_ratio": date_ratio,
                "raw_rows": raw_rows,
                "finite": finite,
                "finite_ratio": finite_ratio,
                "duplicate_rows": duplicate_rows,
                "duplicate_row_ratio": duplicate_rows / raw_rows if raw_rows else 0.0,
                "minimum_monthly_day_ratio": min(monthly_ratios) if monthly_ratios else 0.0,
                "missing_month_count": missing_month_count,
                "min_date": min(observed_dates) if observed_dates else None,
                "max_date": max(observed_dates) if observed_dates else None,
                "decision_status": status,
                "reason_codes": reasons,
            }
        )
    return pd.DataFrame(output)


def build_snapshot_frame(
    metadata: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    as_of_date: date,
    run_id: str,
    data_version: str,
    source_watermark_id: str,
    sample_start: date,
    sample_end: date,
) -> pd.DataFrame:
    required_metadata = {"factor_id", "factor_name", "store_factor", "category"}
    if missing := sorted(required_metadata.difference(metadata.columns)):
        raise ValueError(f"metadata is missing columns: {missing}")
    if "factor_id" not in metrics or "decision_status" not in metrics:
        raise ValueError("raw metrics are missing factor_id/decision_status")
    by_id = metadata.assign(
        factor_id=metadata["factor_id"].map(lambda value: f"yeslab_{int(value)}")
    ).set_index("factor_id")
    rows: list[dict[str, object]] = []
    for metric in metrics.sort_values("factor_id", kind="mergesort").to_dict("records"):
        factor_id = str(metric["factor_id"])
        if factor_id not in by_id.index:
            raise ValueError(f"unknown factor in raw metrics: {factor_id}")
        meta = by_id.loc[factor_id]
        status = str(metric["decision_status"])
        body = {
            "screen_version": SCOPE_ID,
            "factor_name": str(meta["factor_name"]),
            "source_table": str(meta["store_factor"]),
            "category": str(meta["category"]),
            "sample_window": {
                "start": sample_start.isoformat(),
                "end": sample_end.isoformat(),
            },
            "raw_source": {
                key: _json_value(value)
                for key, value in metric.items()
                if key not in {"factor_id", "decision_status", "reason_codes"}
            },
            "decision_status": status,
            "reason_codes": metric.get("reason_codes", []),
            "full_range_pit_dedup_checked": False,
        }
        record = {
            "as_of_date": as_of_date,
            "factor_id": factor_id,
            "factor_version": 1,
            "scope_id": SCOPE_ID,
            "eligibility_status": (
                "PASS" if status.startswith("RAW_SAMPLE_AUDIT_PASS") else "ELIGIBILITY_FAIL"
            ),
            "metrics_json": json.dumps(body, ensure_ascii=False, sort_keys=True),
            "rules_version": RULES_VERSION,
            "run_id": run_id,
            "data_version": data_version,
            "created_at": pd.Timestamp.utcnow().tz_localize(None),
            "source_watermark_id": source_watermark_id,
        }
        record["record_hash"] = _record_hash(record)
        rows.append(record)
    return pd.DataFrame(rows)


def _raw_metrics(
    source: YesLabReadOnlySession,
    *,
    table: str,
    factor_name: str,
    sample_start: date,
    sample_end: date,
) -> dict[str, object]:
    if not _TABLE_PATTERN.fullmatch(table):
        raise ValueError(f"unsafe source table: {table}")
    escaped = factor_name.replace("'", "''")
    script = (
        f"raw = select ts_code,trade_date,value from loadTable('dfs://YesLab','{table}') "
        f"where factor_name='{escaped}' and trade_date between "
        f"date('{sample_start.strftime('%Y.%m.%d')}') "
        f"and date('{sample_end.strftime('%Y.%m.%d')}')\n"
        "dup = select ts_code,trade_date,count(*) as n from raw group by ts_code,trade_date\n"
        "dupCount = select count(*) as duplicate_keys,"
        "sum(iif(n>1,n-1,0)) as duplicate_rows from dup\n"
        "select r.rows,r.finite,r.min_date,r.max_date,d.duplicate_keys,d.duplicate_rows "
        "from (select count(*) as rows,count(iif(isValid(value),1,NULL)) as finite,"
        "min(trade_date) as min_date,max(trade_date) as max_date from raw) r cross join dupCount d"
    )
    result = source.query(script)
    if not isinstance(result, pd.DataFrame) or result.empty:
        return {"rows": 0, "finite": 0, "duplicate_rows": 0}
    row = result.iloc[0]
    metrics = classify_raw_metrics(
        {
            "rows": row.get("rows", 0),
            "finite": row.get("finite", 0),
            "duplicate_rows": row.get("duplicate_rows", 0),
        }
    )
    metrics["min_date"] = _json_value(row.get("min_date"))
    metrics["max_date"] = _json_value(row.get("max_date"))
    metrics["duplicate_keys"] = int(row.get("duplicate_keys", 0) or 0)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-start", type=date.fromisoformat, default=date(2024, 1, 1))
    parser.add_argument("--sample-end", type=date.fromisoformat, default=date(2024, 1, 31))
    parser.add_argument("--as-of", type=date.fromisoformat, default=date(2026, 8, 4))
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    if args.sample_start > args.sample_end:
        raise ValueError("sample-start cannot be after sample-end")
    client = ddb.session()
    client.connect(
        os.getenv("QRV2_DDB_HOST", "10.176.55.211"),
        int(os.getenv("QRV2_DDB_PORT", "8848")),
        os.getenv("QRV2_DDB_USERNAME", "xiahongjie"),
        os.getenv("QRV2_DDB_PASSWORD", "123456"),
    )
    source = YesLabReadOnlySession(client)
    target = V2WriteSession(client, target_database="dfs://QuantResearchV2Test")
    metadata = source.query(
        "select factor_id,factor_name,store_factor,category from loadTable('dfs://YesLab','factor_metadata')"
    )
    structural = target.query(
        "select factor_id,metrics_json from "
        "loadTable('dfs://QuantResearchV2Test','factor_quality_snapshot') "
        f"where scope_id='{STRUCTURAL_SCOPE_ID}'"
    )
    if not isinstance(metadata, pd.DataFrame) or not isinstance(structural, pd.DataFrame):
        raise RuntimeError("raw audit inputs did not return DataFrames")
    candidate_ids = tuple(
        str(row.factor_id)
        for row in structural.itertuples(index=False)
        if json.loads(str(row.metrics_json)).get("decision_status")
        == "STRUCTURAL_DQ_PASS_PENDING_RAW_PIT"
    )
    metadata_by_id = metadata.assign(
        factor_id=metadata["factor_id"].map(lambda value: f"yeslab_{int(value)}")
    ).set_index("factor_id")
    results: list[dict[str, object]] = []
    for factor_id in candidate_ids:
        meta = metadata_by_id.loc[factor_id]
        try:
            metrics = _raw_metrics(
                source,
                table=str(meta["store_factor"]),
                factor_name=str(meta["factor_name"]),
                sample_start=args.sample_start,
                sample_end=args.sample_end,
            )
        except Exception as error:  # record a retryable blocker per factor
            metrics = {
                "rows": 0,
                "finite": 0,
                "duplicate_rows": 0,
                "decision_status": "RAW_SAMPLE_AUDIT_PENDING_QUERY",
                "reason_codes": ["SOURCE_QUERY_ERROR", type(error).__name__],
                "error": str(error)[:500],
            }
        results.append({"factor_id": factor_id, **metrics})
        print(f"{factor_id}: {metrics.get('decision_status')}")
    metrics_frame = pd.DataFrame(results)
    watermark = hashlib.sha256(
        json.dumps(
            {
                "candidate_ids": candidate_ids,
                "sample_start": args.sample_start.isoformat(),
                "sample_end": args.sample_end.isoformat(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    data_version = hashlib.sha256((watermark + str(args.as_of)).encode()).hexdigest()
    frame = build_snapshot_frame(
        metadata,
        metrics_frame,
        as_of_date=args.as_of,
        run_id=str(uuid.uuid4()),
        data_version=data_version,
        source_watermark_id=watermark,
        sample_start=args.sample_start,
        sample_end=args.sample_end,
    )
    print(metrics_frame["decision_status"].value_counts().sort_index().to_string())
    if args.publish:
        print(publish_snapshot(target, frame, scope_id=SCOPE_ID))
    else:
        print("dry-run: no database write")


def _record_hash(record: dict[str, object]) -> str:
    payload = {
        key: value.isoformat() if isinstance(value, (date, pd.Timestamp)) else value
        for key, value in record.items()
        if key != "record_hash"
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True).encode()
    ).hexdigest()


def _json_value(value: object) -> object:
    if isinstance(value, (date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if pd.isna(value):
        return None
    return value


if __name__ == "__main__":
    main()
