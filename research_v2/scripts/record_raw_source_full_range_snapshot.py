"""Record full 2020--2024 raw-source coverage for structural candidates.

Only server-side aggregates are transferred: raw row counts and finite counts
are grouped by factor/date, while a distinct security/date subquery provides a
bounded duplicate-row estimate.  This avoids materializing the multi-million
row YesLab extracts in the client.  PIT conflict resolution is deliberately
kept as a separate pending gate; this scope is not a final admission decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from collections import defaultdict
from datetime import date

import dolphindb as ddb
import pandas as pd
from record_factor_quality_snapshot import publish_snapshot
from record_raw_source_quality_snapshot import summarize_raw_aggregates

from research_v2.db.session import V2WriteSession, YesLabReadOnlySession

SCOPE_ID = "SCREEN_TRAIN2020_2024_RAW_SOURCE_AGG_V1"
STRUCTURAL_SCOPE_ID = "SCREEN_TRAIN2020_2024_STRUCTURAL_COVERAGE_V2"
REFERENCE_FACTOR_ID = "yeslab_0"
_YEARS = tuple(range(2020, 2025))
_GROUP_SIZE = 7


def _query_year_aggregates(
    source: YesLabReadOnlySession,
    *,
    table: str,
    factor_names: tuple[str, ...],
    year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    names = ",".join(f"'{name.replace(chr(39), chr(39) + chr(39))}'" for name in factor_names)
    start = f"{year}.01.01"
    end = f"{year}.12.31"
    base = (
        f"loadTable('dfs://YesLab','{table}') where factor_name in [{names}] "
        f"and trade_date between date('{start}') and date('{end}')"
    )
    rows_query = (
        "select factor_name,trade_date,count(*) as rows,"
        f"count(iif(isValid(value),1,NULL)) as finite from {base} "
        "group by factor_name,trade_date"
    )
    distinct_query = (
        "select factor_name,trade_date,count(*) as securities from ("
        "select distinct factor_name,trade_date,ts_code from "
        f"{base}) group by factor_name,trade_date"
    )
    rows = source.query(rows_query)
    securities = source.query(distinct_query)
    if not isinstance(rows, pd.DataFrame) or not isinstance(securities, pd.DataFrame):
        raise RuntimeError(f"raw aggregate query did not return DataFrames: {table}/{year}")
    return rows, securities


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
    if isinstance(value, (pd.Timestamp, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    if pd.isna(value):
        return None
    return value


def _build_frame(
    metadata: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    as_of_date: date,
    run_id: str,
    data_version: str,
    source_watermark_id: str,
) -> pd.DataFrame:
    required = {"factor_id", "factor_name", "store_factor", "category"}
    if missing := sorted(required.difference(metadata.columns)):
        raise ValueError(f"metadata is missing columns: {missing}")
    by_id = metadata.assign(
        factor_id=metadata["factor_id"].map(lambda value: f"yeslab_{int(value)}")
    ).set_index("factor_id")
    rows: list[dict[str, object]] = []
    for metric in metrics.sort_values("factor_id", kind="mergesort").to_dict("records"):
        factor_id = str(metric["factor_id"])
        if factor_id not in by_id.index:
            raise ValueError(f"metrics contains unknown factor: {factor_id}")
        meta = by_id.loc[factor_id]
        status = str(metric["decision_status"])
        body = {
            "screen_version": SCOPE_ID,
            "factor_name": str(meta["factor_name"]),
            "source_table": str(meta["store_factor"]),
            "category": str(meta["category"]),
            "full_range": {
                key: _json_value(value)
                for key, value in metric.items()
                if key not in {"factor_id", "decision_status", "reason_codes"}
            },
            "decision_status": status,
            "reason_codes": metric.get("reason_codes", []),
            "pit_conflict_materialization_checked": False,
        }
        record = {
            "as_of_date": as_of_date,
            "factor_id": factor_id,
            "factor_version": 1,
            "scope_id": SCOPE_ID,
            "eligibility_status": (
                "PASS" if status.startswith("RAW_FULL_RANGE_PASS") else "ELIGIBILITY_FAIL"
            ),
            "metrics_json": json.dumps(body, ensure_ascii=False, sort_keys=True),
            "rules_version": 1,
            "run_id": run_id,
            "data_version": data_version,
            "created_at": pd.Timestamp.utcnow().tz_localize(None),
            "source_watermark_id": source_watermark_id,
        }
        record["record_hash"] = _record_hash(record)
        rows.append(record)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", type=date.fromisoformat, default=date(2026, 8, 4))
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
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
        "select factor_id,factor_name,store_factor,category "
        "from loadTable('dfs://YesLab','factor_metadata')"
    )
    structural = target.query(
        "select factor_id,metrics_json from "
        "loadTable('dfs://QuantResearchV2Test','factor_quality_snapshot') "
        f"where scope_id='{STRUCTURAL_SCOPE_ID}'"
    )
    expected_result = target.query(
        "select signal_date from loadTable('dfs://QuantResearchV2Test','factor_rank_ic_daily') "
        "where factor_id='yeslab_0' and signal_date between date('2020.01.01') "
        "and date('2024.12.31')"
    )
    if not all(
        isinstance(frame, pd.DataFrame) for frame in (metadata, structural, expected_result)
    ):
        raise RuntimeError("full-range audit inputs did not return DataFrames")
    candidate_ids = tuple(
        str(row.factor_id)
        for row in structural.itertuples(index=False)
        if json.loads(str(row.metrics_json)).get("decision_status")
        == "STRUCTURAL_DQ_PASS_PENDING_RAW_PIT"
    )
    metadata_by_id = metadata.assign(
        factor_id=metadata["factor_id"].map(lambda value: f"yeslab_{int(value)}")
    ).set_index("factor_id")
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for factor_id in candidate_ids:
        meta = metadata_by_id.loc[factor_id]
        grouped[str(meta["store_factor"])].append((factor_id, str(meta["factor_name"])))

    all_metrics: list[pd.DataFrame] = []
    for table in sorted(grouped):
        entries = sorted(grouped[table])
        for offset in range(0, len(entries), _GROUP_SIZE):
            group = entries[offset : offset + _GROUP_SIZE]
            names = tuple(name for _, name in group)
            factor_ids = tuple(factor_id for factor_id, _ in group)
            row_parts: list[pd.DataFrame] = []
            security_parts: list[pd.DataFrame] = []
            for year in _YEARS:
                print(f"query table={table} factors={len(group)} year={year}", flush=True)
                rows, securities = _query_year_aggregates(
                    source, table=table, factor_names=names, year=year
                )
                row_parts.append(rows)
                security_parts.append(securities)
            summary = summarize_raw_aggregates(
                pd.concat(row_parts, ignore_index=True),
                pd.concat(security_parts, ignore_index=True),
                expected_dates=expected_result["signal_date"],
                expected_factor_ids=names,
            )
            summary["factor_id"] = summary["factor_name"].map(
                dict(zip(names, factor_ids, strict=True))
            )
            all_metrics.append(summary)
            print(summary["decision_status"].value_counts().to_string(), flush=True)
    metrics = pd.concat(all_metrics, ignore_index=True)
    watermark = hashlib.sha256(
        json.dumps(
            {
                "candidate_ids": candidate_ids,
                "expected_days": len(expected_result),
                "years": _YEARS,
                "scope": SCOPE_ID,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    data_version = hashlib.sha256((watermark + str(args.as_of)).encode()).hexdigest()
    frame = _build_frame(
        metadata,
        metrics,
        as_of_date=args.as_of,
        run_id=str(uuid.uuid4()),
        data_version=data_version,
        source_watermark_id=watermark,
    )
    print(metrics["decision_status"].value_counts().sort_index().to_string())
    if args.publish:
        print(publish_snapshot(target, frame, scope_id=SCOPE_ID))
    else:
        print("dry-run: no database write")


if __name__ == "__main__":
    main()
