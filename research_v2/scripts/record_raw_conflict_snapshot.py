"""Audit conflicting raw YesLab values for the full training range.

The source commonly contains identical revisions for a security/date key. This
scope distinguishes those from keys carrying multiple distinct finite values;
the latter are quarantined by the production deduplicator and cannot be
admitted without an explicit resolution rule.
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

from research_v2.db.session import V2WriteSession, YesLabReadOnlySession

SCOPE_ID = "SCREEN_TRAIN2020_2024_RAW_CONFLICT_AGG_V1"
RAW_SCOPE_ID = "SCREEN_TRAIN2020_2024_RAW_SOURCE_AGG_V1"
_YEARS = tuple(range(2020, 2025))
_GROUP_SIZE = 7


def classify_conflict_metrics(conflict_keys: int) -> dict[str, object]:
    """Classify distinct-value conflicts independently of identical duplicates."""
    count = int(conflict_keys)
    if count > 0:
        return {
            "conflict_key_count": count,
            "decision_status": "RAW_CONFLICT_CHECK_FAIL",
            "reason_codes": ["CONFLICT_VALUE_KEYS", "DEDUP_QUARANTINE_REQUIRED"],
        }
    return {
        "conflict_key_count": 0,
        "decision_status": "RAW_CONFLICT_CHECK_PASS_PENDING_MATERIALIZATION",
        "reason_codes": ["NO_DISTINCT_VALUE_CONFLICT", "PIT_MATERIALIZATION_PENDING"],
    }


def _query_conflicts(
    source: YesLabReadOnlySession,
    *,
    table: str,
    factor_names: tuple[str, ...],
    year: int,
) -> pd.DataFrame:
    names = ",".join(f"'{name.replace(chr(39), chr(39) + chr(39))}'" for name in factor_names)
    base = (
        f"loadTable('dfs://YesLab','{table}') where factor_name in [{names}] "
        f"and trade_date between date('{year}.01.01') and date('{year}.12.31')"
    )
    script = (
        f"raw = select factor_name,trade_date,ts_code,value from {base}\n"
        "distinctValues = select distinct factor_name,trade_date,ts_code,value from raw\n"
        "keyValues = select factor_name,trade_date,ts_code,count(*) as value_count "
        "from distinctValues group by factor_name,trade_date,ts_code\n"
        "select factor_name,count(*) as conflict_keys from keyValues "
        "where value_count>1 group by factor_name"
    )
    result = source.query(script)
    if not isinstance(result, pd.DataFrame):
        raise RuntimeError(f"conflict query did not return a DataFrame: {table}/{year}")
    return result


def _query_conflicts_resilient(
    source: YesLabReadOnlySession,
    *,
    table: str,
    factor_names: tuple[str, ...],
    year: int,
) -> pd.DataFrame:
    """Retry a conflict aggregate in smaller groups after a server OOM."""
    try:
        return _query_conflicts(
            source, table=table, factor_names=factor_names, year=year
        )
    except RuntimeError:
        if len(factor_names) == 1:
            raise
        midpoint = len(factor_names) // 2
        left = _query_conflicts_resilient(
            source, table=table, factor_names=factor_names[:midpoint], year=year
        )
        right = _query_conflicts_resilient(
            source, table=table, factor_names=factor_names[midpoint:], year=year
        )
        return pd.concat([left, right], ignore_index=True)


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
    if hasattr(value, "item"):
        return value.item()
    if pd.isna(value):
        return None
    return value


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
    raw_scope = target.query(
        "select factor_id,metrics_json from "
        "loadTable('dfs://QuantResearchV2Test','factor_quality_snapshot') "
        f"where scope_id='{RAW_SCOPE_ID}'"
    )
    if not isinstance(metadata, pd.DataFrame) or not isinstance(raw_scope, pd.DataFrame):
        raise RuntimeError("conflict audit inputs did not return DataFrames")
    candidate_ids = tuple(str(row.factor_id) for row in raw_scope.itertuples(index=False))
    by_id = metadata.assign(
        factor_id=metadata["factor_id"].map(lambda value: f"yeslab_{int(value)}")
    ).set_index("factor_id")
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for factor_id in candidate_ids:
        meta = by_id.loc[factor_id]
        grouped[str(meta["store_factor"])].append((factor_id, str(meta["factor_name"])))

    totals: dict[str, int] = {factor_id: 0 for factor_id in candidate_ids}
    for table in sorted(grouped):
        entries = sorted(grouped[table])
        for offset in range(0, len(entries), _GROUP_SIZE):
            group = entries[offset : offset + _GROUP_SIZE]
            names = tuple(name for _, name in group)
            factor_id_by_name = dict((name, factor_id) for factor_id, name in group)
            for year in _YEARS:
                print(f"query table={table} factors={len(group)} year={year}", flush=True)
                result = _query_conflicts_resilient(
                    source, table=table, factor_names=names, year=year
                )
                for row in result.itertuples(index=False):
                    totals[factor_id_by_name[str(row.factor_name)]] += int(row.conflict_keys)

    rows: list[dict[str, object]] = []
    for factor_id in sorted(candidate_ids):
        classification = classify_conflict_metrics(totals[factor_id])
        meta = by_id.loc[factor_id]
        body = {
            "screen_version": SCOPE_ID,
            "factor_name": str(meta["factor_name"]),
            "source_table": str(meta["store_factor"]),
            "category": str(meta["category"]),
            "full_range": {
                "start": "2020-01-01",
                "end": "2024-12-31",
                "years": list(_YEARS),
                "conflict_key_count": classification["conflict_key_count"],
            },
            "decision_status": classification["decision_status"],
            "reason_codes": classification["reason_codes"],
            "pit_materialization_checked": False,
        }
        record = {
            "as_of_date": args.as_of,
            "factor_id": factor_id,
            "factor_version": 1,
            "scope_id": SCOPE_ID,
            "eligibility_status": (
                "PASS"
                if str(classification["decision_status"]).startswith("RAW_CONFLICT_CHECK_PASS")
                else "ELIGIBILITY_FAIL"
            ),
            "metrics_json": json.dumps(body, ensure_ascii=False, sort_keys=True),
            "rules_version": 1,
            "run_id": str(uuid.uuid4()),
            "data_version": "",
            "created_at": pd.Timestamp.utcnow().tz_localize(None),
            "source_watermark_id": "",
        }
        rows.append(record)
    watermark = hashlib.sha256(
        json.dumps(
            {"candidate_ids": candidate_ids, "years": _YEARS, "scope": SCOPE_ID},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    data_version = hashlib.sha256((watermark + str(args.as_of)).encode()).hexdigest()
    for record in rows:
        record["data_version"] = data_version
        record["source_watermark_id"] = watermark
        record["record_hash"] = _record_hash(record)
    frame = pd.DataFrame(rows)
    statuses = frame["metrics_json"].map(lambda value: json.loads(value)["decision_status"])
    print(statuses.value_counts().sort_index().to_string())
    if args.publish:
        print(publish_snapshot(target, frame, scope_id=SCOPE_ID))
    else:
        print("dry-run: no database write")


if __name__ == "__main__":
    main()
