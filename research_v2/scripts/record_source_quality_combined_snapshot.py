"""Combine independent source-quality checks into one auditable gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from datetime import date

import dolphindb as ddb
import pandas as pd
from record_factor_quality_snapshot import publish_snapshot

from research_v2.db.session import V2WriteSession, YesLabReadOnlySession

SCOPE_ID = "SCREEN_TRAIN2020_2024_SOURCE_DQ_COMBINED_V1"
_RAW_SCOPE = "SCREEN_TRAIN2020_2024_RAW_SOURCE_AGG_V1"
_CONFLICT_SCOPE = "SCREEN_TRAIN2020_2024_RAW_CONFLICT_AGG_V1"
_PIT_SCOPE = "SCREEN_TRAIN2020_2024_PIT_KEY_COVERAGE_V1"


def combine_source_statuses(
    *, raw_status: str, conflict_status: str, pit_status: str
) -> dict[str, object]:
    """Return a conservative combined source gate and reason codes."""
    expected = (
        "RAW_FULL_RANGE_PASS_PENDING_PIT_CONFLICT",
        "RAW_CONFLICT_CHECK_PASS_PENDING_MATERIALIZATION",
        "PIT_KEY_DQ_PASS_PENDING_COMPACT_UNIVERSE",
    )
    statuses = (raw_status, conflict_status, pit_status)
    if all(
        status == expected_status
        for status, expected_status in zip(statuses, expected, strict=True)
    ):
        return {
            "decision_status": "SOURCE_DQ_PASS_PENDING_SIMILARITY",
            "reason_codes": [
                "RAW_RANGE_PASS",
                "NO_VALUE_CONFLICT",
                "PIT_KEY_COVERAGE_PASS",
                "SIMILARITY_PENDING",
            ],
        }
    reasons = ["SOURCE_DQ_GATE_FAIL"]
    if raw_status != expected[0]:
        reasons.append("RAW_RANGE_FAIL")
    if conflict_status != expected[1]:
        reasons.append("RAW_CONFLICT_FAIL")
    if pit_status != expected[2]:
        reasons.append("PIT_KEY_COVERAGE_FAIL")
    return {"decision_status": "SOURCE_DQ_FAIL", "reason_codes": reasons}


def _record_hash(record: dict[str, object]) -> str:
    payload = {
        key: value.isoformat() if isinstance(value, (date, pd.Timestamp)) else value
        for key, value in record.items()
        if key != "record_hash"
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True).encode()
    ).hexdigest()


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
    snapshots: dict[str, pd.DataFrame] = {}
    for key, scope in {
        "raw": _RAW_SCOPE,
        "conflict": _CONFLICT_SCOPE,
        "pit": _PIT_SCOPE,
    }.items():
        result = target.query(
            "select factor_id,metrics_json from "
            "loadTable('dfs://QuantResearchV2Test','factor_quality_snapshot') "
            f"where scope_id='{scope}'"
        )
        if not isinstance(result, pd.DataFrame):
            raise RuntimeError(f"source quality scope did not return a DataFrame: {scope}")
        snapshots[key] = result.set_index("factor_id")
    if not isinstance(metadata, pd.DataFrame):
        raise RuntimeError("metadata query did not return a DataFrame")
    by_id = metadata.assign(
        factor_id=metadata["factor_id"].map(lambda value: f"yeslab_{int(value)}")
    ).set_index("factor_id")
    factor_ids = tuple(sorted(set(snapshots["raw"].index)))
    rows: list[dict[str, object]] = []
    for factor_id in factor_ids:
        statuses: dict[str, str] = {}
        details: dict[str, object] = {}
        for key, frame in snapshots.items():
            if factor_id not in frame.index:
                statuses[key] = "MISSING_SCOPE_ROW"
                details[key] = None
                continue
            body = json.loads(str(frame.loc[factor_id, "metrics_json"]))
            statuses[key] = str(body.get("decision_status"))
            details[key] = body
        decision = combine_source_statuses(
            raw_status=statuses["raw"],
            conflict_status=statuses["conflict"],
            pit_status=statuses["pit"],
        )
        meta = by_id.loc[factor_id]
        body = {
            "screen_version": SCOPE_ID,
            "factor_name": str(meta["factor_name"]),
            "source_table": str(meta["store_factor"]),
            "category": str(meta["category"]),
            "source_checks": details,
            "component_statuses": statuses,
            "decision_status": decision["decision_status"],
            "reason_codes": decision["reason_codes"],
            "train_only_similarity_checked": False,
        }
        rows.append(
            {
                "as_of_date": args.as_of,
                "factor_id": factor_id,
                "factor_version": 1,
                "scope_id": SCOPE_ID,
                "eligibility_status": (
                    "PASS"
                    if decision["decision_status"] == "SOURCE_DQ_PASS_PENDING_SIMILARITY"
                    else "ELIGIBILITY_FAIL"
                ),
                "metrics_json": json.dumps(body, ensure_ascii=False, sort_keys=True),
                "rules_version": 1,
                "run_id": str(uuid.uuid4()),
                "data_version": "",
                "created_at": pd.Timestamp.utcnow().tz_localize(None),
                "source_watermark_id": "",
            }
        )
    watermark = hashlib.sha256(
        json.dumps({"factor_ids": factor_ids, "scope": SCOPE_ID}, sort_keys=True).encode()
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
