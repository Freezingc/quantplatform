"""Consolidate all completed gates into a final factor-quality candidate scope."""

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

SCOPE_ID = "SCREEN_TRAIN2020_2024_FACTOR_QUALITY_FINAL_CANDIDATES_V1"
SOURCE_SCOPE_ID = "SCREEN_TRAIN2020_2024_SOURCE_DQ_COMBINED_V1"
VALIDATION_SCOPE_ID = "SCREEN_TRAIN2020_2024_VALIDATION2025_UNIQUE21_V1"
SIMILARITY_SCOPE_ID = "SCREEN_TRAIN2020_2024_SIMILARITY_FULL_DAY18_V2"


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
    scopes: dict[str, pd.DataFrame] = {}
    for key, scope in {
        "source": SOURCE_SCOPE_ID,
        "validation": VALIDATION_SCOPE_ID,
        "similarity": SIMILARITY_SCOPE_ID,
    }.items():
        result = target.query(
            "select factor_id,metrics_json from "
            "loadTable('dfs://QuantResearchV2Test','factor_quality_snapshot') "
            f"where scope_id='{scope}'"
        )
        if not isinstance(result, pd.DataFrame):
            raise RuntimeError(f"quality scope query did not return DataFrame: {scope}")
        scopes[key] = result.set_index("factor_id")
    if not isinstance(metadata, pd.DataFrame):
        raise RuntimeError("metadata query did not return a DataFrame")
    by_id = metadata.assign(
        factor_id=metadata["factor_id"].map(lambda value: f"yeslab_{int(value)}")
    ).set_index("factor_id")
    candidate_ids = tuple(
        sorted(
            set(scopes["validation"].index)
            & set(scopes["similarity"].index)
            & set(scopes["source"].index)
        )
    )
    records: list[dict[str, object]] = []
    for factor_id in candidate_ids:
        source_body = json.loads(str(scopes["source"].loc[factor_id, "metrics_json"]))
        validation_body = json.loads(str(scopes["validation"].loc[factor_id, "metrics_json"]))
        similarity_body = json.loads(str(scopes["similarity"].loc[factor_id, "metrics_json"]))
        source_status = source_body.get("decision_status")
        validation_status = validation_body.get("decision_status")
        similarity_status = similarity_body.get("decision_status")
        final_pass = (
            source_status == "SOURCE_DQ_PASS_PENDING_SIMILARITY"
            and validation_status == "VALIDATION_2025_PASS_PENDING_FINAL_SIMILARITY"
            and similarity_status == "SIMILARITY_FULL_DAY_UNIQUE"
        )
        meta = by_id.loc[factor_id]
        body = {
            "screen_version": SCOPE_ID,
            "factor_name": str(meta["factor_name"]),
            "source_table": str(meta["store_factor"]),
            "category": str(meta["category"]),
            "component_statuses": {
                "source": source_status,
                "validation": validation_status,
                "similarity": similarity_status,
            },
            "validation": validation_body.get("validation", {}),
            "full_day_similarity": similarity_body.get("similarity", {}),
            "decision_status": (
                "FACTOR_QUALITY_PASS_PENDING_PUBLICATION"
                if final_pass
                else "FACTOR_QUALITY_FAIL"
            ),
            "reason_codes": (
                ["SOURCE_DQ_PASS", "VALIDATION_DOUBLE_GATE_PASS", "FULL_DAY_SIMILARITY_UNIQUE"]
                if final_pass
                else ["COMBINED_FACTOR_QUALITY_GATE_FAIL"]
            ),
            "active_factor_set_changed": False,
            "model_training_started": False,
        }
        record = {
            "as_of_date": args.as_of,
            "factor_id": factor_id,
            "factor_version": 1,
            "scope_id": SCOPE_ID,
            "eligibility_status": "PASS" if final_pass else "ELIGIBILITY_FAIL",
            "metrics_json": json.dumps(body, ensure_ascii=False, sort_keys=True),
            "rules_version": 1,
            "run_id": str(uuid.uuid4()),
            "data_version": "",
            "created_at": pd.Timestamp.utcnow().tz_localize(None),
            "source_watermark_id": "",
        }
        records.append(record)
    watermark = hashlib.sha256(
        json.dumps({"factor_ids": candidate_ids, "scope": SCOPE_ID}, sort_keys=True).encode()
    ).hexdigest()
    data_version = hashlib.sha256((watermark + str(args.as_of)).encode()).hexdigest()
    for record in records:
        record["data_version"] = data_version
        record["source_watermark_id"] = watermark
        record["record_hash"] = _record_hash(record)
    frame = pd.DataFrame(records)
    statuses = frame["metrics_json"].map(lambda value: json.loads(value)["decision_status"])
    print(statuses.value_counts().sort_index().to_string())
    if args.publish:
        print(publish_snapshot(target, frame, scope_id=SCOPE_ID))
    else:
        print("dry-run: no database write")


if __name__ == "__main__":
    main()
