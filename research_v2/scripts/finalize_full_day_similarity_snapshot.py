"""Normalize the completed full-day similarity artifact into a formal scope."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from datetime import date
from pathlib import Path

import dolphindb as ddb
import pandas as pd
from record_factor_quality_snapshot import publish_snapshot

from research_v2.db.session import V2WriteSession, YesLabReadOnlySession

SCOPE_ID = "SCREEN_TRAIN2020_2024_SIMILARITY_FULL_DAY26_V3"
ARTIFACT = Path("research_v2/artifacts/diagnostics/train_similarity_full_day_2020_2024.csv")
_THRESHOLD = 0.90


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    parser.add_argument("--as-of", type=date.fromisoformat, default=date(2026, 8, 4))
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    artifact = pd.read_csv(args.artifact)
    required = {
        "factor_id",
        "nearest_factor_id",
        "nearest_similarity",
        "nearest_valid_days",
    }
    if missing := sorted(required.difference(artifact.columns)):
        raise ValueError(f"full-day similarity artifact is missing columns: {missing}")
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
    if not isinstance(metadata, pd.DataFrame):
        raise RuntimeError("metadata query did not return a DataFrame")
    by_id = metadata.assign(
        factor_id=metadata["factor_id"].map(lambda value: f"yeslab_{int(value)}")
    ).set_index("factor_id")
    watermark = hashlib.sha256(
        json.dumps(
            {
                "artifact_rows": len(artifact),
                "artifact": str(args.artifact),
                "scope": SCOPE_ID,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    data_version = hashlib.sha256((watermark + str(args.as_of)).encode()).hexdigest()
    records: list[dict[str, object]] = []
    for metric in artifact.sort_values("factor_id", kind="mergesort").to_dict("records"):
        factor_id = str(metric["factor_id"])
        if factor_id not in by_id.index:
            raise ValueError(f"unknown factor in artifact: {factor_id}")
        similarity = float(metric["nearest_similarity"])
        redundant = similarity >= _THRESHOLD
        status = "SIMILARITY_FULL_DAY_REDUNDANT" if redundant else "SIMILARITY_FULL_DAY_UNIQUE"
        reasons = [
            "FULL_TRAIN_CROSS_SECTIONAL_SIMILARITY",
            "ABOVE_SIMILARITY_THRESHOLD"
            if redundant
            else "BELOW_SIMILARITY_THRESHOLD",
        ]
        meta = by_id.loc[factor_id]
        body = {
            "screen_version": SCOPE_ID,
            "factor_name": str(meta["factor_name"]),
            "source_table": str(meta["store_factor"]),
            "category": str(meta["category"]),
            "evidence_scope": {
                "start": "2020-01-01",
                "end": "2024-12-31",
                "valid_day_count": int(metric["nearest_valid_days"]),
                "threshold": _THRESHOLD,
            },
            "similarity": {
                "nearest_factor_id": metric["nearest_factor_id"],
                "nearest_similarity": similarity,
                "nearest_valid_days": int(metric["nearest_valid_days"]),
            },
            "decision_status": status,
            "reason_codes": reasons,
            "full_day_cross_section_checked": True,
        }
        record = {
            "as_of_date": args.as_of,
            "factor_id": factor_id,
            "factor_version": 1,
            "scope_id": SCOPE_ID,
            "eligibility_status": "PASS",
            "metrics_json": json.dumps(body, ensure_ascii=False, sort_keys=True),
            "rules_version": 2,
            "run_id": str(uuid.uuid4()),
            "data_version": data_version,
            "created_at": pd.Timestamp.utcnow().tz_localize(None),
            "source_watermark_id": watermark,
        }
        record["record_hash"] = hashlib.sha256(
            json.dumps(
                {
                    key: value.isoformat()
                    if isinstance(value, (date, pd.Timestamp))
                    else value
                    for key, value in record.items()
                },
                ensure_ascii=False,
                default=str,
                sort_keys=True,
            ).encode()
        ).hexdigest()
        records.append(record)
    statuses = pd.Series(
        [json.loads(record["metrics_json"])["decision_status"] for record in records]
    )
    print(statuses.value_counts().sort_index().to_string())
    if args.publish:
        print(publish_snapshot(target, pd.DataFrame(records), scope_id=SCOPE_ID))
    else:
        print("dry-run: no database write")


if __name__ == "__main__":
    main()
