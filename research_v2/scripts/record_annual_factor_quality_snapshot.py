"""Record a provisional annual-summary factor quality prefilter.

The existing ``factor_rank_ic_summary`` table contains 2020--2024 annual
summaries for most registered observations.  This command records those
statistics separately from the strict 2025-validation screen.  It is a
prefilter only: no factor set is frozen and no 2026 data is read.
"""

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

SCOPE_ID = "SCREEN_TRAIN2020_2024_ANNUAL_SUMMARY_PREFILTER_V1"
RULES_VERSION = 1


def build_annual_frame(
    metadata: pd.DataFrame,
    annual_summary: pd.DataFrame,
    *,
    as_of_date: date,
    run_id: str,
    data_version: str,
    source_watermark_id: str,
) -> pd.DataFrame:
    required_metadata = {"factor_id", "factor_name", "store_factor", "category"}
    required_summary = {
        "factor_id",
        "factor_version",
        "factor_name",
        "evaluation_scope",
        "valid_day_count",
        "mean_rank_ic",
        "rank_ic_std",
        "icir",
        "positive_day_share",
    }
    _require_columns(metadata, required_metadata, "factor metadata")
    _require_columns(annual_summary, required_summary, "annual RankIC summary")
    scoped = annual_summary.loc[
        annual_summary["evaluation_scope"].astype(str).isin(
            {f"{year}-ANNUAL" for year in range(2020, 2025)}
        )
    ].copy()
    if scoped.duplicated(["factor_id", "factor_version", "evaluation_scope"]).any():
        raise ValueError("annual summary contains duplicate factor/scope rows")
    by_factor = {
        str(factor_id): group.sort_values("evaluation_scope", kind="mergesort")
        for factor_id, group in scoped.groupby("factor_id", sort=True)
    }
    rows: list[dict[str, object]] = []
    for metadata_row in metadata.sort_values("factor_id", kind="mergesort").itertuples(
        index=False
    ):
        factor_id = f"yeslab_{int(metadata_row.factor_id)}"
        group = by_factor.get(factor_id, pd.DataFrame())
        metrics: dict[str, object] = {
            "screen_version": SCOPE_ID,
            "factor_name": str(metadata_row.factor_name),
            "source_table": str(metadata_row.store_factor),
            "category": str(metadata_row.category),
            "evidence_source": "factor_rank_ic_summary",
        }
        if len(group) != 5:
            status = "UNSCREENED"
            reason_codes = ["MISSING_ANNUAL_SUMMARY_SCOPE"]
            eligibility_status = "ELIGIBILITY_FAIL"
        else:
            direction = 1 if float(group["mean_rank_ic"].mean()) >= 0 else -1
            annual_rows: list[dict[str, object]] = []
            failures: list[str] = []
            for row in group.to_dict("records"):
                signed_mean = float(row["mean_rank_ic"]) * direction
                signed_share = (
                    float(row["positive_day_share"])
                    if direction > 0
                    else 1.0 - float(row["positive_day_share"])
                )
                annual_rows.append(
                    {
                        "scope": str(row["evaluation_scope"]),
                        "valid_day_count": int(row["valid_day_count"]),
                        "mean_rank_ic": float(row["mean_rank_ic"]),
                        "rank_ic_std": float(row["rank_ic_std"]),
                        "icir": float(row["icir"]),
                        "signed_mean_rank_ic": signed_mean,
                        "signed_positive_day_share": signed_share,
                    }
                )
                if int(row["valid_day_count"]) < 60:
                    failures.append("VALID_DAY_COUNT")
                if abs(signed_mean) < 0.02:
                    failures.append("MEAN_RANK_IC")
                if abs(float(row["icir"])) < 0.50:
                    failures.append("ICIR")
                if signed_share < 0.55:
                    failures.append("DIRECTIONAL_SHARE")
            metrics["direction"] = direction
            metrics["annual"] = annual_rows
            if failures:
                status = "ANNUAL_PREDICTIVE_FAIL"
                reason_codes = sorted(set(failures))
                eligibility_status = "ELIGIBILITY_FAIL"
            else:
                status = "ANNUAL_STABLE_PROVISIONAL"
                reason_codes = ["FIVE_YEAR_ANNUAL_PREFILTER_PASS", "PENDING_PIT_DQ_FDR"]
                eligibility_status = "PASS"
        metrics["decision_status"] = status
        metrics["reason_codes"] = reason_codes
        record = {
            "as_of_date": as_of_date,
            "factor_id": factor_id,
            "factor_version": 1,
            "scope_id": SCOPE_ID,
            "eligibility_status": eligibility_status,
            "metrics_json": json.dumps(metrics, ensure_ascii=False, sort_keys=True),
            "rules_version": RULES_VERSION,
            "run_id": run_id,
            "data_version": data_version,
            "created_at": pd.Timestamp.utcnow().tz_localize(None),
            "source_watermark_id": source_watermark_id,
        }
        record["record_hash"] = hashlib.sha256(
            json.dumps(
                {key: (value.isoformat() if isinstance(value, (date, pd.Timestamp)) else value)
                 for key, value in record.items() if key != "record_hash"},
                ensure_ascii=False,
                default=str,
                sort_keys=True,
            ).encode()
        ).hexdigest()
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
    summary = target.query(
        "select factor_id,factor_version,factor_name,evaluation_scope,valid_day_count,"
        "mean_rank_ic,rank_ic_std,icir,positive_day_share "
        "from loadTable('dfs://QuantResearchV2Test','factor_rank_ic_summary')"
    )
    if not isinstance(metadata, pd.DataFrame) or not isinstance(summary, pd.DataFrame):
        raise RuntimeError("quality inputs did not return DataFrames")
    watermark_payload = {
        "metadata_rows": len(metadata),
        "summary_rows": len(summary),
        "scope": SCOPE_ID,
    }
    watermark = hashlib.sha256(
        json.dumps(watermark_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    data_version = hashlib.sha256((watermark + str(args.as_of)).encode()).hexdigest()
    frame = build_annual_frame(
        metadata,
        summary,
        as_of_date=args.as_of,
        run_id=str(uuid.uuid4()),
        data_version=data_version,
        source_watermark_id=watermark,
    )
    statuses = frame["metrics_json"].map(lambda value: json.loads(value)["decision_status"])
    print(statuses.value_counts().sort_index().to_string())
    if args.publish:
        print(publish_snapshot(target, frame, scope_id=SCOPE_ID))
    else:
        print("dry-run: no database write")


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"{label} is missing columns: {missing}")


if __name__ == "__main__":
    main()
