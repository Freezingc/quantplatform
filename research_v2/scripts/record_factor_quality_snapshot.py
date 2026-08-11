"""Record the current factor-quality screen in the existing V2 snapshot table.

This command does not recompute labels, read the locked year, or alter any
existing table.  It records one append-only diagnostic snapshot for all factor
metadata rows: factors covered by the strict artifact receive predictive
metrics; all other definitions are explicitly marked ``UNSCREENED``.  The
snapshot is intentionally separate from the frozen factor-set contract.
"""

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

from research_v2.db.session import V2WriteSession, YesLabReadOnlySession

_SCOPE_ID = "SCREEN_TRAIN2020_2024_VALIDATION2025_STRICT26_V1"
_RULES_VERSION = 1
_FROZEN = {
    "yeslab_377",
    "yeslab_380",
    "yeslab_387",
    "yeslab_388",
    "yeslab_391",
    "yeslab_433",
}
_TRAIN_SUMMARY = "gp_candidate_strict_train_2020_2024/summary.csv"
_VALIDATION_SUMMARY = "gp_candidate_strict_validation_2025_summary.csv"


def build_snapshot_frame(
    metadata: pd.DataFrame,
    train_summary: pd.DataFrame,
    validation_summary: pd.DataFrame,
    *,
    as_of_date: date,
    run_id: str,
    data_version: str,
    source_watermark_id: str,
) -> pd.DataFrame:
    """Build canonical rows without performing any database writes."""
    required_metadata = {"factor_id", "factor_name", "store_factor", "category"}
    required_train = {
        "factor_name",
        "valid_day_count",
        "mean_rank_ic",
        "rank_ic_std",
        "icir",
        "positive_day_share",
        "direction",
        "directional_day_share",
        "finite_ratio",
        "minimum_monthly_finite_ratio",
    }
    required_validation = {
        "factor_name",
        "valid_day_count",
        "mean_rank_ic",
        "rank_ic_std",
        "icir",
        "positive_day_share",
    }
    _require_columns(metadata, required_metadata, "factor metadata")
    _require_columns(train_summary, required_train, "TRAIN summary")
    _require_columns(validation_summary, required_validation, "VALIDATION summary")
    if not run_id or not data_version or not source_watermark_id:
        raise ValueError("lineage identifiers cannot be empty")
    if metadata["factor_id"].duplicated().any():
        raise ValueError("factor metadata contains duplicate factor IDs")
    if train_summary["factor_name"].duplicated().any():
        raise ValueError("TRAIN summary contains duplicate factor names")
    if validation_summary["factor_name"].duplicated().any():
        raise ValueError("VALIDATION summary contains duplicate factor names")

    train_by_name = train_summary.set_index("factor_name")
    validation_by_name = validation_summary.set_index("factor_name")
    rows: list[dict[str, object]] = []
    for metadata_row in metadata.sort_values("factor_id", kind="mergesort").itertuples(
        index=False
    ):
        factor_id = f"yeslab_{int(metadata_row.factor_id)}"
        factor_name = str(metadata_row.factor_name)
        metrics: dict[str, object] = {
            "screen_version": _SCOPE_ID,
            "factor_name": factor_name,
            "source_table": str(metadata_row.store_factor),
            "category": str(metadata_row.category),
        }
        if factor_id not in train_by_name.index or factor_id not in validation_by_name.index:
            decision_status = "UNSCREENED"
            reason_codes = ["NO_STRICT_CROSS_WINDOW_ARTIFACT"]
            eligibility_status = "ELIGIBILITY_FAIL"
        else:
            train = train_by_name.loc[factor_id]
            validation = validation_by_name.loc[factor_id]
            train_direction = int(train["direction"])
            validation_signed_mean = float(validation["mean_rank_ic"]) * train_direction
            validation_directional_share = (
                float(validation["positive_day_share"])
                if train_direction > 0
                else 1.0 - float(validation["positive_day_share"])
            )
            validation_gate = (
                int(validation["valid_day_count"]) >= 60
                and abs(float(validation["mean_rank_ic"])) >= 0.02
                and abs(float(validation["icir"])) >= 0.50
                and validation_directional_share >= 0.55
                and validation_signed_mean > 0
            )
            metrics["train"] = _metric_dict(train)
            metrics["validation"] = _metric_dict(validation)
            metrics["direction"] = train_direction
            metrics["validation_directional_share"] = validation_directional_share
            if not validation_gate:
                decision_status = "PREDICTIVE_FAIL"
                reason_codes = ["VALIDATION_PREDICTIVE_GATE"]
                eligibility_status = "ELIGIBILITY_FAIL"
            elif factor_id in _FROZEN:
                decision_status = "FROZEN"
                reason_codes = ["CURRENT_FINAL_FACTOR_SET"]
                eligibility_status = "PASS"
            else:
                decision_status = "PREDICTIVE_STABLE_PENDING_DQ"
                reason_codes = ["TRAIN_VALIDATION_PREDICTIVE_PASS", "PENDING_FULL_DQ"]
                eligibility_status = "PASS"
        metrics["decision_status"] = decision_status
        metrics["reason_codes"] = reason_codes
        record = {
            "as_of_date": as_of_date,
            "factor_id": factor_id,
            "factor_version": 1,
            "scope_id": _SCOPE_ID,
            "eligibility_status": eligibility_status,
            "metrics_json": json.dumps(metrics, ensure_ascii=False, sort_keys=True),
            "rules_version": _RULES_VERSION,
            "run_id": run_id,
            "data_version": data_version,
            "created_at": pd.Timestamp.utcnow().tz_localize(None),
            "source_watermark_id": source_watermark_id,
        }
        record["record_hash"] = _record_hash(record)
        rows.append(record)
    return pd.DataFrame(rows)


def publish_snapshot(
    session: V2WriteSession,
    frame: pd.DataFrame,
    *,
    scope_id: str = _SCOPE_ID,
) -> str:
    """Insert one snapshot idempotently; reject a conflicting same-key snapshot."""
    if frame.empty:
        raise ValueError("snapshot frame cannot be empty")
    as_of = pd.Timestamp(frame["as_of_date"].iloc[0]).strftime("%Y.%m.%d")
    existing = session.query(
        "select factor_id,factor_version,scope_id,data_version,record_hash "
        f"from loadTable('{session.target_database}','factor_quality_snapshot') "
        f"where as_of_date=date('{as_of}') and scope_id='{_quote(scope_id)}'"
    )
    if not isinstance(existing, pd.DataFrame):
        raise RuntimeError("quality snapshot lookup did not return a DataFrame")
    if not existing.empty:
        incoming_versions = set(frame["data_version"].astype(str))
        existing_versions = set(existing["data_version"].astype(str))
        if len(incoming_versions) == 1 and incoming_versions == existing_versions:
            return "skipped"
        raise RuntimeError("conflicting quality snapshot already exists for this scope/date")

    stage = f"quality_snapshot_{uuid.uuid4().hex[:16]}"
    session.upload_frame(stage, frame)
    script = (
        f"typed = select date(as_of_date) as as_of_date, symbol(factor_id) as factor_id, "
        f"int(factor_version) as factor_version, string(scope_id) as scope_id, "
        f"symbol(eligibility_status) as eligibility_status, string(metrics_json) as metrics_json, "
        f"int(rules_version) as rules_version, uuid(run_id) as run_id, "
        f"string(data_version) as data_version, timestamp(created_at) as created_at, "
        f"string(source_watermark_id) as source_watermark_id, string(record_hash) as record_hash "
        f"from {stage}\n"
        "tableInsert{loadTable("
        f"'{session.target_database}','factor_quality_snapshot')}}(typed)\n"
        f"undef(`{stage}, VAR)\nundef(`typed, VAR)"
    )
    session.execute(script)
    return "published"


def _metric_dict(row: pd.Series) -> dict[str, object]:
    return {
        "valid_day_count": int(row["valid_day_count"]),
        "mean_rank_ic": float(row["mean_rank_ic"]),
        "rank_ic_std": float(row["rank_ic_std"]),
        "icir": float(row["icir"]),
        "positive_day_share": float(row["positive_day_share"]),
        **(
            {
                "directional_day_share": float(row["directional_day_share"]),
                "finite_ratio": float(row["finite_ratio"]),
                "minimum_monthly_finite_ratio": float(row["minimum_monthly_finite_ratio"]),
            }
            if "directional_day_share" in row.index
            else {}
        ),
    }


def _record_hash(record: dict[str, object]) -> str:
    payload = {
        key: (value.isoformat() if isinstance(value, (date, pd.Timestamp)) else value)
        for key, value in record.items()
        if key != "record_hash"
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True).encode()
    ).hexdigest()


def _connect() -> tuple[YesLabReadOnlySession, V2WriteSession]:
    client = ddb.session()
    client.connect(
        os.getenv("QRV2_DDB_HOST", "10.176.55.211"),
        int(os.getenv("QRV2_DDB_PORT", "8848")),
        os.getenv("QRV2_DDB_USERNAME", "xiahongjie"),
        os.getenv("QRV2_DDB_PASSWORD", "123456"),
    )
    return YesLabReadOnlySession(client), V2WriteSession(
        client, target_database="dfs://QuantResearchV2Test"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/diagnostics"))
    parser.add_argument("--as-of", type=date.fromisoformat, default=date(2026, 8, 4))
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    source, target = _connect()
    metadata_result = source.query(
        "select factor_id,factor_name,store_factor,category "
        "from loadTable('dfs://YesLab','factor_metadata')"
    )
    if not isinstance(metadata_result, pd.DataFrame):
        raise RuntimeError("factor metadata query did not return a DataFrame")
    train = pd.read_csv(args.artifacts / _TRAIN_SUMMARY)
    validation = pd.read_csv(args.artifacts / _VALIDATION_SUMMARY)
    source_watermark_id = hashlib.sha256(
        json.dumps(
            {"metadata_rows": len(metadata_result), "scope": _SCOPE_ID},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    data_version = hashlib.sha256(
        (source_watermark_id + _SCOPE_ID + str(args.as_of)).encode()
    ).hexdigest()
    run_id = str(uuid.uuid4())
    frame = build_snapshot_frame(
        metadata_result,
        train,
        validation,
        as_of_date=args.as_of,
        run_id=run_id,
        data_version=data_version,
        source_watermark_id=source_watermark_id,
    )
    counts = frame["metrics_json"].map(lambda value: json.loads(value)["decision_status"])
    print(counts.value_counts().sort_index().to_string())
    if args.publish:
        print(publish_snapshot(target, frame))
    else:
        print("dry-run: no database write")


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"{label} is missing columns: {missing}")


def _quote(value: str) -> str:
    return value.replace("'", "''")


if __name__ == "__main__":
    main()
