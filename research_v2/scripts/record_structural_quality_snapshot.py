"""Record train-window structural coverage quality for predictive candidates.

This screen deliberately uses the already materialized daily RankIC sample counts
as a bounded, read-only coverage proxy.  It compares each factor's finite
cross-section with the synthetic reference factor for the same signal date.  It
does not read the 2025/2026 label partitions and it does not claim that raw
source PIT/deduplication has passed; that check remains explicit in the reason
codes of the snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from datetime import date

import dolphindb as ddb
import numpy as np
import pandas as pd
from record_factor_quality_snapshot import publish_snapshot

from research_v2.db.session import V2WriteSession, YesLabReadOnlySession

SCOPE_ID = "SCREEN_TRAIN2020_2024_STRUCTURAL_COVERAGE_V2"
RULES_VERSION = 1
PREDICTIVE_SCOPE_ID = "SCREEN_TRAIN2020_2024_DAILY_HAC_FDR_V2"
REFERENCE_FACTOR_ID = "yeslab_0"


def compute_structural_metrics(
    daily: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    expected_factor_ids: tuple[str, ...],
    minimum_day_ratio: float = 0.80,
    minimum_valid_day_count: int = 120,
    minimum_median_coverage_ratio: float = 0.80,
    minimum_p05_coverage_ratio: float = 0.70,
    minimum_monthly_coverage_ratio: float = 0.70,
) -> pd.DataFrame:
    """Compute deterministic coverage metrics without touching the database."""
    _require_columns(daily, {"factor_id", "signal_date", "sample_count"}, "daily RankIC")
    _require_columns(reference, {"signal_date", "sample_count"}, "reference RankIC")
    if len(set(expected_factor_ids)) != len(expected_factor_ids):
        raise ValueError("expected_factor_ids must be unique")
    if not 0 < minimum_day_ratio <= 1:
        raise ValueError("minimum_day_ratio must be in (0, 1]")

    working = daily.loc[:, ["factor_id", "signal_date", "sample_count"]].copy()
    working["factor_id"] = working["factor_id"].astype(str)
    working["signal_date"] = pd.to_datetime(working["signal_date"], errors="raise").dt.date
    working["sample_count"] = pd.to_numeric(working["sample_count"], errors="coerce")
    reference_frame = reference.loc[:, ["signal_date", "sample_count"]].copy()
    reference_frame["signal_date"] = pd.to_datetime(
        reference_frame["signal_date"], errors="raise"
    ).dt.date
    reference_frame["sample_count"] = pd.to_numeric(
        reference_frame["sample_count"], errors="coerce"
    )
    if reference_frame["signal_date"].duplicated().any():
        raise ValueError("reference contains duplicate signal dates")
    reference_frame = reference_frame.loc[
        np.isfinite(reference_frame["sample_count"].to_numpy(dtype=float))
        & (reference_frame["sample_count"] > 0)
    ].copy()
    if reference_frame.empty:
        raise ValueError("reference has no finite sample counts")
    expected_dates = set(reference_frame["signal_date"])
    expected_day_count = len(expected_dates)
    reference_by_date = reference_frame.set_index("signal_date")["sample_count"]

    rows: list[dict[str, object]] = []
    for factor_id in expected_factor_ids:
        factor = working.loc[working["factor_id"] == factor_id].copy()
        factor = factor.loc[
            factor["signal_date"].isin(expected_dates)
            & np.isfinite(factor["sample_count"].to_numpy(dtype=float))
            & (factor["sample_count"] > 0)
        ].copy()
        duplicate_row_count = int(len(factor) - factor["signal_date"].nunique())
        factor = factor.drop_duplicates("signal_date", keep="last")
        valid_day_count = int(len(factor))
        day_ratio = valid_day_count / expected_day_count if expected_day_count else 0.0
        if valid_day_count:
            ratios = factor["sample_count"].to_numpy(dtype=float) / factor[
                "signal_date"
            ].map(reference_by_date).to_numpy(dtype=float)
            ratios = ratios[np.isfinite(ratios)]
            median_ratio = float(np.median(ratios)) if len(ratios) else 0.0
            p05_ratio = float(np.quantile(ratios, 0.05)) if len(ratios) else 0.0
            min_ratio = float(np.min(ratios)) if len(ratios) else 0.0
            months = pd.Series(sorted(expected_dates)).map(lambda value: value.strftime("%Y-%m"))
            expected_month_counts = months.value_counts()
            observed_month_counts = pd.Series(factor["signal_date"]).map(
                lambda value: value.strftime("%Y-%m")
            ).value_counts()
            monthly_ratios = [
                float(observed_month_counts.get(month, 0) / count)
                for month, count in expected_month_counts.items()
            ]
            min_monthly_ratio = min(monthly_ratios) if monthly_ratios else 0.0
        else:
            median_ratio = p05_ratio = min_ratio = min_monthly_ratio = 0.0
        gate = (
            valid_day_count >= minimum_valid_day_count
            and day_ratio >= minimum_day_ratio
            and median_ratio >= minimum_median_coverage_ratio
            and p05_ratio >= minimum_p05_coverage_ratio
            and min_monthly_ratio >= minimum_monthly_coverage_ratio
            and duplicate_row_count == 0
        )
        if valid_day_count == 0:
            status = "STRUCTURAL_DQ_PENDING_INPUT"
            reasons = ["NO_DAILY_SAMPLE_COUNT"]
        elif gate:
            status = "STRUCTURAL_DQ_PASS_PENDING_RAW_PIT"
            reasons = ["COVERAGE_GATE_PASS", "RAW_PIT_DEDUP_PENDING"]
        else:
            status = "STRUCTURAL_DQ_FAIL"
            reasons = ["COVERAGE_GATE_FAIL"]
            if day_ratio < minimum_day_ratio:
                reasons.append("LOW_DAY_COVERAGE")
            if median_ratio < minimum_median_coverage_ratio:
                reasons.append("LOW_MEDIAN_SAMPLE_COVERAGE")
            if p05_ratio < minimum_p05_coverage_ratio:
                reasons.append("LOW_P05_SAMPLE_COVERAGE")
            if min_monthly_ratio < minimum_monthly_coverage_ratio:
                reasons.append("MISSING_MONTH_OR_LOW_MONTHLY_COVERAGE")
            if duplicate_row_count:
                reasons.append("DUPLICATE_DAILY_KEY")
        rows.append(
            {
                "factor_id": factor_id,
                "expected_day_count": expected_day_count,
                "valid_day_count": valid_day_count,
                "day_ratio": day_ratio,
                "median_coverage_ratio": median_ratio,
                "p05_coverage_ratio": p05_ratio,
                "minimum_coverage_ratio": min_ratio,
                "minimum_monthly_day_ratio": min_monthly_ratio,
                "duplicate_row_count": duplicate_row_count,
                "coverage_gate_pass": gate,
                "decision_status": status,
                "reason_codes": reasons,
            }
        )
    return pd.DataFrame(rows)


def build_snapshot_frame(
    metadata: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    as_of_date: date,
    run_id: str,
    data_version: str,
    source_watermark_id: str,
) -> pd.DataFrame:
    """Build one append-only snapshot row per structurally screened factor."""
    _require_columns(metadata, {"factor_id", "factor_name", "store_factor", "category"}, "metadata")
    _require_columns(metrics, {"factor_id", "decision_status", "reason_codes"}, "metrics")
    metadata_by_id = metadata.assign(
        factor_id=metadata["factor_id"].map(lambda value: f"yeslab_{int(value)}")
    ).set_index("factor_id")
    rows: list[dict[str, object]] = []
    for metric in metrics.sort_values("factor_id", kind="mergesort").to_dict("records"):
        factor_id = str(metric["factor_id"])
        if factor_id not in metadata_by_id.index:
            raise ValueError(f"metrics contains unknown factor: {factor_id}")
        meta = metadata_by_id.loc[factor_id]
        status = str(metric["decision_status"])
        eligibility = (
            "PASS" if status == "STRUCTURAL_DQ_PASS_PENDING_RAW_PIT" else "ELIGIBILITY_FAIL"
        )
        metric_json = {
            "screen_version": SCOPE_ID,
            "factor_name": str(meta["factor_name"]),
            "source_table": str(meta["store_factor"]),
            "category": str(meta["category"]),
            "structural": {
                key: _json_value(value)
                for key, value in metric.items()
                if key not in {"factor_id", "decision_status", "reason_codes"}
            },
            "decision_status": status,
            "reason_codes": metric["reason_codes"],
            "raw_source_pit_dedup_checked": False,
        }
        record = {
            "as_of_date": as_of_date,
            "factor_id": factor_id,
            "factor_version": 1,
            "scope_id": SCOPE_ID,
            "eligibility_status": eligibility,
            "metrics_json": json.dumps(metric_json, ensure_ascii=False, sort_keys=True),
            "rules_version": RULES_VERSION,
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
    prior = target.query(
        "select factor_id,metrics_json from "
        "loadTable('dfs://QuantResearchV2Test','factor_quality_snapshot') "
        f"where scope_id='{PREDICTIVE_SCOPE_ID}'"
    )
    if not isinstance(metadata, pd.DataFrame) or not isinstance(prior, pd.DataFrame):
        raise RuntimeError("quality metadata query did not return DataFrames")
    candidates = tuple(
        str(row.factor_id)
        for row in prior.itertuples(index=False)
        if json.loads(str(row.metrics_json)).get("decision_status")
        == "TRAIN_PREDICTIVE_PASS_PENDING_DQ"
    )
    if not candidates:
        raise RuntimeError("no predictive candidates found in the daily screen")
    quoted = ",".join(f"'{factor_id}'" for factor_id in (REFERENCE_FACTOR_ID, *candidates))
    daily = target.query(
        "select factor_id,signal_date,sample_count "
        "from loadTable('dfs://QuantResearchV2Test','factor_rank_ic_daily') "
        "where signal_date between date('2020.01.01') and date('2024.12.31') "
        f"and factor_id in [{quoted}]"
    )
    if not isinstance(daily, pd.DataFrame):
        raise RuntimeError("daily quality query did not return a DataFrame")
    reference = daily.loc[
        daily["factor_id"] == REFERENCE_FACTOR_ID, ["signal_date", "sample_count"]
    ]
    metrics = compute_structural_metrics(
        daily.loc[daily["factor_id"] != REFERENCE_FACTOR_ID],
        reference,
        expected_factor_ids=candidates,
    )
    source_watermark_id = hashlib.sha256(
        json.dumps(
            {"daily_rows": len(daily), "candidate_ids": candidates, "scope": SCOPE_ID},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    data_version = hashlib.sha256((source_watermark_id + str(args.as_of)).encode()).hexdigest()
    frame = build_snapshot_frame(
        metadata,
        metrics,
        as_of_date=args.as_of,
        run_id=str(uuid.uuid4()),
        data_version=data_version,
        source_watermark_id=source_watermark_id,
    )
    print(metrics["decision_status"].value_counts().sort_index().to_string())
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
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"{label} is missing columns: {missing}")


if __name__ == "__main__":
    main()
