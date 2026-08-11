"""Persist a full 2020--2024 daily RankIC/HAC/FDR prefilter.

The input is the existing V2 ``factor_rank_ic_daily`` table, which contains
daily TRAIN observations through 2024-12-31.  This scope is deliberately
separate from the 2025 validation screen and does not freeze a factor set.
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
from research_v2.evaluation.hac import hac_mean_test

SCOPE_ID = "SCREEN_TRAIN2020_2024_DAILY_HAC_FDR_V2"
RULES_VERSION = 2
_EXCLUDED_FACTOR_IDS = {"yeslab_0", "yeslab_124"}


def compute_train_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    """Compute signed predictive metrics, HAC p-values, and BH q-values."""
    required = {"signal_date", "factor_id", "factor_version", "rank_ic", "sample_count"}
    _require_columns(daily, required, "daily RankIC")
    if daily.duplicated(["signal_date", "factor_id", "factor_version"]).any():
        raise ValueError("daily RankIC contains duplicate factor/date keys")
    working = daily.copy()
    working["signal_date"] = pd.to_datetime(working["signal_date"], errors="raise").dt.date
    working["factor_id"] = working["factor_id"].astype(str)
    working["rank_ic"] = pd.to_numeric(working["rank_ic"], errors="coerce")
    working = working.loc[np.isfinite(working["rank_ic"].to_numpy(dtype=float))].copy()
    rows: list[dict[str, object]] = []
    for (factor_id, factor_version), group in working.groupby(
        ["factor_id", "factor_version"], sort=True
    ):
        group = group.sort_values("signal_date", kind="mergesort")
        values = group["rank_ic"].to_numpy(dtype=float)
        mean_rank_ic = float(values.mean())
        rank_ic_std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        icir = mean_rank_ic / rank_ic_std if rank_ic_std > 0 else float("nan")
        direction = 1 if mean_rank_ic >= 0 else -1
        signed = values * direction
        years = pd.to_datetime(group["signal_date"], errors="raise").dt.year.to_numpy()
        annual = pd.DataFrame({"year": years, "signed_rank_ic": signed}).groupby("year")
        annual_means = annual["signed_rank_ic"].mean()
        positive_years = int((annual_means > 0).sum())
        hac = hac_mean_test(pd.Series(signed), max_lags=5)
        rows.append(
            {
                "factor_id": factor_id,
                "factor_version": int(factor_version),
                "valid_day_count": len(values),
                "mean_rank_ic": mean_rank_ic,
                "rank_ic_std": rank_ic_std,
                "icir": icir,
                "direction": direction,
                "signed_mean_rank_ic": float(signed.mean()),
                "directional_day_share": float((signed > 0).mean()),
                "positive_years": positive_years,
                "annual_signed_mean_rank_ic": {
                    str(int(year)): float(value) for year, value in annual_means.items()
                },
                "hac_t_statistic": float(hac.t_statistic),
                "hac_p_value": float(hac.p_value),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("daily RankIC has no finite observations")
    result["fdr_q_value"] = _benjamini_hochberg(result["hac_p_value"])
    result["prefilter_pass"] = (
        (result["valid_day_count"] >= 120)
        & (result["mean_rank_ic"].abs() >= 0.02)
        & (result["icir"].abs() >= 0.50)
        & (result["directional_day_share"] >= 0.55)
        & (result["positive_years"] >= 3)
        & (result["fdr_q_value"] <= 0.10)
    )
    return result


def build_snapshot_frame(
    metadata: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    as_of_date: date,
    run_id: str,
    data_version: str,
    source_watermark_id: str,
) -> pd.DataFrame:
    _require_columns(metadata, {"factor_id", "factor_name", "store_factor", "category"}, "metadata")
    _require_columns(metrics, {"factor_id", "factor_version", "prefilter_pass"}, "metrics")
    metric_by_id = metrics.set_index(["factor_id", "factor_version"])
    rows: list[dict[str, object]] = []
    for metadata_row in metadata.sort_values("factor_id", kind="mergesort").itertuples(
        index=False
    ):
        factor_id = f"yeslab_{int(metadata_row.factor_id)}"
        key = (factor_id, 1)
        metrics_json: dict[str, object] = {
            "screen_version": SCOPE_ID,
            "factor_name": str(metadata_row.factor_name),
            "source_table": str(metadata_row.store_factor),
            "category": str(metadata_row.category),
            "evidence_source": "factor_rank_ic_daily",
        }
        if factor_id in _EXCLUDED_FACTOR_IDS:
            status = "EXCLUDED_SYNTHETIC"
            reasons = ["EXCLUDED_FACTOR_ID"]
            eligibility = "ELIGIBILITY_FAIL"
        elif key not in metric_by_id.index:
            status = "UNSCREENED"
            reasons = ["NO_TRAIN_DAILY_RANK_IC"]
            eligibility = "ELIGIBILITY_FAIL"
        else:
            metric = metric_by_id.loc[key]
            metric_dict = {
                key_name: _json_value(value)
                for key_name, value in metric.to_dict().items()
                if key_name != "prefilter_pass"
            }
            metrics_json["predictive"] = metric_dict
            if bool(metric["prefilter_pass"]):
                status = "TRAIN_PREDICTIVE_PASS_PENDING_DQ"
                reasons = ["TRAIN_DAILY_HAC_FDR_PASS", "PENDING_STRUCTURAL_DQ"]
                eligibility = "PASS"
            else:
                status = "TRAIN_PREDICTIVE_FAIL"
                reasons = ["TRAIN_DAILY_HAC_FDR_GATE"]
                eligibility = "ELIGIBILITY_FAIL"
        metrics_json["decision_status"] = status
        metrics_json["reason_codes"] = reasons
        record = {
            "as_of_date": as_of_date,
            "factor_id": factor_id,
            "factor_version": 1,
            "scope_id": SCOPE_ID,
            "eligibility_status": eligibility,
            "metrics_json": json.dumps(metrics_json, ensure_ascii=False, sort_keys=True),
            "rules_version": RULES_VERSION,
            "run_id": run_id,
            "data_version": data_version,
            "created_at": pd.Timestamp.utcnow().tz_localize(None),
            "source_watermark_id": source_watermark_id,
        }
        record["record_hash"] = hashlib.sha256(
            json.dumps(
                {
                    key_name: (
                        value.isoformat() if isinstance(value, (date, pd.Timestamp)) else value
                    )
                    for key_name, value in record.items()
                    if key_name != "record_hash"
                },
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
    daily = target.query(
        "select signal_date,factor_id,factor_version,rank_ic,sample_count "
        "from loadTable('dfs://QuantResearchV2Test','factor_rank_ic_daily') "
        "where signal_date between date('2020.01.01') and date('2024.12.31')"
    )
    if not isinstance(metadata, pd.DataFrame) or not isinstance(daily, pd.DataFrame):
        raise RuntimeError("quality inputs did not return DataFrames")
    metrics = compute_train_metrics(daily)
    watermark_payload = {
        "daily_rows": len(daily),
        "factor_count": int(metrics["factor_id"].nunique()),
        "scope": SCOPE_ID,
    }
    watermark = hashlib.sha256(
        json.dumps(watermark_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    data_version = hashlib.sha256((watermark + str(args.as_of)).encode()).hexdigest()
    frame = build_snapshot_frame(
        metadata,
        metrics,
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


def _benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    values = pd.to_numeric(p_values, errors="coerce")
    finite = values.notna() & np.isfinite(values.to_numpy(dtype=float))
    ordered = values.loc[finite].sort_values(kind="mergesort")
    total = len(ordered)
    adjusted = pd.Series(np.nan, index=values.index, dtype=float)
    running = 1.0
    for rank in range(total, 0, -1):
        index = ordered.index[rank - 1]
        running = min(running, float(ordered.iloc[rank - 1]) * total / rank)
        adjusted.loc[index] = running
    return adjusted


def _json_value(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"{label} is missing columns: {missing}")


if __name__ == "__main__":
    main()
