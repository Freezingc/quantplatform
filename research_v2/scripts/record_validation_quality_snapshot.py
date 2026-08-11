"""Record 2025 VALIDATION predictive quality for TRAIN-only unique candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from datetime import date
from pathlib import Path

import dolphindb as ddb
import numpy as np
import pandas as pd
from record_factor_quality_snapshot import publish_snapshot
from record_train_daily_quality_snapshot import _benjamini_hochberg

from research_v2.db.session import V2WriteSession, YesLabReadOnlySession
from research_v2.evaluation.hac import hac_mean_test

SCOPE_ID = "SCREEN_TRAIN2020_2024_VALIDATION2025_UNIQUE21_V1"
TRAIN_SCOPE_ID = "SCREEN_TRAIN2020_2024_DAILY_HAC_FDR_V2"
SIMILARITY_SCOPE_ID = "SCREEN_TRAIN2020_2024_SIMILARITY_MONTHLY_V1"
_DEFAULT_ARTIFACT = Path("research_v2/artifacts/diagnostics/train_unique_validation_2025")


def compute_validation_metrics(
    daily: pd.DataFrame,
    *,
    train_directions: dict[str, int],
    minimum_valid_days: int = 60,
    minimum_abs_mean_rank_ic: float = 0.02,
    minimum_abs_icir: float = 0.50,
    minimum_directional_share: float = 0.55,
    maximum_fdr_q: float = 0.10,
) -> pd.DataFrame:
    """Compute fixed-direction validation metrics and an across-factor FDR gate."""
    required = {"factor_id", "signal_date", "rank_ic"}
    if missing := sorted(required.difference(daily.columns)):
        raise ValueError(f"validation RankIC is missing columns: {missing}")
    frame = daily.loc[:, ["factor_id", "signal_date", "rank_ic"]].copy()
    frame["factor_id"] = frame["factor_id"].astype(str)
    frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="raise").dt.date
    frame["rank_ic"] = pd.to_numeric(frame["rank_ic"], errors="coerce")
    frame = frame.loc[np.isfinite(frame["rank_ic"].to_numpy(dtype=float))].copy()
    rows: list[dict[str, object]] = []
    for factor_id, group in frame.groupby("factor_id", sort=True):
        if factor_id not in train_directions:
            raise ValueError(f"missing TRAIN direction for {factor_id}")
        direction = int(train_directions[factor_id])
        if direction not in {-1, 1}:
            raise ValueError(f"invalid TRAIN direction for {factor_id}: {direction}")
        values = group.sort_values("signal_date", kind="mergesort")["rank_ic"].to_numpy(dtype=float)
        signed = values * direction
        std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        icir = float(values.mean() / std) if np.isfinite(std) and std > 0 else float("nan")
        hac = hac_mean_test(pd.Series(signed), max_lags=5)
        rows.append(
            {
                "factor_id": factor_id,
                "valid_day_count": len(values),
                "mean_rank_ic": float(values.mean()),
                "rank_ic_std": std,
                "icir": icir,
                "direction": direction,
                "signed_mean_rank_ic": float(signed.mean()),
                "directional_day_share": float((signed > 0).mean()),
                "hac_t_statistic": float(hac.t_statistic),
                "hac_p_value": float(hac.p_value),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("validation RankIC has no finite observations")
    result["fdr_q_value"] = _benjamini_hochberg(result["hac_p_value"])
    result["decision_status"] = np.where(
        (result["valid_day_count"] >= minimum_valid_days)
        & (result["mean_rank_ic"].abs() >= minimum_abs_mean_rank_ic)
        & (result["icir"].abs() >= minimum_abs_icir)
        & (result["directional_day_share"] >= minimum_directional_share)
        & (result["signed_mean_rank_ic"] > 0)
        & (result["fdr_q_value"] <= maximum_fdr_q),
        "VALIDATION_2025_PASS_PENDING_FINAL_SIMILARITY",
        "VALIDATION_2025_FAIL",
    )
    result["reason_codes"] = result.apply(_reason_codes, axis=1)
    return result.sort_values("factor_id", kind="mergesort").reset_index(drop=True)


def _reason_codes(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    if int(row["valid_day_count"]) < 60:
        reasons.append("LOW_VALIDATION_DAY_COUNT")
    if abs(float(row["mean_rank_ic"])) < 0.02:
        reasons.append("LOW_VALIDATION_MEAN_RANK_IC")
    if not np.isfinite(float(row["icir"])) or abs(float(row["icir"])) < 0.50:
        reasons.append("LOW_VALIDATION_ICIR")
    if float(row["directional_day_share"]) < 0.55:
        reasons.append("LOW_VALIDATION_DIRECTIONAL_SHARE")
    if float(row["signed_mean_rank_ic"]) <= 0:
        reasons.append("VALIDATION_DIRECTION_REVERSED_OR_NONPOSITIVE")
    if not np.isfinite(float(row["fdr_q_value"])) or float(row["fdr_q_value"]) > 0.10:
        reasons.append("VALIDATION_FDR_FAIL")
    if not reasons:
        reasons.append("VALIDATION_DOUBLE_GATE_PASS")
    return reasons


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
    if isinstance(value, (date, pd.Timestamp)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=_DEFAULT_ARTIFACT)
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
    similarity = target.query(
        "select factor_id,metrics_json from "
        "loadTable('dfs://QuantResearchV2Test','factor_quality_snapshot') "
        f"where scope_id='{SIMILARITY_SCOPE_ID}'"
    )
    train = target.query(
        "select factor_id,metrics_json from "
        "loadTable('dfs://QuantResearchV2Test','factor_quality_snapshot') "
        f"where scope_id='{TRAIN_SCOPE_ID}'"
    )
    if not all(isinstance(frame, pd.DataFrame) for frame in (metadata, similarity, train)):
        raise RuntimeError("validation quality inputs did not return DataFrames")
    candidate_ids = tuple(
        sorted(
            str(row.factor_id)
            for row in similarity.itertuples(index=False)
            if json.loads(str(row.metrics_json)).get("decision_status")
            == "SIMILARITY_SAMPLE_UNIQUE_PENDING_FULL"
        )
    )
    train_directions: dict[str, int] = {}
    for row in train.itertuples(index=False):
        factor_id = str(row.factor_id)
        if factor_id not in candidate_ids:
            continue
        body = json.loads(str(row.metrics_json))
        direction = body.get("predictive", {}).get("direction")
        if direction is not None:
            train_directions[factor_id] = int(direction)
    daily = pd.read_parquet(args.artifacts / "daily_rank_ic.parquet")
    if "factor_id" not in daily.columns and "factor_name" in daily.columns:
        daily = daily.rename(columns={"factor_name": "factor_id"})
    metrics = compute_validation_metrics(daily, train_directions=train_directions)
    args.artifacts.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.artifacts / "quality_summary.csv", index=False)
    metadata_by_id = metadata.assign(
        factor_id=metadata["factor_id"].map(lambda value: f"yeslab_{int(value)}")
    ).set_index("factor_id")
    watermark = hashlib.sha256(
        json.dumps(
            {"candidate_ids": candidate_ids, "validation_rows": len(daily), "scope": SCOPE_ID},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    data_version = hashlib.sha256((watermark + str(args.as_of)).encode()).hexdigest()
    records: list[dict[str, object]] = []
    metric_by_id = metrics.set_index("factor_id")
    for factor_id in candidate_ids:
        if factor_id not in metric_by_id.index:
            raise RuntimeError(f"validation artifact missing candidate: {factor_id}")
        metric = metric_by_id.loc[factor_id]
        meta = metadata_by_id.loc[factor_id]
        metric_dict = {
            key: _json_value(value)
            for key, value in metric.to_dict().items()
            if key not in {"factor_id", "decision_status", "reason_codes"}
        }
        body = {
            "screen_version": SCOPE_ID,
            "factor_name": str(meta["factor_name"]),
            "source_table": str(meta["store_factor"]),
            "category": str(meta["category"]),
            "evidence_scope": {"start": "2025-01-01", "end": "2025-12-31"},
            "validation": metric_dict,
            "decision_status": str(metric["decision_status"]),
            "reason_codes": metric["reason_codes"],
            "locked_year_read": False,
        }
        record = {
            "as_of_date": args.as_of,
            "factor_id": factor_id,
            "factor_version": 1,
            "scope_id": SCOPE_ID,
            "eligibility_status": (
                "PASS"
                if str(metric["decision_status"]).startswith("VALIDATION_2025_PASS")
                else "ELIGIBILITY_FAIL"
            ),
            "metrics_json": json.dumps(body, ensure_ascii=False, sort_keys=True),
            "rules_version": 1,
            "run_id": str(uuid.uuid4()),
            "data_version": data_version,
            "created_at": pd.Timestamp.utcnow().tz_localize(None),
            "source_watermark_id": watermark,
        }
        record["record_hash"] = _record_hash(record)
        records.append(record)
    frame = pd.DataFrame(records)
    print(metrics["decision_status"].value_counts().sort_index().to_string())
    if args.publish:
        print(publish_snapshot(target, frame, scope_id=SCOPE_ID))
    else:
        print("dry-run: no database write")


if __name__ == "__main__":
    main()
