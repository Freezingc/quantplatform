"""Full TRAIN-window RankIC-series dependence diagnostic."""

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

from research_v2.db.session import V2WriteSession

SCOPE_ID = "SCREEN_TRAIN2020_2024_RANKIC_SERIES_SIM_V1"
SOURCE_SCOPE_ID = "SCREEN_TRAIN2020_2024_SOURCE_DQ_COMBINED_V1"
_MIN_VALID_DAYS = 60
_THRESHOLD = 0.90


def rankic_similarity_pairs(
    daily: pd.DataFrame, *, factor_ids: tuple[str, ...], minimum_valid_days: int = _MIN_VALID_DAYS
) -> pd.DataFrame:
    """Return absolute Pearson dependence of daily RankIC series in TRAIN only."""
    required = {"signal_date", "factor_id", "rank_ic"}
    if missing := sorted(required.difference(daily.columns)):
        raise ValueError(f"daily RankIC is missing columns: {missing}")
    frame = daily.loc[:, ["signal_date", "factor_id", "rank_ic"]].copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="raise").dt.date
    frame["factor_id"] = frame["factor_id"].astype(str)
    frame["rank_ic"] = pd.to_numeric(frame["rank_ic"], errors="coerce")
    pivot = frame.pivot_table(
        index="signal_date", columns="factor_id", values="rank_ic", aggfunc="first"
    ).reindex(columns=list(factor_ids))
    correlation = pivot.corr(min_periods=minimum_valid_days)
    rows: list[dict[str, object]] = []
    for index, left_id in enumerate(factor_ids):
        for right_id in factor_ids[index + 1 :]:
            valid_days = int(pivot[[left_id, right_id]].dropna().shape[0])
            value = correlation.loc[left_id, right_id]
            rows.append(
                {
                    "factor_id_left": left_id,
                    "factor_id_right": right_id,
                    "abs_rankic_correlation": abs(float(value))
                    if pd.notna(value) and np.isfinite(float(value))
                    else np.nan,
                    "valid_days": valid_days,
                }
            )
    return pd.DataFrame(rows)


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
    target = V2WriteSession(client, target_database="dfs://QuantResearchV2Test")
    source_scope = target.query(
        "select factor_id,metrics_json from "
        "loadTable('dfs://QuantResearchV2Test','factor_quality_snapshot') "
        f"where scope_id='{SOURCE_SCOPE_ID}'"
    )
    if not isinstance(source_scope, pd.DataFrame):
        raise RuntimeError("RankIC similarity source scope did not return a DataFrame")
    factor_ids = tuple(
        sorted(
            str(row.factor_id)
            for row in source_scope.itertuples(index=False)
            if json.loads(str(row.metrics_json)).get("decision_status")
            == "SOURCE_DQ_PASS_PENDING_SIMILARITY"
        )
    )
    quoted = ",".join(f"'{factor_id}'" for factor_id in factor_ids)
    daily = target.query(
        "select signal_date,factor_id,rank_ic from "
        "loadTable('dfs://QuantResearchV2Test','factor_rank_ic_daily') "
        "where signal_date between date('2020.01.01') and date('2024.12.31') "
        f"and factor_id in [{quoted}] and factor_version=1"
    )
    if not isinstance(daily, pd.DataFrame):
        raise RuntimeError("RankIC daily query did not return a DataFrame")
    pairs = rankic_similarity_pairs(daily, factor_ids=factor_ids)
    rows: list[dict[str, object]] = []
    for factor_id in factor_ids:
        left = pairs.loc[pairs["factor_id_left"] == factor_id]
        right = pairs.loc[pairs["factor_id_right"] == factor_id]
        candidates = pd.concat([left, right], ignore_index=True)
        candidates = candidates.loc[
            (candidates["valid_days"] >= _MIN_VALID_DAYS)
            & np.isfinite(candidates["abs_rankic_correlation"].to_numpy(dtype=float))
        ].sort_values("abs_rankic_correlation", ascending=False, kind="mergesort")
        if candidates.empty:
            status = "RANKIC_SERIES_SIM_INSUFFICIENT"
            partner = None
            similarity = None
            valid_days = 0
            reasons = ["NO_VALID_RANKIC_PAIR"]
        else:
            nearest = candidates.iloc[0]
            partner = (
                str(nearest["factor_id_right"])
                if str(nearest["factor_id_left"]) == factor_id
                else str(nearest["factor_id_left"])
            )
            similarity = float(nearest["abs_rankic_correlation"])
            valid_days = int(nearest["valid_days"])
            status = (
                "RANKIC_SERIES_SIM_HIGH_PENDING_CROSS_SECTION"
                if similarity >= _THRESHOLD
                else "RANKIC_SERIES_SIM_UNIQUE_PENDING_CROSS_SECTION"
            )
            reasons = ["FULL_TRAIN_RANKIC_SERIES", "CROSS_SECTION_SIMILARITY_PENDING"]
        rows.append(
            {
                "factor_id": factor_id,
                "nearest_factor_id": partner,
                "abs_rankic_correlation": similarity,
                "nearest_valid_days": valid_days,
                "decision_status": status,
                "reason_codes": reasons,
            }
        )
    metrics = pd.DataFrame(rows)
    watermark = hashlib.sha256(
        json.dumps({"factor_ids": factor_ids, "scope": SCOPE_ID}, sort_keys=True).encode()
    ).hexdigest()
    data_version = hashlib.sha256((watermark + str(args.as_of)).encode()).hexdigest()
    records: list[dict[str, object]] = []
    for metric in metrics.to_dict("records"):
        body = {
            "screen_version": SCOPE_ID,
            "evidence_scope": {"start": "2020-01-01", "end": "2024-12-31"},
            "rankic_series_similarity": metric,
            "decision_status": metric["decision_status"],
            "reason_codes": metric["reason_codes"],
            "cross_section_similarity_checked": False,
        }
        record = {
            "as_of_date": args.as_of,
            "factor_id": metric["factor_id"],
            "factor_version": 1,
            "scope_id": SCOPE_ID,
            "eligibility_status": "PASS",
            "metrics_json": json.dumps(body, ensure_ascii=False, sort_keys=True),
            "rules_version": 1,
            "run_id": str(uuid.uuid4()),
            "data_version": data_version,
            "created_at": pd.Timestamp.utcnow().tz_localize(None),
            "source_watermark_id": watermark,
        }
        record["record_hash"] = _record_hash(record)
        records.append(record)
    print(metrics["decision_status"].value_counts().sort_index().to_string())
    if args.publish:
        print(publish_snapshot(target, pd.DataFrame(records), scope_id=SCOPE_ID))
    else:
        print("dry-run: no database write")


if __name__ == "__main__":
    main()
