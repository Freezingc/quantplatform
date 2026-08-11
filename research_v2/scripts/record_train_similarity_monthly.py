"""Stream a monthly TRAIN-only cross-sectional similarity diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from collections import defaultdict
from datetime import date
from pathlib import Path

import dolphindb as ddb
import numpy as np
import pandas as pd
from record_factor_quality_snapshot import publish_snapshot
from record_train_similarity_sample import (
    _pairwise_sample_similarity,
    _query_resilient,
    classify_similarity_pairs,
)

from research_v2.db.session import V2WriteSession, YesLabReadOnlySession
from research_v2.factors.materializer import build_technical_factor_batches
from research_v2.factors.source_backfill import (
    FactorMonth,
    build_factor_definitions,
    build_factor_month_context,
)

SCOPE_ID = "SCREEN_TRAIN2020_2024_SIMILARITY_MONTHLY_V1"
SOURCE_SCOPE_ID = "SCREEN_TRAIN2020_2024_SOURCE_DQ_COMBINED_V1"
_FACTOR_GROUP_SIZE = 3
_MIN_COMMON_MEMBERS = 30
_THRESHOLD = 0.90


def _monthly_sample_dates(reference: pd.DataFrame) -> tuple[date, ...]:
    dates = pd.to_datetime(reference["signal_date"], errors="raise").dt.date
    frame = pd.DataFrame({"signal_date": dates})
    months = frame["signal_date"].map(lambda value: value.strftime("%Y-%m"))
    return tuple(
        sorted(
            group["signal_date"].min()
            for _, group in frame.groupby(months, sort=True)
        )
    )


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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research_v2/artifacts/diagnostics/train_similarity_monthly_2020_2024.csv"),
    )
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
        "select factor_id,factor_name,creator,update_frequency,store_factor,category "
        "from loadTable('dfs://YesLab','factor_metadata')"
    )
    source_scope = target.query(
        "select factor_id,metrics_json from "
        "loadTable('dfs://QuantResearchV2Test','factor_quality_snapshot') "
        f"where scope_id='{SOURCE_SCOPE_ID}'"
    )
    reference = target.query(
        "select signal_date from loadTable('dfs://QuantResearchV2Test','factor_rank_ic_daily') "
        "where factor_id='yeslab_0' and signal_date between date('2020.01.01') "
        "and date('2024.12.31')"
    )
    if not all(isinstance(frame, pd.DataFrame) for frame in (metadata, source_scope, reference)):
        raise RuntimeError("monthly similarity inputs did not return DataFrames")
    factor_ids = tuple(
        sorted(
            str(row.factor_id)
            for row in source_scope.itertuples(index=False)
            if json.loads(str(row.metrics_json)).get("decision_status")
            == "SOURCE_DQ_PASS_PENDING_SIMILARITY"
        )
    )
    sample_dates = _monthly_sample_dates(reference)
    definitions = build_factor_definitions(metadata, factor_ids)
    context = build_factor_month_context(
        FactorMonth.from_partition(sample_dates[0].strftime("%Y-%m")),
        definitions,
        attempt_nonce="train-similarity-monthly-v1",
    )
    grouped: dict[str, list] = defaultdict(list)
    for definition in definitions:
        grouped[definition.source_table].append(definition)
    pair_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for date_index, sample_date in enumerate(sample_dates, start=1):
        daily_parts: list[pd.DataFrame] = []
        for table in sorted(grouped):
            entries = sorted(grouped[table], key=lambda definition: definition.factor_id)
            for offset in range(0, len(entries), _FACTOR_GROUP_SIZE):
                group = tuple(entries[offset : offset + _FACTOR_GROUP_SIZE])
                raw = _query_resilient(
                    source,
                    table=table,
                    factor_names=tuple(definition.display_name for definition in group),
                    dates=(sample_date,),
                )
                for definition in group:
                    materialized = build_technical_factor_batches(
                        raw,
                        definition,
                        source_factor_name=definition.display_name,
                        source_watermark_id=f"train-similarity-monthly|{table}",
                        context=context,
                        record_hash_strategy="fast",
                        include_lineage=False,
                    )
                    daily_parts.append(
                        materialized.clean_batch.frame.loc[
                            :, ["signal_date", "ts_code", "factor_id", "factor_version", "value"]
                        ].copy()
                    )
        daily = pd.concat(daily_parts, ignore_index=True)
        pairs = _pairwise_sample_similarity(daily, factor_ids)
        for pair in pairs.itertuples(index=False):
            if int(pair.valid_days) and pd.notna(pair.median_daily_abs_spearman):
                pair_values[(str(pair.factor_id_left), str(pair.factor_id_right))].append(
                    float(pair.median_daily_abs_spearman)
                )
        print(
            f"monthly similarity date={sample_date.isoformat()} "
            f"progress={date_index}/{len(sample_dates)} rows={len(daily)}",
            flush=True,
        )
    pair_rows = [
        {
            "factor_id_left": left,
            "factor_id_right": right,
            "median_daily_abs_spearman": float(np.median(values))
            if values
            else np.nan,
            "valid_days": len(values),
        }
        for (left, right), values in pair_values.items()
    ]
    pairs = pd.DataFrame(pair_rows)
    classified = classify_similarity_pairs(
        pairs,
        factor_ids=factor_ids,
        threshold=_THRESHOLD,
        minimum_valid_days=20,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    classified.to_csv(args.output, index=False)
    print(
        f"monthly similarity complete: factors={len(factor_ids)} dates={len(sample_dates)} "
        f"output={args.output}",
        flush=True,
    )
    watermark = hashlib.sha256(
        json.dumps(
            {"factor_ids": factor_ids, "sample_dates": [d.isoformat() for d in sample_dates]},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    data_version = hashlib.sha256((watermark + str(args.as_of)).encode()).hexdigest()
    metadata_by_id = metadata.assign(
        factor_id=metadata["factor_id"].map(lambda value: f"yeslab_{int(value)}")
    ).set_index("factor_id")
    records: list[dict[str, object]] = []
    for metric in classified.to_dict("records"):
        meta = metadata_by_id.loc[str(metric["factor_id"])]
        body = {
            "screen_version": SCOPE_ID,
            "factor_name": str(meta["factor_name"]),
            "source_table": str(meta["store_factor"]),
            "category": str(meta["category"]),
            "evidence_scope": {
                "start": "2020-01-01",
                "end": "2024-12-31",
                "sample_dates": [d.isoformat() for d in sample_dates],
                "sample_date_count": len(sample_dates),
            },
            "similarity": metric,
            "decision_status": metric["decision_status"],
            "reason_codes": metric["reason_codes"],
            "full_train_cross_section_checked": False,
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
    if args.publish:
        print(publish_snapshot(target, pd.DataFrame(records), scope_id=SCOPE_ID))
    else:
        print("dry-run: no database write")


if __name__ == "__main__":
    main()
