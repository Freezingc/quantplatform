"""Scope-bounded TRAIN-only similarity diagnostic for source-qualified factors.

The sample dates are selected only from 2020--2024 TRAIN.  Results are
explicitly diagnostic: a high pairwise similarity is recorded as redundant
pending a full-range similarity pass, never as a final factor-set decision.
"""

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

from research_v2.db.session import V2WriteSession, YesLabReadOnlySession
from research_v2.factors.materializer import build_technical_factor_batches
from research_v2.factors.source_backfill import (
    FactorMonth,
    build_factor_definitions,
    build_factor_month_context,
)

SCOPE_ID = "SCREEN_TRAIN2020_2024_SIMILARITY_SAMPLE_V2"
SOURCE_SCOPE_ID = "SCREEN_TRAIN2020_2024_SOURCE_DQ_COMBINED_V1"
REFERENCE_FACTOR_ID = "yeslab_0"
_SAMPLE_DATES_PER_YEAR = 4
_FACTOR_GROUP_SIZE = 3
_SIMILARITY_THRESHOLD = 0.90
_MIN_COMMON_MEMBERS = 30


def classify_similarity_pairs(
    pairs: pd.DataFrame,
    *,
    factor_ids: tuple[str, ...],
    threshold: float = _SIMILARITY_THRESHOLD,
    minimum_valid_days: int = 1,
) -> pd.DataFrame:
    """Classify each factor by its highest sampled pairwise similarity."""
    required = {
        "factor_id_left",
        "factor_id_right",
        "median_daily_abs_spearman",
        "valid_days",
    }
    if missing := sorted(required.difference(pairs.columns)):
        raise ValueError(f"similarity pairs are missing columns: {missing}")
    rows: list[dict[str, object]] = []
    for factor_id in factor_ids:
        left = pairs.loc[pairs["factor_id_left"].astype(str) == factor_id]
        right = pairs.loc[pairs["factor_id_right"].astype(str) == factor_id]
        candidates = pd.concat([left, right], ignore_index=True)
        candidates = candidates.loc[
            pd.to_numeric(candidates["valid_days"], errors="coerce") >= minimum_valid_days
        ].copy()
        candidates["median_daily_abs_spearman"] = pd.to_numeric(
            candidates["median_daily_abs_spearman"], errors="coerce"
        )
        candidates = candidates.loc[np.isfinite(candidates["median_daily_abs_spearman"])]
        if candidates.empty:
            rows.append(
                {
                    "factor_id": factor_id,
                    "nearest_factor_id": None,
                    "nearest_similarity": np.nan,
                    "nearest_valid_days": 0,
                    "decision_status": "SIMILARITY_SAMPLE_INSUFFICIENT_PENDING_FULL",
                    "reason_codes": ["NO_VALID_SAMPLE_PAIR"],
                }
            )
            continue
        candidates = candidates.sort_values(
            ["median_daily_abs_spearman", "valid_days"],
            ascending=[False, False],
            kind="mergesort",
        )
        nearest = candidates.iloc[0]
        partner = (
            str(nearest["factor_id_right"])
            if str(nearest["factor_id_left"]) == factor_id
            else str(nearest["factor_id_left"])
        )
        similarity = float(nearest["median_daily_abs_spearman"])
        redundant = similarity >= threshold
        rows.append(
            {
                "factor_id": factor_id,
                "nearest_factor_id": partner,
                "nearest_similarity": similarity,
                "nearest_valid_days": int(nearest["valid_days"]),
                "decision_status": (
                    "SIMILARITY_SAMPLE_REDUNDANT_PENDING_FULL"
                    if redundant
                    else "SIMILARITY_SAMPLE_UNIQUE_PENDING_FULL"
                ),
                "reason_codes": [
                    "TRAIN_ONLY_SAMPLE_EVIDENCE",
                    "HIGH_SIMILARITY_PENDING_FULL_RANGE"
                    if redundant
                    else "BELOW_SIMILARITY_THRESHOLD",
                ],
            }
        )
    return pd.DataFrame(rows)


def _sample_dates(reference: pd.DataFrame) -> tuple[date, ...]:
    dates = pd.to_datetime(reference["signal_date"], errors="raise").dt.date
    selected: list[date] = []
    for _, group in pd.DataFrame({"signal_date": dates}).groupby(
        dates.map(lambda value: value.year), sort=True
    ):
        ordered = sorted(set(group["signal_date"]))
        indices = (0, len(ordered) // 3, (2 * len(ordered)) // 3, len(ordered) - 1)
        selected.extend(ordered[index] for index in indices)
    return tuple(sorted(set(selected)))


def _query_sample(
    source: YesLabReadOnlySession,
    *,
    table: str,
    factor_names: tuple[str, ...],
    dates: tuple[date, ...],
) -> pd.DataFrame:
    names = ",".join(f"'{name.replace(chr(39), chr(39) + chr(39))}'" for name in factor_names)
    dates_literal = ",".join(f"date('{value.strftime('%Y.%m.%d')}')" for value in dates)
    script = (
        "select ts_code,trade_date,factor_name,value from "
        f"loadTable('dfs://YesLab','{table}') where factor_name in [{names}] "
        f"and trade_date in [{dates_literal}]"
    )
    result = source.query(script)
    if not isinstance(result, pd.DataFrame):
        raise RuntimeError(f"similarity sample query did not return a DataFrame: {table}")
    return result


def _query_resilient(
    source: YesLabReadOnlySession,
    *,
    table: str,
    factor_names: tuple[str, ...],
    dates: tuple[date, ...],
) -> pd.DataFrame:
    try:
        return _query_sample(source, table=table, factor_names=factor_names, dates=dates)
    except RuntimeError:
        if len(factor_names) == 1:
            raise
        midpoint = len(factor_names) // 2
        return pd.concat(
            [
                _query_resilient(
                    source, table=table, factor_names=factor_names[:midpoint], dates=dates
                ),
                _query_resilient(
                    source, table=table, factor_names=factor_names[midpoint:], dates=dates
                ),
            ],
            ignore_index=True,
        )


def _pairwise_sample_similarity(
    observations: pd.DataFrame, factor_ids: tuple[str, ...]
) -> pd.DataFrame:
    pivot = observations.pivot_table(
        index=["signal_date", "ts_code"],
        columns="factor_id",
        values="value",
        aggfunc="first",
    ).reindex(columns=list(factor_ids))
    pair_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    pair_days: dict[tuple[str, str], int] = defaultdict(int)
    for _, day in pivot.groupby(level="signal_date", sort=True):
        values = day.droplevel("signal_date")
        finite = values.notna().to_numpy(dtype=np.int64)
        common = finite.T @ finite
        ranks = values.rank(method="average", axis=0)
        correlations = ranks.corr(min_periods=_MIN_COMMON_MEMBERS)
        for left_index, left_id in enumerate(factor_ids):
            for right_index in range(left_index + 1, len(factor_ids)):
                right_id = factor_ids[right_index]
                if common[left_index, right_index] < _MIN_COMMON_MEMBERS:
                    continue
                correlation = correlations.iloc[left_index, right_index]
                if pd.notna(correlation) and np.isfinite(float(correlation)):
                    pair = (left_id, right_id)
                    pair_values[pair].append(abs(float(correlation)))
                    pair_days[pair] += 1
    rows: list[dict[str, object]] = []
    for left_id, right_id in (
        (left_id, right_id)
        for index, left_id in enumerate(factor_ids)
        for right_id in factor_ids[index + 1 :]
    ):
        values = pair_values[(left_id, right_id)]
        rows.append(
            {
                "factor_id_left": left_id,
                "factor_id_right": right_id,
                "median_daily_abs_spearman": float(np.median(values))
                if values
                else np.nan,
                "valid_days": pair_days[(left_id, right_id)],
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research_v2/artifacts/diagnostics/train_similarity_sample_2020_2024_v2.csv"),
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
        raise RuntimeError("similarity inputs did not return DataFrames")
    factor_ids = tuple(
        sorted(
            str(row.factor_id)
            for row in source_scope.itertuples(index=False)
            if json.loads(str(row.metrics_json)).get("decision_status")
            == "SOURCE_DQ_PASS_PENDING_SIMILARITY"
        )
    )
    definitions = build_factor_definitions(metadata, factor_ids)
    sample_dates = _sample_dates(reference)
    context = build_factor_month_context(
        FactorMonth.from_partition(sample_dates[0].strftime("%Y-%m")),
        definitions,
        attempt_nonce="train-similarity-sample-v1",
    )
    grouped: dict[str, list] = defaultdict(list)
    for definition in definitions:
        grouped[definition.source_table].append(definition)
    collected: list[pd.DataFrame] = []
    for table in sorted(grouped):
        entries = sorted(grouped[table], key=lambda definition: definition.factor_id)
        for offset in range(0, len(entries), _FACTOR_GROUP_SIZE):
            group = tuple(entries[offset : offset + _FACTOR_GROUP_SIZE])
            names = tuple(definition.display_name for definition in group)
            print(f"query table={table} factors={len(group)} dates={len(sample_dates)}", flush=True)
            raw = _query_resilient(
                source, table=table, factor_names=names, dates=sample_dates
            )
            for definition in group:
                materialized = build_technical_factor_batches(
                    raw,
                    definition,
                    source_factor_name=definition.display_name,
                    source_watermark_id=f"train-similarity|{table}",
                    context=context,
                    record_hash_strategy="fast",
                    include_lineage=False,
                )
                clean = materialized.clean_batch.frame.loc[
                    :, ["signal_date", "ts_code", "factor_id", "factor_version", "value"]
                ].copy()
                collected.append(clean)
    observations = pd.concat(collected, ignore_index=True)
    observations["signal_date"] = pd.to_datetime(
        observations["signal_date"], errors="raise"
    ).dt.date
    if observations.duplicated(["signal_date", "ts_code", "factor_id", "factor_version"]).any():
        raise RuntimeError("similarity sample contains duplicate business keys")
    pairs = _pairwise_sample_similarity(observations, factor_ids)
    classified = classify_similarity_pairs(pairs, factor_ids=factor_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    classified.to_csv(args.output, index=False)
    print(
        f"similarity sample complete: factors={len(factor_ids)} observations={len(observations)} "
        f"dates={len(sample_dates)} output={args.output}",
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
        factor_id = str(metric["factor_id"])
        meta = metadata_by_id.loc[factor_id]
        body = {
            "screen_version": SCOPE_ID,
            "factor_name": str(meta["factor_name"]),
            "source_table": str(meta["store_factor"]),
            "category": str(meta["category"]),
            "evidence_scope": {
                "start": "2020-01-01",
                "end": "2024-12-31",
                "sample_dates": [d.isoformat() for d in sample_dates],
                "sample_dates_per_year": _SAMPLE_DATES_PER_YEAR,
            },
            "similarity": {
                key: (value if not isinstance(value, float) or np.isfinite(value) else None)
                for key, value in metric.items()
                if key not in {"factor_id", "decision_status", "reason_codes"}
            },
            "decision_status": metric["decision_status"],
            "reason_codes": metric["reason_codes"],
            "full_train_similarity_checked": False,
        }
        record = {
            "as_of_date": args.as_of,
            "factor_id": factor_id,
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
