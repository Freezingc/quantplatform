"""Read-only full-day similarity for the 2025 validation representatives."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import date
from pathlib import Path

import dolphindb as ddb
import numpy as np
import pandas as pd
from evaluate_gp_candidates_strict import _metadata
from record_train_similarity_full_day import (
    _FACTOR_GROUP_SIZE,
    _MIN_VALID_DAYS,
    _THRESHOLD,
    _query_resilient,
    daily_similarity_values,
    query_clean_factor_month,
    vectorized_clean_technical_sample,
    write_pair_artifact,
)
from record_train_similarity_sample import classify_similarity_pairs

from research_v2.db.session import V2WriteSession, YesLabReadOnlySession
from research_v2.factors.source_backfill import build_factor_definitions

DEFAULT_FACTOR_IDS: tuple[str, ...] = (
    "yeslab_115",
    "yeslab_118",
    "yeslab_163",
    "yeslab_357",
    "yeslab_363",
    "yeslab_365",
    "yeslab_376",
    "yeslab_380",
    "yeslab_383",
    "yeslab_385",
    "yeslab_396",
    "yeslab_95",
    "yeslab_96",
    "yeslab_98",
)
DEFAULT_DAILY_INPUT = Path(
    "research_v2/artifacts/diagnostics/train26_representatives_validation_2025_v1/daily_rank_ic.parquet"
)
DEFAULT_OUTPUT = Path(
    "research_v2/artifacts/diagnostics/validation_similarity_full_day_2025_representatives_v1.csv"
)
DEFAULT_PAIRS_OUTPUT = Path(
    "research_v2/artifacts/diagnostics/validation_similarity_pairs_2025_representatives_v1.csv"
)


def build_month_dates(
    daily: pd.DataFrame,
    *,
    start: date,
    end: date,
) -> list[tuple[str, tuple[date, ...]]]:
    """Build sorted logical-month date groups and enforce the date boundary."""

    if start > end:
        raise ValueError("validation start must not be after end")
    if "signal_date" not in daily.columns:
        raise ValueError("daily validation input is missing signal_date")
    dates = pd.to_datetime(daily["signal_date"], errors="raise").dt.date
    outside = (dates < start) | (dates > end)
    if outside.any():
        values = sorted(set(dates.loc[outside]))
        raise ValueError(f"validation dates outside validation window: {values[:3]}")
    frame = pd.DataFrame({"signal_date": sorted(set(dates))})
    month_keys = frame["signal_date"].map(lambda value: value.strftime("%Y-%m"))
    return [
        (str(month), tuple(sorted(set(group["signal_date"]))))
        for month, group in frame.groupby(month_keys, sort=True)
    ]


def _parse_factor_ids(value: str) -> tuple[str, ...]:
    factor_ids = tuple(item.strip() for item in value.split(",") if item.strip())
    if not factor_ids or len(set(factor_ids)) != len(factor_ids):
        raise ValueError("factor-ids must be non-empty and unique")
    return factor_ids


def _connect() -> tuple[object, object]:
    host = os.getenv("QRV2_DDB_HOST", "10.176.55.211")
    port = int(os.getenv("QRV2_DDB_PORT", "8848"))
    username = os.getenv("QRV2_DDB_USERNAME", "xiahongjie")
    password = os.getenv("QRV2_DDB_PASSWORD", "123456")
    source_client = ddb.session()
    source_client.connect(host, port, username, password)
    v2_client = ddb.session()
    v2_client.connect(host, port, username, password)
    return source_client, v2_client


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--factor-ids", default=",".join(DEFAULT_FACTOR_IDS))
    parser.add_argument("--daily-input", type=Path, default=DEFAULT_DAILY_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pairs-output", type=Path, default=DEFAULT_PAIRS_OUTPUT)
    args = parser.parse_args()

    start_date = pd.Timestamp(args.start).date()
    end_date = pd.Timestamp(args.end).date()
    if end_date > date(2025, 12, 31):
        raise ValueError("validation similarity cannot read dates after 2025-12-31")
    factor_ids = _parse_factor_ids(args.factor_ids)
    daily_input = pd.read_parquet(args.daily_input)
    month_dates = build_month_dates(daily_input, start=start_date, end=end_date)
    source_client, v2_client = _connect()
    source = YesLabReadOnlySession(source_client)
    target = V2WriteSession(v2_client, target_database="dfs://QuantResearchV2Test")
    metadata = _metadata(source, factor_ids)
    definitions = build_factor_definitions(metadata, factor_ids)
    metadata_ids = {f"yeslab_{int(value)}" for value in metadata["factor_id"]}
    missing_metadata = sorted(set(factor_ids).difference(metadata_ids))
    if missing_metadata:
        raise ValueError(f"factor metadata is missing requested IDs: {missing_metadata}")
    grouped: dict[str, list] = defaultdict(list)
    for definition in definitions:
        grouped[definition.source_table].append(definition)
    pair_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for month_index, (logical_month, signal_dates) in enumerate(month_dates, start=1):
        observations = query_clean_factor_month(
            target,
            factor_ids=factor_ids,
            start=signal_dates[0],
            end=signal_dates[-1],
        )
        present = set(observations["factor_id"].astype(str))
        missing_factor_ids = tuple(
            factor_id for factor_id in factor_ids if factor_id not in present
        )
        if missing_factor_ids:
            fallback_parts: list[pd.DataFrame] = []
            for table in sorted(grouped):
                entries = [
                    definition
                    for definition in grouped[table]
                    if definition.factor_id in missing_factor_ids
                ]
                for offset in range(0, len(entries), _FACTOR_GROUP_SIZE):
                    group = tuple(entries[offset : offset + _FACTOR_GROUP_SIZE])
                    raw = _query_resilient(
                        source,
                        table=table,
                        factor_names=tuple(definition.display_name for definition in group),
                        dates=signal_dates,
                    )
                    fallback_parts.append(vectorized_clean_technical_sample(raw, group))
            if fallback_parts:
                observations = pd.concat([observations, *fallback_parts], ignore_index=True)
        for pair, values in daily_similarity_values(observations, factor_ids).items():
            pair_values[pair].extend(values)
        print(
            f"validation similarity month={logical_month} "
            f"progress={month_index}/{len(month_dates)} rows={len(observations)}",
            flush=True,
        )
    pair_rows = []
    for left_index, left_id in enumerate(factor_ids):
        for right_id in factor_ids[left_index + 1 :]:
            values = pair_values[(left_id, right_id)]
            pair_rows.append(
                {
                    "factor_id_left": left_id,
                    "factor_id_right": right_id,
                    "median_daily_abs_spearman": float(np.median(values))
                    if values
                    else np.nan,
                    "valid_days": len(values),
                }
            )
    pairs = pd.DataFrame(pair_rows)
    write_pair_artifact(pairs, args.pairs_output)
    classified = classify_similarity_pairs(
        pairs,
        factor_ids=factor_ids,
        threshold=_THRESHOLD,
        minimum_valid_days=_MIN_VALID_DAYS,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    classified.to_csv(args.output, index=False)
    metadata_payload = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "factor_ids": list(factor_ids),
        "daily_input": str(args.daily_input),
        "valid_date_count": sum(len(dates) for _, dates in month_dates),
        "target_database": "dfs://QuantResearchV2Test",
        "factor_value_source": "V2_clean_pointer_plus_YesLab_fallback",
        "threshold": _THRESHOLD,
        "minimum_valid_days": _MIN_VALID_DAYS,
        "locked_test_read": False,
        "database_writes": False,
        "pair_count": len(pairs),
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"validation similarity complete: factors={len(factor_ids)} "
        f"dates={metadata_payload['valid_date_count']} output={args.output}",
        flush=True,
    )
    print(classified["decision_status"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
