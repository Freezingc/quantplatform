"""Strict, write-free evaluation of a training-window GP factor shortlist.

The script deliberately separates the two data boundaries used by the research
workflow: raw technical-factor observations are read through the YesLab
read-only session and the universe/forward-return inputs are read through the
V2 latest-manifest reader.  It never reads the locked 2026 label partition and
never writes either database.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path

import dolphindb as ddb
import pandas as pd

from research_v2.compact.publisher import read_latest_partition
from research_v2.db.session import V2WriteSession, YesLabReadOnlySession
from research_v2.factors.source_backfill import build_factor_definitions
from research_v2.factors.source_evaluation import (
    evaluate_source_factor_range,
    summarize_factor_rank_ic,
)

DEFAULT_FACTOR_IDS: tuple[str, ...] = (
    "yeslab_376",
    "yeslab_464",
    "yeslab_393",
    "yeslab_405",
    "yeslab_381",
    "yeslab_404",
    "yeslab_398",
    "yeslab_380",
    "yeslab_442",
    "yeslab_432",
    "yeslab_383",
    "yeslab_396",
    "yeslab_443",
    "yeslab_387",
    "yeslab_385",
    "yeslab_391",
    "yeslab_388",
    "yeslab_425",
    "yeslab_431",
    "yeslab_458",
    "yeslab_418",
    "yeslab_395",
    "yeslab_400",
    "yeslab_444",
    "yeslab_377",
    "yeslab_433",
)


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


def _parse_date(value: str) -> date:
    return pd.Timestamp(value).date()


def _month_inputs(
    v2_session: V2WriteSession,
    *,
    start_date: date,
    end_date: date,
    purge_last_dates: int,
) -> tuple[dict[str, tuple[pd.DataFrame, pd.DataFrame]], tuple[date, ...]]:
    """Load manifest-selected universe and labels, then purge the segment tail."""
    periods = tuple(str(period) for period in pd.period_range(start_date, end_date, freq="M"))
    universe_parts: list[pd.DataFrame] = []
    label_parts: list[pd.DataFrame] = []
    for logical_partition in periods:
        universe = read_latest_partition(
            v2_session,
            object_name="universe_daily_latest",
            logical_partition=logical_partition,
            columns=("batch_id", "signal_date", "ts_code", "is_member"),
        )
        labels = read_latest_partition(
            v2_session,
            object_name="label_daily_latest",
            logical_partition=logical_partition,
            columns=(
                "batch_id",
                "signal_date",
                "ts_code",
                "raw_return",
                "label_valid",
                "exit_date",
            ),
        )
        if universe.empty or labels.empty:
            raise RuntimeError(f"compact inputs are missing for {logical_partition}")
        universe_parts.append(universe)
        label_parts.append(labels)

    universe_frame = pd.concat(universe_parts, ignore_index=True)
    label_frame = pd.concat(label_parts, ignore_index=True)
    universe_frame["signal_date"] = pd.to_datetime(
        universe_frame["signal_date"], errors="raise"
    ).dt.date
    label_frame["signal_date"] = pd.to_datetime(
        label_frame["signal_date"], errors="raise"
    ).dt.date
    all_dates = tuple(sorted(set(label_frame["signal_date"])))
    if purge_last_dates < 0:
        raise ValueError("purge_last_dates cannot be negative")
    purged_dates = all_dates[-purge_last_dates:] if purge_last_dates else ()
    if purged_dates:
        purge_set = set(purged_dates)
        universe_frame = universe_frame.loc[~universe_frame["signal_date"].isin(purge_set)].copy()
        label_frame = label_frame.loc[~label_frame["signal_date"].isin(purge_set)].copy()

    month_inputs: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for logical_partition in periods:
        period = pd.Period(logical_partition, freq="M")
        month_inputs[logical_partition] = (
            universe_frame.loc[
                universe_frame["signal_date"].between(
                    period.start_time.date(), period.end_time.date()
                )
            ].copy(),
            label_frame.loc[
                label_frame["signal_date"].between(period.start_time.date(), period.end_time.date())
            ].copy(),
        )
    return month_inputs, purged_dates


def _metadata(source_session: YesLabReadOnlySession, factor_ids: tuple[str, ...]) -> pd.DataFrame:
    numeric_ids = ", ".join(str(int(factor_id.removeprefix("yeslab_"))) for factor_id in factor_ids)
    result = source_session.query(
        "select factor_id,factor_name,creator,update_frequency,store_factor "
        "from loadTable('dfs://YesLab','factor_metadata') "
        f"where factor_id in [{numeric_ids}]"
    )
    if not isinstance(result, pd.DataFrame):
        raise RuntimeError("factor metadata query did not return a DataFrame")
    return result.copy(deep=True)


def _coverage(evaluations: tuple[object, ...]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for evaluation in evaluations:
        coverage = evaluation.coverage.copy(deep=True)
        coverage["logical_partition"] = evaluation.logical_partition
        rows.append(coverage)
    if not rows:
        raise RuntimeError("strict evaluation returned no monthly coverage")
    frame = pd.concat(rows, ignore_index=True)
    numeric = frame.assign(
        finite_count=pd.to_numeric(frame["finite_count"], errors="raise"),
        mature_row_count=pd.to_numeric(frame["mature_row_count"], errors="raise"),
    )
    grouped = numeric.groupby("factor_id", sort=True, as_index=False).agg(
        finite_count=("finite_count", "sum"),
        mature_row_count=("mature_row_count", "sum"),
        minimum_monthly_finite_ratio=("finite_ratio", "min"),
        mean_monthly_finite_ratio=("finite_ratio", "mean"),
    )
    grouped["finite_ratio"] = grouped["finite_count"] / grouped["mature_row_count"]
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--purge-last-dates", type=int, default=6)
    parser.add_argument("--chunk-months", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factor-ids", default=",".join(DEFAULT_FACTOR_IDS))
    args = parser.parse_args()

    start_date = _parse_date(args.start)
    end_date = _parse_date(args.end)
    factor_ids = tuple(item.strip() for item in args.factor_ids.split(",") if item.strip())
    if not factor_ids or len(set(factor_ids)) != len(factor_ids):
        raise ValueError("factor-ids must be non-empty and unique")

    source_client, v2_client = _connect()
    source_session = YesLabReadOnlySession(source_client)
    v2_session = V2WriteSession(v2_client, target_database="dfs://QuantResearchV2Test")
    metadata = _metadata(source_session, factor_ids)
    definitions = build_factor_definitions(metadata, factor_ids)
    month_inputs, purged_dates = _month_inputs(
        v2_session,
        start_date=start_date,
        end_date=end_date,
        purge_last_dates=args.purge_last_dates,
    )
    evaluations = evaluate_source_factor_range(
        source_session,
        metadata,
        factor_ids=factor_ids,
        start_date=start_date,
        end_date=end_date,
        month_inputs=month_inputs,
        minimum_cross_section=30,
        chunk_months=args.chunk_months,
    )
    daily = pd.concat(
        [evaluation.daily_rank_ic for evaluation in evaluations], ignore_index=True
    )
    summary = summarize_factor_rank_ic(daily)
    coverage = _coverage(evaluations)
    id_to_name = {definition.factor_id: definition.display_name for definition in definitions}
    summary["factor_id"] = summary["factor_name"].astype(str)
    summary["source_factor_name"] = summary["factor_id"].map(id_to_name)
    summary = summary.merge(coverage, on="factor_id", how="left", validate="one_to_one")
    summary["direction"] = summary["mean_rank_ic"].map(lambda value: 1 if value > 0 else -1)
    summary["directional_day_share"] = summary.apply(
        lambda row: row["positive_day_share"]
        if row["direction"] > 0
        else 1.0 - row["positive_day_share"],
        axis=1,
    )
    summary = summary.sort_values("factor_id", kind="mergesort").reset_index(drop=True)

    args.output.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(args.output / "daily_rank_ic.parquet", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    coverage.to_csv(args.output / "coverage.csv", index=False)
    metadata_payload = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "purged_signal_dates": [value.isoformat() for value in purged_dates],
        "factor_ids": list(factor_ids),
        "factor_names": [definition.display_name for definition in definitions],
        "target_database": "dfs://QuantResearchV2Test",
        "locked_year_read": False,
        "chunk_months": args.chunk_months,
        "minimum_cross_section": 30,
        "source_materialization": "YesLab strict PIT normalization/deduplication",
        "output_rows_daily_rank_ic": len(daily),
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"strict evaluation complete: factors={len(factor_ids)} months={len(evaluations)} "
        f"daily_rows={len(daily)} output={args.output}"
    )


if __name__ == "__main__":
    main()
