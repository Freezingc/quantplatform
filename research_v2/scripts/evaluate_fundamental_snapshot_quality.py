"""Read-only 5-day RankIC evaluation for materialized PIT financial factors.

This command evaluates only the new ``fundamental_pit_snapshot`` table against
the manifest-selected SH/SZ universe and raw forward-open five-trading-day
labels.  It never mutates the source tables, snapshot table, formal factor
set, or model inputs, and rejects access to the locked 2026 test period.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import dolphindb as ddb
import pandas as pd

from research_v2.compact.publisher import read_latest_partition
from research_v2.db.session import V2WriteSession
from research_v2.evaluation.factor_rank_ic import (
    compute_factor_daily_rank_ic,
    summarize_factor_rank_ic,
)

TARGET_DATABASE = "dfs://QuantResearchV2Test"
SNAPSHOT_OBJECT = "fundamental_pit_snapshot"
LOCKED_TEST_START = date(2026, 1, 1)
DEFAULT_METRICS = ("roa", "roe", "cfoa")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2020, 1, 2))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2020, 12, 31))
    parser.add_argument("--metric", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.metric = tuple(dict.fromkeys(args.metric or DEFAULT_METRICS))
    if args.start > args.end or args.end >= LOCKED_TEST_START:
        parser.error("quality range must be before 2026 and start <= end")
    if not args.metric or any(not value.isidentifier() for value in args.metric):
        parser.error("metrics must be non-empty identifiers")
    return args


def summarize_snapshot_quality(
    snapshot: pd.DataFrame,
    universe: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    minimum_cross_section: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate coverage and daily/factor RankIC without pooling dates."""
    required_snapshot = {"signal_date", "ts_code", "metric_id", "value"}
    required_universe = {"signal_date", "ts_code", "is_member"}
    required_labels = {"signal_date", "ts_code", "raw_return", "label_valid"}
    for name, frame, required in (
        ("snapshot", snapshot, required_snapshot),
        ("universe", universe, required_universe),
        ("labels", labels, required_labels),
    ):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} is missing columns: {missing}")
    keys = ["signal_date", "ts_code"]
    snapshot_frame = snapshot.loc[:, keys + ["metric_id", "value"]].copy()
    universe_frame = universe.loc[:, keys + ["is_member"]].copy()
    label_frame = labels.loc[:, keys + ["raw_return", "label_valid"]].copy()
    for frame in (snapshot_frame, universe_frame, label_frame):
        frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="raise").dt.date
        frame["ts_code"] = frame["ts_code"].astype(str)
    if snapshot_frame.duplicated(keys + ["metric_id"], keep=False).any():
        raise ValueError("snapshot contains duplicate metric/date/security keys")
    if universe_frame.duplicated(keys, keep=False).any() or label_frame.duplicated(keys, keep=False).any():
        raise ValueError("universe or labels contain duplicate date/security keys")
    mature = universe_frame.loc[universe_frame["is_member"].fillna(False).astype(bool), keys].merge(
        label_frame.loc[label_frame["label_valid"].fillna(False).astype(bool), keys],
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    coverage_rows: list[dict[str, object]] = []
    for metric_id, group in snapshot_frame.groupby("metric_id", sort=True):
        merged = mature.merge(group, on=keys, how="left", validate="one_to_one")
        finite = pd.to_numeric(merged["value"], errors="coerce").notna().sum()
        coverage_rows.append(
            {
                "metric_id": str(metric_id),
                "finite_count": int(finite),
                "mature_row_count": len(merged),
                "finite_ratio": float(finite / len(merged)) if len(merged) else 0.0,
            }
        )
    daily = compute_factor_daily_rank_ic(
        snapshot_frame.rename(columns={"metric_id": "factor_name"}),
        label_frame,
        minimum_cross_section=minimum_cross_section,
    )
    summary = summarize_factor_rank_ic(daily).rename(columns={"factor_name": "metric_id"})
    return (
        pd.DataFrame(coverage_rows).sort_values("metric_id", ignore_index=True),
        daily,
        summary.sort_values("metric_id", ignore_index=True),
    )


def _connect() -> V2WriteSession:
    client = ddb.session()
    client.connect(
        os.getenv("QRV2_DDB_HOST", "10.176.55.211"),
        int(os.getenv("QRV2_DDB_PORT", "8848")),
        os.getenv("QRV2_DDB_USERNAME", "xiahongjie"),
        os.getenv("QRV2_DDB_PASSWORD", ""),
    )
    return V2WriteSession(client, target_database=TARGET_DATABASE)


def _query_snapshot(
    session: V2WriteSession, *, start: date, end: date, metrics: tuple[str, ...]
) -> pd.DataFrame:
    metric_symbols = "`" + "`".join(sorted(metrics))
    result = session.query(
        "select signal_date,ts_code,metric_id,value,available_date,stale_days "
        f"from loadTable('{session.target_database}','{SNAPSHOT_OBJECT}') "
        f"where signal_date between date('{start:%Y.%m.%d}') and date('{end:%Y.%m.%d}') "
        f"and metric_id in {metric_symbols}"
    )
    if not isinstance(result, pd.DataFrame):
        raise RuntimeError("fundamental snapshot query did not return a DataFrame")
    if result.empty:
        raise RuntimeError("fundamental snapshot has no rows for the requested range")
    if (pd.to_datetime(result["available_date"]) > pd.to_datetime(result["signal_date"])).any():
        raise RuntimeError("fundamental snapshot has a PIT violation")
    if (pd.to_numeric(result["stale_days"], errors="raise") > 180).any():
        raise RuntimeError("fundamental snapshot has stale values")
    return result


def _query_month_inputs(
    session: V2WriteSession, *, start: date, end: date
) -> tuple[pd.DataFrame, pd.DataFrame]:
    universes: list[pd.DataFrame] = []
    labels: list[pd.DataFrame] = []
    for period in pd.period_range(start, end, freq="M"):
        logical_partition = str(period)
        universes.append(
            read_latest_partition(
                session,
                object_name="universe_daily_latest",
                logical_partition=logical_partition,
                columns=("batch_id", "signal_date", "ts_code", "is_member"),
            )
        )
        labels.append(
            read_latest_partition(
                session,
                object_name="label_daily_latest",
                logical_partition=logical_partition,
                columns=("batch_id", "signal_date", "ts_code", "raw_return", "label_valid"),
            )
        )
    universe = pd.concat(universes, ignore_index=True)
    labels_frame = pd.concat(labels, ignore_index=True)
    universe["signal_date"] = pd.to_datetime(universe["signal_date"], errors="raise").dt.date
    labels_frame["signal_date"] = pd.to_datetime(
        labels_frame["signal_date"], errors="raise"
    ).dt.date
    universe["ts_code"] = universe["ts_code"].astype(str)
    labels_frame["ts_code"] = labels_frame["ts_code"].astype(str)
    shsz = universe["ts_code"].str.endswith((".SH", ".SZ"))
    return (
        universe.loc[shsz, ["signal_date", "ts_code", "is_member"]].drop_duplicates(),
        labels_frame.loc[:, ["signal_date", "ts_code", "raw_return", "label_valid"]].drop_duplicates(),
    )


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    session = _connect()
    snapshot = _query_snapshot(session, start=args.start, end=args.end, metrics=args.metric)
    universe, labels = _query_month_inputs(session, start=args.start, end=args.end)
    coverage, daily, summary = summarize_snapshot_quality(snapshot, universe, labels)
    summary = summary.merge(coverage, on="metric_id", how="left", validate="one_to_one")
    summary["direction"] = summary["mean_rank_ic"].map(lambda value: 1 if value >= 0 else -1)
    summary["directional_day_share"] = summary.apply(
        lambda row: row["positive_day_share"]
        if row["direction"] > 0
        else 1.0 - row["positive_day_share"],
        axis=1,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.output / "daily_rank_ic.csv", index=False)
    coverage.to_csv(args.output / "coverage.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    metadata = {
        "scope": "FUNDAMENTAL_PIT_SNAPSHOT_QUALITY_V1",
        "market_scope": "SH/SZ",
        "excluded_exchanges": ["BJ"],
        "label": "raw 5-trading-day forward adjusted return",
        "period": f"{args.start.isoformat()}/{args.end.isoformat()}",
        "metrics": list(args.metric),
        "minimum_cross_section": 30,
        "input_source": "QuantResearchV2Test.fundamental_pit_snapshot and latest V2 universe/labels",
        "database_writes": False,
        "locked_test_read": False,
        "snapshot_row_count": len(snapshot),
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    return metadata


if __name__ == "__main__":
    main()
