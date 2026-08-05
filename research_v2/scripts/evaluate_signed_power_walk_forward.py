"""Evaluate isolated signed-power factor variants with Tushare market inputs.

The script is read-only.  ``stock_factor_daily`` supplies the Tushare-imported
``vol``/``pct_chg`` history, while ``daily_basic`` is fetched directly from
Tushare for the turnover overlay that is missing on known source dates.  V2
latest universe and raw five-day labels are used only as the evaluation
boundary.  No 2026 rows are allowed and no database writes are performed.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from audit_signed_power_variants import compute_signed_power_variants
from research_v2.compact.publisher import read_latest_partition
from research_v2.db.session import V2WriteSession, YesLabReadOnlySession
from research_v2.evaluation.factor_rank_ic import compute_factor_daily_rank_ic

LOCKED_TEST_START = date(2026, 1, 1)
TRAIN_START = date(2020, 1, 1)
TRAIN_END = date(2024, 12, 31)
VALIDATION_START = date(2025, 1, 1)
VALIDATION_END = date(2025, 12, 31)
LOOKBACK_DAYS = 180
PURGE_LAST_DATES = 6


def merge_turnover_overlay(market: pd.DataFrame, overlay: pd.DataFrame) -> pd.DataFrame:
    """Merge direct Tushare turnover values without changing market keys."""

    required_market = {"trade_date", "ts_code", "turnover_rate"}
    required_overlay = required_market
    if missing := sorted(required_market.difference(market.columns)):
        raise ValueError(f"market is missing columns: {missing}")
    if missing := sorted(required_overlay.difference(overlay.columns)):
        raise ValueError(f"turnover overlay is missing columns: {missing}")
    result = market.copy(deep=True)
    direct = overlay.loc[:, ["trade_date", "ts_code", "turnover_rate"]].copy()
    for frame in (result, direct):
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="raise").dt.normalize()
        frame["ts_code"] = frame["ts_code"].astype(str)
        frame["turnover_rate"] = pd.to_numeric(frame["turnover_rate"], errors="coerce")
    if direct.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("turnover overlay contains duplicate market keys")
    result = result.merge(
        direct.rename(columns={"turnover_rate": "tushare_turnover_rate"}),
        on=["trade_date", "ts_code"],
        how="left",
        validate="one_to_one",
    )
    result["turnover_rate"] = result["tushare_turnover_rate"].combine_first(
        result["turnover_rate"]
    )
    return result.drop(columns=["tushare_turnover_rate"])


def summarize_walk_forward_metrics(
    daily: pd.DataFrame,
    *,
    train_direction: int | None = None,
) -> pd.DataFrame:
    """Summarize RankIC and freeze direction when evaluating validation."""

    required = {"factor_id", "signal_date", "rank_ic", "sample_count"}
    if missing := sorted(required.difference(daily.columns)):
        raise ValueError(f"daily RankIC is missing columns: {missing}")
    frame = daily.copy(deep=True)
    frame["rank_ic"] = pd.to_numeric(frame["rank_ic"], errors="coerce")
    frame["sample_count"] = pd.to_numeric(frame["sample_count"], errors="coerce")
    frame = frame.loc[np.isfinite(frame["rank_ic"]) & np.isfinite(frame["sample_count"])]
    rows: list[dict[str, object]] = []
    for factor_id, group in frame.groupby("factor_id", sort=True):
        values = group["rank_ic"].to_numpy(dtype=float)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        direction = int(train_direction) if train_direction is not None else (1 if mean >= 0 else -1)
        signed = values * direction
        row: dict[str, object] = {
            "factor_id": str(factor_id),
            "valid_day_count": len(values),
            "mean_rank_ic": mean,
            "rank_ic_std": std,
            "icir": mean / std if np.isfinite(std) and std > 0 else float("nan"),
            "direction": direction,
            "signed_mean_rank_ic": float(signed.mean()),
            "directional_day_share": float((signed > 0).mean()),
            "mean_sample_count": float(group["sample_count"].mean()),
        }
        if "finite_ratio" in group:
            ratios = pd.to_numeric(group["finite_ratio"], errors="coerce")
            row["minimum_finite_ratio"] = float(ratios.min())
            row["mean_finite_ratio"] = float(ratios.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _read_daily_basic_cache(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_json(path, orient="table")
    except (OSError, ValueError, TypeError):
        return pd.DataFrame()
    required = {"ts_code", "turnover_rate"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    if "trade_date" not in frame.columns:
        frame["trade_date"] = path.stem.removeprefix("daily_basic_")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    frame["ts_code"] = frame["ts_code"].astype(str)
    frame["turnover_rate"] = pd.to_numeric(frame["turnover_rate"], errors="coerce")
    return frame.loc[:, ["trade_date", "ts_code", "turnover_rate"]]


def fetch_daily_basic(
    dates: Sequence[date],
    *,
    token: str,
    cache_dir: Path,
) -> pd.DataFrame:
    """Fetch/calculate a cached direct-Tushare turnover panel."""

    if not token.strip():
        raise ValueError("QRV2_TUSHARE_TOKEN is required")
    import tushare as ts  # type: ignore[import-untyped]

    pro = ts.pro_api(token)
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for current in sorted(set(dates)):
        if current >= LOCKED_TEST_START:
            raise ValueError("daily_basic fetch cannot read 2026 locked-test data")
        path = cache_dir / f"daily_basic_{current:%Y%m%d}.json"
        frame = _read_daily_basic_cache(path) if path.is_file() else pd.DataFrame()
        if frame.empty:
            raw = pro.daily_basic(
                trade_date=current.strftime("%Y%m%d"),
                fields="ts_code,turnover_rate",
            )
            frame = raw if isinstance(raw, pd.DataFrame) else pd.DataFrame()
            if not frame.empty:
                frame["trade_date"] = pd.Timestamp(current)
                path.write_text(frame.to_json(orient="table", date_format="iso"), encoding="utf-8")
                frame = _read_daily_basic_cache(path)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["trade_date", "ts_code", "turnover_rate"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        ["trade_date", "ts_code"], keep="last"
    )


def _query_open_dates(source: YesLabReadOnlySession, start: date, end: date) -> tuple[date, ...]:
    result = source.query(
        "select cal_date from loadTable('dfs://YesLab','trade_calendar') "
        "where exchange='SSE' and is_open == 1 "
        f"and cal_date between date('{start:%Y.%m.%d}') and date('{end:%Y.%m.%d}') "
        "order by cal_date"
    )
    if not isinstance(result, pd.DataFrame) or "cal_date" not in result.columns:
        raise RuntimeError("Tushare-derived trade calendar query failed")
    dates = pd.to_datetime(result["cal_date"], errors="coerce").dropna().dt.date
    return tuple(sorted(current for current in dates if current < LOCKED_TEST_START))


def _query_market(
    source: YesLabReadOnlySession,
    start: date,
    end: date,
) -> pd.DataFrame:
    result = source.query(
        "select ts_code,trade_date,vol,amount,pct_chg,turnover_rate "
        "from loadTable('dfs://YesLab','stock_factor_daily') "
        f"where trade_date between date('{start:%Y.%m.%d}') and date('{end:%Y.%m.%d}')"
    )
    if not isinstance(result, pd.DataFrame):
        raise RuntimeError("Tushare-derived stock market query failed")
    frame = result.copy(deep=True)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="raise").dt.normalize()
    frame["ts_code"] = frame["ts_code"].astype(str)
    for column in ("vol", "amount", "pct_chg", "turnover_rate"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.duplicated(["trade_date", "ts_code"]).any():
        raise RuntimeError("Tushare-derived market contains duplicate keys")
    return frame


def _query_inputs(
    target: V2WriteSession,
    logical_partition: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    universe = read_latest_partition(
        target,
        object_name="universe_daily_latest",
        logical_partition=logical_partition,
        columns=("batch_id", "signal_date", "ts_code", "is_member"),
    )
    labels = read_latest_partition(
        target,
        object_name="label_daily_latest",
        logical_partition=logical_partition,
        columns=("batch_id", "signal_date", "ts_code", "raw_return", "label_valid", "exit_date"),
    )
    if universe.empty or labels.empty:
        raise RuntimeError(f"V2 latest inputs are missing for {logical_partition}")
    universe["signal_date"] = pd.to_datetime(universe["signal_date"], errors="raise").dt.normalize()
    labels["signal_date"] = pd.to_datetime(labels["signal_date"], errors="raise").dt.normalize()
    universe = universe.loc[universe["is_member"].astype(bool), ["signal_date", "ts_code"]]
    labels = labels.loc[labels["label_valid"].astype(bool), ["signal_date", "ts_code", "raw_return", "label_valid"]]
    return universe.drop_duplicates(["signal_date", "ts_code"]), labels.drop_duplicates(
        ["signal_date", "ts_code"]
    )


def _month_ranges(start: date, end: date) -> tuple[tuple[str, date, date], ...]:
    periods = pd.period_range(start, end, freq="M")
    return tuple((str(period), period.start_time.date(), period.end_time.date()) for period in periods)


def evaluate_phase(
    source: YesLabReadOnlySession,
    target: V2WriteSession,
    *,
    start: date,
    end: date,
    turnover: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate one phase month-by-month with bounded market working sets."""

    rows: list[pd.DataFrame] = []
    for logical_partition, month_start, month_end in _month_ranges(start, end):
        market = _query_market(source, month_start - timedelta(days=LOOKBACK_DAYS), month_end)
        basic = turnover.loc[
            turnover["trade_date"].between(
                pd.Timestamp(month_start - timedelta(days=LOOKBACK_DAYS)), pd.Timestamp(month_end)
            )
        ]
        market = merge_turnover_overlay(market, basic)
        variants = compute_signed_power_variants(market)
        factors = pd.concat(
            [frame.loc[frame["signal_date"].between(pd.Timestamp(month_start), pd.Timestamp(month_end))]
             for frame in variants.values()],
            ignore_index=True,
        )
        universe, labels = _query_inputs(target, logical_partition)
        factors = factors.rename(columns={"factor_id": "factor_name"})
        daily = compute_factor_daily_rank_ic(
            factors,
            labels.merge(universe, on=["signal_date", "ts_code"], how="inner", validate="one_to_one"),
            minimum_cross_section=30,
            label_column="raw_return",
        )
        daily["signal_date"] = pd.to_datetime(daily["signal_date"], errors="raise").dt.normalize()
        mature_counts = (
            labels.merge(universe, on=["signal_date", "ts_code"], how="inner", validate="one_to_one")
            .groupby("signal_date", sort=True)
            .size()
            .rename("mature_count")
            .reset_index()
        )
        mature_counts["signal_date"] = pd.to_datetime(
            mature_counts["signal_date"], errors="raise"
        ).dt.normalize()
        daily = daily.merge(mature_counts, on="signal_date", how="left", validate="many_to_one")
        daily["finite_ratio"] = daily["sample_count"] / daily["mature_count"]
        rows.append(daily.rename(columns={"factor_name": "factor_id"}))
    if not rows:
        return pd.DataFrame(columns=["factor_id", "signal_date", "rank_ic", "sample_count"])
    result = pd.concat(rows, ignore_index=True)
    all_dates = sorted(pd.to_datetime(result["signal_date"], errors="raise").dt.date.unique())
    purge = set(all_dates[-PURGE_LAST_DATES:])
    if purge:
        result = result.loc[
            ~pd.to_datetime(result["signal_date"], errors="raise").dt.date.isin(purge)
        ].copy()
    return result.sort_values(["factor_id", "signal_date"], ignore_index=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/qrv2-tushare-replay-cache"))
    parser.add_argument("--token", default=os.getenv("QRV2_TUSHARE_TOKEN", ""))
    args = parser.parse_args(argv)
    if args.output is None:
        raise ValueError("output is required")
    return args


def _connect() -> tuple[YesLabReadOnlySession, V2WriteSession]:
    import dolphindb as ddb

    host = os.getenv("QRV2_DDB_HOST", "10.176.55.211")
    port = int(os.getenv("QRV2_DDB_PORT", "8848"))
    user = os.getenv("QRV2_DDB_USERNAME", "xiahongjie")
    password = os.getenv("QRV2_DDB_PASSWORD", "123456")
    source_client = ddb.session()
    source_client.connect(host, port, user, password)
    target_client = ddb.session()
    target_client.connect(host, port, user, password)
    return YesLabReadOnlySession(source_client), V2WriteSession(
        target_client, target_database="dfs://QuantResearchV2Test"
    )


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    source, target = _connect()
    open_dates = _query_open_dates(source, TRAIN_START - timedelta(days=LOOKBACK_DAYS), VALIDATION_END)
    turnover = fetch_daily_basic(open_dates, token=args.token, cache_dir=args.cache_dir)
    train_daily = evaluate_phase(source, target, start=TRAIN_START, end=TRAIN_END, turnover=turnover)
    train_summary = summarize_walk_forward_metrics(train_daily)
    validation_daily = evaluate_phase(
        source, target, start=VALIDATION_START, end=VALIDATION_END, turnover=turnover
    )
    directions = {
        str(row.factor_id): int(row.direction)
        for row in train_summary.itertuples(index=False)
    }
    validation_summary = pd.concat(
        [
            summarize_walk_forward_metrics(
                validation_daily.loc[validation_daily["factor_id"] == factor_id],
                train_direction=directions.get(factor_id),
            )
            for factor_id in sorted(directions)
        ],
        ignore_index=True,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    train_daily.to_csv(args.output / "train_daily_rank_ic.csv", index=False)
    train_daily.to_csv(args.output / "train_daily_rank_ic_with_coverage.csv", index=False)
    validation_daily.to_csv(args.output / "validation_daily_rank_ic.csv", index=False)
    validation_daily.to_csv(args.output / "validation_daily_rank_ic_with_coverage.csv", index=False)
    train_summary.to_csv(args.output / "train_summary.csv", index=False)
    validation_summary.to_csv(args.output / "validation_summary.csv", index=False)
    payload = {
        "scope": "SIGNED_POWER_WALK_FORWARD_TUSHARE_V2",
        "train_start": TRAIN_START.isoformat(),
        "train_end": TRAIN_END.isoformat(),
        "validation_start": VALIDATION_START.isoformat(),
        "validation_end": VALIDATION_END.isoformat(),
        "purge_last_dates": PURGE_LAST_DATES,
        "lookback_days": LOOKBACK_DAYS,
        "factor_ids": ["yeslab_371", "yeslab_389", "yeslab_390"],
        "market_source": "Tushare.daily_basic overlay + YesLab stock_factor_daily imported from Tushare",
        "label_source": "QuantResearchV2Test.label_daily_latest.raw_return",
        "locked_test_read": False,
        "database_writes": False,
        "turnover_rows": int(len(turnover)),
        "train_daily_rows": int(len(train_daily)),
        "validation_daily_rows": int(len(validation_daily)),
    }
    (args.output / "metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


if __name__ == "__main__":
    main()
