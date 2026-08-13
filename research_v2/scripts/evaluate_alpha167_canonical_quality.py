"""Evaluate Alpha167/Alpha99 on the canonical V2 SH/SZ panel.

Historical Alpha99 replay is blocked because the stored factor was produced
with a different market source/cross-section.  This command intentionally
does not compare stored factor values.  It rematerializes the exact calculator
formula from the local Tushare daily cache, using the complete V2 SH/SZ code
union for each monthly panel, then evaluates the fixed 2020-2024 direction on
the raw five-trading-day label.  No database writes and no 2026 reads occur.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path

import dolphindb as ddb  # type: ignore[import-untyped]
import numpy as np
import pandas as pd

from audit_formula_replay_tushare import normalize_tushare_market
from research_v2.compact.publisher import read_latest_partition
from research_v2.db.session import V2WriteSession

LOCKED_TEST_START = date(2026, 1, 1)
DEFAULT_CACHE_DIR = Path("research_v2/artifacts/tushare_cache")
FROZEN_DIRECTION = 1
WARMUP_CALENDAR_DAYS = 120


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2020, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.start > args.end:
        parser.error("start date must not be after end date")
    if args.end >= LOCKED_TEST_START:
        parser.error("evaluation range must be before 2026 locked-test")
    return args


def _connect() -> object:
    client = ddb.session()
    client.connect(
        os.getenv("QRV2_DDB_HOST", "10.176.55.211"),
        int(os.getenv("QRV2_DDB_PORT", "8848")),
        os.getenv("QRV2_DDB_USERNAME", "xiahongjie"),
        os.getenv("QRV2_DDB_PASSWORD", "123456"),
    )
    return client


def _periods(start: date, end: date) -> tuple[str, ...]:
    return tuple(str(value) for value in pd.period_range(start, end, freq="M"))


def load_period_inputs(client: object, period: str) -> tuple[pd.DataFrame, tuple[str, ...]]:
    session = V2WriteSession(client, target_database="dfs://QuantResearchV2Test")
    universe = read_latest_partition(
        session,
        object_name="universe_daily_latest",
        logical_partition=period,
        columns=("batch_id", "signal_date", "ts_code", "is_member"),
    )
    labels = read_latest_partition(
        session,
        object_name="label_daily_latest",
        logical_partition=period,
        columns=("batch_id", "signal_date", "ts_code", "raw_return", "label_valid"),
    )
    for frame in (universe, labels):
        frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="raise").dt.normalize()
        frame["ts_code"] = frame["ts_code"].astype(str)
    universe = universe.loc[
        universe["is_member"].fillna(False).astype(bool)
        & universe["ts_code"].str.endswith((".SH", ".SZ")),
        ["signal_date", "ts_code"],
    ].drop_duplicates(["signal_date", "ts_code"], keep="last")
    codes = tuple(sorted(universe["ts_code"].unique()))
    labels = labels.loc[
        labels["label_valid"].fillna(False).astype(bool)
        & labels["ts_code"].str.endswith((".SH", ".SZ")),
        ["signal_date", "ts_code", "raw_return"],
    ].copy()
    labels["raw_return"] = pd.to_numeric(labels["raw_return"], errors="coerce")
    labels = labels.loc[np.isfinite(labels["raw_return"].to_numpy(float))]
    labels = universe.merge(
        labels, on=["signal_date", "ts_code"], how="inner", validate="one_to_one"
    ).sort_values(["signal_date", "ts_code"], kind="mergesort").reset_index(drop=True)
    return labels, codes


def _cache_dates(cache_dir: Path, start: date, end: date) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(cache_dir.glob("daily_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].json")):
        token = path.stem.removeprefix("daily_")
        try:
            current = date.fromisoformat(f"{token[:4]}-{token[4:6]}-{token[6:8]}")
        except ValueError:
            continue
        if start <= current <= end:
            paths.append(path)
    if not paths:
        raise FileNotFoundError(f"no Tushare daily cache files in {start}..{end}")
    return paths


def load_cached_market(cache_dir: Path, start: date, end: date, codes: Sequence[str]) -> pd.DataFrame:
    wanted = {str(code) for code in codes}
    frames: list[pd.DataFrame] = []
    for path in _cache_dates(cache_dir, start, end):
        frame = pd.read_json(path, orient="table")
        frame["ts_code"] = frame["ts_code"].astype(str)
        frame = frame.loc[frame["ts_code"].isin(wanted)]
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise ValueError("Tushare cache has no overlap with V2 panel codes")
    return normalize_tushare_market(
        pd.concat(frames, ignore_index=True), codes=tuple(sorted(wanted)), include_daily_basic=False
    )


def compute_alpha99(market: pd.DataFrame, *, window: int = 5) -> pd.DataFrame:
    """Implement Alpha99: -cross-sectional_rank(rolling_cov(rank(close), rank(vol)))."""

    required = {"trade_date", "ts_code", "close", "vol"}
    if missing := sorted(required.difference(market.columns)):
        raise ValueError(f"market frame is missing columns: {missing}")
    close = market.pivot(index="trade_date", columns="ts_code", values="close").sort_index()
    volume = market.pivot(index="trade_date", columns="ts_code", values="vol").sort_index()
    rank_close = close.rank(axis=1, pct=True)
    rank_volume = volume.rank(axis=1, pct=True)
    covariance = rank_close.rolling(window=window, min_periods=window).cov(rank_volume)
    factor = -covariance.rank(axis=1, pct=True)
    return factor.replace([np.inf, -np.inf], np.nan)


def compute_daily_rank_ic(panel: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for signal_date, group in labels.groupby("signal_date", sort=True):
        if signal_date not in panel.index:
            continue
        values = panel.loc[signal_date].rename("factor_value").to_frame()
        values.index.name = "ts_code"
        joined = group.set_index("ts_code")["raw_return"].rename("raw_return").to_frame().join(values, how="inner")
        joined = joined.replace([np.inf, -np.inf], np.nan).dropna()
        if len(joined) < 30:
            continue
        rank_ic = joined["factor_value"].rank(method="average").corr(joined["raw_return"].rank(method="average"))
        if np.isfinite(rank_ic):
            rows.append({"signal_date": signal_date, "rank_ic": float(rank_ic), "sample_count": int(len(joined))})
    return pd.DataFrame(rows, columns=["signal_date", "rank_ic", "sample_count"])


def summarize_daily(daily: pd.DataFrame) -> dict[str, object]:
    values = pd.to_numeric(daily["rank_ic"], errors="coerce")
    values = values.loc[np.isfinite(values.to_numpy(float))].to_numpy(float)
    std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
    signed = values * FROZEN_DIRECTION
    return {
        "valid_day_count": int(len(values)),
        "mean_rank_ic": float(values.mean()) if len(values) else float("nan"),
        "icir": float(values.mean() / std) if np.isfinite(std) and std > 0 else float("nan"),
        "signed_mean_rank_ic": float(signed.mean()) if len(values) else float("nan"),
        "directional_day_share": float((signed > 0).mean()) if len(values) else float("nan"),
    }


def summarize_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    columns = ["month", "valid_day_count", "mean_rank_ic", "icir", "signed_mean_rank_ic", "directional_day_share"]
    if daily.empty:
        return pd.DataFrame(columns=columns)
    frame = daily.copy()
    frame["month"] = pd.to_datetime(frame["signal_date"]).dt.to_period("M").astype(str)
    rows = [{"month": month, **summarize_daily(group)} for month, group in frame.groupby("month", sort=True)]
    return pd.DataFrame(rows, columns=columns)


def run_evaluation(args: argparse.Namespace, *, client: object | None = None) -> dict[str, object]:
    own_client = client is None
    client = client or _connect()
    args.output.mkdir(parents=True, exist_ok=True)
    daily_parts: list[pd.DataFrame] = []
    market_rows = label_rows = 0
    for period in _periods(args.start, args.end):
        period_start = pd.Period(period, freq="M").start_time.date()
        period_end = pd.Period(period, freq="M").end_time.date()
        labels, panel_codes = load_period_inputs(client, period)
        market = load_cached_market(
            args.cache_dir,
            period_start - timedelta(days=WARMUP_CALENDAR_DAYS),
            period_end,
            panel_codes,
        )
        factor = compute_alpha99(market)
        daily_parts.append(compute_daily_rank_ic(factor, labels))
        market_rows += len(market)
        label_rows += len(labels)
        del factor, market, labels
    daily = pd.concat(daily_parts, ignore_index=True).sort_values("signal_date", kind="mergesort")
    monthly = summarize_monthly(daily)
    daily.to_csv(args.output / "daily_rank_ic.csv", index=False)
    monthly.to_csv(args.output / "monthly_rank_ic.csv", index=False)
    train = daily.loc[pd.to_datetime(daily["signal_date"]).dt.year <= 2024]
    validation = daily.loc[pd.to_datetime(daily["signal_date"]).dt.year == 2025]
    train_month = summarize_monthly(train)
    validation_month = summarize_monthly(validation)
    annual: dict[str, dict[str, object]] = {}
    for year in range(2020, 2025):
        year_daily = train.loc[pd.to_datetime(train["signal_date"]).dt.year == year]
        annual[str(year)] = summarize_daily(year_daily)
    payload = {
        "scope": "ALPHA167_CANONICAL_TUSHARE_QUALITY_V1",
        "factor_id": "yeslab_167",
        "factor_name": "Alpha99",
        "market_scope": "SH/SZ",
        "excluded_exchanges": ["BJ"],
        "market_source": "local Tushare.daily cache",
        "label": "forward_open_5d:v1 raw five-trading-day forward return",
        "frozen_direction": FROZEN_DIRECTION,
        "database_writes": False,
        "locked_test_read": False,
        "market_rows": int(market_rows),
        "label_rows": int(label_rows),
        "train_2020_2024": summarize_daily(train),
        "validation_2025": summarize_daily(validation),
        "annual": annual,
        "strict_annual_icir_pass_years": int(sum(abs(row["icir"]) >= 0.5 for row in annual.values())),
        "train_monthly_signed_mean_pass_months": int((train_month["signed_mean_rank_ic"] >= .02).sum()),
        "validation_monthly_signed_mean_pass_months": int((validation_month["signed_mean_rank_ic"] >= .02).sum()),
        "train_monthly_directional_share_pass_months": int((train_month["directional_day_share"] >= .55).sum()),
        "validation_monthly_directional_share_pass_months": int((validation_month["directional_day_share"] >= .55).sum()),
    }
    (args.output / "quality_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if own_client:
        client.close()
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(run_evaluation(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
