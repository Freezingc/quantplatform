"""Evaluate isolated market-benchmark variants of the Terri volatility factor.

The source ``TerriStdCalculator`` estimates market return as the percentage
change of the cross-sectional mean *price*.  That is a fragile benchmark when
the stock panel changes or high-priced names dominate the mean.  This command
recomputes the factor from the read-only Tushare cache using alternative
cross-sectional return benchmarks, then evaluates the same V2 raw five-day
forward-return label.  It never writes factor values or reads the locked test.
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
DEFAULT_START = date(2020, 1, 1)
DEFAULT_END = date(2025, 12, 31)
FROZEN_DIRECTION = -1
VARIANT_NAMES = (
    "price_mean_change",
    "equal_weight_return",
    "median_return",
    "trimmed_return_10pct",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=DEFAULT_START)
    parser.add_argument("--end", type=date.fromisoformat, default=DEFAULT_END)
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


def load_v2_inputs(client: object, start: date, end: date) -> pd.DataFrame:
    """Read the compact V2 member/label panel with no database writes."""

    session = V2WriteSession(client, target_database="dfs://QuantResearchV2Test")
    universes: list[pd.DataFrame] = []
    labels: list[pd.DataFrame] = []
    for period in _periods(start, end):
        universes.append(
            read_latest_partition(
                session,
                object_name="universe_daily_latest",
                logical_partition=period,
                columns=("batch_id", "signal_date", "ts_code", "is_member"),
            )
        )
        labels.append(
            read_latest_partition(
                session,
                object_name="label_daily_latest",
                logical_partition=period,
                columns=("batch_id", "signal_date", "ts_code", "raw_return", "label_valid"),
            )
        )
    universe = pd.concat(universes, ignore_index=True)
    label = pd.concat(labels, ignore_index=True)
    for frame in (universe, label):
        frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="raise").dt.normalize()
        frame["ts_code"] = frame["ts_code"].astype(str)
    universe = universe.loc[
        universe["is_member"].fillna(False).astype(bool)
        & universe["ts_code"].str.endswith((".SH", ".SZ")),
        ["signal_date", "ts_code"],
    ].drop_duplicates(["signal_date", "ts_code"], keep="last")
    label = label.loc[
        label["label_valid"].fillna(False).astype(bool)
        & label["ts_code"].str.endswith((".SH", ".SZ")),
        ["signal_date", "ts_code", "raw_return"],
    ].copy()
    label["raw_return"] = pd.to_numeric(label["raw_return"], errors="coerce")
    label = label.loc[np.isfinite(label["raw_return"].to_numpy(float))]
    return universe.merge(
        label, on=["signal_date", "ts_code"], how="inner", validate="one_to_one"
    ).sort_values(["signal_date", "ts_code"], kind="mergesort").reset_index(drop=True)


def load_v2_inputs_period(client: object, period: str) -> pd.DataFrame:
    """Load one logical month so six years never reside in memory at once."""

    session = V2WriteSession(client, target_database="dfs://QuantResearchV2Test")
    universe = read_latest_partition(
        session,
        object_name="universe_daily_latest",
        logical_partition=period,
        columns=("batch_id", "signal_date", "ts_code", "is_member"),
    )
    label = read_latest_partition(
        session,
        object_name="label_daily_latest",
        logical_partition=period,
        columns=("batch_id", "signal_date", "ts_code", "raw_return", "label_valid"),
    )
    for frame in (universe, label):
        frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="raise").dt.normalize()
        frame["ts_code"] = frame["ts_code"].astype(str)
    universe = universe.loc[
        universe["is_member"].fillna(False).astype(bool)
        & universe["ts_code"].str.endswith((".SH", ".SZ")),
        ["signal_date", "ts_code"],
    ].drop_duplicates(["signal_date", "ts_code"], keep="last")
    label = label.loc[
        label["label_valid"].fillna(False).astype(bool)
        & label["ts_code"].str.endswith((".SH", ".SZ")),
        ["signal_date", "ts_code", "raw_return"],
    ].copy()
    label["raw_return"] = pd.to_numeric(label["raw_return"], errors="coerce")
    label = label.loc[np.isfinite(label["raw_return"].to_numpy(float))]
    return universe.merge(
        label, on=["signal_date", "ts_code"], how="inner", validate="one_to_one"
    ).sort_values(["signal_date", "ts_code"], kind="mergesort").reset_index(drop=True)


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
        raise FileNotFoundError(f"no Tushare daily cache files in {start}..{end}: {cache_dir}")
    return paths


def load_cached_market(cache_dir: Path, start: date, end: date) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in _cache_dates(cache_dir, start, end):
        frame = pd.read_json(path, orient="table")
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    codes = raw["ts_code"].astype(str)
    raw = raw.loc[codes.str.endswith((".SH", ".SZ"))].copy()
    return normalize_tushare_market(raw, codes=tuple(raw["ts_code"].astype(str).unique()), include_daily_basic=False)


def restrict_market_to_panel_codes(market: pd.DataFrame, panel_codes: Sequence[str]) -> pd.DataFrame:
    """Match the replay convention: calculate on the period's V2 code union."""

    codes = {str(code) for code in panel_codes}
    if not codes:
        raise ValueError("panel code union must not be empty")
    result = market.loc[market["ts_code"].astype(str).isin(codes)].copy()
    if result.empty:
        raise ValueError("market cache has no overlap with the V2 panel code union")
    return result


def market_benchmarks(close: pd.DataFrame, pct_chg: pd.DataFrame) -> dict[str, pd.Series]:
    """Return benchmark return series, all calculated using same-day data only."""

    price_mean = close.mean(axis=1).pct_change() * 100.0
    equal_weight = pct_chg.mean(axis=1)
    median_return = pct_chg.median(axis=1)
    lower = pct_chg.quantile(0.10, axis=1)
    upper = pct_chg.quantile(0.90, axis=1)
    trimmed = pct_chg.clip(lower=lower, upper=upper, axis=0).mean(axis=1)
    return {
        "price_mean_change": price_mean,
        "equal_weight_return": equal_weight,
        "median_return": median_return,
        "trimmed_return_10pct": trimmed,
    }


def compute_variant_panel(market: pd.DataFrame) -> dict[str, pd.DataFrame]:
    required = {"trade_date", "ts_code", "close", "pct_chg"}
    if missing := sorted(required.difference(market.columns)):
        raise ValueError(f"market frame is missing columns: {missing}")
    close = market.pivot(index="trade_date", columns="ts_code", values="close").sort_index()
    pct_chg = market.pivot(index="trade_date", columns="ts_code", values="pct_chg").sort_index()
    factors: dict[str, pd.DataFrame] = {}
    for name, benchmark in market_benchmarks(close, pct_chg).items():
        bias = pct_chg.sub(benchmark, axis=0).abs()
        bench = pct_chg.abs().add(benchmark.abs(), axis=0) + 0.1
        terri = bias / bench
        factors[name] = (terri * pct_chg).rolling(20, min_periods=20).std(ddof=1)
    return factors


def compute_daily_rank_ic(factor: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for signal_date, group in labels.groupby("signal_date", sort=True):
        if signal_date not in factor.index:
            continue
        values = factor.loc[signal_date].rename("factor_value").to_frame()
        values.index.name = "ts_code"
        joined = group.set_index("ts_code")["raw_return"].rename("raw_return").to_frame().join(values, how="inner")
        joined = joined.replace([np.inf, -np.inf], np.nan).dropna()
        if len(joined) < 30:
            continue
        rank_ic = joined["factor_value"].rank(method="average").corr(joined["raw_return"].rank(method="average"))
        if np.isfinite(rank_ic):
            rows.append({"signal_date": signal_date, "rank_ic": float(rank_ic), "sample_count": int(len(joined))})
    return pd.DataFrame(rows, columns=["signal_date", "rank_ic", "sample_count"])


def summarize_daily(daily: pd.DataFrame, *, direction: int = FROZEN_DIRECTION) -> dict[str, object]:
    values = pd.to_numeric(daily["rank_ic"], errors="coerce")
    values = values.loc[np.isfinite(values.to_numpy(float))].to_numpy(float)
    signed = values * direction
    std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
    return {
        "valid_day_count": int(len(values)),
        "mean_rank_ic": float(values.mean()) if len(values) else float("nan"),
        "icir": float(values.mean() / std) if np.isfinite(std) and std > 0 else float("nan"),
        "signed_mean_rank_ic": float(signed.mean()) if len(values) else float("nan"),
        "directional_day_share": float((signed > 0).mean()) if len(values) else float("nan"),
    }


def summarize_months(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy()
    frame["month"] = pd.to_datetime(frame["signal_date"]).dt.to_period("M").astype(str)
    rows: list[dict[str, object]] = []
    for month, group in frame.groupby("month", sort=True):
        summary = summarize_daily(group)
        rows.append({"month": month, **summary})
    return pd.DataFrame(rows)


def run_evaluation(args: argparse.Namespace, *, client: object | None = None) -> dict[str, object]:
    own_client = client is None
    client = client or _connect()
    args.output.mkdir(parents=True, exist_ok=True)
    periods = _periods(args.start, args.end)
    daily_by_variant: dict[str, list[pd.DataFrame]] = {name: [] for name in VARIANT_NAMES}
    market_rows = 0
    label_rows = 0
    for period in periods:
        period_start = pd.Period(period, freq="M").start_time.date()
        period_end = pd.Period(period, freq="M").end_time.date()
        labels = load_v2_inputs_period(client, period)
        market = load_cached_market(args.cache_dir, period_start - timedelta(days=45), period_end)
        market = restrict_market_to_panel_codes(market, labels["ts_code"].unique())
        factors = compute_variant_panel(market)
        market_rows += len(market)
        label_rows += len(labels)
        for name, panel in factors.items():
            daily_by_variant[name].append(compute_daily_rank_ic(panel, labels))
        del factors, market, labels
    summaries: list[dict[str, object]] = []
    for name, parts in daily_by_variant.items():
        daily = pd.concat(parts, ignore_index=True).sort_values("signal_date", kind="mergesort")
        daily.to_csv(args.output / f"{name}_daily_rank_ic.csv", index=False)
        monthly = summarize_months(daily)
        monthly.to_csv(args.output / f"{name}_monthly_rank_ic.csv", index=False)
        summary = summarize_daily(daily)
        summary.update(
            {
                "variant": name,
                "monthly_count": int(len(monthly)),
                "monthly_signed_mean_pass_months": int((monthly["signed_mean_rank_ic"] >= 0.02).sum()) if len(monthly) else 0,
                "monthly_directional_share_pass_months": int((monthly["directional_day_share"] >= 0.55).sum()) if len(monthly) else 0,
                "monthly_strict_icir_pass_months": int((monthly["icir"].abs() >= 0.5).sum()) if len(monthly) else 0,
            }
        )
        summaries.append(summary)
    payload = {
        "scope": "TERRI_MARKET_BENCHMARK_VARIANT_EVALUATION_V1",
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "market_start": (args.start - timedelta(days=45)).isoformat(),
        "market_source": "local Tushare.daily cache",
        "market_scope": "SH/SZ",
        "excluded_exchanges": ["BJ"],
        "label": "forward_open_5d:v1 raw five-trading-day forward return",
        "database_writes": False,
        "locked_test_read": False,
        "factor_id": "yeslab_114",
        "frozen_direction": FROZEN_DIRECTION,
        "market_rows": int(market_rows),
        "label_rows": int(label_rows),
        "variants": summaries,
    }
    (args.output / "quality_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if own_client:
        client.close()
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_evaluation(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
