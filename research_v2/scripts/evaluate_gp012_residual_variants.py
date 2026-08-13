"""Evaluate isolated de-correlation variants for GP_012 against GP_036.

GP_012 (Alpha381) and GP_036 (Alpha405) are both strong SH/SZ Tushare
candidates, but their validation cross-sectional Spearman similarity is high.
This read-only experiment residualizes GP_012 against GP_036 independently on
each signal date.  It evaluates whether the residual retains predictive
information while reducing pair redundancy.  No factor values are published,
no existing table is modified, and dates at or after the locked 2026 test are
rejected.
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
FROZEN_DIRECTION = -1
VARIANTS = ("gp012_raw", "gp012_residual_ols", "gp012_residual_rank")


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


def load_panel_inputs_period(client: object, period: str) -> tuple[pd.DataFrame, tuple[str, ...]]:
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
    panel_codes = tuple(sorted(universe["ts_code"].astype(str).unique()))
    return labels, panel_codes


def load_panel_period(client: object, period: str) -> pd.DataFrame:
    """Compatibility wrapper returning only the valid label panel."""

    labels, _ = load_panel_inputs_period(client, period)
    return labels


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


def compute_gp_panels(market: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"trade_date", "ts_code", "amount", "pct_chg"}
    if missing := sorted(required.difference(market.columns)):
        raise ValueError(f"market frame is missing columns: {missing}")
    pct = market.pivot(index="trade_date", columns="ts_code", values="pct_chg").sort_index()
    amount = market.pivot(index="trade_date", columns="ts_code", values="amount").sort_index()
    z = pct.sub(pct.mean(axis=1), axis=0).div(pct.std(axis=1) + 1e-10, axis=0)
    gp012 = (z * amount).rolling(5, min_periods=5).max()
    gp036 = z.rolling(5, min_periods=5).max()
    return gp012.replace([np.inf, -np.inf], np.nan), gp036.replace([np.inf, -np.inf], np.nan)


def cross_sectional_residual(y: pd.Series, x: pd.Series, *, rank: bool = False) -> pd.Series:
    """OLS residual with intercept; optionally regress cross-sectional ranks."""

    frame = pd.concat([y.rename("y"), x.rename("x")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    result = pd.Series(np.nan, index=y.index, dtype=float)
    if len(frame) < 3:
        return result
    left = frame["y"].rank(method="average", pct=True) if rank else frame["y"]
    right = frame["x"].rank(method="average", pct=True) if rank else frame["x"]
    matrix = np.column_stack([np.ones(len(frame)), right.to_numpy(float)])
    coefficients, *_ = np.linalg.lstsq(matrix, left.to_numpy(float), rcond=None)
    result.loc[frame.index] = left.to_numpy(float) - matrix @ coefficients
    return result


def build_variants(gp012: pd.DataFrame, gp036: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if not gp012.index.equals(gp036.index) or not gp012.columns.equals(gp036.columns):
        raise ValueError("GP panels must have matching dates and securities")

    def residual_panel(y: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
        valid = y.notna() & x.notna()
        y_mean = y.where(valid).mean(axis=1)
        x_mean = x.where(valid).mean(axis=1)
        yc = y.sub(y_mean, axis=0).where(valid, 0.0)
        xc = x.sub(x_mean, axis=0).where(valid, 0.0)
        denominator = (xc * xc).sum(axis=1)
        beta = (xc * yc).sum(axis=1).div(denominator.replace(0.0, np.nan))
        result = y.sub(y_mean, axis=0).sub(x.sub(x_mean, axis=0).mul(beta, axis=0))
        return result.where(valid & beta.notna().to_numpy()[:, None])

    residual = residual_panel(gp012, gp036)
    rank_residual = residual_panel(
        gp012.rank(axis=1, method="average", pct=True),
        gp036.rank(axis=1, method="average", pct=True),
    )
    return {
        "gp012_raw": gp012,
        "gp012_residual_ols": residual,
        "gp012_residual_rank": rank_residual,
    }


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


def summarize_daily(daily: pd.DataFrame, direction: int = FROZEN_DIRECTION) -> dict[str, object]:
    values = pd.to_numeric(daily["rank_ic"], errors="coerce")
    values = values.loc[np.isfinite(values.to_numpy(float))].to_numpy(float)
    std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
    signed = values * direction
    return {
        "valid_day_count": int(len(values)),
        "mean_rank_ic": float(values.mean()) if len(values) else float("nan"),
        "icir": float(values.mean() / std) if np.isfinite(std) and std > 0 else float("nan"),
        "signed_mean_rank_ic": float(signed.mean()) if len(values) else float("nan"),
        "directional_day_share": float((signed > 0).mean()) if len(values) else float("nan"),
    }


def summarize_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "month",
        "valid_day_count",
        "mean_rank_ic",
        "icir",
        "signed_mean_rank_ic",
        "directional_day_share",
    ]
    if daily.empty:
        return pd.DataFrame(columns=columns)
    frame = daily.copy()
    frame["month"] = pd.to_datetime(frame["signal_date"]).dt.to_period("M").astype(str)
    rows: list[dict[str, object]] = []
    for month, group in frame.groupby("month", sort=True):
        rows.append({"month": month, **summarize_daily(group)})
    return pd.DataFrame(rows, columns=columns)


def run_evaluation(args: argparse.Namespace, *, client: object | None = None) -> dict[str, object]:
    own_client = client is None
    client = client or _connect()
    args.output.mkdir(parents=True, exist_ok=True)
    daily_parts: dict[str, list[pd.DataFrame]] = {name: [] for name in VARIANTS}
    pair_parts: list[pd.DataFrame] = []
    market_rows = 0
    label_rows = 0
    for period in _periods(args.start, args.end):
        period_start = pd.Period(period, freq="M").start_time.date()
        period_end = pd.Period(period, freq="M").end_time.date()
        labels, panel_codes = load_panel_inputs_period(client, period)
        market = load_cached_market(
            args.cache_dir, period_start - timedelta(days=10), period_end, panel_codes
        )
        gp012, gp036 = compute_gp_panels(market)
        variants = build_variants(gp012, gp036)
        market_rows += len(market)
        label_rows += len(labels)
        for name, panel in variants.items():
            daily_parts[name].append(compute_daily_rank_ic(panel, labels))
        # Daily pair similarity is computed only on the current month, then
        # aggregated by month to avoid retaining stock-level panels.
        for signal_date in labels["signal_date"].drop_duplicates().sort_values():
            if signal_date not in gp012.index:
                continue
            joined = pd.DataFrame({"gp012": gp012.loc[signal_date], "gp036": gp036.loc[signal_date], "residual": variants["gp012_residual_ols"].loc[signal_date]}).replace([np.inf, -np.inf], np.nan).dropna()
            if len(joined) >= 30:
                pair_parts.append(pd.DataFrame([{
                    "signal_date": signal_date,
                    "raw_vs_gp036": joined["gp012"].rank().corr(joined["gp036"].rank()),
                    "residual_vs_gp036": joined["residual"].rank().corr(joined["gp036"].rank()),
                    "sample_count": len(joined),
                }]))
        del variants, gp012, gp036, market, labels
    summaries: list[dict[str, object]] = []
    for name, parts in daily_parts.items():
        daily = pd.concat(parts, ignore_index=True).sort_values("signal_date", kind="mergesort")
        monthly = summarize_monthly(daily)
        daily.to_csv(args.output / f"{name}_daily_rank_ic.csv", index=False)
        monthly.to_csv(args.output / f"{name}_monthly_rank_ic.csv", index=False)
        summary = summarize_daily(daily)
        train = daily.loc[pd.to_datetime(daily["signal_date"]).dt.year <= 2024]
        validation = daily.loc[pd.to_datetime(daily["signal_date"]).dt.year == 2025]
        train_month = summarize_monthly(train)
        validation_month = summarize_monthly(validation)
        summary.update({
            "variant": name,
            "train_2020_2024": summarize_daily(train),
            "validation_2025": summarize_daily(validation),
            "train_monthly_signed_mean_pass_months": int((train_month["signed_mean_rank_ic"] >= .02).sum()),
            "validation_monthly_signed_mean_pass_months": int((validation_month["signed_mean_rank_ic"] >= .02).sum()),
            "train_monthly_directional_share_pass_months": int((train_month["directional_day_share"] >= .55).sum()),
            "validation_monthly_directional_share_pass_months": int((validation_month["directional_day_share"] >= .55).sum()),
        })
        summaries.append(summary)
    pair = pd.concat(pair_parts, ignore_index=True)
    pair["month"] = pd.to_datetime(pair["signal_date"]).dt.to_period("M").astype(str)
    pair.to_csv(args.output / "pair_similarity_daily.csv", index=False)
    payload = {
        "scope": "GP012_RESIDUAL_VARIANT_EVALUATION_V1",
        "factor_ids": ["yeslab_381", "yeslab_405"],
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "market_source": "local Tushare.daily cache",
        "market_scope": "SH/SZ",
        "excluded_exchanges": ["BJ"],
        "label": "forward_open_5d:v1 raw five-trading-day forward return",
        "database_writes": False,
        "locked_test_read": False,
        "frozen_direction": FROZEN_DIRECTION,
        "market_rows": int(market_rows),
        "label_rows": int(label_rows),
        "pair_similarity": {
            "raw_vs_gp036_median_abs_spearman": float(pair["raw_vs_gp036"].abs().median()),
            "residual_vs_gp036_median_abs_spearman": float(pair["residual_vs_gp036"].abs().median()),
            "raw_vs_gp036_validation_median_abs_spearman": float(pair.loc[pair["month"].str.startswith("2025"), "raw_vs_gp036"].abs().median()),
            "residual_vs_gp036_validation_median_abs_spearman": float(pair.loc[pair["month"].str.startswith("2025"), "residual_vs_gp036"].abs().median()),
            "daily_count": int(len(pair)),
        },
        "variants": summaries,
    }
    (args.output / "quality_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if own_client:
        client.close()
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(run_evaluation(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
