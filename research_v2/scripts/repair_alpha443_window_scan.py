"""Window-variant repair exploration for yeslab_443 (GP_074).

GP_074 formula: ts_max_M(ts_regression_residual_R(cs_mean(cs_rank(amount)), amount*pct_chg))
with canonical R=30, M=5. The factor collapses in the 2017 large-cap-dominant
pre-training window (official-universe ICIR 0.262) and is WATCH in 2018. This
read-only exploration screens (R, M) variants to see whether any variant
recovers the 2017 window without being tuned on it: variants are screened on
2017 and would still need to clear 2018/2019 and the 2020-2024 training gates
in a follow-up to qualify as a repair. No database writes, no locked-test reads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "research_v2" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "research_v2" / "src"))

from audit_formula_replay_tushare import (  # noqa: E402
    _connect,
    _query_open_dates,
    fetch_full_tushare_shsz_market,
)

CACHE = REPO_ROOT / "research_v2" / "artifacts" / "tushare_cache"
REG_WINDOWS = (30, 40, 60, 90)
MAX_WINDOWS = (15, 20)
DIRECTION = -1


def compute_gp074_variant(market_wide: dict[str, pd.DataFrame], reg_window: int, max_window: int) -> pd.DataFrame:
    amount = market_wide["amount"]
    pct_chg = market_wide["pct_chg"]
    x = amount.rank(axis=1, pct=True).mean(axis=1)  # cross-sectional rank mean (Series)
    y = amount * pct_chg
    mx = x.rolling(reg_window, min_periods=reg_window).mean()
    varx = x.rolling(reg_window, min_periods=reg_window).var().replace(0.0, np.nan)
    my = y.rolling(reg_window, min_periods=reg_window).mean()
    cov = y.mul(x, axis=0).rolling(reg_window, min_periods=reg_window).mean().sub(my.mul(mx, axis=0), axis=0)
    # cov above is E[xy]-E[x]E[y]; rolling cov uses sample correction, negligible for screening
    beta = cov.div(varx, axis=0)
    residual = (y - my).sub(beta.mul(x - mx, axis=0), axis=0)
    return residual.rolling(max_window, min_periods=max_window).max()


def spearman_daily(factor: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    common_cols = factor.columns.intersection(labels.columns)
    f = factor.loc[:, common_cols].rank(axis=1)
    lab = labels.loc[:, common_cols].rank(axis=1)
    idx = f.index.intersection(lab.index)
    rows = []
    for dt in idx:
        fv, lv = f.loc[dt], lab.loc[dt]
        both = fv.notna() & lv.notna()
        if int(both.sum()) < 100:
            continue
        a, b = fv[both].to_numpy(float), lv[both].to_numpy(float)
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        rows.append({"signal_date": dt, "ic": float(np.corrcoef(a, b)[0, 1]), "n": int(both.sum())})
    return pd.DataFrame(rows)


def summarize(daily: pd.DataFrame) -> dict:
    if daily.empty:
        return {"days": 0}
    ic = daily["ic"]
    return {
        "days": int(len(ic)),
        "signed_mean_rank_ic": round(float(DIRECTION * ic.mean()), 6),
        "signed_icir": round(float(DIRECTION * ic.mean() / ic.std()), 4) if ic.std() > 0 else None,
        "directional_day_share": round(float((DIRECTION * ic > 0).mean()), 4),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2017, 1, 3))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2017, 12, 29))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reg-windows", type=str, default=None, help="comma-separated override, e.g. 30,40")
    parser.add_argument("--max-windows", type=str, default=None, help="comma-separated override, e.g. 15,20")
    parser.add_argument("--out-name", type=str, default="alpha443_window_scan.json")
    parser.add_argument("--official-universe", action="store_true", help="also compute IC restricted to official cn_liquid_a members")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    reg_windows = tuple(int(v) for v in args.reg_windows.split(",")) if args.reg_windows else REG_WINDOWS
    max_windows = tuple(int(v) for v in args.max_windows.split(",")) if args.max_windows else MAX_WINDOWS

    client = _connect()
    market_start = args.start - timedelta(days=90 + 60)  # warmup for reg_window<=60
    market_end = args.end + timedelta(days=15)
    open_dates = _query_open_dates(client, market_start, market_end)
    market = fetch_full_tushare_shsz_market(
        open_dates, token=os.getenv("QRV2_TUSHARE_TOKEN", ""), cache_dir=CACHE, include_daily_basic=False
    )
    market["trade_date"] = pd.to_datetime(market["trade_date"])
    market["ts_code"] = market["ts_code"].astype(str)
    wide = {
        field: market.pivot_table(index="trade_date", columns="ts_code", values=field, aggfunc="last").sort_index()
        for field in ("amount", "pct_chg", "open_hfq")
    }
    # labels: adjusted open T+1 -> T+6
    prices = wide["open_hfq"]
    trading = prices.index
    pos = {t: i for i, t in enumerate(trading)}
    mat = prices.to_numpy()
    codes = list(prices.columns)
    lab_rows = []
    sig_dates = [d for d in trading if pd.Timestamp(args.start) <= d <= pd.Timestamp(args.end)]
    for sd in sig_dates:
        i = pos[sd]
        if i + 6 >= len(trading):
            continue
        for j, c in enumerate(codes):
            ep, xp = mat[i + 1, j], mat[i + 6, j]
            if np.isfinite(ep) and np.isfinite(xp) and ep > 0 and xp > 0:
                lab_rows.append((sd, c, xp / ep - 1))
    labels = pd.DataFrame(lab_rows, columns=["signal_date", "ts_code", "ret"]).pivot_table(
        index="signal_date", columns="ts_code", values="ret", aggfunc="last"
    )
    print(f"market {market.shape[0]} rows; labels {labels.shape}", flush=True)

    official_members = None
    if args.official_universe:
        uni = client.run(
            "select signal_date, ts_code, data_version "
            "from loadTable('dfs://QuantResearchV2Test','universe_daily') "
            f"where signal_date between {args.start:%Y.%m.%d} and {args.end:%Y.%m.%d} "
            "and universe_id='cn_liquid_a' and is_member=1"
        )
        if uni is None or len(uni) == 0:
            raise RuntimeError("official cn_liquid_a universe has no members in window")
        uni["signal_date"] = pd.to_datetime(uni["signal_date"]).dt.normalize()
        uni["ts_code"] = uni["ts_code"].astype(str)
        latest = uni.groupby("signal_date")["data_version"].max().rename("latest_version")
        uni = uni.merge(latest, on="signal_date", how="left")
        uni = uni[uni["data_version"] == uni["latest_version"]]
        official_members = {dt: set(g["ts_code"]) for dt, g in uni.groupby("signal_date")}
        print(f"official universe dates: {len(official_members)}", flush=True)

    results = {}
    for reg in reg_windows:
        for mxw in max_windows:
            factor = compute_gp074_variant(wide, reg, mxw)
            factor = factor.reindex(labels.index)
            daily = spearman_daily(factor, labels)
            s = summarize(daily)
            key = f"reg{reg}_max{mxw}"
            results[key] = {"reg_window": reg, "max_window": mxw, "full_market": s}
            if not daily.empty:
                annual = {}
                daily_y = daily.copy()
                daily_y["year"] = pd.to_datetime(daily_y["signal_date"]).dt.year
                for year, grp in daily_y.groupby("year"):
                    annual[str(year)] = summarize(grp)
                if len(annual) > 1:
                    results[key]["annual_full_market"] = annual
            if official_members is not None:
                keep = pd.DataFrame(False, index=factor.index, columns=factor.columns)
                for dt in keep.index:
                    members = official_members.get(dt)
                    if members:
                        keep.loc[dt, keep.columns.intersection(list(members))] = True
                daily_off = spearman_daily(factor.where(keep, np.nan), labels.where(keep.reindex(index=labels.index), np.nan))
                results[key]["official_cn_liquid_a"] = summarize(daily_off)
                if not daily_off.empty:
                    annual_off = {}
                    daily_off_y = daily_off.copy()
                    daily_off_y["year"] = pd.to_datetime(daily_off_y["signal_date"]).dt.year
                    for year, grp in daily_off_y.groupby("year"):
                        annual_off[str(year)] = summarize(grp)
                    if len(annual_off) > 1:
                        results[key]["annual_official_cn_liquid_a"] = annual_off
            print(f"{key}: {results[key]}", flush=True)

    payload = {
        "scope": "ALPHA443_WINDOW_VARIANT_REPAIR_SCREEN_V2",
        "window": f"{args.start}/{args.end}",
        "formula": "ts_max_M(regression_residual_R(cs_mean(cs_rank(amount)), amount*pct_chg)); direction -1",
        "canonical": "reg30_max5",
        "windows_screened": {"reg": list(reg_windows), "max": list(max_windows)},
        "market_scope": "SH/SZ",
        "label": "forward_open_5d:v1 equivalent (Tushare adjusted open T+1->T+6)",
        "universe": "full_market (screening); official-universe follow-up only for qualifying variants",
        "variants": results,
        "database_writes": False,
        "locked_test_read": False,
        "formal_factor_set_changed": False,
        "model_training_started": False,
    }
    (args.output / args.out_name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output / args.out_name}")
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
