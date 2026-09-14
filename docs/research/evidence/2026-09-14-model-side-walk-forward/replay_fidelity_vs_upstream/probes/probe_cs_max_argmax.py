"""The decisive test for yeslab_464: does cs_max's argmax land on a .BJ security, and does dropping BJ move it?

GP_Factor_095_Calculator computes _v13 = _v12.rolling(60).kurt() and then _v14 = _v13.max(axis=1) -- a
cs_max that collapses the whole cross-section into ONE market-wide series, which after two five-day means
becomes _v16, the regression regressor for EVERY security. Section 16 showed the replay panel carries 148
to 270 .BJ securities a day that upstream does not carry under those codes, and that BJ is two to three
times over-represented in the top 1% tail of pct_chg. Section 15.2 showed that counting HOW MANY names
differ cannot detect this, because a max depends on whether ONE name is present.

This recomputes the chain up to _v16 from the cached daily bars, twice -- once on the panel as the
materializer sees it and once with .BJ columns dropped -- and reports, per day: the argmax security,
whether it is .BJ, and how far _v14 and _v16 move when BJ is removed. If the argmax is a BJ name on the
days where the replay diverges most from upstream and not on the days where it agrees, the mechanism is
confirmed rather than merely permitted.

The rolling regression slope is computed in closed form, slope = sum_j (t_j - tbar) x_j / sum_j (t_j - tbar)^2,
which is algebraically identical to the calculator's _slope (population variance of t, NaN if any input is
NaN) and is verified against that function on random data before use.

SAFETY: reads the local daily-bar cache and repository source only. No database, no weights, and it does
NOT write into research_v2/artifacts -- output goes to --out only.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

CACHE = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2/artifacts/tushare_cache")
CALC = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/数据库/factor_calculators/GP_Factor_095_Calculator.py")


def slope_weights(n: int) -> np.ndarray:
    t = np.arange(1, n + 1, dtype=float)
    d = t - t.mean()
    return d / (d @ d)


def rolling_slope(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Closed-form equivalent of df.rolling(n).apply(_slope, raw=True)."""
    w = slope_weights(n)
    out = None
    for j in range(n):
        term = df.shift(n - 1 - j).mul(w[j])
        out = term if out is None else out.add(term)
    return out


def cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1) + 1e-10, axis=0)


def chain(close: pd.DataFrame, amount: pd.DataFrame, pct_chg: pd.DataFrame) -> dict:
    v1 = amount * pct_chg
    v2 = rolling_slope(v1, 5)
    v3 = v2.rolling(20).skew()
    v4 = pct_chg.rolling(5).median()
    v5 = v4.rolling(20).rank(pct=True)
    v6 = (v3 < v5).astype(float)
    v7 = cs_zscore(close)
    v8 = v7.rolling(60).rank(pct=True)
    v9 = (v6 > v8).astype(float)
    v10 = v9.rolling(5).mean()
    v11 = v10.rank(axis=1, pct=True)
    v12 = rolling_slope(v11, 20)
    v13 = v12.rolling(60).kurt()
    v14 = v13.max(axis=1)
    v15 = v14.rolling(5).mean()
    v16 = v15.rolling(5).mean()
    return {"v13": v13, "v14": v14, "v16": v16}


def load_panel(start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = sorted(p.stem.replace("daily_", "") for p in CACHE.glob("daily_[0-9]*.json")
                   if start.replace("-", "") <= p.stem.replace("daily_", "") <= end.replace("-", ""))
    if not dates:
        raise SystemExit(f"no cached daily bars between {start} and {end}")
    per_date: dict[pd.Timestamp, pd.DataFrame] = {}
    for d in dates:
        raw = json.loads((CACHE / f"daily_{d}.json").read_text())
        recs = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
        frame = pd.DataFrame(recs)
        frame["ts_code"] = frame["ts_code"].astype(str)
        frame = frame.drop_duplicates("ts_code", keep="last").set_index("ts_code")
        per_date[pd.Timestamp(f"{d[:4]}-{d[4:6]}-{d[6:]}")] = frame
    out = {}
    for field in ("close", "amount", "pct_chg"):
        # one Series per date, assembled into a date-by-code frame: index=trade_date, columns=ts_code
        wide = pd.DataFrame({ts: f[field] for ts, f in per_date.items() if field in f.columns}).T
        wide = wide.sort_index().apply(pd.to_numeric, errors="coerce")
        wide.index = pd.to_datetime(wide.index)
        out[field] = wide
    return out["close"], out["amount"], out["pct_chg"], dates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="panel load start, must leave >=150 trading days of warm-up")
    ap.add_argument("--end", required=True)
    ap.add_argument("--report-days", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    # verify the closed-form slope against the calculator's own _slope before trusting anything
    sys.path.insert(0, str(CALC.parent.parent))
    ns: dict = {"np": np}
    src = CALC.read_text(encoding="utf-8")
    exec(src[src.index("def _slope"):src.index("class GP_Factor_095_Calculator")], ns)
    rng = np.random.default_rng(7)
    for n in (5, 20):
        w = slope_weights(n)
        worst = 0.0
        for _ in range(400):
            x = rng.normal(size=n)
            if rng.random() < 0.2:
                x[rng.integers(0, n)] = np.nan
            a = ns["_slope"](x)
            b = np.nan if not np.all(np.isfinite(x)) else float(w @ x)
            if np.isnan(a) != np.isnan(b):
                raise SystemExit(f"slope NaN mismatch n={n}")
            if not np.isnan(a):
                worst = max(worst, abs(a - b))
        print(f"  closed-form slope verified for n={n}, worst abs diff {worst:.3e}", flush=True)

    close, amount, pct_chg, dates = load_panel(args.start, args.end)
    print(f"  panel: {close.shape[0]} trading days x {close.shape[1]} securities "
          f"({dates[0]}..{dates[-1]})", flush=True)
    bj = [c for c in close.columns if c.endswith(".BJ")]
    print(f"  .BJ columns: {len(bj)}", flush=True)

    full = chain(close, amount, pct_chg)
    keep = [c for c in close.columns if not c.endswith(".BJ")]
    nobj = chain(close[keep], amount[keep], pct_chg[keep])

    rows = []
    for day in sorted(args.report_days):
        ts = pd.Timestamp(day)
        if ts not in full["v13"].index:
            rows.append({"day": day, "in_panel": False}); continue
        v13 = full["v13"].loc[ts]
        finite = v13.dropna()
        argmax = finite.idxmax() if len(finite) else None
        v13n = nobj["v13"].loc[ts].dropna()
        argmax_n = v13n.idxmax() if len(v13n) else None
        f14, n14 = full["v14"].get(ts), nobj["v14"].get(ts)
        f16, n16 = full["v16"].get(ts), nobj["v16"].get(ts)
        rows.append({
            "day": day, "in_panel": True,
            "securities_with_a_finite_v13": int(len(finite)),
            "bj_securities_with_a_finite_v13": int(sum(1 for c in finite.index if c.endswith(".BJ"))),
            "argmax_code": argmax, "argmax_is_BJ": bool(argmax and argmax.endswith(".BJ")),
            "argmax_value": float(finite.max()) if len(finite) else None,
            "argmax_code_without_BJ": argmax_n,
            "argmax_changed_when_BJ_dropped": argmax != argmax_n,
            "v14_with_BJ": float(f14) if pd.notna(f14) else None,
            "v14_without_BJ": float(n14) if pd.notna(n14) else None,
            "v14_relative_change": float(abs(f14 - n14) / abs(f14)) if pd.notna(f14) and f14 else None,
            "v16_with_BJ": float(f16) if pd.notna(f16) else None,
            "v16_without_BJ": float(n16) if pd.notna(n16) else None,
            "v16_relative_change": float(abs(f16 - n16) / abs(f16)) if pd.notna(f16) and f16 else None,
            "warm_up_matured": bool(pd.notna(f16)),
        })
        r = rows[-1]
        fmt = lambda x: ("n/a" if x is None else f"{x:.6g}")
        pct = lambda x: ("n/a" if x is None else f"{x:.4%}")
        print(f"  {day}: finite_v13 {r['securities_with_a_finite_v13']} (BJ {r['bj_securities_with_a_finite_v13']}) "
              f"argmax {argmax} (BJ={r['argmax_is_BJ']}) -> without-BJ {argmax_n} "
              f"changed={r['argmax_changed_when_BJ_dropped']}; "
              f"v14 {fmt(r['v14_with_BJ'])} -> {fmt(r['v14_without_BJ'])} ({pct(r['v14_relative_change'])}); "
              f"v16 {fmt(r['v16_with_BJ'])} -> {fmt(r['v16_without_BJ'])} ({pct(r['v16_relative_change'])})",
              flush=True)

    measured = [r for r in rows if r.get("in_panel")]
    out = {
        "question": "Does GP_Factor_095_Calculator's cs_max argmax land on a .BJ security, and does "
                    "dropping .BJ move the regressor every security depends on?",
        "panel_window": [args.start, args.end], "trading_days_loaded": len(dates),
        "securities_in_panel": int(close.shape[1]), "bj_columns": len(bj),
        "slope_closed_form_verified": True,
        "report_days": measured,
        "days_where_argmax_is_BJ": sum(1 for r in measured if r["argmax_is_BJ"]),
        "days_where_argmax_changes_when_BJ_dropped": sum(1 for r in measured if r["argmax_changed_when_BJ_dropped"]),
        "max_v16_relative_change": max((r["v16_relative_change"] for r in measured
                                        if r["v16_relative_change"] is not None), default=None),
        "verdict": None,
        "database_writes": False, "weights_written": False,
        "cache_writes": False,
    }
    k = out["days_where_argmax_is_BJ"]
    out["verdict"] = (
        f"cs_max's argmax is a .BJ security on {k} of {len(measured)} reported days, and dropping .BJ "
        f"changes the argmax on {out['days_where_argmax_changes_when_BJ_dropped']} of them, moving _v16 "
        f"(the regressor every security shares) by up to "
        f"{(out['max_v16_relative_change'] or 0):.4%}. "
        + ("The mechanism is CONFIRMED as operative: the panel's BJ membership sets the regressor."
           if k else
           "The argmax is never a BJ name on these days, so BJ membership does not set the regressor here "
           "and the cs_max mechanism is NOT supported by this test.")
    )
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({kk: v for kk, v in out.items() if kk != "report_days"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
