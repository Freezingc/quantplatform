"""Raw close or adjusted close? Recompute yeslab_464 both ways and see which matches the stored GP_095.

Eight candidates for yeslab_464's divergence have now been excluded, the last being the BJ/cs_max
mechanism -- real and material (dropping BJ moves the shared regressor _v16 by up to 10.62% and _v14 by up
to 40.28%) but unrelated to the day-to-day fidelity swing. This tests a ninth, and it is cheap because it
is decisive in one run: GP_Factor_095_Calculator's _v7 = cs_zscore(close) feeds a 60-day rolling rank and
then the whole remaining chain, so if production computed on adjusted prices while the replay used raw
ones -- or the reverse -- every downstream value differs while the SCALE stays pinned by the z-score and
rank steps. That is exactly the observed shape: std ratio 0.852-1.144, Spearman swinging 0.636-0.988.

merge_tushare_adjustment_factor (audit_formula_replay_tushare.py:420-441) only ATTACHES adj_factor; it
does not overwrite close. So the replay's close is the raw Tushare daily close. This recomputes the full
factor for one day on both close and close*adj_factor and compares each against the stored GP_095.

The rolling regression slope uses the same closed form verified to 4e-16 against the calculator's own
_slope in probe_cs_max_argmax.py.

SAFETY: reads the local daily-bar and adj_factor caches and SELECT-only dfs://YesLab. No weights, and it
does NOT write into research_v2/artifacts -- output goes to --out only.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

CACHE = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2/artifacts/tushare_cache")
REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
TABLE, NAME, FID = "factor_data10", "GP_095", "yeslab_464"


def slope_weights(n):
    t = np.arange(1, n + 1, dtype=float); d = t - t.mean(); return d / (d @ d)


def rolling_slope(df, n):
    w = slope_weights(n); out = None
    for j in range(n):
        term = df.shift(n - 1 - j).mul(w[j])
        out = term if out is None else out.add(term)
    return out


def cs_zscore(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1) + 1e-10, axis=0)


def gp095(close, amount, pct_chg):
    """GP_Factor_095_Calculator._calculate, transcribed step for step from the calculator source."""
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
    v17 = cs_zscore(pct_chg)
    mx = v16.rolling(10).mean()
    varx = v16.rolling(10).var().replace(0, np.nan)
    v18 = pd.DataFrame(index=v17.index, columns=v17.columns, dtype=float)
    for c in v17.columns:
        my = v17[c].rolling(10).mean()
        cov = v16.rolling(10).cov(v17[c])
        beta = cov / varx
        v18[c] = ((v17[c] - my) - beta * (v16 - mx)).replace([np.inf, -np.inf], np.nan)
    return v18.rolling(5).max().replace([np.inf, -np.inf], np.nan)


def load(start, end):
    dates = sorted(p.stem.replace("daily_", "") for p in CACHE.glob("daily_[0-9]*.json")
                   if start.replace("-", "") <= p.stem.replace("daily_", "") <= end.replace("-", ""))
    close, amount, pct, adj = {}, {}, {}, {}
    for d in dates:
        ts = pd.Timestamp(f"{d[:4]}-{d[4:6]}-{d[6:]}")
        raw = json.loads((CACHE / f"daily_{d}.json").read_text())
        recs = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
        f = pd.DataFrame(recs); f["ts_code"] = f["ts_code"].astype(str)
        f = f.drop_duplicates("ts_code", keep="last").set_index("ts_code")
        close[ts] = f["close"]; amount[ts] = f["amount"]; pct[ts] = f["pct_chg"]
        ap = CACHE / f"adj_factor_{d}.json"
        if ap.is_file():
            ar = json.loads(ap.read_text())
            arec = ar["data"] if isinstance(ar, dict) and "data" in ar else ar
            af = pd.DataFrame(arec); af["ts_code"] = af["ts_code"].astype(str)
            adj[ts] = af.drop_duplicates("ts_code", keep="last").set_index("ts_code")["adj_factor"]
    wide = lambda dd: pd.DataFrame(dd).T.sort_index().apply(pd.to_numeric, errors="coerce")
    C, A, P = wide(close), wide(amount), wide(pct)
    for w in (C, A, P):
        w.index = pd.to_datetime(w.index)
    AD = wide(adj); AD.index = pd.to_datetime(AD.index)
    AD = AD.reindex(index=C.index, columns=C.columns)
    return C, A, P, AD, dates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True); ap.add_argument("--end", required=True)
    ap.add_argument("--day", required=True); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    C, A, P, AD, dates = load(a.start, a.end)
    print(f"  panel {C.shape[0]} days x {C.shape[1]} codes ({dates[0]}..{dates[-1]}), "
          f"adj_factor coverage {AD.notna().mean().mean():.4f}", flush=True)
    CH = (C * AD)                      # 后复权 close
    res_raw = gp095(C, A, P)
    res_hfq = gp095(CH, A, P)
    ts = pd.Timestamp(a.day)

    sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
    from walk_forward_common import TARGET_DATABASE, current_versions, new_holder, query
    h = new_holder(); fp = current_versions(h, "factor_daily")
    part = f"{a.day[:4]}-{a.day[5:7]}"; dl = a.day.replace("-", ".")
    ok, fd = query(h, f"select ts_code, value from loadTable('{TARGET_DATABASE}','factor_daily') "
                      f"where signal_date = date('{dl}') and factor_id = '{FID}' "
                      f"and data_version in ['{fp[part]}']")
    ok2, up = query(h, f"select ts_code, value from loadTable('dfs://YesLab','{TABLE}') "
                       f"where factor_name = '{NAME}' and trade_date = date('{dl}')")
    if not ok or not ok2:
        raise SystemExit("query failed")
    def ser(df):
        df = df.copy(); df["ts_code"] = df["ts_code"].astype(str)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df.dropna(subset=["value"]).groupby("ts_code")["value"].first()
    stored_fd, stored_up = ser(fd), ser(up)

    def compare(label, computed):
        row = computed.loc[ts].dropna() if ts in computed.index else pd.Series(dtype=float)
        out = {}
        for target, s in (("factor_daily", stored_fd), ("upstream_GP_095", stored_up)):
            j = pd.concat([row.rename("mine"), s.rename("theirs")], axis=1, join="inner").dropna()
            if len(j) < 50:
                out[target] = {"pairs": int(len(j))}; continue
            diff = (j["mine"] - j["theirs"]).abs()
            out[target] = {"pairs": int(len(j)),
                           "spearman": float(j["mine"].corr(j["theirs"], method="spearman")),
                           "pearson": float(np.corrcoef(j["mine"], j["theirs"])[0, 1]),
                           "exact_match_rate": float((diff == 0).mean()),
                           "median_relative_difference": float((diff / j["theirs"].abs().replace(0, np.nan)).median()),
                           "std_ratio": float(j["mine"].std() / j["theirs"].std())}
        print(f"  {label}:", json.dumps(out, ensure_ascii=False), flush=True)
        return out

    raw_cmp = compare("recomputed_with_RAW_close", res_raw)
    hfq_cmp = compare("recomputed_with_HFQ_close", res_hfq)
    out = {
        "question": "Does GP_095 / yeslab_464 diverge because the replay uses raw close where production "
                    "used adjusted close, or the reverse?",
        "day": a.day, "panel_window": [a.start, a.end], "trading_days": C.shape[0],
        "securities": int(C.shape[1]),
        "adj_factor_cell_coverage": float(AD.notna().mean().mean()),
        "recomputed_raw_close": raw_cmp, "recomputed_hfq_close": hfq_cmp,
        "verdict": None, "database_writes": False, "weights_written": False, "cache_writes": False,
    }
    def sp(c, t): return (c.get(t) or {}).get("spearman")
    r_up, h_up = sp(raw_cmp, "upstream_GP_095"), sp(hfq_cmp, "upstream_GP_095")
    r_fd, h_fd = sp(raw_cmp, "factor_daily"), sp(hfq_cmp, "factor_daily")
    out["verdict"] = (
        f"Against upstream GP_095 the raw-close recomputation reaches Spearman {r_up} and the "
        f"hfq-close one {h_up}; against the published factor_daily the same two reach {r_fd} and {h_fd}. "
        + ("ADJUSTMENT IS THE DIFFERENCE -- one of the two clearly beats the other against factor_daily, "
           "so the close convention explains the divergence."
           if (r_fd is not None and h_fd is not None and abs(r_fd - h_fd) > 0.05) else
           "ADJUSTMENT IS NOT THE DIFFERENCE -- both conventions land about equally close to factor_daily, "
           "so the raw-versus-adjusted close hypothesis is excluded too.")
    )
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("recomputed_")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
