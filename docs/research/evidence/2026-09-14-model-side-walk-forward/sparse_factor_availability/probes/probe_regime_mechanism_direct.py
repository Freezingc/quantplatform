#!/usr/bin/env python3
"""Read-only: test §BD's mechanism directly -- is each sparse factor available exactly where its power's base is positive?

§BD read the six calculators and found each contains `base ** exponent` with a base that can go
negative, and that the two empirically measured availability clusters (§BA.5) map one-to-one onto the
two bases: cluster A {372, 389, 390} uses `pct_chg`, cluster B {375, 384} uses `tanh` of the 20-day
change in `vol` and {406} uses `tanh` of the 20-day change in `turnover_rate`. That was source
structure plus a matching shape, not a test. This is the test.

For every one of the 969 OOS days it computes, over that day's panel cross-section, the share of
securities whose base is positive:

    cluster A base > 0   <=>  pct_chg > 0
    GP_006 / GP_015      <=>  vol > vol 20 trading days earlier
    GP_037               <=>  turnover_rate > turnover_rate 20 trading days earlier

and compares each with the factor's actual non-NaN share in the frozen panel that day. If §BD is right
the two series should be nearly the same series, per day, for all six factors -- and the two dense
controls, whose expressions contain no power, should show no such relationship with any of the three
bases.

Market data comes from the canonical Tushare cache the materializer itself used
(research_v2/artifacts/tushare_cache), read in place. That is deliberate: it is the same input the
frozen values were replayed from, so a match is evidence about the replay rather than about a
re-download.

Read-only. No database, no network, no calculator executed, nothing written into any artifact
directory, and the cache directory is only ever opened for reading.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import date, datetime
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd

ART = Path("/data/zhangyudong/qrv2_research_artifacts")
REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX")
CACHE = REPO / "research_v2" / "artifacts" / "tushare_cache"
FOLD_DATASETS = {
    "F1": ("226a8dfc497c2de6b07516711f058f26a9270675916c9e67dbf7f087a07d2538", 2022),
    "F2": ("eb9bba1b5a0bd66d0ad6c82536b295de1fc079aec125cb27bb68491bdb21a234", 2023),
    "F3": ("9766e69f342511c5bda30188df15a7b9b17d936bdc26b9b60ba244e8e7e0c9e7", 2024),
    "F4": ("6f9478d7645c8c745a73f00c03b6122e17d0c06c05cc159ece531fb00df9fb9d", 2025),
}
WORKSPACES = ("m2_v1", "m0r_v2", "m1_full_v1", "m0_v3", "m0_v2")
COL = {"yeslab_372": "f02", "yeslab_375": "f03", "yeslab_384": "f05", "yeslab_389": "f06",
       "yeslab_390": "f07", "yeslab_406": "f08", "yeslab_381": "f04", "yeslab_425": "f09",
       "yeslab_357": "f01", "yeslab_471": "f12"}
BASE_OF = {
    "yeslab_372": "pct_chg_up", "yeslab_389": "pct_chg_up", "yeslab_390": "pct_chg_up",
    "yeslab_375": "vol_up_20", "yeslab_384": "vol_up_20", "yeslab_406": "turnover_up_20",
}
LOOKBACK = 20


def read_cache_day(kind: str, day: date) -> pd.DataFrame | None:
    path = CACHE / f"{kind}_{day:%Y%m%d}.json"
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as fh:
        blob = json.load(fh)
    return pd.DataFrame(blob["data"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, default=ART)
    ap.add_argument("--out", type=Path, default=ART / "regime_mechanism")
    ap.add_argument("--warmup-from", default="2021-11-01")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # ---- 1. per-day panel cross-section and per-factor non-NaN counts -------------
    need = sorted(set(COL.values()))
    rows = []
    for fold, (ds, _year) in FOLD_DATASETS.items():
        for ws in WORKSPACES:
            paths = sorted(glob.glob(str(args.artifacts / ws / "datasets" / ds / "split=test" / "*" / "*.parquet")))
            if paths:
                break
        te = pd.concat(
            [pd.read_parquet(p, columns=["signal_date", "ts_code", "label_valid", "raw_return_5d"] + need)
             for p in paths], ignore_index=True)
        te["signal_date"] = pd.to_datetime(te["signal_date"]).dt.date
        te = te.loc[te["label_valid"].fillna(False).astype(bool)
                    & np.isfinite(pd.to_numeric(te["raw_return_5d"], errors="coerce"))]
        for day, g in te.groupby("signal_date"):
            rec = {"signal_date": day, "n": len(g), "codes": set(g["ts_code"].astype(str))}
            for name, col in COL.items():
                rec[name] = int(pd.to_numeric(g[col], errors="coerce").notna().sum())
            rows.append(rec)
        print(f"  {fold}: {te['signal_date'].nunique()} days", flush=True)
        del te
    panel = pd.DataFrame(rows).sort_values("signal_date").reset_index(drop=True)

    # ---- 2. market data from the canonical cache, with a 20-trading-day lookback ---
    import re
    # "daily_*.json" also matches "daily_basic_*.json", whose name[6:14] is not a date; take only the
    # eight-digit ones.
    day_files = sorted(m.group(1) for p in CACHE.glob("daily_*.json")
                       if (m := re.fullmatch(r"daily_(\d{8})\.json", p.name)))
    all_days = [datetime.strptime(s, "%Y%m%d").date() for s in day_files]
    start = date.fromisoformat(args.warmup_from)
    all_days = [d for d in all_days if d >= start]
    print(f"  cache days from {all_days[0]} to {all_days[-1]}: {len(all_days)}", flush=True)

    frames = []
    for i, d in enumerate(all_days):
        daily = read_cache_day("daily", d)
        basic = read_cache_day("daily_basic", d)
        if daily is None:
            continue
        keep = daily.loc[:, ["ts_code", "pct_chg", "vol"]].copy()
        if basic is not None:
            keep = keep.merge(basic.loc[:, ["ts_code", "turnover_rate"]], on="ts_code", how="left")
        else:
            keep["turnover_rate"] = np.nan
        keep["trade_date"] = d
        frames.append(keep)
        if i % 200 == 0:
            print(f"    loaded {i}/{len(all_days)}", flush=True)
    mkt = pd.concat(frames, ignore_index=True)
    mkt["ts_code"] = mkt["ts_code"].astype(str)
    for c in ("pct_chg", "vol", "turnover_rate"):
        mkt[c] = pd.to_numeric(mkt[c], errors="coerce")
    mkt = mkt.sort_values(["ts_code", "trade_date"], kind="mergesort")
    mkt["vol_20"] = mkt.groupby("ts_code")["vol"].shift(LOOKBACK)
    mkt["turnover_20"] = mkt.groupby("ts_code")["turnover_rate"].shift(LOOKBACK)
    print(f"  market rows {len(mkt)}", flush=True)

    # ---- 3. per day, over the panel's own cross-section ---------------------------
    idx = mkt.set_index(["trade_date", "ts_code"])
    out_rows = []
    for rec in panel.to_dict("records"):
        day, codes, n = rec["signal_date"], rec["codes"], rec["n"]
        # idx.loc[(day, list_of_codes)] is not valid MultiIndex fancy indexing and silently raised
        # KeyError for every day; slice the date level first, then reindex to that day's cross-section.
        try:
            day_frame = idx.xs(day, level="trade_date")
        except KeyError:
            continue
        sub = day_frame.reindex(sorted(codes))
        pct = pd.to_numeric(sub["pct_chg"], errors="coerce")
        vol = pd.to_numeric(sub["vol"], errors="coerce")
        vol20 = pd.to_numeric(sub["vol_20"], errors="coerce")
        to = pd.to_numeric(sub["turnover_rate"], errors="coerce")
        to20 = pd.to_numeric(sub["turnover_20"], errors="coerce")
        out_rows.append({
            "signal_date": day, "n_panel": n,
            "share_pct_chg_up": float((pct > 0).mean()),
            "share_vol_up_20": float((vol > vol20).mean()) if vol20.notna().any() else None,
            "share_turnover_up_20": float((to > to20).mean()) if to20.notna().any() else None,
            "share_vol_defined": float(vol20.notna().mean()),
            **{f"cov_{name}": rec[name] / n for name in COL},
        })
    if not out_rows:
        raise SystemExit(f"no day matched the market cache; panel days {len(panel)}, "
                         f"cache days {len(all_days)}")
    daily = pd.DataFrame(out_rows).sort_values("signal_date").reset_index(drop=True)
    # the first LOOKBACK days of the window have no 20-day-ago market row inside the loaded range
    daily = daily[daily["share_vol_defined"] > 0.5].reset_index(drop=True)

    result = {
        "question": (
            "Is each sparse factor's daily availability the share of its cross-section whose power base "
            "is positive, as §BD's source reading predicts?"
        ),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "read_only": True, "database_access": False, "network_access": False,
        "calculator_executed": False,
        "market_source": str(CACHE),
        "days": int(len(daily)),
        "base_assignment": BASE_OF,
        "controls_with_no_power_in_their_expression": ["yeslab_381", "yeslab_425", "yeslab_357", "yeslab_471"],
        "per_factor": {},
    }
    bases = {"pct_chg_up": "share_pct_chg_up", "vol_up_20": "share_vol_up_20",
             "turnover_up_20": "share_turnover_up_20"}
    for name in COL:
        cov = daily[f"cov_{name}"]
        entry = {"mean_coverage": float(cov.mean()), "min_coverage": float(cov.min()),
                 "max_coverage": float(cov.max()), "correlation_with_each_base": {}}
        for bkey, bcol in bases.items():
            b = daily[bcol]
            ok = b.notna()
            pearson = float(np.corrcoef(cov[ok], b[ok])[0, 1]) if ok.sum() > 2 else None
            spearman = float(pd.Series(cov[ok]).corr(pd.Series(b[ok].to_numpy()), method="spearman")) \
                if ok.sum() > 2 else None
            entry["correlation_with_each_base"][bkey] = {
                "pearson": pearson, "spearman": spearman,
                "mean_coverage_minus_mean_base": float(cov[ok].mean() - b[ok].mean()),
                "mean_absolute_day_level_gap": float((cov[ok] - b[ok]).abs().mean()),
                "share_of_days_coverage_within_0.05_of_base": float(((cov[ok] - b[ok]).abs() <= 0.05).mean()),
            }
        entry["predicted_base"] = BASE_OF.get(name)
        if entry["predicted_base"]:
            entry["verdict"] = entry["correlation_with_each_base"][entry["predicted_base"]]
        result["per_factor"][name] = entry

    best = {}
    for name, e in result["per_factor"].items():
        ranked = sorted(e["correlation_with_each_base"].items(),
                        key=lambda kv: -(kv[1]["pearson"] or -9))
        best[name] = {"best_base": ranked[0][0], "best_pearson": ranked[0][1]["pearson"],
                      "predicted_base": e["predicted_base"],
                      "prediction_matches": e["predicted_base"] == ranked[0][0]}
    result["does_the_predicted_base_win_for_every_sparse_factor"] = {
        k: v for k, v in best.items()
    }
    result["sparse_predictions_all_correct"] = all(
        v["prediction_matches"] for k, v in best.items() if k in BASE_OF
    )

    out = args.out / "regime_mechanism_direct.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    daily.to_parquet(args.out / "regime_mechanism_daily.parquet", index=False)
    print(f"wrote {out}")
    print(f"\n{'factor':14} {'predicted':16} {'best':16} {'pearson':>8} {'spearman':>9} "
          f"{'mean gap':>9} {'|gap|<=.05':>10} match")
    for name, e in result["per_factor"].items():
        b = best[name]
        v = e["correlation_with_each_base"][b["best_base"]]
        print(f"{name:14} {str(e['predicted_base']):16} {b['best_base']:16} "
              f"{v['pearson']:8.4f} {v['spearman']:9.4f} "
              f"{v['mean_coverage_minus_mean_base']:+9.4f} "
              f"{v['share_of_days_coverage_within_0.05_of_base']:10.3f} {b['prediction_matches']}")


if __name__ == "__main__":
    main()
