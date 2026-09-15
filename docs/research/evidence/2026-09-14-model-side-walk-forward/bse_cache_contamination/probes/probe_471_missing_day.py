"""Why does the yeslab_471 transcription diverge from the published value on exactly 21 trading days?

The diagnosis probe settled two things. The divergence is NOT a date offset (no neighbouring computed day
matches better), and it has a sharp onset and a sharp recovery: Spearman against the published value is
0.99995 through 2022-02-24, falls to 0.52-0.58 from 2022-02-25 through 2022-03-25, and is back to 0.99995
from 2022-03-28. The row counts do not move (published 3024-3061, computed groups 4210-4237 on the affected
days, the same as on the clean days around them), so the universe is not the cause either.

Twenty-one affected trading days is the length of the attention rolling window
(demeaned pct_chg squared, rolling(21, min_periods=21).mean()). A rolling mean of fixed length turns a
defect in ONE input day into a defect of CONSTANT size across exactly the following 21 output days, and
then removes it abruptly -- which is the observed shape. A warm start with a relaxed min_periods would
instead produce a gradually improving agreement, which is not what is observed. So two mechanisms are
tested against the published values, over every candidate day in the panel:

  MISSING_DAY   the pipeline that produced the published value did not have trading day d0 in its panel,
                so its 21-day window is the compacted 21 days that skip d0
  WARM_START    the pipeline that produced the published value started its panel at day S and used
                min_periods=1, so the first 20 days after S use a truncated window

Also asked here, because it decides who owns the defect: are the published factor_daily rows for those
days simply a copy of the upstream dfs://YesLab AttentionSpillover column? If they are, the divergence
lives upstream of the materializer and no repository code produced it.

SAFETY: local caches plus SELECT-only against dfs://YesLab and dfs://QuantResearchV2Test. No writes
except --out. No weights, no repository file touched.
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
CACHE = REPO / "artifacts" / "tushare_cache"
WINDOW = 21


def prep(m, code, l1):
    m = m.rename(columns={code: "code", l1: "l1"})
    m["in_p"] = pd.to_datetime(m["in_date"].astype(str), format="%Y%m%d", errors="coerce")
    o = m["out_date"]
    m["out_p"] = pd.to_datetime(o.astype(str).str.replace(r"\.0$", "", regex=True), format="%Y%m%d", errors="coerce")
    m.loc[o.isna() | o.astype(str).isin(["None", "nan", "", "NaT"]), "out_p"] = pd.NaT
    return m.dropna(subset=["in_p"])[["code", "l1", "in_p", "out_p"]]


class Labels:
    def __init__(self, m, name_of):
        self.codes = m["code"].to_numpy(); self.l1 = m["l1"].to_numpy()
        self.ins = m["in_p"].to_numpy(); self.outs = m["out_p"].to_numpy()
        self.open_ended = pd.isna(m["out_p"]).to_numpy()
        order = np.lexsort((self.l1, self.ins, self.codes))
        for a in ("codes", "l1", "ins", "outs", "open_ended"):
            setattr(self, a, getattr(self, a)[order])
        self.name_of = name_of

    def on(self, day):
        dv = np.datetime64(day)
        mask = (self.ins <= dv) & (self.open_ended | (self.outs > dv))
        out = {}
        for c, l in zip(self.codes[mask], self.l1[mask]):
            n = self.name_of.get(l)
            if n is not None:
                out[c] = n
        return out


def spillover_day(att, mv, lab):
    df = pd.DataFrame({"attention": att, "mv": mv, "industry": lab}).dropna(
        subset=["attention", "mv", "industry"])
    out = pd.Series(np.nan, index=att.index, dtype=float)
    for _, ind in df.groupby("industry", sort=False):
        ind = ind.sort_values("mv").copy()
        n = len(ind)
        pct = (np.arange(1, n + 1, dtype=float) - 0.5) / n
        ind["bucket"] = np.where(pct <= 0.3, "small", np.where(pct > 0.7, "big", "mid"))
        for _, b in ind.groupby("bucket", sort=False):
            a = b["attention"]; m = len(a)
            if m > 1:
                out.loc[b.index] = (((a.sum() - a) / (m - 1)) - a).to_numpy(dtype=float)
    return out


def spearman(a, b):
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    if len(j) < 50:
        return None, len(j)
    return float(j["a"].corr(j["b"], method="spearman")), len(j)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-start", default="2021-08-01")
    ap.add_argument("--panel-end", default="2022-04-30")
    ap.add_argument("--target-start", default="2022-02-14")
    ap.add_argument("--target-end", default="2022-04-08")
    ap.add_argument("--candidate-start", default="2021-12-01")
    ap.add_argument("--classification", type=Path, required=True)
    ap.add_argument("--scan-days", default="2022-02-24,2022-02-25,2022-03-04,2022-03-15,2022-03-25,2022-03-28")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    all_dates = sorted({pd.Timestamp(p.stem.replace("daily_", "")) for p in CACHE.glob("daily_[0-9]*.json")})
    window = [d for d in all_dates if pd.Timestamp(args.panel_start) <= d <= pd.Timestamp(args.panel_end)]
    targets = [d for d in all_dates if pd.Timestamp(args.target_start) <= d <= pd.Timestamp(args.target_end)]
    cands = [d for d in window if d >= pd.Timestamp(args.candidate_start)]
    print(f"  panel {len(window)} days, targets {len(targets)}, candidates {len(cands)}", flush=True)

    def wide(field, fname):
        per = {}
        for d in window:
            raw = json.loads((CACHE / f"{fname}_{d:%Y%m%d}.json").read_text())
            recs = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
            f = pd.DataFrame(recs); f["ts_code"] = f["ts_code"].astype(str)
            per[d] = pd.to_numeric(f.drop_duplicates("ts_code", keep="last").set_index("ts_code")[field],
                                   errors="coerce")
        w = pd.DataFrame(per).T.sort_index(); w.index = pd.to_datetime(w.index); return w

    pct = wide("pct_chg", "daily"); mv = wide("total_mv", "daily_basic")
    mv = mv.reindex(index=pct.index, columns=pct.columns)
    dsq = pct.sub(pct.mean(axis=1, skipna=True), axis=0).pow(2)
    print(f"  panel {pct.shape[0]} x {pct.shape[1]}", flush=True)

    name_of = {r["index_code"]: r["industry_name"] for r in json.loads(args.classification.read_text())}
    A = prep(pd.concat([pd.read_json(f, orient="table")
                        for f in sorted(glob.glob(str(CACHE / "pit_industry_members_*_SI.json")))],
                       ignore_index=True), "con_code", "index_code")
    LA = Labels(A, name_of)
    lab_cache = {d: pd.Series({c: LA.on(d).get(c) for c in pct.columns}, dtype=object) for d in targets}

    sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
    from walk_forward_common import TARGET_DATABASE, current_versions, new_holder, query
    h = new_holder(); fp = current_versions(h, "factor_daily")
    parts = sorted({f"{d:%Y-%m}" for d in targets})
    vl = ", ".join("'" + fp[p] + "'" for p in parts if p in fp)
    ok, fd = query(h,
        f"select signal_date, ts_code, value from loadTable('{TARGET_DATABASE}','factor_daily') "
        f"where signal_date between {targets[0]:%Y.%m.%d} and {targets[-1]:%Y.%m.%d} "
        f"and factor_id = 'yeslab_471' and data_version in [{vl}]")
    fd["day"] = pd.to_datetime(fd["signal_date"]).dt.normalize()
    fd["ts_code"] = fd["ts_code"].astype(str); fd["value"] = pd.to_numeric(fd["value"], errors="coerce")
    pub = {d: g.dropna(subset=["value"]).groupby("ts_code")["value"].first() for d, g in fd.groupby("day")}
    print(f"  published days {len(pub)}", flush=True)

    up_days = {}
    for d in targets:
        ok2, up = query(h,
            f"select ts_code, value from loadTable('dfs://YesLab','factor_data10') "
            f"where factor_name = 'AttentionSpillover' and trade_date = date('{d:%Y.%m.%d}')")
        if ok2 and isinstance(up, pd.DataFrame) and len(up):
            up["ts_code"] = up["ts_code"].astype(str); up["value"] = pd.to_numeric(up["value"], errors="coerce")
            up_days[d] = up.dropna(subset=["value"]).groupby("ts_code")["value"].first()
    print(f"  upstream days {len(up_days)}", flush=True)

    # --- who owns the published value? published vs upstream, per day
    pvu = []
    for d in sorted(pub):
        if d not in up_days:
            continue
        j = pd.concat([pub[d].rename("pub"), up_days[d].rename("up")], axis=1, join="inner").dropna()
        sp, n = spearman(pub[d], up_days[d])
        rel = float(((j["pub"] - j["up"]).abs() / j["up"].abs().replace(0, np.nan)).median()) if n else None
        exact = float((np.isclose(j["pub"], j["up"], rtol=0, atol=0)).mean()) if n else None
        pvu.append({"day": str(d.date()), "pairs": n, "spearman": sp, "median_rel_diff": rel,
                    "bitwise_equal_fraction": exact})
    pvu_df = pd.DataFrame(pvu)

    def variant_series(kind, anchor, day):
        if kind == "baseline":
            att = dsq.rolling(WINDOW, min_periods=WINDOW).mean()
        elif kind == "missing_day":
            att = dsq.drop(index=[anchor]).rolling(WINDOW, min_periods=WINDOW).mean()
        elif kind == "warm_start":
            att = dsq.loc[dsq.index >= anchor].rolling(WINDOW, min_periods=1).mean().reindex(dsq.index)
        else:
            raise ValueError(kind)
        if day not in att.index:
            return None
        return spillover_day(att.loc[day], mv.loc[day], lab_cache[day])

    scan = [pd.Timestamp(x) for x in args.scan_days.split(",")]
    scan = [d for d in scan if d in pub]
    CONTROL_DAYS = {"2022-02-24", "2022-03-28"}   # the clean days either side of the window
    rows = []
    for kind in ("baseline", "missing_day", "warm_start"):
        anchors = [None] if kind == "baseline" else cands
        for a in anchors:
            per_day = {}
            for d in scan:
                s = variant_series(kind, a, d)
                per_day[str(d.date())] = (spearman(s, pub[d])[0] if s is not None else None)
            aff = [v for k, v in per_day.items() if v is not None and k not in CONTROL_DAYS]
            mean_aff = float(np.mean(aff)) if aff else float("nan")
            min_aff = float(np.min(aff)) if aff else float("nan")
            rows.append({"mechanism": kind, "anchor": (str(a.date()) if a is not None else None),
                         "spearman_by_scan_day": per_day,
                         "min_over_affected_scan_days": min_aff,
                         "mean_over_affected_scan_days": mean_aff})
            print(f"  {kind:12} {str(a.date()) if a is not None else '-':12} "
                  f"mean_affected={mean_aff:.4f}", flush=True)

    best = max([r for r in rows if r["mechanism"] != "baseline"],
               key=lambda r: (-1e9 if np.isnan(r["mean_over_affected_scan_days"])
                              else r["mean_over_affected_scan_days"]))
    base = next(r for r in rows if r["mechanism"] == "baseline")

    out = {
        "question": "What mechanism produces yeslab_471's 21-day divergence window in Feb-Mar 2022, and who owns it?",
        "window": [args.target_start, args.target_end],
        "attention_window_days": WINDOW,
        "affected_days": 21,
        "published_vs_upstream": {
            "days": len(pvu_df),
            "min_spearman": float(pvu_df["spearman"].min()) if len(pvu_df) else None,
            "median_rel_diff_range": [float(pvu_df["median_rel_diff"].min()), float(pvu_df["median_rel_diff"].max())] if len(pvu_df) else None,
            "bitwise_equal_fraction_range": [float(pvu_df["bitwise_equal_fraction"].min()), float(pvu_df["bitwise_equal_fraction"].max())] if len(pvu_df) else None,
            "verdict": None,
            "per_day": pvu,
        },
        "mechanism_scan": {
            "baseline": base,
            "best_non_baseline": best,
            "all_rows": sorted(rows, key=lambda r: (-1e9 if np.isnan(r["mean_over_affected_scan_days"])
                                                    else -r["mean_over_affected_scan_days"]))[:12],
            "scan_days": [str(d.date()) for d in scan],
        },
        "database_writes": False, "weights_written": False,
    }
    eq = out["published_vs_upstream"]["bitwise_equal_fraction_range"]
    out["published_vs_upstream"]["verdict"] = (
        "published IS the upstream column (bitwise equal on every day in the window) -- the divergence is upstream's, not the materializer's"
        if eq and eq[0] > 0.999 else
        "published is NOT a bitwise copy of upstream -- the materializer transformed it" if eq else "not measured")
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))

    print("\n=== published vs upstream ===")
    print(pvu_df.to_string(index=False))
    print("\n=== best mechanisms (mean Spearman over affected scan days) ===")
    for r in out["mechanism_scan"]["all_rows"]:
        print(f"{r['mechanism']:12} {str(r['anchor']):12} {r['mean_over_affected_scan_days']:.4f}  "
              + " ".join(f"{k[5:]}={v:.4f}" for k, v in r["spearman_by_scan_day"].items() if v is not None))
    print("\nverdict:", out["published_vs_upstream"]["verdict"])


if __name__ == "__main__":
    main()
