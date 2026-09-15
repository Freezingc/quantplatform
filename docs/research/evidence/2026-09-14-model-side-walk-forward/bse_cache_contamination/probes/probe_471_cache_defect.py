"""Is yeslab_471's 21-day divergence window caused by six corrupt rows in one cached trading day?

The diagnosis probe localised the divergence to 2022-02-25 .. 2022-03-25 inclusive -- 21 trading days,
which is exactly the length of the attention rolling window -- with clean agreement on 2022-02-24 and from
2022-03-28. A fixed-length rolling mean turns a defect in ONE input day into a defect of constant size
across exactly the following N output days and then removes it abruptly, so the shape points at one day.
Inspecting the cache found that day: daily_20220225.json carries six rows whose pct_chg is outside any
Chinese price limit, five of them BSE securities under the 920xxx code vintage that did not exist in 2022,
the worst being 920227.BJ with pre_close 0.01 and pct_chg 200400. The cross-sectional demeaning that
precedes the squaring turns that one row into a shift of the whole day's demeaned-squared returns, and
200400 squared is eleven orders of magnitude above a normal day.

Tested here, against the published factor_daily values on all 38 days of the window, are four panels: the
cache as-is, and three pre-declared cleanings -- drop 920xxx rows, drop rows beyond any price limit
(|pct_chg| > 44), and restrict to SH/SZ. None of the three is tuned to the outcome; each is a defect class
already documented elsewhere in this evidence pack. Also measured: the same four panels against the
upstream dfs://YesLab column, and a census of EVERY cached trading day for rows beyond a price limit, so
the blast radius is a count of days rather than an anecdote.

SAFETY: local caches plus SELECT-only against dfs://YesLab and dfs://QuantResearchV2Test. No writes except
--out. No weights, no repository file touched, no cache file modified.
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
CACHE = REPO / "artifacts" / "tushare_cache"
WINDOW = 21
LIMIT = 44.0          # no Chinese board allows a daily move beyond 44% (30% BSE, 20% STAR/ChiNext, 44% IPO day 1)
SHSZ = ("60", "00", "30", "68")


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
        return {c: self.name_of[l] for c, l in zip(self.codes[mask], self.l1[mask])
                if self.name_of.get(l) is not None}


def spillover_day(att, mv, lab):
    df = pd.DataFrame({"attention": att, "mv": mv, "industry": lab}).dropna(
        subset=["attention", "mv", "industry"])
    out = pd.Series(np.nan, index=att.index, dtype=float)
    for _, ind in df.groupby("industry", sort=False):
        ind = ind.sort_values("mv").copy()
        n = len(ind)
        p = (np.arange(1, n + 1, dtype=float) - 0.5) / n
        ind["bucket"] = np.where(p <= 0.3, "small", np.where(p > 0.7, "big", "mid"))
        for _, b in ind.groupby("bucket", sort=False):
            a = b["attention"]; m = len(a)
            if m > 1:
                out.loc[b.index] = (((a.sum() - a) / (m - 1)) - a).to_numpy(dtype=float)
    return out


def cmp(a, b):
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    if len(j) < 50:
        return None, None, 0
    sp = float(j["a"].corr(j["b"], method="spearman"))
    rel = float(((j["a"] - j["b"]).abs() / j["b"].abs().replace(0, np.nan)).median())
    return sp, rel, len(j)


def read_day(d):
    raw = json.loads((CACHE / f"daily_{d:%Y%m%d}.json").read_text())
    recs = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
    f = pd.DataFrame(recs); f["ts_code"] = f["ts_code"].astype(str)
    f = f.drop_duplicates("ts_code", keep="last")
    return pd.to_numeric(f.set_index("ts_code")["pct_chg"], errors="coerce")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-start", default="2021-08-01")
    ap.add_argument("--panel-end", default="2022-04-30")
    ap.add_argument("--census-start", default="2014-01-01")
    ap.add_argument("--target-start", default="2022-02-14")
    ap.add_argument("--target-end", default="2022-04-08")
    ap.add_argument("--affected-start", default="2022-02-25")
    ap.add_argument("--affected-end", default="2022-03-25")
    ap.add_argument("--skip-census", action="store_true")
    ap.add_argument("--classification", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    all_dates = sorted({pd.Timestamp(p.stem.replace("daily_", "")) for p in CACHE.glob("daily_[0-9]*.json")})
    panel_days = [d for d in all_dates if pd.Timestamp(args.panel_start) <= d <= pd.Timestamp(args.panel_end)]
    print(f"  panel {len(panel_days)} days; cache spans {all_dates[0]:%Y-%m-%d}..{all_dates[-1]:%Y-%m-%d}", flush=True)

    # ---------- part 1: census of every cached day for rows beyond a price limit ----------
    census = []
    for d in ([] if args.skip_census else all_dates):
        if d < pd.Timestamp(args.census_start):
            continue
        s = read_day(d)
        over = s[s.abs() > LIMIT]
        if len(over):
            bj920 = int(over.index.str.startswith("920").sum())
            census.append({"day": str(d.date()), "rows": int(len(s)), "rows_beyond_limit": int(len(over)),
                           "max_abs_pct_chg": float(over.abs().max()),
                           "bj_920xx_rows": bj920,
                           "codes": sorted(over.index.tolist())[:20],
                           "worst": {str(k): float(v) for k, v in over.reindex(
                               over.abs().sort_values(ascending=False).index[:5]).items()}})
        if len(census) and len(census) % 25 == 0:
            print(f"    census {d:%Y-%m-%d}: {len(census)} bad days so far", flush=True)
    print(f"  census: {len(census)} day(s) carry rows beyond |pct_chg| > {LIMIT}", flush=True)

    # ---------- part 2: four panels, 38 days, against published and upstream ----------
    per = {d: read_day(d) for d in panel_days}
    pct = pd.DataFrame(per).T.sort_index(); pct.index = pd.to_datetime(pct.index)
    mvper = {}
    for d in panel_days:
        raw = json.loads((CACHE / f"daily_basic_{d:%Y%m%d}.json").read_text())
        recs = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
        f = pd.DataFrame(recs); f["ts_code"] = f["ts_code"].astype(str)
        mvper[d] = pd.to_numeric(f.drop_duplicates("ts_code", keep="last").set_index("ts_code")["total_mv"],
                                 errors="coerce")
    mv = pd.DataFrame(mvper).T.sort_index(); mv.index = pd.to_datetime(mv.index)
    mv = mv.reindex(index=pct.index, columns=pct.columns)

    variants = {
        "cache_as_is": pct,
        "drop_920xx": pct.drop(columns=[c for c in pct.columns if c.startswith("920")]),
        "drop_beyond_limit": pct.where(pct.abs() <= LIMIT),
        "shsz_only": pct[[c for c in pct.columns if c.startswith(SHSZ)]],
    }
    attn = {k: v.sub(v.mean(axis=1, skipna=True), axis=0).pow(2).rolling(WINDOW, min_periods=WINDOW).mean()
            for k, v in variants.items()}
    dropped = {k: int(pct.shape[1] - v.shape[1]) for k, v in variants.items()}
    print(f"  columns dropped per variant: {dropped}", flush=True)

    name_of = {r["index_code"]: r["industry_name"] for r in json.loads(args.classification.read_text())}
    A = prep(pd.concat([pd.read_json(f, orient="table")
                        for f in sorted(glob.glob(str(CACHE / "pit_industry_members_*_SI.json")))],
                       ignore_index=True), "con_code", "index_code")
    LA = Labels(A, name_of)

    sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
    from walk_forward_common import TARGET_DATABASE, current_versions, new_holder, query
    h = new_holder(); fp = current_versions(h, "factor_daily")
    parts = sorted({f"{d:%Y-%m}" for d in panel_days if d >= pd.Timestamp(args.target_start)})
    vl = ", ".join("'" + fp[p] + "'" for p in parts if p in fp)
    ok, fd = query(h,
        f"select signal_date, ts_code, value from loadTable('{TARGET_DATABASE}','factor_daily') "
        f"where signal_date between {pd.Timestamp(args.target_start):%Y.%m.%d} and {pd.Timestamp(args.target_end):%Y.%m.%d} "
        f"and factor_id = 'yeslab_471' and data_version in [{vl}]")
    fd["day"] = pd.to_datetime(fd["signal_date"]).dt.normalize()
    fd["ts_code"] = fd["ts_code"].astype(str); fd["value"] = pd.to_numeric(fd["value"], errors="coerce")
    pub = {d: g.dropna(subset=["value"]).groupby("ts_code")["value"].first() for d, g in fd.groupby("day")}
    targets = sorted(pub)
    print(f"  published days {len(targets)}", flush=True)

    up = {}
    for d in targets:
        ok2, u = query(h, f"select ts_code, value from loadTable('dfs://YesLab','factor_data10') "
                          f"where factor_name = 'AttentionSpillover' and trade_date = date('{d:%Y.%m.%d}')")
        if ok2 and isinstance(u, pd.DataFrame) and len(u):
            u["ts_code"] = u["ts_code"].astype(str); u["value"] = pd.to_numeric(u["value"], errors="coerce")
            up[d] = u.dropna(subset=["value"]).groupby("ts_code")["value"].first()
    print(f"  upstream days {len(up)}", flush=True)

    rows = []
    for d in targets:
        lab = pd.Series({c: LA.on(d).get(c) for c in pct.columns}, dtype=object)
        rec = {"day": str(d.date()), "published_rows": int(len(pub[d])),
               "upstream_rows": int(len(up.get(d, [])))}
        for k in variants:
            s = spillover_day(attn[k].loc[d], mv.loc[d].reindex(attn[k].columns), lab)
            sp, rel, n = cmp(s, pub[d])
            rec[f"{k}__vs_published_spearman"] = sp
            rec[f"{k}__vs_published_median_rel"] = rel
            rec[f"{k}__pairs"] = n
            if d in up:
                spu, relu, _ = cmp(s, up[d])
                rec[f"{k}__vs_upstream_spearman"] = spu
                rec[f"{k}__vs_upstream_median_rel"] = relu
        rows.append(rec)
        print(f"  {rec['day']}  as_is={rec['cache_as_is__vs_published_spearman']:.4f} "
              f"drop920={rec['drop_920xx__vs_published_spearman']:.4f} "
              f"droplim={rec['drop_beyond_limit__vs_published_spearman']:.4f} "
              f"shsz={rec['shsz_only__vs_published_spearman']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    aff = df[(df.day >= args.affected_start) & (df.day <= args.affected_end)]
    cln = df.drop(aff.index)
    summary = {}
    for k in variants:
        col = f"{k}__vs_published_spearman"
        summary[k] = {
            "affected_days": int(len(aff)),
            "affected_min_spearman_vs_published": float(aff[col].min()),
            "affected_median_spearman_vs_published": float(aff[col].median()),
            "clean_min_spearman_vs_published": float(cln[col].min()),
            "clean_median_spearman_vs_published": float(cln[col].median()),
            "affected_median_rel_vs_published": float(aff[f"{k}__vs_published_median_rel"].median()),
            "clean_median_rel_vs_published": float(cln[f"{k}__vs_published_median_rel"].median()),
            "affected_min_spearman_vs_upstream": float(aff[f"{k}__vs_upstream_spearman"].min()),
            "clean_median_spearman_vs_upstream": float(cln[f"{k}__vs_upstream_spearman"].median()),
            "columns_dropped": dropped[k],
        }
    out = {
        "question": "Do six corrupt rows in daily_20220225.json explain yeslab_471's 21-day divergence window?",
        "attention_window_days": WINDOW, "price_limit_used": LIMIT,
        "affected_window": [args.affected_start, args.affected_end],
        "target_window": [args.target_start, args.target_end],
        "cache_census_beyond_price_limit": {
            "days_scanned": int(sum(1 for d in all_dates if d >= pd.Timestamp(args.census_start))),
            "cache_span": [str(all_dates[0].date()), str(all_dates[-1].date())],
            "days_with_rows_beyond_limit": len(census),
            "days": census,
        },
        "summary_by_panel_variant": summary,
        "per_day": rows,
        "database_writes": False, "weights_written": False, "cache_writes": False,
    }
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n=== cache census: days with rows beyond |pct_chg| > %s ===" % LIMIT)
    for c in census:
        print(f"  {c['day']}  rows={c['rows_beyond_limit']:3d} max={c['max_abs_pct_chg']:12.1f} "
              f"bj920={c['bj_920xx_rows']:3d} worst={list(c['worst'].items())[:2]}")
    print("\n=== summary by panel variant (vs published) ===")
    print(pd.DataFrame(summary).T.to_string())


if __name__ == "__main__":
    main()
