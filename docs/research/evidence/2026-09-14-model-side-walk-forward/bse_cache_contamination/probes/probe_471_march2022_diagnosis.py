"""Why does the yeslab_471 transcription match the published value on 399 of 420 pre-cutover days but not on 21, all in March 2022?

The pre-cutover census found yeslab_357's transcription exact on every day (minimum Spearman
0.9999999991) while yeslab_471 fell to 0.5222 on 21 days, all of them between 2022-03-01 and 2022-03-25.
Same panel, same labels, same days -- so the panel and the labels are not the cause. total_mv is fully
populated in the daily_basic cache for those dates, and the cache files all share one mtime, so neither a
partial cache nor a re-fetch explains it either.

Two hypotheses are testable without guessing. A DATE SHIFT: the published value for day d might correspond
to a recomputation anchored on a different day, which would show up as a neighbouring day matching better
than the same day. An ONSET: something changed in the input on a specific date, which would show up as the
agreement being high before that date and low after, with no neighbouring day matching better. This
computes 471 for every day in a window around March 2022 and cross-correlates each computed day against
the published values of that day and its neighbours.

SAFETY: local caches plus SELECT-only against dfs://QuantResearchV2Test. No writes except --out.
"""
from __future__ import annotations
import argparse, glob, json, sys
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
CACHE = REPO / "artifacts" / "tushare_cache"


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
    return out, len(df), int(df["attention"].notna().sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-start", default="2021-08-01")
    ap.add_argument("--panel-end", default="2022-04-30")
    ap.add_argument("--target-start", default="2022-02-14")
    ap.add_argument("--target-end", default="2022-04-08")
    ap.add_argument("--classification", type=Path, required=True)
    ap.add_argument("--shift", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    all_dates = sorted({pd.Timestamp(p.stem.replace("daily_", "")) for p in CACHE.glob("daily_[0-9]*.json")})
    window = [d for d in all_dates if pd.Timestamp(args.panel_start) <= d <= pd.Timestamp(args.panel_end)]
    targets = [d for d in all_dates if pd.Timestamp(args.target_start) <= d <= pd.Timestamp(args.target_end)]
    print(f"  panel {len(window)} days, targets {len(targets)} days", flush=True)

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
    codes = list(pct.columns)
    print(f"  panel {pct.shape[0]} x {pct.shape[1]}", flush=True)

    name_of = {r["index_code"]: r["industry_name"] for r in json.loads(args.classification.read_text())}
    A = prep(pd.concat([pd.read_json(f, orient="table")
                        for f in sorted(glob.glob(str(CACHE / "pit_industry_members_*_SI.json")))],
                       ignore_index=True), "con_code", "index_code")
    LA = Labels(A, name_of)

    attention = pct.sub(pct.mean(axis=1, skipna=True), axis=0).pow(2).rolling(window=21, min_periods=21).mean()

    mine = {}
    for d in targets:
        lab = pd.Series({c: LA.on(d).get(c) for c in codes}, dtype=object)
        s, grouped, att_ok = spillover_day(attention.loc[d], mv.loc[d], lab)
        mine[d] = {"series": s, "grouped": grouped, "attention_finite": att_ok}
    print(f"  computed {len(mine)} days", flush=True)

    sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
    from walk_forward_common import TARGET_DATABASE, current_versions, new_holder, query
    h = new_holder(); fp = current_versions(h, "factor_daily")
    parts = sorted({f"{d:%Y-%m}" for d in targets})
    vers = sorted({fp[p] for p in parts if p in fp})
    vl = ", ".join("'" + v + "'" for v in vers)
    ok, fd = query(h,
        f"select signal_date, ts_code, value from loadTable('{TARGET_DATABASE}','factor_daily') "
        f"where signal_date between {targets[0]:%Y.%m.%d} and {targets[-1]:%Y.%m.%d} "
        f"and factor_id = 'yeslab_471' and data_version in [{vl}]")
    fd["day"] = pd.to_datetime(fd["signal_date"]).dt.normalize()
    fd["ts_code"] = fd["ts_code"].astype(str); fd["value"] = pd.to_numeric(fd["value"], errors="coerce")
    pub = {d: g.dropna(subset=["value"]).groupby("ts_code")["value"].first() for d, g in fd.groupby("day")}
    print(f"  published days {len(pub)}", flush=True)

    tlist = sorted(pub)
    rows = []
    for d in tlist:
        best = None
        for k in range(-args.shift, args.shift + 1):
            i = tlist.index(d) + k
            if not (0 <= i < len(tlist)):
                continue
            other = tlist[i]
            j = pd.concat([mine[other]["series"].rename("mine"), pub[d].rename("pub")], axis=1, join="inner").dropna()
            if len(j) < 50:
                continue
            sp = float(j["mine"].corr(j["pub"], method="spearman"))
            if best is None or sp > best["spearman"]:
                best = {"best_matching_computed_day": str(other.date()), "spearman": sp, "offset_days": k,
                        "pairs": int(len(j))}
        same = pd.concat([mine[d]["series"].rename("mine"), pub[d].rename("pub")], axis=1, join="inner").dropna()
        rows.append({
            "published_day": str(d.date()), "published_rows": int(len(pub[d])),
            "computed_grouped": mine[d]["grouped"], "computed_attention_finite": mine[d]["attention_finite"],
            "same_day_spearman": float(same["mine"].corr(same["pub"], method="spearman")) if len(same) > 50 else None,
            "same_day_pairs": int(len(same)),
            "same_day_median_rel": float(((same["mine"] - same["pub"]).abs() /
                                          same["pub"].abs().replace(0, np.nan)).median()) if len(same) > 50 else None,
            "best_shift": best,
            "shift_beats_same_day": bool(best and best["offset_days"] != 0 and same is not None and
                                         best["spearman"] > (float(same["mine"].corr(same["pub"], method="spearman"))
                                                             if len(same) > 50 else -2) + 0.01),
        })

    df = pd.DataFrame(rows)
    bad = df[df["same_day_spearman"].fillna(1) < 0.99]
    out = {
        "question": "Why does the yeslab_471 transcription fail on 21 days, all in March 2022?",
        "window": [args.target_start, args.target_end], "days": len(df),
        "days_below_0_99": int(len(bad)),
        "first_day_below_0_99": str(bad["published_day"].iloc[0]) if len(bad) else None,
        "last_day_below_0_99": str(bad["published_day"].iloc[-1]) if len(bad) else None,
        "date_shift_hypothesis": {
            "days_where_a_neighbouring_computed_day_matches_better": int(df["shift_beats_same_day"].sum()),
            "verdict": None,
        },
        "onset_pattern": {
            "spearman_by_day": {r["published_day"]: r["same_day_spearman"] for r in rows},
        },
        "inputs_on_the_affected_days": {
            "computed_grouped_range": [int(df.loc[df.published_day.isin(bad.published_day), "computed_grouped"].min()),
                                       int(df.loc[df.published_day.isin(bad.published_day), "computed_grouped"].max())] if len(bad) else None,
            "attention_finite_range": [int(df.loc[df.published_day.isin(bad.published_day), "computed_attention_finite"].min()),
                                       int(df.loc[df.published_day.isin(bad.published_day), "computed_attention_finite"].max())] if len(bad) else None,
            "published_rows_range": [int(df.loc[df.published_day.isin(bad.published_day), "published_rows"].min()),
                                     int(df.loc[df.published_day.isin(bad.published_day), "published_rows"].max())] if len(bad) else None,
        },
        "per_day": rows, "database_writes": False, "weights_written": False,
    }
    n_shift = out["date_shift_hypothesis"]["days_where_a_neighbouring_computed_day_matches_better"]
    out["date_shift_hypothesis"]["verdict"] = (
        "REFUTED -- no neighbouring computed day matches better, so this is not a date offset"
        if n_shift == 0 else f"POSSIBLE -- {n_shift} day(s) match a neighbouring computed day better")
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_day", "onset_pattern")}, ensure_ascii=False, indent=1))
    print(f"\n{'published_day':14}{'pub_rows':>9}{'grouped':>9}{'att_ok':>8}{'same_sp':>9}{'best_day':>13}{'best_sp':>9}{'off':>5}")
    for r in rows:
        b = r["best_shift"] or {}
        print(f"{r['published_day']:14}{r['published_rows']:9d}{r['computed_grouped']:9d}{r['computed_attention_finite']:8d}"
              f"{(r['same_day_spearman'] or 0):9.4f}{str(b.get('best_matching_computed_day','-')):>13}"
              f"{(b.get('spearman') or 0):9.4f}{b.get('offset_days',0):5d}")


if __name__ == "__main__":
    main()
