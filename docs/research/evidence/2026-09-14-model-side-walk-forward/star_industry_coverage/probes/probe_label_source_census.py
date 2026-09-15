"""Census, not sample: the label-source sensitivity on EVERY post-cutover trading day in the cache.

Section 17.10 measured 24 of 24 problem days but only 27 of 549 post-cutover days, and said so: the
"repair lands" half of the recommendation is census-grade while the "clean days barely move" half is
sample-grade. This closes that asymmetry by running every cached post-cutover trading day.

It is affordable because the expensive parts are built once rather than per day: the market panel is loaded
once for the whole window plus warm-up, both label matrices are built once for every date, and the
published 357/471 values are fetched in a handful of chunked queries instead of one per day.

Source A is validated rather than assumed. Section 17.8 found the provider and a direct read of the cache
agreeing on 5,088 of 5,088 shared securities on one day; this checks that on several days and REFUSES to
run the census if they disagree, because the census uses the fast direct read for all 594 dates.

SAFETY: reads the local caches and repository source, plus SELECT-only queries against
dfs://QuantResearchV2Test. No writes except --out. Nothing is written into research_v2/artifacts.
No weights.
"""
from __future__ import annotations
import argparse, glob, json, sys
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
CACHE = REPO / "artifacts" / "tushare_cache"
STAR = ("688", "689")


def prep(m, code, l1):
    m = m.rename(columns={code: "code", l1: "l1"})
    m["in_p"] = pd.to_datetime(m["in_date"].astype(str), format="%Y%m%d", errors="coerce")
    o = m["out_date"]
    m["out_p"] = pd.to_datetime(o.astype(str).str.replace(r"\.0$", "", regex=True), format="%Y%m%d", errors="coerce")
    m.loc[o.isna() | o.astype(str).isin(["None", "nan", "", "NaT"]), "out_p"] = pd.NaT
    return m.dropna(subset=["in_p"])[["code", "l1", "in_p", "out_p"]]


class Labels:
    """Active label per security per day, with the provider's own tie-break, resolved in numpy."""

    def __init__(self, m: pd.DataFrame, name_of: dict, drop_unmapped: bool = True):
        self.codes = m["code"].to_numpy()
        self.l1 = m["l1"].to_numpy()
        self.ins = m["in_p"].to_numpy()
        self.outs = m["out_p"].to_numpy()
        self.open_ended = pd.isna(m["out_p"]).to_numpy()
        # build_pit_snapshot sorts by (con_code, in_date, index_code) and keeps the LAST row, so the
        # tie-break is latest in_date then largest index_code. Sorting the whole table that way once and
        # letting a later assignment overwrite an earlier one reproduces it exactly.
        order = np.lexsort((self.l1, self.ins, self.codes))
        self.codes, self.l1 = self.codes[order], self.l1[order]
        self.ins, self.outs, self.open_ended = self.ins[order], self.outs[order], self.open_ended[order]
        self.name_of = name_of
        self.drop_unmapped = drop_unmapped

    def on(self, day) -> dict:
        dv = np.datetime64(day)
        mask = (self.ins <= dv) & (self.open_ended | (self.outs > dv))
        out = {}
        for c, l in zip(self.codes[mask], self.l1[mask]):
            name = self.name_of.get(l)
            if name is None:
                if self.drop_unmapped:
                    continue          # build_pit_snapshot merges classifications with how="inner"
                name = l
            out[c] = name             # later (larger in_date, larger index_code) wins, matching keep="last"
        return out


def industry_zscore_abs(ret_row, labels_row):
    valid = ret_row.notna() & labels_row.notna()
    z = pd.Series(np.nan, index=ret_row.index, dtype=float)
    if not valid.any():
        return z
    g = ret_row[valid].groupby(labels_row[valid], sort=False)
    z.loc[valid] = ((ret_row[valid] - g.transform("mean")) / g.transform("std")).abs().to_numpy(dtype=float)
    return z.replace([np.inf, -np.inf], np.nan)


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
            if m <= 1:
                continue
            out.loc[b.index] = (((a.sum() - a) / (m - 1)) - a).to_numpy(dtype=float)
    return out, len(df)


def compare(a: pd.Series, b: pd.Series) -> dict:
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    if len(j) < 20 or j["a"].std() == 0 or j["b"].std() == 0:
        return {"pairs": int(len(j))}
    d = (j["a"] - j["b"]).abs()
    rel = (d / j["b"].abs().replace(0, np.nan)).dropna()
    return {"pairs": int(len(j)), "changed": int((d > 0).sum()), "changed_rate": float((d > 0).mean()),
            "spearman": float(j["a"].corr(j["b"], method="spearman")),
            "median_rel": float(rel.median()) if len(rel) else None,
            "p95_rel": float(rel.quantile(0.95)) if len(rel) else None,
            "max_abs": float(d.max())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-09-25")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--warmup-trading-days", type=int, default=45)
    ap.add_argument("--member-all", type=Path, required=True)
    ap.add_argument("--classification", type=Path, required=True)
    ap.add_argument("--validate-days", nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    all_dates = sorted({pd.Timestamp(p.stem.replace("daily_", "")) for p in CACHE.glob("daily_[0-9]*.json")})
    targets = [d for d in all_dates if pd.Timestamp(args.start) <= d <= pd.Timestamp(args.end)]
    first_idx = all_dates.index(targets[0])
    if first_idx < args.warmup_trading_days:
        raise SystemExit("not enough cached warm-up before the window")
    window = all_dates[first_idx - args.warmup_trading_days: all_dates.index(targets[-1]) + 1]
    print(f"  targets {len(targets)} days ({targets[0].date()}..{targets[-1].date()}), "
          f"panel window {len(window)} days", flush=True)

    def wide(field, fname):
        per = {}
        for d in window:
            raw = json.loads((CACHE / f"{fname}_{d:%Y%m%d}.json").read_text())
            recs = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
            f = pd.DataFrame(recs); f["ts_code"] = f["ts_code"].astype(str)
            per[d] = pd.to_numeric(f.drop_duplicates("ts_code", keep="last").set_index("ts_code")[field],
                                   errors="coerce")
        w = pd.DataFrame(per).T.sort_index()
        w.index = pd.to_datetime(w.index)
        return w

    close = wide("close", "daily"); pct = wide("pct_chg", "daily"); mv = wide("total_mv", "daily_basic")
    mv = mv.reindex(index=pct.index, columns=pct.columns)
    codes = list(pct.columns)
    print(f"  panel {pct.shape[0]} x {pct.shape[1]}", flush=True)

    name_of = {r["index_code"]: r["industry_name"] for r in json.loads(args.classification.read_text())}
    A = prep(pd.concat([pd.read_json(f, orient="table")
                        for f in sorted(glob.glob(str(CACHE / "pit_industry_members_*_SI.json")))],
                       ignore_index=True), "con_code", "index_code")
    B = prep(pd.DataFrame(json.loads(args.member_all.read_text())), "ts_code", "l1_code")
    LA, LBm = Labels(A, name_of), Labels(B, name_of)

    # ---- validate the fast label reader against the real provider before trusting 594 dates of it ----
    vdays = [pd.Timestamp(d) for d in args.validate_days] or [targets[0], targets[len(targets) // 2], targets[-1]]
    sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
    from audit_tushare_pit_industry import VersionedPITIndustryProvider
    provider = VersionedPITIndustryProvider.from_cache(CACHE, cutover=date(2021, 12, 10))
    validation = []
    for d in vdays:
        prov_row = provider.get_industry_labels([d.date()], codes).iloc[0]
        fast = LA.on(d)
        prov = {c: v for c, v in prov_row.items() if pd.notna(v)}
        same = sum(1 for c in prov if fast.get(c) == prov[c])
        validation.append({"day": str(d.date()), "provider_labeled": len(prov), "fast_labeled": len(fast),
                           "identical": same, "identical_rate": same / len(prov) if prov else None,
                           "only_provider": len(set(prov) - set(fast)), "only_fast": len(set(fast) - set(prov))})
        print(f"  validate {d.date()}: provider {len(prov)} vs fast {len(fast)}, identical {same}", flush=True)
    if any(v["identical_rate"] is None or v["identical_rate"] < 1.0 for v in validation):
        raise SystemExit(f"fast label reader disagrees with the provider; refusing to run the census: {validation}")

    # ---- label matrices and factor values, both scopes ----
    ret = close / close.shift(1) - 1
    cross_mean = pct.mean(axis=1, skipna=True)
    attention = pct.sub(cross_mean, axis=0).pow(2).rolling(window=21, min_periods=21).mean()

    scopes = ("shsz", "all")
    per_day = {s: [] for s in scopes}
    for i, d in enumerate(targets):
        la = LA.on(d)
        mb = LBm.on(d)
        # The UNION rule, identical to probe_label_source_sensitivity.py: member_all wins where both
        # sources speak, and fills where the cache is silent -- but under the shsz scope it may only FILL
        # for non-BJ securities, leaving BSE codes exactly as the pipeline in force leaves them. Building
        # this as member_all alone would be a REPLACE, which section 13 showed loses labels, and an
        # earlier version of this census made exactly that mistake: it reported negative coverage gains
        # (min -88) and a negative median groupable gain, which is the replace behaviour, not the union's.
        def union(scope: str) -> dict:
            merged = dict(la)
            merged.update({k: v for k, v in mb.items() if k in la})
            fills = {k: v for k, v in mb.items() if k not in la}
            if scope == "shsz":
                fills = {k: v for k, v in fills.items() if not k.endswith(".BJ")}
            merged.update(fills)
            return merged

        labmats = {"shsz": union("shsz"), "all": union("all")}
        replace_only = mb
        ret_row = ret.loc[d]
        z_A = industry_zscore_abs(ret_row, pd.Series({c: la.get(c) for c in codes}, dtype=object))
        att_row, mv_row = attention.loc[d], mv.loc[d]
        s_A, gA = spillover_day(att_row, mv_row, pd.Series({c: la.get(c) for c in codes}, dtype=object))
        for s in scopes:
            lm = pd.Series({c: labmats[s].get(c) for c in codes}, dtype=object)
            z_B = industry_zscore_abs(ret_row, lm)
            s_B, gB = spillover_day(att_row, mv_row, lm)
            diff = sum(1 for c in codes if (la.get(c) or "~") != (labmats[s].get(c) or "~"))
            per_day[s].append({
                "day": str(d.date()), "labels_differing": diff,
                "coverage_replace_only_member_all": sum(1 for c in codes if replace_only.get(c)),
                "coverage_A": sum(1 for c in codes if la.get(c)), "coverage_B": sum(1 for c in codes if labmats[s].get(c)),
                "grouped_471_A": gA, "grouped_471_B": gB,
                "f357": compare(z_A, z_B), "f471": compare(s_A, s_B),
            })
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(targets)} days", flush=True)

    # ---- published values, fetched in chunks, for the transcription check ----
    from walk_forward_common import TARGET_DATABASE, current_versions, new_holder, query
    holder = new_holder(); fpins = current_versions(holder, "factor_daily")
    pub = {"yeslab_357": {}, "yeslab_471": {}}
    for year in sorted({d.year for d in targets}):
        yd = [d for d in targets if d.year == year]
        parts = sorted({f"{d:%Y-%m}" for d in yd})
        miss = [p for p in parts if p not in fpins]
        if miss:
            print(f"  WARNING unpublished partitions for {year}: {miss[:4]}", flush=True); continue
        versions = sorted({fpins[p] for p in parts})
        vl = ", ".join("'" + v + "'" for v in versions)
        lo, hi = yd[0], yd[-1]
        ok, fd = query(holder,
            f"select signal_date, ts_code, factor_id, value from loadTable('{TARGET_DATABASE}','factor_daily') "
            f"where signal_date between {lo:%Y.%m.%d} and {hi:%Y.%m.%d} "
            f"and factor_id in ['yeslab_357','yeslab_471'] and data_version in [{vl}]")
        if not ok or not isinstance(fd, pd.DataFrame):
            print(f"  WARNING published fetch failed for {year}", flush=True); continue
        fd["day"] = pd.to_datetime(fd["signal_date"]).dt.strftime("%Y-%m-%d")
        fd["ts_code"] = fd["ts_code"].astype(str); fd["value"] = pd.to_numeric(fd["value"], errors="coerce")
        for fid, g in fd.dropna(subset=["value"]).groupby("factor_id"):
            for (day, code), v in g.groupby(["day", "ts_code"])["value"].first().items():
                pub[fid].setdefault(day, {})[code] = float(v)
        print(f"  published {year}: {len(yd)} days", flush=True)

    trans = []
    for i, d in enumerate(targets):
        ds = str(d.date())
        if ds not in pub["yeslab_357"]:
            continue
        la = LA.on(d)
        lm = pd.Series({c: la.get(c) for c in codes}, dtype=object)
        z_A = industry_zscore_abs(ret.loc[d], lm)
        att_row, mv_row = attention.loc[d], mv.loc[d]
        s_A, _ = spillover_day(att_row, mv_row, lm)
        c357 = compare(z_A, pd.Series(pub["yeslab_357"][ds]))
        c471 = compare(s_A, pd.Series(pub["yeslab_471"].get(ds, {})))
        trans.append({"day": ds, "f357_spearman": c357.get("spearman"), "f471_spearman": c471.get("spearman"),
                      "f357_pairs": c357.get("pairs"), "f471_pairs": c471.get("pairs")})

    def agg(rows, key):
        vals = [r[key] for r in rows if r[key] is not None]
        return {"min": min(vals), "median": sorted(vals)[len(vals) // 2], "max": max(vals), "n": len(vals)} if vals else None

    summary = {}
    for s in scopes:
        rows = per_day[s]
        summary[s] = {
            "days": len(rows),
            "labels_differing": agg(rows, "labels_differing"),
            "coverage_gain": agg([r["coverage_B"] - r["coverage_A"] for r in rows] and
                                 [{"v": r["coverage_B"] - r["coverage_A"]} for r in rows], "v"),
            "grouped_471_gain": agg([{"v": r["grouped_471_B"] - r["grouped_471_A"]} for r in rows], "v"),
            "f357_changed_rate": agg([r["f357"] for r in rows if "changed_rate" in r["f357"]], "changed_rate"),
            "f357_spearman": agg([r["f357"] for r in rows if "spearman" in r["f357"]], "spearman"),
            "f357_median_rel": agg([r["f357"] for r in rows if r["f357"].get("median_rel") is not None], "median_rel"),
            "f357_p95_rel": agg([r["f357"] for r in rows if r["f357"].get("p95_rel") is not None], "p95_rel"),
            "f471_changed_rate": agg([r["f471"] for r in rows if "changed_rate" in r["f471"]], "changed_rate"),
            "f471_spearman": agg([r["f471"] for r in rows if "spearman" in r["f471"]], "spearman"),
            "f471_median_rel": agg([r["f471"] for r in rows if r["f471"].get("median_rel") is not None], "median_rel"),
            "f471_p95_rel": agg([r["f471"] for r in rows if r["f471"].get("p95_rel") is not None], "p95_rel"),
        }
    t357 = [t["f357_spearman"] for t in trans if t["f357_spearman"] is not None]
    t471 = [t["f471_spearman"] for t in trans if t["f471_spearman"] is not None]
    out = {
        "question": "Over EVERY cached post-cutover trading day, does the SH/SZ-restricted union leave the "
                    "published 357/471 essentially unchanged, and how does that compare with the all-codes union?",
        "window": [args.start, args.end], "target_days": len(targets),
        "warmup_trading_days": args.warmup_trading_days,
        "label_reader_validation": validation,
        "label_reader_validation_verdict": "fast reader identical to the provider on all validated days",
        "summary_by_scope": summary,
        "transcription_check_against_published": {
            "days_compared": len(trans),
            "f357_spearman_min": min(t357) if t357 else None,
            "f357_spearman_median": sorted(t357)[len(t357) // 2] if t357 else None,
            "f471_spearman_min": min(t471) if t471 else None,
            "f471_spearman_median": sorted(t471)[len(t471) // 2] if t471 else None,
        },
        "per_day": per_day, "transcription_per_day": trans,
        "database_writes": False, "weights_written": False, "cache_writes": False,
    }
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_day", "transcription_per_day")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
