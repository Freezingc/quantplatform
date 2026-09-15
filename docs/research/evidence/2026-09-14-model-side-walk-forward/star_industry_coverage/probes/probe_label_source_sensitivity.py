"""What does swapping the industry-label source actually do to yeslab_357 and yeslab_471?

The decision sheet's remaining precondition for option 1 was rewritten in section 7.5: the question is no
longer "can a backfilled replay reproduce the YesLab stored values" -- section Q showed the replay never
reproduced them even where it had labels -- but "are values replayed with the new labels consistent with
the replay's OWN convention on days where it already had labels?" That is measurable read-only, and this
measures it.

Method, and why it is sound despite the transcription not being exact. Both factors are computed twice
from the SAME transcription of their calculators, differing only in the industry-label matrix:

  source A  VersionedPITIndustryProvider.from_cache(tushare_cache, cutover=2021-12-10) -- the exact
            object materialize_frozen_factor_daily.py passes to replay._replay_factor for PIT_FACTORS
  source B  the union rule recommended in option1_feasibility.md section 13: index_member_all's active L1
            where it has one, else the cache's, with L1 codes mapped to SW2021 industry names

Section W established that a hand transcription of this kind reproduces the published factor to about
Spearman 0.97 rather than 1.000, so A-versus-published is reported as a TRANSCRIPTION CHECK and not as a
claim of exactness. The quantity that matters is A versus B, where the transcription is identical on both
sides and therefore cancels: whatever it gets wrong, it gets wrong the same way twice, so the difference
isolates the label source.

Also reported, because option 1 is usually described as "adds the missing STAR rows": how many securities
change value at all, and how many change materially, on a POST-cutover day where the STAR labels already
exist and the only label differences are the roughly 136 a day measured in section 14.

SAFETY: reads the local caches and repository source, plus SELECT-only queries against
dfs://QuantResearchV2Test and dfs://YesLab for the published comparison. No writes except --out. It does
NOT write into research_v2/artifacts. No weights.
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


def load_wide(dates, field, fname):
    per = {}
    for d in dates:
        p = CACHE / f"{fname}_{d:%Y%m%d}.json"
        if not p.is_file():
            raise SystemExit(f"missing cache file {p}")
        raw = json.loads(p.read_text())
        recs = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
        f = pd.DataFrame(recs)
        f["ts_code"] = f["ts_code"].astype(str)
        f = f.drop_duplicates("ts_code", keep="last").set_index("ts_code")[field]
        per[pd.Timestamp(d)] = pd.to_numeric(f, errors="coerce")
    w = pd.DataFrame(per).T.sort_index()
    w.index = pd.to_datetime(w.index)
    return w


def prep_members(m, code, l1):
    m = m.rename(columns={code: "code", l1: "l1"})
    m["in_p"] = pd.to_datetime(m["in_date"].astype(str), format="%Y%m%d", errors="coerce")
    o = m["out_date"]
    m["out_p"] = pd.to_datetime(o.astype(str).str.replace(r"\.0$", "", regex=True), format="%Y%m%d", errors="coerce")
    m.loc[o.isna() | o.astype(str).isin(["None", "nan", "", "NaT"]), "out_p"] = pd.NaT
    return m.dropna(subset=["in_p"])[["code", "l1", "in_p", "out_p"]]


def active_labels(m, day, name_of):
    dv = np.datetime64(day)
    mask = (m["in_p"].to_numpy() <= dv) & (pd.isna(m["out_p"]).to_numpy() | (m["out_p"].to_numpy() > dv))
    sub = m[mask]
    out = {}
    for c, l in zip(sub["code"].to_numpy(), sub["l1"].to_numpy()):
        out.setdefault(c, set()).add(l)
    # the provider's own tie-break: latest in_date, then largest index_code (base.py -> build_pit_snapshot)
    resolved = {}
    for c, ls in out.items():
        if len(ls) == 1:
            resolved[c] = next(iter(ls))
        else:
            cand = sub[(sub["code"] == c)]
            cand = cand.sort_values(["in_p", "l1"], kind="stable")
            resolved[c] = cand["l1"].iloc[-1]
    return {c: name_of.get(v, v) for c, v in resolved.items()}


def industry_zscore(ret_row, labels_row):
    """base.py:102-123 industry_zscore_and_skew, restricted to the z-score half, one date at a time."""
    valid = ret_row.notna() & labels_row.notna()
    z = pd.Series(np.nan, index=ret_row.index, dtype=float)
    if not valid.any():
        return z
    g = ret_row[valid].groupby(labels_row[valid], sort=False)
    z.loc[valid] = ((ret_row[valid] - g.transform("mean")) / g.transform("std")).to_numpy(dtype=float)
    return z


def attention_spillover(pct_chg, total_mv, labels, target):
    """AttentionSpilloverCalculator._calculate, transcribed, evaluated on one target date."""
    cross_mean = pct_chg.mean(axis=1, skipna=True)
    abnormal_sq = pct_chg.sub(cross_mean, axis=0).pow(2)
    attention = abnormal_sq.rolling(window=21, min_periods=21).mean()
    att = attention.loc[target]
    mv = total_mv.loc[target]
    lab = labels.loc[target]
    df = pd.DataFrame({"attention": att, "mv": mv, "industry": lab}).dropna(
        subset=["attention", "mv", "industry"])
    out = pd.Series(np.nan, index=att.index, dtype=float)
    if df.empty:
        return out, 0
    for _, ind in df.groupby("industry", sort=False):
        ind = ind.sort_values("mv").copy()
        n = len(ind)
        pct = (np.arange(1, n + 1, dtype=float) - 0.5) / n
        ind["bucket"] = np.where(pct <= 0.3, "small", np.where(pct > 0.7, "big", "mid"))
        for _, b in ind.groupby("bucket", sort=False):
            a = b["attention"]; m = len(a)
            if m <= 1:
                continue
            peer = (a.sum() - a) / (m - 1)
            out.loc[b.index] = (peer - a).to_numpy(dtype=float)
    return out, len(df)


def compare(a: pd.Series, b: pd.Series) -> dict:
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    if len(j) < 20:
        return {"pairs": int(len(j))}
    d = (j["a"] - j["b"]).abs()
    rel = (d / j["b"].abs().replace(0, np.nan)).dropna()
    return {
        "pairs": int(len(j)),
        "identical": int((d == 0).sum()),
        "identical_rate": float((d == 0).mean()),
        "within_1e_12": int((d <= 1e-12).sum()),
        "changed": int((d > 0).sum()),
        "changed_rate": float((d > 0).mean()),
        "changed_materially_gt_1e_6": int((d > 1e-6).sum()),
        "spearman": float(j["a"].corr(j["b"], method="spearman")),
        "pearson": float(np.corrcoef(j["a"], j["b"])[0, 1]),
        "max_abs_difference": float(d.max()),
        "median_abs_difference": float(d.median()),
        "median_relative_difference": float(rel.median()) if len(rel) else None,
        "p95_relative_difference": float(rel.quantile(0.95)) if len(rel) else None,
        "std_ratio_a_over_b": float(j["a"].std() / j["b"].std()) if j["b"].std() else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", required=True)
    ap.add_argument("--warmup-trading-days", type=int, default=45)
    ap.add_argument("--member-all", type=Path, required=True)
    ap.add_argument("--classification", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--union-scope", choices=("all", "shsz"), default="all",
                    help="which securities the union may ADD labels for. 'all' fills every code the "
                         "alternative source knows; 'shsz' fills only SH/SZ codes, leaving .BJ unlabelled "
                         "exactly as the pipeline in force leaves them. The two differ because the union "
                         "finds labels for BSE securities that the current provider has none for, and "
                         "adding those to the industry groups is a separate decision from backfilling STAR.")
    args = ap.parse_args()
    target = pd.Timestamp(args.day)

    dates_all = sorted({pd.Timestamp(p.stem.split("_")[1]) for p in CACHE.glob("daily_[0-9]*.json")})
    dates = [d for d in dates_all if d <= target][-args.warmup_trading_days:]
    if dates[-1] != target:
        raise SystemExit(f"{args.day} is not a cached trading day")
    print(f"  window {dates[0].date()}..{dates[-1].date()} ({len(dates)} trading days)", flush=True)

    close = load_wide(dates, "close", "daily")
    pct_chg = load_wide(dates, "pct_chg", "daily")
    total_mv = load_wide(dates, "total_mv", "daily_basic")
    codes = sorted(close.columns)
    print(f"  panel {close.shape[0]} days x {len(codes)} codes", flush=True)

    # ---- label source A: the exact provider object the materializer uses ----
    sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
    from audit_tushare_pit_industry import VersionedPITIndustryProvider
    provider = VersionedPITIndustryProvider.from_cache(CACHE, cutover=date(2021, 12, 10))
    LA = provider.get_industry_labels([d.date() for d in dates], codes)
    LA.index = pd.to_datetime(LA.index)

    # ---- label source B: the union rule, L1 code -> SW2021 industry name ----
    cls = json.loads(args.classification.read_text())
    name_of = {r["index_code"]: r["industry_name"] for r in cls}
    A = prep_members(pd.concat([pd.read_json(f, orient="table")
                                for f in sorted(glob.glob(str(CACHE / "pit_industry_members_*_SI.json")))],
                               ignore_index=True), "con_code", "index_code")
    B = prep_members(pd.DataFrame(json.loads(args.member_all.read_text())), "ts_code", "l1_code")
    rowsB, decomp = {}, []
    LA_t = LA.loc[target]          # one date's label row, so .get returns a scalar not a Series
    for d in dates:
        ma, mb = active_labels(A, d, name_of), active_labels(B, d, name_of)
        merged = dict(ma)
        merged.update({k: v for k, v in mb.items() if k in ma})   # member_all wins where both speak
        fills = {k: v for k, v in mb.items() if k not in ma}      # and fills where the cache is silent
        if args.union_scope == "shsz":
            fills = {k: v for k, v in fills.items() if not k.endswith(".BJ")}
        merged.update(fills)
        rowsB[d] = pd.Series({c: merged.get(c) for c in codes}, dtype=object)
        if d == target:
            panel = set(codes)
            both = [c for c in panel if c in ma and c in mb]
            decomp = [{"both_labeled": len(both),
                       "name_differs_genuine_disagreement": sum(1 for c in both if ma[c] != mb[c]),
                       "added_by_union_all": sum(1 for c in panel if c not in ma and c in mb),
                       "added_by_union_bj": sum(1 for c in panel if c not in ma and c in mb and c.endswith(".BJ")),
                       "added_by_union_shsz": sum(1 for c in panel if c not in ma and c in mb and not c.endswith(".BJ")),
                       "lost_under_member_all": sum(1 for c in panel if c in ma and c not in mb),
                       "provider_matches_cache_raw_on": sum(1 for c in both if LA_t.get(c) == ma[c]),
                       "applied_fill_after_scope_filter": len(fills)}]
    LB = pd.DataFrame(rowsB).T.sort_index()
    LB.index = pd.to_datetime(LB.index)

    diff_labels = int((LA.loc[target].fillna("~") != LB.loc[target].fillna("~")).sum())
    print(f"  labels differing on {args.day}: {diff_labels} of {len(codes)}", flush=True)
    print(f"  label coverage on {args.day}: cache/provider {int(LA.loc[target].notna().sum())}, "
          f"union {int(LB.loc[target].notna().sum())}", flush=True)

    # ---- yeslab_357: |within-industry z-score of the 1-day return| ----
    ret = close / close.shift(1) - 1
    z357_A = industry_zscore(ret.loc[target], LA.loc[target]).abs().replace([np.inf, -np.inf], np.nan)
    z357_B = industry_zscore(ret.loc[target], LB.loc[target]).abs().replace([np.inf, -np.inf], np.nan)

    # ---- yeslab_471 ----
    s471_A, nA = attention_spillover(pct_chg, total_mv, LA, target)
    s471_B, nB = attention_spillover(pct_chg, total_mv, LB, target)
    print(f"  471 grouped securities: A {nA}, B {nB}", flush=True)

    # ---- published values, for the transcription check ----
    from walk_forward_common import TARGET_DATABASE, current_versions, new_holder, query
    holder = new_holder()
    fp = current_versions(holder, "factor_daily")
    part = f"{args.day[:4]}-{args.day[5:7]}"; dl = args.day.replace("-", ".")
    ok, fd = query(holder,
        f"select ts_code, factor_id, value from loadTable('{TARGET_DATABASE}','factor_daily') "
        f"where signal_date = date('{dl}') and factor_id in ['yeslab_357','yeslab_471'] "
        f"and data_version in ['{fp[part]}']")
    pub = {}
    if ok and isinstance(fd, pd.DataFrame):
        fd["ts_code"] = fd["ts_code"].astype(str); fd["value"] = pd.to_numeric(fd["value"], errors="coerce")
        for fid, g in fd.groupby("factor_id"):
            pub[fid] = g.dropna(subset=["value"]).groupby("ts_code")["value"].first()

    out = {
        "question": "What does swapping the industry-label source do to yeslab_357 and yeslab_471, on a "
                    "day where the replay already had labels?",
        "day": args.day, "window": [str(dates[0].date()), str(dates[-1].date())],
        "trading_days": len(dates), "panel_codes": len(codes),
        "label_sources": {
            "A": "VersionedPITIndustryProvider.from_cache(tushare_cache, cutover=2021-12-10) -- the object "
                 "materialize_frozen_factor_daily.py passes to replay._replay_factor for PIT_FACTORS",
            "B": "union rule (index_member_all preferred, cache fallback), L1 code mapped to SW2021 name",
        },
        "labels_differing_on_the_target_day": diff_labels,
        "union_scope": args.union_scope,
        "label_difference_decomposition_on_the_target_day": decomp,
        "label_coverage": {"A_provider": int(LA.loc[target].notna().sum()),
                           "B_union": int(LB.loc[target].notna().sum()),
                           "gain": int(LB.loc[target].notna().sum() - LA.loc[target].notna().sum())},
        "methodology_note": "A-versus-published is a TRANSCRIPTION CHECK, not a claim of exactness; "
                            "section W measured this kind of transcription at about Spearman 0.97 against "
                            "the published factor. The decisive comparison is A-versus-B, where the "
                            "transcription is identical on both sides and cancels.",
        "yeslab_357": {
            "A_vs_B_label_source_effect": compare(z357_A, z357_B),
            "A_vs_published_transcription_check": compare(z357_A, pub["yeslab_357"]) if "yeslab_357" in pub else None,
            "B_vs_published": compare(z357_B, pub["yeslab_357"]) if "yeslab_357" in pub else None,
        },
        "yeslab_471": {
            "A_vs_B_label_source_effect": compare(s471_A, s471_B),
            "A_vs_published_transcription_check": compare(s471_A, pub["yeslab_471"]) if "yeslab_471" in pub else None,
            "B_vs_published": compare(s471_B, pub["yeslab_471"]) if "yeslab_471" in pub else None,
            "grouped_securities_A": nA, "grouped_securities_B": nB,
        },
        "database_writes": False, "weights_written": False, "cache_writes": False,
    }
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for fid in ("yeslab_357", "yeslab_471"):
        print(f"\n=== {fid} ===")
        for k, v in out[fid].items():
            if isinstance(v, dict):
                print(f"  {k}: pairs {v.get('pairs')} identical_rate {v.get('identical_rate')} "
                      f"changed {v.get('changed')} spearman {v.get('spearman')} "
                      f"median_rel {v.get('median_relative_difference')} p95_rel {v.get('p95_relative_difference')} "
                      f"max_abs {v.get('max_abs_difference')}")
            else:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
