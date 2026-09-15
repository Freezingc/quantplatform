"""Who are the non-STAR rows that option 1 still leaves out of the dense-nine intersection?

project_option1.py counted a gain only for STAR valid-label rows missing 357/471, and left the rest of the
gap as `residual_gap_not_star` (median 45.5 rows/day over the 420 pre-cutover days, maximum 171 on
2022-07-29). That residual is what decides the single day that still fails the proposed 0.95 bar, and
problems_found.md §AA recorded it as measured but unattributed.

It may be understating option 1. The SH/SZ union restores labels from index_member_all for NON-STAR
securities too -- the pre-cutover census measured 400-687 differing labels per day in the SH/SZ scope, of
which STAR is only part -- so a non-STAR row that is missing exactly 357 and 471 because the cache has no
label for it is fixable by the same union, and counting it as residual is too pessimistic.

This attributes every residual row on the days with the largest residuals: which of the nine factors it
lacks, whether the cache has a label for it, whether index_member_all does, and therefore whether the
SH/SZ union would lift it. It then recomputes coverage under the union's full fixable set rather than the
STAR-only one.

SAFETY: SELECT-only against dfs://QuantResearchV2Test, plus read-only local caches. No writes except
--out. No weights, no repository file touched.
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
CACHE = REPO / "artifacts" / "tushare_cache"
DENSE9 = ["yeslab_206", "yeslab_357", "yeslab_425", "yeslab_381", "yeslab_443_reg30_max20_v1",
          "yeslab_464", "yeslab_471", "yeslab_484", "yeslab_96"]
INDUSTRY = {"yeslab_357", "yeslab_471"}
STAR = ("688", "689")


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
        dv = np.datetime64(pd.Timestamp(day))
        mask = (self.ins <= dv) & (self.open_ended | (self.outs > dv))
        return {c: self.name_of[l] for c, l in zip(self.codes[mask], self.l1[mask])
                if self.name_of.get(l) is not None}


def sql_list(v) -> str:
    return ", ".join("'" + str(x).replace("'", "''") + "'" for x in v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, required=True, help="star_rows_in_denominator_420.json")
    ap.add_argument("--days", type=int, default=8, help="take the N days with the largest residual")
    ap.add_argument("--classification", type=Path, required=True)
    ap.add_argument("--member-all", type=Path, required=True)
    ap.add_argument("--bar", type=float, default=0.95)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    src = json.loads(args.rows.read_text())
    per = pd.DataFrame(src["per_day"])
    per["cov_after_star_only"] = (per.intersection_rows + per.gap_fixable_by_star_labels) / per.valid_label_rows
    days = per.nlargest(args.days, "residual_gap_not_star")["day"].tolist()
    print(f"  attributing the {len(days)} largest residuals: {days}", flush=True)

    name_of = {r["index_code"]: r["industry_name"] for r in json.loads(args.classification.read_text())}
    A = prep(pd.concat([pd.read_json(f, orient="table")
                        for f in sorted(glob.glob(str(CACHE / "pit_industry_members_*_SI.json")))],
                       ignore_index=True), "con_code", "index_code")
    B = prep(pd.DataFrame(json.loads(args.member_all.read_text())), "ts_code", "l1_code")
    LA, LB = Labels(A, name_of), Labels(B, name_of)
    print(f"  label sources: cache {len(A)} memberships, member_all {len(B)}", flush=True)

    sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
    from walk_forward_common import TARGET_DATABASE, current_versions, new_holder, query
    h = new_holder()
    lpins, fpins = current_versions(h, "label_daily"), current_versions(h, "factor_daily")
    ids = sql_list(DENSE9)

    out_days = []
    for year in sorted({d[:4] for d in days}):
        ydays = [d for d in days if d.startswith(year)]
        parts = sorted({f"{d[:4]}-{d[5:7]}" for d in ydays})
        ll = sql_list(sorted({lpins[p] for p in parts}))
        fl = sql_list(sorted({fpins[p] for p in parts}))
        dl = ", ".join(d.replace("-", ".") for d in ydays)
        ok, lab = query(h, f"select signal_date, ts_code from loadTable('{TARGET_DATABASE}','label_daily') "
                           f"where signal_date in [{dl}] and label_valid=1 and data_version in [{ll}]")
        if not ok:
            raise SystemExit(f"label query failed {year}: {lab}")
        script = (f"fac = select signal_date, ts_code, factor_id from loadTable('{TARGET_DATABASE}','factor_daily') "
                  f"where signal_date in [{dl}] and factor_id in [{ids}] and data_version in [{fl}];"
                  "dk = select signal_date, ts_code, factor_id from fac group by signal_date, ts_code, factor_id;"
                  "select signal_date, ts_code, factor_id from dk")
        ok2, fac = query(h, script)
        if not ok2:
            raise SystemExit(f"factor query failed {year}: {fac}")
        for df in (lab, fac):
            df["day"] = pd.to_datetime(df["signal_date"]).dt.strftime("%Y-%m-%d")
            df["ts_code"] = df["ts_code"].astype(str)
        lab_sets = {d: set(g.ts_code) for d, g in lab.groupby("day")}
        have = {d: {c: set(g.factor_id) for c, g in gg.groupby("ts_code")}
                for d, gg in fac.groupby("day")}
        print(f"  {year}: label rows {len(lab)}, factor rows {len(fac)}", flush=True)

        for day in ydays:
            V = lab_sets.get(day, set())
            H = have.get(day, {})
            inter = {c for c in V if len(H.get(c, ()) & set(DENSE9)) == len(DENSE9)}
            gap = V - inter
            star_gap = {c for c in gap if c.startswith(STAR)}
            resid = gap - star_gap
            la, lb = LA.on(pd.Timestamp(day)), LB.on(pd.Timestamp(day))
            cls, miss_counter, examples = {}, {}, []
            for c in sorted(resid):
                missing = sorted(set(DENSE9) - H.get(c, set()))
                only_industry = set(missing) <= INDUSTRY
                cache_has, ma_has = bool(la.get(c)), bool(lb.get(c))
                if only_industry and not cache_has and ma_has and not c.endswith(".BJ"):
                    k = "FIXABLE_BY_UNION_LABEL"
                elif only_industry and not cache_has and not ma_has:
                    k = "NO_LABEL_IN_EITHER_SOURCE"
                elif only_industry and cache_has:
                    k = "LABEL_PRESENT_BUT_INDUSTRY_FACTOR_MISSING"
                elif c.endswith(".BJ"):
                    k = "BSE_SECURITY"
                else:
                    k = "MISSING_NON_INDUSTRY_FACTOR"
                cls[k] = cls.get(k, 0) + 1
                for f in missing:
                    miss_counter[f] = miss_counter.get(f, 0) + 1
                if len(examples) < 40:
                    examples.append({"ts_code": c, "missing": missing, "cache_label": la.get(c),
                                     "member_all_label": lb.get(c), "class": k})
            fix_extra = cls.get("FIXABLE_BY_UNION_LABEL", 0)
            row = per[per.day == day].iloc[0]
            N, Vn = int(row.intersection_rows), int(row.valid_label_rows)
            star_gain = int(row.gap_fixable_by_star_labels)
            out_days.append({
                "day": day, "valid_label_rows": len(V), "valid_label_rows_per_probe": Vn,
                "denominator_matches": len(V) == Vn,
                "intersection_rows": len(inter), "intersection_rows_per_probe": N,
                "numerator_matches": len(inter) == N,
                "gap_rows": len(gap), "star_gap_rows": len(star_gap), "residual_not_star": len(resid),
                "residual_not_star_per_probe": int(row.residual_gap_not_star),
                "residual_matches": len(resid) == int(row.residual_gap_not_star),
                "residual_classification": cls,
                "missing_factor_histogram_over_residual": miss_counter,
                "coverage_after_star_only": (N + star_gain) / Vn,
                "coverage_after_star_plus_union_labels": (N + star_gain + fix_extra) / Vn,
                "clears_bar_star_only": (N + star_gain) / Vn >= args.bar,
                "clears_bar_star_plus_union": (N + star_gain + fix_extra) / Vn >= args.bar,
                "rows_short_of_bar_star_only": max(0, int(np.ceil(args.bar * Vn)) - (N + star_gain)),
                "rows_short_of_bar_star_plus_union": max(0, int(np.ceil(args.bar * Vn)) - (N + star_gain + fix_extra)),
                "examples": examples,
            })
            print(f"  {day}: V={len(V)} N={len(inter)} gap={len(gap)} star={len(star_gap)} resid={len(resid)} "
                  f"{cls}", flush=True)

    df = pd.DataFrame(out_days)
    ok = bool(df.denominator_matches.all() and df.numerator_matches.all() and df.residual_matches.all())
    out = {
        "question": "Who are the non-STAR residual rows option 1 leaves out, and would the SH/SZ union lift them?",
        "days": len(df), "day_list": days, "bar": args.bar,
        "reproduces_the_decomposition": {
            "verdict": "EXACT" if ok else "MISMATCH -- attribution below is not trustworthy",
            "denominator_identical_on": int(df.denominator_matches.sum()),
            "numerator_identical_on": int(df.numerator_matches.sum()),
            "residual_identical_on": int(df.residual_matches.sum()), "days": len(df)},
        "classification_totals": {k: int(v) for k, v in
                                  pd.DataFrame([r["residual_classification"] for r in out_days]).fillna(0).sum().items()},
        "missing_factor_histogram_totals": {k: int(v) for k, v in
                                            pd.DataFrame([r["missing_factor_histogram_over_residual"]
                                                          for r in out_days]).fillna(0).sum().items()},
        "worst_day": {
            "day": str(df.loc[df.residual_not_star.idxmax(), "day"]),
            "residual_not_star": int(df.residual_not_star.max()),
            "coverage_after_star_only": float(df.loc[df.residual_not_star.idxmax(), "coverage_after_star_only"]),
            "coverage_after_star_plus_union_labels": float(
                df.loc[df.residual_not_star.idxmax(), "coverage_after_star_plus_union_labels"]),
            "rows_short_of_bar_star_only": int(df.loc[df.residual_not_star.idxmax(), "rows_short_of_bar_star_only"]),
            "rows_short_of_bar_star_plus_union": int(
                df.loc[df.residual_not_star.idxmax(), "rows_short_of_bar_star_plus_union"])},
        "days_clearing_the_bar": {
            "star_only": int(df.clears_bar_star_only.sum()),
            "star_plus_union_labels": int(df.clears_bar_star_plus_union.sum()), "of": len(df)},
        "answer": None, "per_day": out_days,
        "database_writes": False, "weights_written": False, "cache_writes": False,
    }
    w = out["worst_day"]
    out["answer"] = (
        f"On the {len(df)} days with the largest residual, the {int(df.residual_not_star.sum())} non-STAR gap rows "
        f"classify as {out['classification_totals']}. On the binding day {w['day']} the union's extra fixable rows "
        f"move coverage from {w['coverage_after_star_only']:.6f} to {w['coverage_after_star_plus_union_labels']:.6f}, "
        f"so the shortfall against the {args.bar} bar goes from {w['rows_short_of_bar_star_only']} rows to "
        f"{w['rows_short_of_bar_star_plus_union']}.")
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n" + out["answer"])
    print(json.dumps({k: out[k] for k in ("reproduces_the_decomposition", "classification_totals",
                                          "missing_factor_histogram_totals", "days_clearing_the_bar")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
