"""Classify every gap row on the 24 problem days by whether a label backfill can fix it.

The union projection so far counted only STAR rows gaining labels. But index_member_all also carries
460-614 NON-STAR securities per day that the cache has no label for, so the projection is conservative
in a second direction -- and the binding day misses the proposed 0.95 bar by only four rows, which is
small enough that this second term could decide it.

Per security per day, this pulls how many of the nine dense factors are present, then splits the gap
rows (valid label, not all nine) into four classes:

  fixable_star          STAR row; the cache has no STAR label before 2023-09-25
  fixable_non_star      non-STAR row the cache has no label for but index_member_all does
  not_fixable_labelled  row that already HAS a cache label, so it is missing one of the other seven
                        factors and a label backfill cannot help it
  not_fixable_no_label  row neither source has a label for

It also records, for each fixable class, how many rows are missing EXACTLY the two industry factors
(count == 7), which is the security-level form of the mechanism claim.

SAFETY: SELECT-only against dfs://QuantResearchV2Test. No writes anywhere except --out. No weights.
"""
from __future__ import annotations
import argparse, glob, json, math, sys
from pathlib import Path
import pandas as pd

REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
CACHE = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2/artifacts/tushare_cache")
DENSE9 = ["yeslab_206", "yeslab_357", "yeslab_381", "yeslab_425", "yeslab_443_reg30_max20_v1",
          "yeslab_464", "yeslab_471", "yeslab_484", "yeslab_96"]
STAR = ("688", "689")
FLOOR = 0.90


def sql_list(v) -> str:
    return ", ".join("'" + str(x).replace("'", "''") + "'" for x in v)


def prep(m, code, l1):
    m = m.rename(columns={code: "code", l1: "l1"})
    m["in_p"] = pd.to_datetime(m["in_date"].astype(str), format="%Y%m%d", errors="coerce")
    o = m["out_date"]
    m["out_p"] = pd.to_datetime(o.astype(str).str.replace(r"\.0$", "", regex=True), format="%Y%m%d", errors="coerce")
    m.loc[o.isna() | o.astype(str).isin(["None", "nan", "", "NaT"]), "out_p"] = pd.NaT
    return m.dropna(subset=["in_p"])[["code", "in_p", "out_p"]]


def labeled(m, day):
    s = m[(m.in_p <= day) & (m.out_p.isna() | (m.out_p > day))]
    return set(s["code"].astype(str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=Path, required=True)
    ap.add_argument("--member-all", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    src = json.loads(args.days.read_text())
    days = sorted(src["days"])
    ref = src.get("per_day", {})

    A = prep(pd.concat([pd.read_json(f, orient="table")
                        for f in sorted(glob.glob(str(CACHE / "pit_industry_members_*_SI.json")))],
                       ignore_index=True), "con_code", "index_code")
    B = prep(pd.DataFrame(json.loads(args.member_all.read_text())), "ts_code", "l1_code")

    sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
    from walk_forward_common import TARGET_DATABASE, current_versions, new_holder, query
    holder = new_holder()
    lpins, fpins = current_versions(holder, "label_daily"), current_versions(holder, "factor_daily")
    ids = sql_list(DENSE9)

    lab_frames, cnt_frames = [], []
    for year in sorted({d[:4] for d in days}):
        yd = [d for d in days if d.startswith(year)]
        parts = sorted({f"{d[:4]}-{d[5:7]}" for d in yd})
        miss = [p for p in parts if p not in lpins or p not in fpins]
        if miss:
            raise SystemExit(f"unpublished partitions for {year}: {miss}")
        ll = sql_list(sorted({lpins[p] for p in parts})); fl = sql_list(sorted({fpins[p] for p in parts}))
        dl = ", ".join(d.replace("-", ".") for d in yd)
        ok, lab = query(holder,
            f"select signal_date, ts_code from loadTable('{TARGET_DATABASE}','label_daily') "
            f"where signal_date in [{dl}] and label_valid=1 and data_version in [{ll}]")
        if not ok or not isinstance(lab, pd.DataFrame):
            raise SystemExit(f"label query failed {year}: {lab}")
        ok2, cnt = query(holder,
            f"fac = select signal_date, ts_code, factor_id from loadTable('{TARGET_DATABASE}','factor_daily') "
            f"where signal_date in [{dl}] and factor_id in [{ids}] and data_version in [{fl}];"
            "dk = select signal_date, ts_code, factor_id from fac group by signal_date, ts_code, factor_id;"
            "select signal_date, ts_code, count(*) as cnt from dk group by signal_date, ts_code")
        if not ok2 or not isinstance(cnt, pd.DataFrame):
            raise SystemExit(f"count query failed {year}: {cnt}")
        lab_frames.append(lab); cnt_frames.append(cnt)
        print(f"  {year}: {len(lab)} label rows, {len(cnt)} securities with >=1 factor", flush=True)

    lab = pd.concat(lab_frames, ignore_index=True); cnt = pd.concat(cnt_frames, ignore_index=True)
    for df in (lab, cnt):
        df["day"] = pd.to_datetime(df["signal_date"]).dt.strftime("%Y-%m-%d")
        df["ts_code"] = df["ts_code"].astype(str)
    merged = lab.merge(cnt[["day", "ts_code", "cnt"]], on=["day", "ts_code"], how="left")
    merged["cnt"] = merged["cnt"].fillna(0).astype(int)

    rows = []
    for day, g in merged.groupby("day"):
        ts = pd.Timestamp(day)
        la, lb = labeled(A, ts), labeled(B, ts)
        V = len(g); N = int((g["cnt"] == 9).sum())
        gap = g[g["cnt"] < 9]
        is_star = gap["ts_code"].str.startswith(STAR)
        has_a = gap["ts_code"].isin(la); has_b = gap["ts_code"].isin(lb)
        fix_star = gap[is_star]; fix_non = gap[~is_star & ~has_a & has_b]
        not_lab = gap[~is_star & has_a]; not_any = gap[~is_star & ~has_a & ~has_b]
        r = ref.get(day, {})
        rows.append({
            "day": day, "valid_label_rows": V, "intersection_rows": N,
            "matches_instrument": (V == r.get("valid_label_rows") and N == r.get("intersection_rows")),
            "gap_rows": len(gap),
            "fixable_star": len(fix_star), "fixable_star_missing_exactly_the_two": int((fix_star["cnt"] == 7).sum()),
            "fixable_non_star": len(fix_non), "fixable_non_star_missing_exactly_the_two": int((fix_non["cnt"] == 7).sum()),
            "not_fixable_already_labelled": len(not_lab),
            "not_fixable_no_label_anywhere": len(not_any),
            "coverage_now": N / V,
            "coverage_union_star_only": (N + len(fix_star)) / V,
            "coverage_union_star_and_non_star": (N + len(fix_star) + len(fix_non)) / V,
            "coverage_union_strict": (N + int((fix_star["cnt"] == 7).sum())
                                      + int((fix_non["cnt"] == 7).sum())) / V,
            "rows_needed_for_floor": math.ceil(FLOOR * V) - N,
            "rows_needed_for_095": math.ceil(0.95 * V) - N,
        })

    repro = all(r["matches_instrument"] for r in rows)
    worst = min(rows, key=lambda r: r["coverage_union_star_and_non_star"])
    worst_strict = min(rows, key=lambda r: r["coverage_union_strict"])
    out = {
        "question": "Splitting the gap rows four ways, does a UNION backfill clear the proposed 0.95 bar?",
        "database": TARGET_DATABASE, "floor": FLOOR, "proposed_bar": 0.95, "days": len(rows),
        "reproduces_the_instrument": "EXACT" if repro else "MISMATCH -- not trustworthy",
        "gap_rows_total": sum(r["gap_rows"] for r in rows),
        "fixable_star_total": sum(r["fixable_star"] for r in rows),
        "fixable_non_star_total": sum(r["fixable_non_star"] for r in rows),
        "not_fixable_already_labelled_total": sum(r["not_fixable_already_labelled"] for r in rows),
        "not_fixable_no_label_anywhere_total": sum(r["not_fixable_no_label_anywhere"] for r in rows),
        "fixable_rows_missing_exactly_the_two_industry_factors":
            sum(r["fixable_star_missing_exactly_the_two"] + r["fixable_non_star_missing_exactly_the_two"]
                for r in rows),
        "fixable_rows_total": sum(r["fixable_star"] + r["fixable_non_star"] for r in rows),
        "worst_day": {
            "star_only": {"day": min(rows, key=lambda r: r["coverage_union_star_only"])["day"],
                          "coverage": min(r["coverage_union_star_only"] for r in rows)},
            "star_and_non_star": {"day": worst["day"], "coverage": worst["coverage_union_star_and_non_star"],
                                  "clears_floor": worst["coverage_union_star_and_non_star"] >= FLOOR,
                                  "clears_095": worst["coverage_union_star_and_non_star"] >= 0.95},
            "strict_only_rows_missing_exactly_two": {
                "day": worst_strict["day"], "coverage": worst_strict["coverage_union_strict"],
                "clears_floor": worst_strict["coverage_union_strict"] >= FLOOR,
                "clears_095": worst_strict["coverage_union_strict"] >= 0.95},
        },
        "days_clearing_095_star_and_non_star": sum(1 for r in rows if r["coverage_union_star_and_non_star"] >= 0.95),
        "days_clearing_095_strict": sum(1 for r in rows if r["coverage_union_strict"] >= 0.95),
        "per_day": rows, "database_writes": False, "weights_written": False,
    }
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_day"}, ensure_ascii=False, indent=1))
    print(f"{'day':12}{'V':>6}{'gap':>5}{'fStar':>6}{'fNon':>5}{'nlLab':>6}{'nlAny':>6}{'now':>9}{'uS':>9}{'uSN':>9}{'strict':>9}")
    for r in sorted(rows, key=lambda x: x["coverage_union_star_and_non_star"])[:6]:
        print(f"{r['day']:12}{r['valid_label_rows']:6d}{r['gap_rows']:5d}{r['fixable_star']:6d}"
              f"{r['fixable_non_star']:5d}{r['not_fixable_already_labelled']:6d}"
              f"{r['not_fixable_no_label_anywhere']:6d}{r['coverage_now']:9.6f}"
              f"{r['coverage_union_star_only']:9.6f}{r['coverage_union_star_and_non_star']:9.6f}"
              f"{r['coverage_union_strict']:9.6f}")


if __name__ == "__main__":
    main()
