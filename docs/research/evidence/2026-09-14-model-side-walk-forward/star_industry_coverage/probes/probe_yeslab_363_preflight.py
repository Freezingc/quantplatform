"""Pre-flight for yeslab_363 as a candidate dense factor: does it carry the same STAR label dependency?

The frozen factor set is still being decided, and yeslab_363 is the third member of the materializer's
PIT_FACTORS = {yeslab_357, yeslab_363, yeslab_471} -- the factors that receive an industry provider at
replay time. §M established that 357 and 471 are missing for every STAR security before 2023-09-25 because
the SW2021 industry cache has no STAR membership before that date, and that this is what makes 24 OOS days
unevaluable. 363 was never tested, and it is not in the frozen set, so nobody has measured what adding it
would cost.

If 363 shares the dependency, adding it to a dense subset would reintroduce exactly the failure the decision
sheet is trying to repair. This measures 363's presence in factor_daily, its per-day row count against the
resolved universe, and its STAR versus non-STAR split on days either side of the cutover, alongside 357 and
471 as in-family controls and 484 as an out-of-family control that takes no industry provider.

SAFETY: SELECT-only against dfs://QuantResearchV2Test. No writes except --out. No weights.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd

REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
UNIVERSE_ID = "cn_liquid_a"
FACTORS = ["yeslab_363", "yeslab_357", "yeslab_471", "yeslab_484"]
STAR = ("688", "689")
PIT = {"yeslab_357", "yeslab_363", "yeslab_471"}


def sql_list(v) -> str:
    return ", ".join("'" + str(x).replace("'", "''") + "'" for x in v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", default="2022-06-30,2023-07-05,2023-09-22,2023-09-25,2024-06-17,2024-12-20,2025-06-16")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    days = args.days.split(",")

    sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
    from walk_forward_common import TARGET_DATABASE, current_versions, new_holder, query
    h = new_holder()
    fpins, upins = current_versions(h, "factor_daily"), current_versions(h, "universe_daily")

    ok0, span = query(h,
        f"select min(signal_date) as lo, max(signal_date) as hi, count(*) as n "
        f"from loadTable('{TARGET_DATABASE}','factor_daily') where factor_id='yeslab_363'")
    print(f"  yeslab_363 in factor_daily: {ok0 and isinstance(span, pd.DataFrame) and len(span)}", flush=True)

    rows = []
    for day in days:
        dl = day.replace("-", "."); part = day[:7]
        fl = sql_list([fpins[part]]) if part in fpins else "''"
        ul = sql_list([upins[part]]) if part in upins else "''"
        ok, uni = query(h,
            f"select ts_code, data_version from loadTable('{TARGET_DATABASE}','universe_daily') "
            f"where signal_date = date('{dl}') and universe_id='{UNIVERSE_ID}' and is_member=1 "
            f"and data_version in [{ul}]")
        members = set(uni.ts_code.astype(str)) if ok and isinstance(uni, pd.DataFrame) else set()
        ok2, fac = query(h,
            f"select ts_code, factor_id from loadTable('{TARGET_DATABASE}','factor_daily') "
            f"where signal_date = date('{dl}') and factor_id in [{sql_list(FACTORS)}] and data_version in [{fl}]")
        fac["ts_code"] = fac.ts_code.astype(str)
        rec = {"day": day, "universe_members": len(members),
               "star_universe_members": sum(1 for c in members if c.startswith(STAR))}
        for f in FACTORS:
            s = set(fac.loc[fac.factor_id == f, "ts_code"])
            star = {c for c in s if c.startswith(STAR)}
            rec[f"{f}__rows"] = len(s)
            rec[f"{f}__rows_in_universe"] = len(s & members)
            rec[f"{f}__coverage_of_universe"] = len(s & members) / len(members) if members else None
            rec[f"{f}__star_rows"] = len(star)
            rec[f"{f}__star_coverage"] = (len(star) / rec["star_universe_members"]) if rec["star_universe_members"] else None
            rec[f"{f}__non_star_coverage"] = (
                len((s - star) & (members - {c for c in members if c.startswith(STAR)}))
                / max(1, len(members) - rec["star_universe_members"]))
        rows.append(rec)
        print(f"  {day}: universe {len(members)} (star {rec['star_universe_members']}) | "
              + " ".join(f"{f.replace('yeslab_','')}={rec[f + '__rows']}(star {rec[f + '__star_rows']})"
                         for f in FACTORS), flush=True)

    df = pd.DataFrame(rows)
    span_rec = recs = None
    if ok0 and isinstance(span, pd.DataFrame) and len(span):
        span_rec = {"min_signal_date": str(span.lo.iloc[0]), "max_signal_date": str(span.hi.iloc[0]),
                    "rows": int(span.n.iloc[0])}
    pre = df[df.day < "2023-09-25"]; post = df[df.day >= "2023-09-25"]
    out = {
        "question": "Does yeslab_363 carry the same STAR industry-label dependency as 357 and 471?",
        "pit_factors_in_the_materializer": sorted(PIT),
        "controls": {"in_family": ["yeslab_357", "yeslab_471"],
                     "out_of_family_no_industry_provider": ["yeslab_484"]},
        "yeslab_363_span_in_factor_daily": span_rec,
        "per_day": rows,
        "summary": {
            f: {"pre_cutover_star_coverage": (pre[f"{f}__star_coverage"].dropna().tolist() or None),
                "post_cutover_star_coverage": (post[f"{f}__star_coverage"].dropna().tolist() or None),
                "pre_cutover_universe_coverage_median": float(pre[f"{f}__coverage_of_universe"].median()),
                "post_cutover_universe_coverage_median": float(post[f"{f}__coverage_of_universe"].median())}
            for f in FACTORS},
        "answer": None, "database_writes": False, "weights_written": False,
    }
    s363 = out["summary"]["yeslab_363"]; s484 = out["summary"]["yeslab_484"]
    pre363 = [x for x in (s363["pre_cutover_star_coverage"] or []) if x is not None]
    post363 = [x for x in (s363["post_cutover_star_coverage"] or []) if x is not None]
    out["answer"] = (
        f"yeslab_363 spans {span_rec['min_signal_date'][:10] if span_rec else '?'} to "
        f"{span_rec['max_signal_date'][:10] if span_rec else '?'} with {span_rec['rows'] if span_rec else '?'} rows. "
        f"Its STAR coverage pre-cutover is {[round(x,4) for x in pre363]} and post-cutover "
        f"{[round(x,4) for x in post363]}, against universe coverage medians of "
        f"{s363['pre_cutover_universe_coverage_median']:.4f} and {s363['post_cutover_universe_coverage_median']:.4f}; "
        f"the out-of-family control yeslab_484 has STAR coverage "
        f"{[round(x,4) for x in (s484['pre_cutover_star_coverage'] or []) if x is not None]} pre-cutover.")
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n" + out["answer"])
    cols = ["day", "universe_members", "star_universe_members"] + \
           [f"{f}__rows" for f in FACTORS] + [f"{f}__star_rows" for f in FACTORS]
    pd.set_option("display.width", 300)
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
