"""Switching the membership API is not purely additive: it moves the annual churn hole to another day.

The cache (index_member) expires 143 memberships on 2022-07-29 and re-registers 145 on 2022-08-01.
index_member_all expires 243 on 2022-07-28 and re-registers 255 on 2022-07-29. So a backfill would
repair 2022-07-29 and simultaneously expose 2022-07-28, which today passes. Any price for option 1
that only counts added rows is wrong in the unsafe direction.

This computes, for every OOS business day, the securities that would GAIN a label and the securities
that would LOSE one under the switch. Days with losers are the ones that need an exact re-measurement
against the evaluator's valid-label row set. Reads local JSON only. No database, no weights.
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import pandas as pd

CACHE = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2/artifacts/tushare_cache")


def prep(m, code, l1):
    m = m.rename(columns={code: "code", l1: "l1"})
    m["in_p"] = pd.to_datetime(m["in_date"].astype(str), format="%Y%m%d", errors="coerce")
    out = m["out_date"]
    m["out_p"] = pd.to_datetime(out.astype(str).str.replace(r"\.0$", "", regex=True), format="%Y%m%d", errors="coerce")
    m.loc[out.isna() | out.astype(str).isin(["None", "nan", "", "NaT"]), "out_p"] = pd.NaT
    return m.dropna(subset=["in_p"])[["code", "l1", "in_p", "out_p"]]


def labeled(m, day):
    sel = m[(m.in_p <= day) & (m.out_p.isna() | (m.out_p > day))]
    return {c: "|".join(sorted(set(v))) for c, v in sel.groupby("code")["l1"].apply(list).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member-all", type=Path, required=True)
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    A = prep(pd.concat([pd.read_json(f, orient="table")
                        for f in sorted(glob.glob(str(CACHE / "pit_industry_members_*_SI.json")))],
                       ignore_index=True), "con_code", "index_code")
    B = prep(pd.DataFrame(json.loads(args.member_all.read_text())), "ts_code", "l1_code")

    days = pd.bdate_range(args.start, args.end)
    rows = []
    for day in days:
        ma, mb = labeled(A, day), labeled(B, day)
        gain = set(mb) - set(ma)
        lose = set(ma) - set(mb)
        changed = {k for k in set(ma) & set(mb) if ma[k] != mb[k]}
        rows.append({"day": str(day.date()), "labeled_cache": len(ma), "labeled_member_all": len(mb),
                     "gain": len(gain), "lose": len(lose), "label_changed": len(changed),
                     "lose_codes": sorted(lose)[:20], "lose_is_star": sum(1 for c in lose if c.startswith(("688", "689"))),
                     "gain_is_star": sum(1 for c in gain if c.startswith(("688", "689")))})
    with_losers = [r for r in rows if r["lose"] > 0]
    out = {
        "question": "Which OOS days would LOSE industry labels if the cache were rebuilt from "
                    "index_member_all, i.e. where is option 1 not purely additive?",
        "window": [args.start, args.end], "business_days": len(rows),
        "days_with_losers": len(with_losers),
        "days_with_losers_list": [r["day"] for r in with_losers],
        "max_losers_on_any_day": max((r["lose"] for r in rows), default=0),
        "days_with_label_changed": sum(1 for r in rows if r["label_changed"] > 0),
        "total_label_changed_pairs": sum(r["label_changed"] for r in rows),
        "min_gain_over_window": min(r["gain"] for r in rows),
        "median_gain_over_window": sorted(r["gain"] for r in rows)[len(rows) // 2],
        "loser_days": with_losers,
        "per_day": rows,
        "database_writes": False, "weights_written": False,
    }
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_day", "loser_days")}, ensure_ascii=False, indent=1))
    for r in with_losers:
        print(f"  {r['day']}  lose {r['lose']:4d} (star {r['lose_is_star']:3d})  gain {r['gain']:4d}  changed {r['label_changed']}  {r['lose_codes'][:6]}")


if __name__ == "__main__":
    main()
