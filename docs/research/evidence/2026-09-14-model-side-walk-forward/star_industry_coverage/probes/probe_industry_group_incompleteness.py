"""How incomplete were the industry groups that 357 and 471 were actually computed on?

base.py:112-118 (industry_zscore_and_skew) does:

    valid = values.notna() & groups.notna()
    grouped = values.loc[valid].groupby(groups.loc[valid], sort=False)
    means = grouped.transform("mean"); stds = grouped.transform("std")
    daily_zscore = (values.loc[valid] - means) / stds

and AttentionSpilloverCalculator does row_df.dropna(subset=["attention","mv","industry"]) before grouping
by industry and market-cap tercile. In both, a security with NO industry label is excluded from its
industry's statistics -- so the mean, the standard deviation and the peer set that every OTHER member of
that industry is measured against were computed without it.

That makes the STAR label gap a two-sided defect. The visible side is the one already documented: STAR
rows get no 357/471 at all, the nine-factor intersection shrinks, and 24 OOS days fall below the floor.
The invisible side has not been written down anywhere: every non-STAR security in every affected industry
was standardised against an incomplete group, on every day before 2023-09-25 -- including the 945 days
that passed the floor and the training years.

This sizes the invisible side per industry per day: how many members the group actually had, how many it
should have had under the recommended union rule, and therefore what share of each group was missing.

Reads the two local membership tables only. No database, no weights, no writes outside --out.
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np
import pandas as pd

CACHE = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2/artifacts/tushare_cache")
STAR = ("688", "689")


class Table:
    def __init__(self, m):
        self.codes = m["code"].to_numpy(); self.l1 = m["l1"].to_numpy()
        self.ins = m["in_p"].to_numpy(); self.outs = m["out_p"].to_numpy()
        self.open_ended = pd.isna(m["out_p"]).to_numpy()

    def groups(self, day):
        dv = np.datetime64(day)
        mask = (self.ins <= dv) & (self.open_ended | (self.outs > dv))
        out = {}
        for c, l in zip(self.codes[mask], self.l1[mask]):
            out.setdefault(l, set()).add(c)
        return out


def prep(m, code, l1):
    m = m.rename(columns={code: "code", l1: "l1"})
    m["in_p"] = pd.to_datetime(m["in_date"].astype(str), format="%Y%m%d", errors="coerce")
    o = m["out_date"]
    m["out_p"] = pd.to_datetime(o.astype(str).str.replace(r"\.0$", "", regex=True), format="%Y%m%d", errors="coerce")
    m.loc[o.isna() | o.astype(str).isin(["None", "nan", "", "NaT"]), "out_p"] = pd.NaT
    return m.dropna(subset=["in_p"])[["code", "l1", "in_p", "out_p"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--member-all", type=Path, required=True)
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2023-09-24")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    A = Table(prep(pd.concat([pd.read_json(f, orient="table")
                              for f in sorted(glob.glob(str(CACHE / "pit_industry_members_*_SI.json")))],
                             ignore_index=True), "con_code", "index_code"))
    B = Table(prep(pd.DataFrame(json.loads(args.member_all.read_text())), "ts_code", "l1_code"))

    days = pd.bdate_range(args.start, args.end)
    per_day, per_industry = [], []
    for day in days:
        ga, gb = A.groups(day), B.groups(day)
        gu = {l1: set(m) for l1, m in gb.items()}
        have_b = {c for s in gb.values() for c in s}
        for l1, members in ga.items():
            gu.setdefault(l1, set()).update(members - have_b)
        shares, missing = [], []
        for l1 in sorted(set(ga) | set(gu)):
            had, should = len(ga.get(l1, ())), len(gu.get(l1, ()))
            if should == 0:
                continue
            miss = should - had
            shares.append(miss / should); missing.append(miss)
            per_industry.append({"day": str(day.date()), "l1": l1, "members_used": had,
                                 "members_under_union": should, "missing": miss,
                                 "missing_share": miss / should})
        per_day.append({"day": str(day.date()), "industries": len(shares),
                        "industries_with_any_missing_member": int(sum(1 for s in shares if s > 0)),
                        "median_missing_share": float(np.median(shares)) if shares else None,
                        "max_missing_share": float(max(shares)) if shares else None,
                        "total_members_used": int(sum(len(v) for v in ga.values())),
                        "total_members_under_union": int(sum(len(v) for v in gu.values()))})

    d = pd.DataFrame(per_day); i = pd.DataFrame(per_industry)
    out = {
        "question": "How incomplete were the industry groups that yeslab_357 and yeslab_471 were computed on?",
        "code_basis": {
            "yeslab_357": "base.py:112-118 industry_zscore_and_skew -- valid = values.notna() & "
                          "groups.notna(); the group mean and std are transformed over `valid` only",
            "yeslab_471": "AttentionSpilloverCalculator._calculate -- row_df.dropna(subset=['attention',"
                          "'mv','industry']) before groupby('industry') and market-cap tercile bucketing",
            "consequence": "a security with no label is excluded from its industry's statistics, so every "
                           "OTHER member of that industry is measured against an incomplete group",
        },
        "union_rule": "index_member_all preferred, cache as fallback",
        "window": [args.start, args.end], "business_days": int(len(d)),
        "per_day": {
            "industries_with_any_missing_member": {
                "median": float(d["industries_with_any_missing_member"].median()),
                "max": int(d["industries_with_any_missing_member"].max()),
                "of_industries_median": float(d["industries"].median())},
            "median_missing_share_of_a_group": {
                "min": float(d["median_missing_share"].min()), "median": float(d["median_missing_share"].median()),
                "max": float(d["median_missing_share"].max())},
            "worst_industry_missing_share_on_any_day": float(d["max_missing_share"].max()),
            "total_members_used_median": float(d["total_members_used"].median()),
            "total_members_under_union_median": float(d["total_members_under_union"].median()),
        },
        "per_industry_distribution_of_missing_share": {
            "industry_days": int(len(i)),
            "industry_days_with_a_missing_member": int((i["missing"] > 0).sum()),
            "share_of_industry_days_affected": float((i["missing"] > 0).mean()),
            "missing_count_quantiles": {q: float(i["missing"].quantile(q)) for q in (0.5, 0.9, 0.99)},
            "missing_share_quantiles": {q: float(i["missing_share"].quantile(q)) for q in (0.5, 0.9, 0.99)},
            "worst_ten_industry_days": i.nlargest(10, "missing_share")[
                ["day", "l1", "members_used", "members_under_union", "missing", "missing_share"]
            ].to_dict(orient="records"),
        },
        "what_this_does_not_say": (
            "This sizes the INCIDENCE -- how much of each group was missing -- not the MAGNITUDE of the "
            "resulting change in any security's factor value. Magnitude needs a replay, which is the "
            "owner-gated next step and was not run here."
        ),
        "per_day_rows": per_day,
        "database_writes": False, "weights_written": False,
    }
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_day_rows"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
