"""How many securities' 357/471 values change when industry labels are backfilled -- not just how many rows are added.

yeslab_471 (AttentionSpilloverCalculator) groups securities by industry and then by market-cap tercile
WITHIN each industry, and computes a leave-one-out peer mean of the attention proxy. yeslab_357
(IndustryStandAbsCalculator, 行业内收益标准化绝对值) standardises returns WITHIN industry. Both are
group statistics: the leave-one-out peer mean of security i in group G is (sum(G) - a_i)/(|G|-1) - a_i, so
ANY change to G's membership changes the value of EVERY member of G, not only of the security that joined
or left.

That makes the usual framing of option 1 -- "it adds the missing STAR rows" -- an undercount by a wide
margin. Every security sharing an industry with a newly labelled one gets a different 357 and 471 too.
This counts, per day, how many securities sit in an industry whose member set changes, under the union
rule recommended in option1_feasibility.md section 13 (index_member_all preferred, cache as fallback).

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
    def __init__(self, m: pd.DataFrame):
        self.codes = m["code"].to_numpy(); self.l1 = m["l1"].to_numpy()
        self.ins = m["in_p"].to_numpy(); self.outs = m["out_p"].to_numpy()
        self.open_ended = pd.isna(m["out_p"]).to_numpy()

    def groups(self, day) -> dict[str, set]:
        dv = np.datetime64(day)
        mask = (self.ins <= dv) & (self.open_ended | (self.outs > dv))
        out: dict[str, set] = {}
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
    rows = []
    for day in days:
        ga, gb = A.groups(day), B.groups(day)
        # union rule: member_all wins where it has a label, cache fills the rest
        gu: dict[str, set] = {}
        for l1, members in gb.items():
            gu.setdefault(l1, set()).update(members)
        for code in {c for s in ga.values() for c in s} - {c for s in gb.values() for c in s}:
            for l1, members in ga.items():
                if code in members:
                    gu.setdefault(l1, set()).add(code)
        # a security's value changes if ITS industry's member set differs between cache and union
        affected = set()
        added = set()
        for l1, after in gu.items():
            before = ga.get(l1, set())
            if before != after:
                affected |= after          # every member of a changed group, incl. the newcomers
                added |= (after - before)
        labelled_before = {c for s in ga.values() for c in s}
        labelled_after = {c for s in gu.values() for c in s}
        rows.append({
            "day": str(day.date()),
            "labelled_before": len(labelled_before), "labelled_after": len(labelled_after),
            "labels_added": len(labelled_after - labelled_before),
            "labels_added_star": sum(1 for c in labelled_after - labelled_before if c.startswith(STAR)),
            "industries_before": len(ga), "industries_after": len(gu),
            "industries_whose_membership_changed": sum(1 for l1 in gu if ga.get(l1, set()) != gu[l1]),
            "securities_in_a_changed_group": len(affected),
            "share_of_labelled_securities_whose_value_changes":
                len(affected) / len(labelled_after) if labelled_after else None,
        })

    df = pd.DataFrame(rows)
    out = {
        "question": "When industry labels are backfilled, how many securities' 357/471 values change -- "
                    "as opposed to how many rows are merely added?",
        "why_it_is_not_just_the_added_rows": (
            "both factors are group statistics. AttentionSpilloverCalculator computes a leave-one-out peer "
            "mean within industry x market-cap tercile, so (sum(G) - a_i)/(|G|-1) - a_i changes for EVERY "
            "member of G when G's membership changes. 行业内收益标准化绝对值 standardises within industry "
            "and behaves the same way."
        ),
        "union_rule": "index_member_all preferred, cache as fallback (option1_feasibility.md section 13)",
        "window": [args.start, args.end], "business_days": int(len(df)),
        "rows_added_per_day": {"min": int(df["labels_added"].min()), "median": int(df["labels_added"].median()),
                               "max": int(df["labels_added"].max()),
                               "of_which_star_median": int(df["labels_added_star"].median())},
        "securities_whose_value_changes_per_day": {
            "min": int(df["securities_in_a_changed_group"].min()),
            "median": int(df["securities_in_a_changed_group"].median()),
            "max": int(df["securities_in_a_changed_group"].max()),
        },
        "share_of_labelled_securities_whose_value_changes": {
            "min": float(df["share_of_labelled_securities_whose_value_changes"].min()),
            "median": float(df["share_of_labelled_securities_whose_value_changes"].median()),
            "max": float(df["share_of_labelled_securities_whose_value_changes"].max()),
        },
        "amplification_factor_median_value_changes_over_rows_added":
            float((df["securities_in_a_changed_group"] / df["labels_added"].replace(0, np.nan)).median()),
        "industries_whose_membership_changed": {
            "median": float(df["industries_whose_membership_changed"].median()),
            "max": int(df["industries_whose_membership_changed"].max()),
            "industries_before_median": float(df["industries_before"].median()),
        },
        "per_day": rows,
        "database_writes": False, "weights_written": False,
    }
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_day"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
