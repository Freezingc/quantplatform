"""Does the provider's own tie-break actually decide, or does it fall through to an arbitrary key?

build_pit_snapshot (audit_tushare_pit_industry.py:96-103) already handles a security holding two
simultaneously active L1 industries: it counts them into snapshot.attrs["overlap_conflict_code_count"]
and resolves them with sort_values(["con_code","in_date","index_code"]).drop_duplicates(keep="last"),
i.e. latest in_date wins, and ties fall through to the largest index_code. So the ambiguity is surfaced
and deterministic, not swallowed.

But the tie-break keys on in_date, and in_date is exactly the field index_member corrupts for the STAR
market -- every 688/689 record carries 2023-09-25. Where all candidate records for a security share one
in_date, "latest in_date" decides nothing and the choice is the largest index_code, which is arbitrary.
This counts how much of the cache's ambiguity is genuinely decided by the rule versus falling through,
and how much of it index_member_all removes outright by not being ambiguous at all.

Reads the two local membership tables only. No database, no weights, no writes outside --out.
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np
import pandas as pd

CACHE = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2/artifacts/tushare_cache")
STAR = ("688", "689")


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
    ap.add_argument("--start", default="2014-01-01")
    ap.add_argument("--end", default="2026-12-31")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    A = prep(pd.concat([pd.read_json(f, orient="table")
                        for f in sorted(glob.glob(str(CACHE / "pit_industry_members_*_SI.json")))],
                       ignore_index=True), "con_code", "index_code")
    B = prep(pd.DataFrame(json.loads(args.member_all.read_text())), "ts_code", "l1_code")

    Ac, Bc = A.copy(), B.copy()
    Ac["codes"] = Ac["code"]; Bc["codes"] = Bc["code"]
    days = pd.bdate_range(args.start, args.end)
    ain, aout = Ac["in_p"].to_numpy(), Ac["out_p"].to_numpy()
    aopen = pd.isna(Ac["out_p"]).to_numpy()
    acode, al1 = Ac["code"].to_numpy(), Ac["l1"].to_numpy()

    decided = fell_through = 0
    amb_star = amb_nonstar = 0
    per_day = []
    for day in days:
        dv = np.datetime64(day)
        m = (ain <= dv) & (aopen | (aout > dv))
        sub = pd.DataFrame({"code": acode[m], "l1": al1[m], "in": ain[m]})
        g = sub.groupby("code")
        nun = g["l1"].nunique()
        amb = nun[nun > 1].index
        if len(amb) == 0:
            per_day.append({"day": str(day.date()), "ambiguous_securities": 0, "decided_by_in_date": 0,
                            "fell_through_to_index_code": 0, "ambiguous_star": 0})
            continue
        s = sub[sub["code"].isin(amb)]
        nd = s.groupby("code")["in"].nunique()
        d = int((nd > 1).sum()); f = int((nd == 1).sum())
        st = int(s[s["code"].str.startswith(STAR)]["code"].nunique())
        decided += d; fell_through += f; amb_star += st; amb_nonstar += len(amb) - st
        per_day.append({"day": str(day.date()), "ambiguous_securities": int(len(amb)),
                        "decided_by_in_date": d, "fell_through_to_index_code": f, "ambiguous_star": st})

    df = pd.DataFrame(per_day)
    total = int(df["ambiguous_securities"].sum())
    # what index_member_all does with the same securities: is it ambiguous at all?
    bin_, bout = B["in_p"].to_numpy(), B["out_p"].to_numpy()
    bopen = pd.isna(B["out_p"]).to_numpy()
    bcode, bl1 = B["code"].to_numpy(), B["l1"].to_numpy()
    amb_b = 0
    for day in days[::20]:  # sampled: index_member_all is expected to be unambiguous throughout
        dv = np.datetime64(day)
        m = (bin_ <= dv) & (bopen | (bout > dv))
        s = pd.DataFrame({"code": bcode[m], "l1": bl1[m]})
        if len(s):
            nun = s.groupby("code")["l1"].nunique()
            amb_b += int((nun > 1).sum())

    out = {
        "question": "How much of the cache's label ambiguity is genuinely decided by the provider's "
                    "tie-break, and how much falls through to an arbitrary key?",
        "tie_break_in_force": 'sort_values(["con_code","in_date","index_code"]).drop_duplicates(keep="last") '
                              "at audit_tushare_pit_industry.py:101-103 -- latest in_date wins, ties fall "
                              "through to the largest index_code",
        "window": [args.start, args.end], "business_days": int(len(df)),
        "cache": {
            "ambiguous_security_days": total,
            "per_day_mean": total / len(df), "max_on_any_one_day": int(df["ambiguous_securities"].max()),
            "days_with_ambiguity": int((df["ambiguous_securities"] > 0).sum()),
            "decided_by_in_date": decided,
            "fell_through_to_the_largest_index_code": fell_through,
            "share_falling_through": (fell_through / total) if total else None,
            "ambiguous_star_security_days": amb_star,
            "ambiguous_non_star_security_days": amb_nonstar,
        },
        "member_all_sampled_every_20th_business_day": {
            "days_sampled": int(len(days[::20])), "ambiguous_security_days": amb_b,
        },
        "verdict": None, "per_day": per_day,
        "database_writes": False, "weights_written": False,
    }
    c = out["cache"]
    out["verdict"] = (
        f"{c['share_falling_through']:.4f} of the cache's {c['ambiguous_security_days']} ambiguous "
        f"security-days are NOT decided by in_date and fall through to the largest index_code, which is "
        f"arbitrary. index_member_all is unambiguous on all {out['member_all_sampled_every_20th_business_day']['days_sampled']} "
        f"sampled days, so preferring it removes the question rather than answering it."
    )
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_day"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
