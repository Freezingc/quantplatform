"""Re-run the bulk-expiry census on the alternative API: which of the nine dates survive?

finding.md section 10.3 concluded the vendor expires industry memberships in bulk every year around
end-June, naming nine fixed dates, and 2022-07-29 -- the single dropped OOS day of 2022 -- was one of
them. That census read the cache built from index_member. Since index_member turns out to clamp and
bulk-register (912 securities on 2014-02-21, 559 on 2023-09-25, 535 expirations on 2021-12-10), the
nine dates may be properties of that API rather than of the vendor's calendar. This lists every day
with a large membership change under BOTH sources, side by side.

Reads local JSON only. No database, no weights, no writes outside --out.
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np
import pandas as pd

CACHE = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2/artifacts/tushare_cache")
CITED_NINE = ["2015-06-30", "2016-06-30", "2017-06-30", "2018-07-13", "2019-07-05",
              "2021-12-10", "2022-07-29", "2024-07-30", "2026-07-01"]


def prep(m, code, l1):
    m = m.rename(columns={code: "code", l1: "l1"})
    m["in_p"] = pd.to_datetime(m["in_date"].astype(str), format="%Y%m%d", errors="coerce")
    out = m["out_date"]
    m["out_p"] = pd.to_datetime(out.astype(str).str.replace(r"\.0$", "", regex=True), format="%Y%m%d", errors="coerce")
    m.loc[out.isna() | out.astype(str).isin(["None", "nan", "", "NaT"]), "out_p"] = pd.NaT
    return m.dropna(subset=["in_p"])


def big_days(m, days, threshold):
    ins = np.sort(m["in_p"].to_numpy()); outs = np.sort(m["out_p"].dropna().to_numpy())
    dv = days.to_numpy()
    active = np.searchsorted(ins, dv, "right") - np.searchsorted(outs, dv, "right")
    s = pd.Series(active, index=days)
    d = s.diff()
    reg = pd.Series(np.searchsorted(ins, dv, "right") - np.searchsorted(ins, dv - np.timedelta64(1, "D"), "right"), index=days)
    exp = pd.Series(np.searchsorted(outs, dv, "right") - np.searchsorted(outs, dv - np.timedelta64(1, "D"), "right"), index=days)
    hit = s[(d <= -threshold) | (d >= threshold)]
    return [{"day": str(x.date()), "active": int(s[x]), "delta": int(d[x]),
             "registrations": int(reg[x]), "expirations": int(exp[x])} for x in hit.index]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member-all", type=Path, required=True)
    ap.add_argument("--threshold", type=int, default=50)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    days = pd.bdate_range("2014-01-01", "2026-12-31")
    A = prep(pd.concat([pd.read_json(f, orient="table")
                        for f in sorted(glob.glob(str(CACHE / "pit_industry_members_*_SI.json")))],
                       ignore_index=True), "con_code", "index_code")
    B = prep(pd.DataFrame(json.loads(args.member_all.read_text())), "ts_code", "l1_code")
    a, b = big_days(A, days, args.threshold), big_days(B, days, args.threshold)
    a_days = {r["day"] for r in a}; b_days = {r["day"] for r in b}
    out = {
        "question": "Are the nine bulk-expiry dates a property of the vendor's calendar, or of the "
                    "index_member API the cache was built from?",
        "threshold": args.threshold,
        "cache_index_member": {"count": len(a), "days": a},
        "member_all": {"count": len(b), "days": b},
        "the_nine_dates_cited_in_finding_10_3": {
            d: {"in_cache_list": d in a_days, "in_member_all_list": d in b_days} for d in CITED_NINE
        },
        "survive_the_switch": sorted(a_days & b_days),
        "cache_only_artifacts": sorted(a_days - b_days),
        "member_all_only": sorted(b_days - a_days),
        "database_writes": False, "weights_written": False,
    }
    out["verdict"] = (
        f"{len(a_days - b_days)} of {len(a_days)} large-change days in the cache disappear under "
        f"index_member_all, including {sum(1 for d in CITED_NINE if d in a_days and d not in b_days)} "
        f"of the nine dates cited in finding.md 10.3; {len(a_days & b_days)} survive."
    )
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out["verdict"], ensure_ascii=False))
    print("cache days     :", [r["day"] for r in a])
    print("member_all days:", [r["day"] for r in b])
    print("nine cited     :", json.dumps(out["the_nine_dates_cited_in_finding_10_3"], ensure_ascii=False))


if __name__ == "__main__":
    main()
