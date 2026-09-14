"""Do the other two documented failure shapes survive the switch to index_member_all?

root_cause.md documents three ways the PIT industry dependency fails: the STAR board-wide absence
(2022-01..2023-09, the only one inside the OOS window), a 31-day incomplete left edge
(2014-01-02..2014-02-20, 17 of 28 industries until 912 securities register on 2014-02-21), and a
one-day hole on PIT_CUTOVER (2021-12-10, 535 memberships expire right-exclusive while their 393
replacements start 2021-12-13). The second and third land in the TRAINING window, where the dense-nine
intersection is already below the floor on 1,045 of 1,948 days.

If the alternative API also repairs those two, option 1 is not only an OOS fix. This measures all
three shapes on both sources from local JSON only. No database, no weights, no writes outside --out.
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np
import pandas as pd

CACHE = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2/artifacts/tushare_cache")
CUTOVER = pd.Timestamp("2021-12-10")
LEFT_EDGE = (pd.Timestamp("2014-01-02"), pd.Timestamp("2014-02-20"))
STAR_STEP = pd.Timestamp("2023-09-25")


def prep(m: pd.DataFrame, code: str, l1: str) -> pd.DataFrame:
    m = m.rename(columns={code: "code", l1: "l1"})
    m["in_p"] = pd.to_datetime(m["in_date"].astype(str), format="%Y%m%d", errors="coerce")
    out = m["out_date"]
    m["out_p"] = pd.to_datetime(out.astype(str).str.replace(r"\.0$", "", regex=True),
                                format="%Y%m%d", errors="coerce")
    m.loc[out.isna() | out.astype(str).isin(["None", "nan", "", "NaT"]), "out_p"] = pd.NaT
    return m.dropna(subset=["in_p"])


def series(m: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    ins = np.sort(m["in_p"].to_numpy())
    outs = np.sort(m["out_p"].dropna().to_numpy())
    dv = days.to_numpy()
    star = m[m["code"].astype(str).str.startswith(("688", "689"))]
    si = np.sort(star["in_p"].to_numpy()); so = np.sort(star["out_p"].dropna().to_numpy())
    return pd.DataFrame({
        "active": np.searchsorted(ins, dv, "right") - np.searchsorted(outs, dv, "right"),
        "active_star": np.searchsorted(si, dv, "right") - np.searchsorted(so, dv, "right"),
        "industries": [m[(m.in_p <= d) & (m.out_p.isna() | (m.out_p > d))]["l1"].nunique() for d in days],
        "expirations": np.searchsorted(outs, dv, "right") - np.searchsorted(outs, dv - np.timedelta64(1, "D"), "right"),
        "registrations": np.searchsorted(ins, dv, "right") - np.searchsorted(ins, dv - np.timedelta64(1, "D"), "right"),
    }, index=days)


def shape(name: str, m: pd.DataFrame, days: pd.DatetimeIndex) -> dict:
    s = series(m, days)
    lo, hi = LEFT_EDGE
    edge = s.loc[lo:hi]
    cut = s.loc[CUTOVER] if CUTOVER in s.index else None
    pre = s.loc[:STAR_STEP - pd.Timedelta(days=1), "active_star"].max()
    drops = s["active"].diff()
    worst = drops.idxmin()
    return {
        "source": name,
        "shape_1_star_board_wide_absence": {
            "max_active_star_before_2023_09_25": int(pre),
            "active_star_on_2022_07_29": int(s.loc[pd.Timestamp("2022-07-29"), "active_star"]),
            "active_star_on_2023_07_05": int(s.loc[pd.Timestamp("2023-07-05"), "active_star"]),
            "first_day_active_star_nonzero": str(s[s.active_star > 0].index.min().date()),
        },
        "shape_2_left_edge_2014": {
            "industries_on_2014_01_02": int(s.loc[lo, "industries"]),
            "industries_on_2014_02_20": int(s.loc[hi, "industries"]),
            "industries_on_2014_02_21": int(s.loc[pd.Timestamp("2014-02-21"), "industries"]),
            "active_min_over_edge": int(edge["active"].min()),
            "active_on_2014_02_21": int(s.loc[pd.Timestamp("2014-02-21"), "active"]),
            "registrations_on_2014_02_21": int(s.loc[pd.Timestamp("2014-02-21"), "registrations"]),
            "edge_days_below_full_industry_count": int((edge["industries"] < s["industries"].max()).sum()),
        },
        "shape_3_cutover_hole_2021_12_10": {
            "expirations_on_2021_12_10": int(cut["expirations"]) if cut is not None else None,
            "registrations_on_2021_12_10": int(cut["registrations"]) if cut is not None else None,
            "registrations_on_2021_12_13": int(s.loc[pd.Timestamp("2021-12-13"), "registrations"]),
            "active_on_2021_12_09": int(s.loc[pd.Timestamp("2021-12-09"), "active"]),
            "active_on_2021_12_10": int(cut["active"]) if cut is not None else None,
            "active_on_2021_12_13": int(s.loc[pd.Timestamp("2021-12-13"), "active"]),
            "hole_present": bool(cut is not None and cut["active"] < s.loc[pd.Timestamp("2021-12-09"), "active"]),
        },
        "largest_single_day_drop": {"day": str(worst.date()), "delta": int(drops.min())},
        "days_with_a_drop_of_50_or_more": int((drops <= -50).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--member-all", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    days = pd.bdate_range("2014-01-01", "2026-12-31")
    A = prep(pd.concat([pd.read_json(f, orient="table")
                        for f in sorted(glob.glob(str(CACHE / "pit_industry_members_*_SI.json")))],
                       ignore_index=True), "con_code", "index_code")
    B = prep(pd.DataFrame(json.loads(args.member_all.read_text())), "ts_code", "l1_code")

    out = {
        "question": "Do the 2014 left edge and the 2021-12-10 cutover hole also disappear under "
                    "index_member_all, i.e. is option 1 a training-window fix and not only an OOS fix?",
        "interval_rule": "active(d) = in_date <= d < out_date (right-exclusive), the vendor's own rule",
        "sources": {"cache_index_member": shape("cache (index_member)", A, days),
                    "member_all": shape("index_member_all", B, days)},
        "database_writes": False, "weights_written": False,
    }
    a, b = out["sources"]["cache_index_member"], out["sources"]["member_all"]
    verdict = []
    for key, label in (("shape_1_star_board_wide_absence", "STAR board-wide absence"),
                       ("shape_2_left_edge_2014", "2014 left edge"),
                       ("shape_3_cutover_hole_2021_12_10", "2021-12-10 cutover hole")):
        verdict.append({"shape": label, "cache": a[key], "member_all": b[key]})
    out["verdict"] = verdict
    out["answer"] = {
        "shape_1": "REPAIRED by member_all" if b["shape_1_star_board_wide_absence"]["max_active_star_before_2023_09_25"] > 0
                   else "still absent",
        "shape_2": ("REPAIRED" if b["shape_2_left_edge_2014"]["industries_on_2014_01_02"]
                    > a["shape_2_left_edge_2014"]["industries_on_2014_01_02"] else "UNCHANGED")
                   + f" (industries on 2014-01-02: cache {a['shape_2_left_edge_2014']['industries_on_2014_01_02']}"
                     f" vs member_all {b['shape_2_left_edge_2014']['industries_on_2014_01_02']})",
        "shape_3": ("REPAIRED" if not b["shape_3_cutover_hole_2021_12_10"]["hole_present"] else "STILL PRESENT")
                   + f" (expirations on 2021-12-10: cache {a['shape_3_cutover_hole_2021_12_10']['expirations_on_2021_12_10']}"
                     f" vs member_all {b['shape_3_cutover_hole_2021_12_10']['expirations_on_2021_12_10']})",
    }
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out["answer"], ensure_ascii=False, indent=1))
    for v in verdict:
        print("---", v["shape"])
        print("   cache     :", json.dumps(v["cache"], ensure_ascii=False))
        print("   member_all:", json.dumps(v["member_all"], ensure_ascii=False))
    print("largest drop cache     :", a["largest_single_day_drop"], "days>=50:", a["days_with_a_drop_of_50_or_more"])
    print("largest drop member_all:", b["largest_single_day_drop"], "days>=50:", b["days_with_a_drop_of_50_or_more"])


if __name__ == "__main__":
    main()
