"""Design and validate the label-precedence rule option 1 needs, instead of leaving it as a footnote.

The decision sheet recommends building option 1 as a UNION of the two membership APIs rather than a
replacement, and then flags that a union needs an explicit precedence rule because the two disagree on
114 label pairs across 39 days -- and that the rule is neither implemented nor validated. It also notes
an unquantified cost: a union can mask a genuine expiry, because a security that really left an industry
still carries a label from the half that has not expired it yet.

This proposes one rule and tests it against the properties that matter, over the whole 2014-2026 window:

  RULE  label(security, day) = index_member_all's active L1 if it has one,
                               else the cache's active L1 if it has one,
                               else unlabelled.
        index_member_all wins because it is the only source carrying true in_date values for the
        STAR market, and because on the overlap window the two agree on 34,570 of 34,571 pairs, so
        preferring it changes almost nothing that was already settled.

Tested properties, each of which a bad rule would fail:
  P1 determinism    no (security, day) is left ambiguous -- at most one label survives the rule
  P2 no new holes   the union's daily active count is never below EITHER source's
  P3 repairs all three documented failure shapes (STAR absence, 2014 left edge, 2021-12-10 cutover)
  P4 churn smoothing  how many of the ±50 large-change days survive
  P5 the price      how many (security, day) labels the union keeps that index_member_all considers
                    already expired -- the masking cost, counted rather than hand-waved
  P6 precedence actually bites  how many labels come from the cache fallback rather than the preferred
                    source, i.e. whether the rule is doing anything at all

Reads the two local membership tables only. No database, no weights, no writes outside --out.
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np
import pandas as pd

CACHE = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2/artifacts/tushare_cache")
STAR = ("688", "689")


def prep(m: pd.DataFrame, code: str, l1: str) -> pd.DataFrame:
    m = m.rename(columns={code: "code", l1: "l1"})
    m["in_p"] = pd.to_datetime(m["in_date"].astype(str), format="%Y%m%d", errors="coerce")
    o = m["out_date"]
    m["out_p"] = pd.to_datetime(o.astype(str).str.replace(r"\.0$", "", regex=True), format="%Y%m%d", errors="coerce")
    m.loc[o.isna() | o.astype(str).isin(["None", "nan", "", "NaT"]), "out_p"] = pd.NaT
    return m.dropna(subset=["in_p"])[["code", "l1", "in_p", "out_p"]]


class Active:
    """Membership active on a day, right-exclusive, with the per-day mask done in numpy not groupby.

    The obvious implementation filters the frame per day and groups by security; over 3,143 business days
    that is tens of millions of pandas row operations and does not finish. Numpy boolean masks over the
    two columns plus a zip into a dict is the same answer in seconds.
    """

    def __init__(self, m: pd.DataFrame):
        self.codes = m["code"].to_numpy()
        self.l1 = m["l1"].to_numpy()
        self.ins = m["in_p"].to_numpy()
        outs = m["out_p"].to_numpy()
        self.outs = outs
        self.open_ended = pd.isna(m["out_p"]).to_numpy()
        # earliest expiry per security, for the masking test: a cache-fallback label on day D masks a
        # closed membership iff the preferred source has SOME record for that security expiring by D
        closed = m.dropna(subset=["out_p"])
        self.first_out = closed.groupby("code")["out_p"].min().to_dict() if len(closed) else {}

    def on(self, day: pd.Timestamp) -> dict[str, set]:
        dv = np.datetime64(day)
        mask = (self.ins <= dv) & (self.open_ended | (self.outs > dv))
        out: dict[str, set] = {}
        for c, l in zip(self.codes[mask], self.l1[mask]):
            out.setdefault(c, set()).add(l)
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--member-all", type=Path, required=True)
    ap.add_argument("--start", default="2014-01-01")
    ap.add_argument("--end", default="2026-12-31")
    ap.add_argument("--threshold", type=int, default=50)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prefer", choices=("member_all", "cache", "surgical"), default="member_all",
                    help="which source wins where BOTH carry a label. member_all is the higher-quality "
                         "table (zero internal ambiguity, true in_date values) but restates ~136 "
                         "securities a day; cache changes nothing that already has a label, so it is "
                         "purely additive, but inherits the cache's ambiguity. surgical is cache-preferred "
                         "restricted to STAR securities before the 2023-09-25 cutover: the smallest change "
                         "that repairs the 24 OOS days, and nothing else.")
    args = ap.parse_args()

    A = prep(pd.concat([pd.read_json(f, orient="table")
                        for f in sorted(glob.glob(str(CACHE / "pit_industry_members_*_SI.json")))],
                       ignore_index=True), "con_code", "index_code")
    B = prep(pd.DataFrame(json.loads(args.member_all.read_text())), "ts_code", "l1_code")
    days = pd.bdate_range(args.start, args.end)

    actA, actB = Active(A), Active(B)
    if args.prefer == "member_all":
        actP, actS = actB, actA
    elif args.prefer == "cache":
        actP, actS = actA, actB
    else:  # surgical: the cache is authoritative everywhere it speaks; member_all only fills STAR pre-cutover
        actP, actS = actA, None
    rows, masks, from_cache = [], 0, 0
    agree = amb_pick = amb_other = genuine = 0
    ambA = ambB = ambU = 0
    for day in days:
        ma, mb = actA.on(day), actB.on(day)
        mp = actP.on(day)
        if actS is None:
            star_pre = day < pd.Timestamp("2023-09-25")
            ms = {c: v for c, v in mb.items() if star_pre and c.startswith(STAR)}
        else:
            ms = actS.on(day)
        ambA += sum(1 for v in ma.values() if len(v) > 1)
        ambB += sum(1 for v in mb.values() if len(v) > 1)
        union: dict[str, set] = {}
        day_from_cache = day_masks = 0
        day_agree = day_cache_ambiguous_member_all_picks_one = 0
        day_cache_ambiguous_other = day_genuine_disagreement = 0
        for code in set(ma) | set(mb):
            lab = mp.get(code) or ms.get(code)          # RULE: the preferred source wins, the other fills
            if lab is None:
                continue
            union[code] = lab
            if code not in mp and code in ms:
                day_from_cache += 1
                # the price: does the preferred source consider this membership already closed?
                fo = actP.first_out.get(code)
                if fo is not None and fo <= day:
                    day_masks += 1
            if code in ma and code in mb:
                if ma[code] == mb[code]:
                    day_agree += 1
                elif len(ma[code]) > 1 and len(mb[code]) == 1 and mb[code] <= ma[code]:
                    day_cache_ambiguous_member_all_picks_one += 1
                elif len(ma[code]) > 1:
                    day_cache_ambiguous_other += 1
                else:
                    day_genuine_disagreement += 1
        ambU += sum(1 for v in union.values() if len(v) > 1)
        star_u = sum(1 for c in union if c.startswith(STAR))
        rows.append({"day": str(day.date()),
                     # SECURITIES, not labels: the cache can list one security under two L1s at once, so a
                     # label count would make the union look like it lost rows when it only lost a duplicate
                     "active_cache": len(ma), "active_member_all": len(mb), "active_union": len(union),
                     "labels_cache": sum(len(v) for v in ma.values()),
                     "labels_member_all": sum(len(v) for v in mb.values()),
                     "labels_union": sum(len(v) for v in union.values()),
                     "ambiguous_securities_cache": sum(1 for v in ma.values() if len(v) > 1),
                     "ambiguous_securities_member_all": sum(1 for v in mb.values() if len(v) > 1),
                     "ambiguous_securities_union": sum(1 for v in union.values() if len(v) > 1),
                     "star_active_cache": sum(1 for c in ma if c.startswith(STAR)),
                     "star_active_member_all": sum(1 for c in mb if c.startswith(STAR)),
                     "star_active_union": star_u,
                     "labels_from_cache_fallback": day_from_cache,
                     "labels_masking_an_expired_membership": day_masks,
                     "both_sources_labelled_and_agree": day_agree,
                     "cache_ambiguous_member_all_picks_one_of_them": day_cache_ambiguous_member_all_picks_one,
                     "cache_ambiguous_other": day_cache_ambiguous_other,
                     "genuine_single_label_disagreement": day_genuine_disagreement,
                     "industries_union": len({next(iter(v)) for v in union.values() if len(v) == 1})})
        masks += day_masks; from_cache += day_from_cache
        agree += day_agree; amb_pick += day_cache_ambiguous_member_all_picks_one
        amb_other += day_cache_ambiguous_other; genuine += day_genuine_disagreement

    df = pd.DataFrame(rows)
    df["d_union"] = df["active_union"].diff()
    big = df[(df["d_union"] <= -args.threshold) | (df["d_union"] >= args.threshold)]

    def window(sub: pd.DataFrame, col: str, lo: str, hi: str) -> dict:
        w = sub[(sub.day >= lo) & (sub.day <= hi)]
        return {"min": int(w[col].min()), "max": int(w[col].max())} if len(w) else {}

    p2_violations = df[df["active_union"] < df[["active_cache", "active_member_all"]].max(axis=1)]
    out = {
        "prefer": args.prefer,
        "rule": (f"label = the {args.prefer} active L1 if any, else the other source's active L1 if any, "
                 "else unlabelled"),
        "window": [args.start, args.end], "business_days": int(len(df)),
        "P1_determinism": {
            "ambiguous_security_days_in_cache": ambA,
            "ambiguous_security_days_in_member_all": ambB,
            "ambiguous_security_days_after_the_rule": ambU,
            "verdict": ("PASS" if ambU == 0 else
                        "FAIL -- the rule inherits the CACHE's ambiguity on the days the preferred source "
                        "has no label; a tie-break inside the cache is also required"),
        },
        "P2_no_new_holes": {
            "days_where_union_is_below_either_source": int(len(p2_violations)),
            "verdict": "PASS" if len(p2_violations) == 0 else "FAIL",
        },
        "P3_the_three_failure_shapes": {
            "star_before_cutover": {
                "cache": window(df[df.day < "2023-09-25"], "star_active_cache", "2022-01-01", "2023-09-24"),
                "member_all": window(df[df.day < "2023-09-25"], "star_active_member_all", "2022-01-01", "2023-09-24"),
                "union": window(df[df.day < "2023-09-25"], "star_active_union", "2022-01-01", "2023-09-24"),
            },
            "left_edge_2014": {
                "industries_on_2014_01_02": int(df.loc[df.day == "2014-01-02", "industries_union"].iloc[0]),
                "active_on_2014_01_02": int(df.loc[df.day == "2014-01-02", "active_union"].iloc[0]),
            },
            "cutover_2021_12_10": {
                "active_2021_12_09": int(df.loc[df.day == "2021-12-09", "active_union"].iloc[0]),
                "active_2021_12_10": int(df.loc[df.day == "2021-12-10", "active_union"].iloc[0]),
                "active_2021_12_13": int(df.loc[df.day == "2021-12-13", "active_union"].iloc[0]),
                "hole_present": bool(int(df.loc[df.day == "2021-12-10", "active_union"].iloc[0])
                                     < int(df.loc[df.day == "2021-12-09", "active_union"].iloc[0])),
            },
        },
        "P4_churn_smoothing": {
            "threshold": args.threshold,
            "large_change_days_union": int(len(big)),
            "large_change_days": [{"day": r["day"], "delta": int(r["d_union"])} for _, r in big.iterrows()],
        },
        "P5_the_price_masking_a_genuine_expiry": {
            "security_days_where_the_union_keeps_a_label_member_all_considers_expired": masks,
            "as_a_share_of_all_union_labels": None,
            "max_on_any_one_day": int(df["labels_masking_an_expired_membership"].max()),
            "days_where_it_is_nonzero": int((df["labels_masking_an_expired_membership"] > 0).sum()),
        },
        "P6_does_the_rule_do_anything": {
            "labels_supplied_by_the_cache_fallback": from_cache,
            "max_from_cache_on_any_one_day": int(df["labels_from_cache_fallback"].max()),
        },
        "P7_agreement_census_over_the_whole_universe": {
            "note": ("security-days where BOTH sources carry a label. The earlier 0.999971 agreement "
                     "figure was measured on STAR securities only; this is the whole universe, and the "
                     "cache's own ambiguity is separated out from genuine disagreement."),
            "both_labelled_and_identical": agree,
            "cache_ambiguous_and_member_all_picks_one_of_them": amb_pick,
            "cache_ambiguous_other_shape": amb_other,
            "genuine_single_label_disagreement": genuine,
            "pairs_compared": agree + amb_pick + amb_other + genuine,
            "agreement_rate_all_pairs": agree / max(1, agree + amb_pick + amb_other + genuine),
            "genuine_disagreement_rate": genuine / max(1, agree + amb_pick + amb_other + genuine),
            "genuine_disagreement_per_day_mean": genuine / max(1, len(df)),
            "days_with_any_genuine_disagreement": int((df["genuine_single_label_disagreement"] > 0).sum()),
        },
        "P8_the_cache_is_internally_ambiguous": {
            "note": ("a security with more than one simultaneously active L1 industry, under the "
                     "right-exclusive interval rule the provider itself uses"),
            "ambiguous_security_days_in_cache": ambA,
            "ambiguous_security_days_in_member_all": ambB,
            "ambiguous_security_days_after_the_rule": ambU,
            "per_day_mean_in_cache": ambA / max(1, len(df)),
            "max_on_any_one_day_in_cache": int(df["ambiguous_securities_cache"].max()),
            "days_where_the_cache_is_ambiguous": int((df["ambiguous_securities_cache"] > 0).sum()),
        },
        "per_day": rows,
        "database_writes": False, "weights_written": False,
    }
    total = int(df["active_union"].sum())
    out["P5_the_price_masking_a_genuine_expiry"]["as_a_share_of_all_union_labels"] = masks / total if total else None
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_day"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
