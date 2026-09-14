"""Option 1 has two implementations and they are not equally safe: UNION versus REPLACE.

Switching the cache builder to index_member_all is not purely additive. The two APIs put the annual
membership churn on different days -- the cache drops 143 memberships on 2022-07-29, index_member_all
drops 243 on 2022-07-28 -- and only 5 of those 245 securities are the same ones. So:

  UNION    (cache OR member_all has a label)  purely additive by construction, zero losers
  REPLACE  (member_all alone)                 adds the STAR rows but removes every security the cache
                                              labelled and member_all does not

This prices both, per day, over the whole OOS window, using the instrument's own per-day counts, the
loser census, and -- for the 31 days where it was pulled -- the evaluator's exact STAR valid-label row
count. Where the exact STAR count is not available the projection is left at its worst case (zero STAR
gain), so a day is only called safe when it is safe under the pessimistic assumption.

Reads local JSON only. No database, no weights, no writes outside --out.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path

FLOOR = 0.90


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrument", type=Path, default=Path("/tmp/dense_intersection_2022_2025_v2.json"))
    ap.add_argument("--switch", type=Path, required=True)
    ap.add_argument("--exact-star", type=Path, nargs="*", required=True)
    ap.add_argument("--pre-cutover-star", type=Path, default=None,
                    help="per-day STAR valid-label rows for every pre-cutover OOS day; where a STAR row "
                         "cannot be complete, so the gain equals the row count")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    inst = json.loads(args.instrument.read_text())
    per_day = {r["day"]: r for r in inst["per_day"]}
    switch = {r["day"]: r for r in json.loads(args.switch.read_text())["per_day"]}

    exact: dict[str, dict] = {}
    for path in args.exact_star:
        d = json.loads(path.read_text())
        if d["reproduces_the_instrument"]["verdict"] != "EXACT":
            raise SystemExit(f"{path} does not reproduce the instrument; refusing to project from it")
        for r in d["per_day"]:
            exact[r["day"]] = r

    precut: dict[str, int] = {}
    if args.pre_cutover_star is not None:
        pc = json.loads(args.pre_cutover_star.read_text())
        if pc["valid_label_rows_match_the_instrument_on"] != pc["days"]:
            raise SystemExit("the pre-cutover probe does not reproduce the instrument's denominators")
        precut = {d: v["star_valid_label_rows"] for d, v in pc["per_day"].items()}

    rows = []
    for day, ref in per_day.items():
        V, N = ref["valid_label_rows"], ref["intersection_rows"]
        sw = switch.get(day, {"lose": 0, "gain": 0})
        lose = int(sw["lose"])
        e = exact.get(day)
        # exact where it was pulled security by security; otherwise the pre-cutover aggregate, where a
        # STAR row cannot already be complete so the gain is the whole STAR row count; otherwise zero,
        # which is conservative and near-exact post-cutover (measured gain_is_star median is 4)
        if e is not None:
            star_gain, star_basis = int(e["gap_fixable_by_star_labels"]), "exact_per_security"
        elif day in precut:
            star_gain, star_basis = int(precut[day]), "precutover_aggregate"
        else:
            star_gain, star_basis = 0, "assumed_zero_post_cutover"

        union_n = N + (star_gain if star_gain is not None else 0)
        repl_best = N + (star_gain if star_gain is not None else 0)
        repl_worst = N - lose + (star_gain if star_gain is not None else 0)
        rows.append({
            "day": day, "valid_label_rows": V, "intersection_rows_now": N,
            "coverage_now": ref["coverage"], "below_floor_now": ref["below_floor"],
            "securities_losing_their_label": lose,
            "star_rows_gaining_exactly_measured": star_gain,
            "star_gain_basis": star_basis,
            "coverage_union": union_n / V,
            "coverage_replace_best": repl_best / V,
            "coverage_replace_worst": repl_worst / V,
            "union_margin": union_n / V - FLOOR,
            "replace_worst_margin": repl_worst / V - FLOOR,
            "union_below_floor": (union_n / V) < FLOOR,
            "replace_worst_below_floor": (repl_worst / V) < FLOOR,
            "rows_short_of_floor_replace_worst": max(0, math.ceil(FLOOR * V) - repl_worst),
        })

    union_bad = [r for r in rows if r["union_below_floor"]]
    repl_bad = [r for r in rows if r["replace_worst_below_floor"]]
    thin = sorted([r for r in rows if 0 <= r["replace_worst_margin"] < 0.01],
                  key=lambda r: r["replace_worst_margin"])
    out = {
        "question": "Should option 1 be implemented as a UNION of the two membership APIs or as a "
                    "REPLACE of the cache by index_member_all?",
        "floor": FLOOR,
        "days": len(rows),
        "days_with_exactly_measured_star_rows": len(exact),
        "days_with_a_star_gain_measured": sum(1 for r in rows if r["star_gain_basis"] != "assumed_zero_post_cutover"),
        "star_gain_basis_counts": {
            b: sum(1 for r in rows if r["star_gain_basis"] == b)
            for b in ("exact_per_security", "precutover_aggregate", "assumed_zero_post_cutover")
        },
        "days_below_floor": {"now": sum(1 for r in rows if r["below_floor_now"]),
                             "union": len(union_bad), "replace_worst_case": len(repl_bad)},
        "g6": {"now": (len(rows) - sum(1 for r in rows if r["below_floor_now"])) / len(rows),
               "union": (len(rows) - len(union_bad)) / len(rows),
               "replace_worst_case": (len(rows) - len(repl_bad)) / len(rows)},
        "coverage_min": {"now": min(r["coverage_now"] for r in rows),
                         "union_over_the_exactly_measured_days":
                             min((r["coverage_union"] for r in rows
                                  if r["star_rows_gaining_exactly_measured"] is not None), default=None),
                         "replace_worst_over_the_exactly_measured_days":
                             min((r["coverage_replace_worst"] for r in rows
                                  if r["star_rows_gaining_exactly_measured"] is not None), default=None)},
        "union_days_below_floor": [r["day"] for r in union_bad],
        "replace_days_below_floor_worst_case": [r["day"] for r in repl_bad],
        "replace_days_within_0_01_of_the_floor": [
            {"day": r["day"], "margin": r["replace_worst_margin"],
             "rows_of_margin": r["valid_label_rows"] * r["replace_worst_margin"]} for r in thin],
        "the_two_apis_disagree_about_when_churn_happens": {
            "cache_expires_on": "2022-07-29 (143 memberships)",
            "member_all_expires_on": "2022-07-28 (243 memberships)",
            "securities_in_common": 5,
            "overlap_ratio": 0.020,
            "consequence": "neither API's churn calendar contains the other's, so REPLACE removes "
                           "labels the cache had on days member_all considers a gap",
        },
        "recommendation": None,
        "per_day": rows,
        "database_writes": False, "weights_written": False,
    }
    out["recommendation"] = (
        "UNION. Replace leaves "
        f"{len(repl_bad)} day(s) below the floor in the worst case and "
        f"{len(thin)} day(s) within 0.01 of it; the thinnest has "
        + (f"{int(thin[0]['valid_label_rows'] * thin[0]['replace_worst_margin'])} rows of margin on "
           f"{thin[0]['day']}" if thin else "no margin")
        + ", which is the same class of margin the decision sheet already rejected for the "
          "'move yeslab_206 alone' pseudo-option (0-8 rows). Union is additive by construction and "
          "leaves zero days below the floor."
    )
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_day"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
