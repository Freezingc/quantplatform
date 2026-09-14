"""Exact, not upper-bound: how many of the 24 days' missing rows are STAR rows a label backfill could lift.

The feasibility measurement said index_member_all supplies 446-551 STAR industry memberships per problem
day while the 0.90 floor needs 1-36 more intersection rows. That is an upper bound: a membership only
helps if the security is also in that day's valid-label row set. This reads the evaluator's own two
tables -- the same ones the screen reads, at the same published data_versions, resolved per year exactly
as the screen resolves them -- and counts, per problem day, how many valid-label rows are STAR, how many
of those already carry all nine dense factors, and therefore how big the fixable part of the gap is.

The numerator and denominator are cross-checked against the instrument's own per-day output; if either
fails to reproduce, the decomposition is not trustworthy and the run says so.

SAFETY: SELECT-only against dfs://QuantResearchV2Test. No writes anywhere except --out. No weights.
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path

REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
DENSE9 = ["yeslab_206", "yeslab_357", "yeslab_381", "yeslab_425", "yeslab_443_reg30_max20_v1",
          "yeslab_464", "yeslab_471", "yeslab_484", "yeslab_96"]
STAR = ("688", "689")


def sql_list(values) -> str:
    return ", ".join("'" + str(v).replace("'", "''") + "'" for v in values)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--floor", type=float, default=0.90)
    args = ap.parse_args()

    src = json.loads(args.days.read_text())
    days = sorted(src["days"] if "days" in src else src["summary"]["days_below_floor_list"])
    per_day = src.get("per_day") or {r["day"]: r for r in src.get("instrument", {}).get("per_day", [])}

    sys.path.insert(0, str(REPO / "scripts"))
    sys.path.insert(0, str(REPO / "src"))
    import pandas as pd
    from walk_forward_common import TARGET_DATABASE, current_versions, new_holder, query

    holder = new_holder()
    lpins, fpins = current_versions(holder, "label_daily"), current_versions(holder, "factor_daily")
    ids = sql_list(DENSE9)

    lab_frames, comp_frames, starfac_frames = [], [], []
    # per year, with that year's own published versions -- the screen's resolution rule, not a union
    for year in sorted({d[:4] for d in days}):
        ydays = [d for d in days if d.startswith(year)]
        parts = sorted({f"{d[:4]}-{d[5:7]}" for d in ydays})
        miss = [p for p in parts if p not in lpins or p not in fpins]
        if miss:
            raise SystemExit(f"unpublished partitions for {year}: {miss}")
        ll = sql_list(sorted({lpins[p] for p in parts}))
        fl = sql_list(sorted({fpins[p] for p in parts}))
        dl = ", ".join(d.replace("-", ".") for d in ydays)

        ok, lab = query(holder,
            f"select signal_date, ts_code from loadTable('{TARGET_DATABASE}','label_daily') "
            f"where signal_date in [{dl}] and label_valid=1 and data_version in [{ll}]")
        if not ok or not isinstance(lab, pd.DataFrame):
            raise SystemExit(f"label query failed for {year}: {lab}")
        lab_frames.append(lab)

        # two separate selects, never a right-table column through a join: the numerator set on its own
        script = (
            f"fac = select signal_date, ts_code, factor_id from loadTable('{TARGET_DATABASE}','factor_daily') "
            f"where signal_date in [{dl}] and factor_id in [{ids}] and data_version in [{fl}];"
            "dk = select signal_date, ts_code, factor_id from fac group by signal_date, ts_code, factor_id;"
            f"select signal_date, ts_code from dk group by signal_date, ts_code having count(*) = {len(DENSE9)}"
        )
        ok, comp = query(holder, script)
        if not ok or not isinstance(comp, pd.DataFrame):
            raise SystemExit(f"intersection query failed for {year}: {comp}")
        comp_frames.append(comp)

        script2 = (
            f"fac = select signal_date, ts_code, factor_id from loadTable('{TARGET_DATABASE}','factor_daily') "
            f"where signal_date in [{dl}] and factor_id in [{ids}] and data_version in [{fl}];"
            "dk = select signal_date, ts_code, factor_id from fac group by signal_date, ts_code, factor_id;"
            "p = select signal_date, factor_id, substr(ts_code,0,3) as pfx from dk;"
            "select signal_date, factor_id, pfx, count(*) as n from p group by signal_date, factor_id, pfx"
        )
        ok2, bf = query(holder, script2)
        if ok2 and isinstance(bf, pd.DataFrame):
            starfac_frames.append(bf)
        print(f"  {year}: label rows {len(lab)}, complete rows {len(comp)}", flush=True)

    lab = pd.concat(lab_frames, ignore_index=True)
    comp = pd.concat(comp_frames, ignore_index=True)
    for df in (lab, comp):
        df["day"] = pd.to_datetime(df["signal_date"]).dt.strftime("%Y-%m-%d")
        df["ts_code"] = df["ts_code"].astype(str)

    lab_sets = {d: set(g["ts_code"]) for d, g in lab.groupby("day")}
    comp_sets = {d: set(g["ts_code"]) for d, g in comp.groupby("day")}

    bf = pd.concat(starfac_frames, ignore_index=True) if starfac_frames else pd.DataFrame(
        columns=["signal_date", "factor_id", "pfx", "n"])
    if len(bf):
        bf["day"] = pd.to_datetime(bf["signal_date"]).dt.strftime("%Y-%m-%d")
        bf["pfx"] = bf["pfx"].astype(str)
    bf_star = bf[bf["pfx"].isin(STAR)].groupby(["day", "factor_id"])["n"].sum() if len(bf) else {}
    bf_all = bf.groupby(["day", "factor_id"])["n"].sum() if len(bf) else {}

    rows = []
    for day in days:
        L, C = lab_sets.get(day, set()), comp_sets.get(day, set())
        C = C & L  # the instrument restricts the numerator to valid-label rows
        V, N = len(L), len(C)
        star_rows = {c for c in L if c.startswith(STAR)}
        star_with9 = star_rows & C
        fixable = len(star_rows) - len(star_with9)
        need = math.ceil(args.floor * V) - N
        ref = per_day.get(day, {})
        rows.append({
            "day": day,
            "valid_label_rows": V,
            "valid_label_rows_per_instrument": ref.get("valid_label_rows"),
            "denominator_matches_instrument": V == ref.get("valid_label_rows"),
            "intersection_rows": N,
            "intersection_rows_per_instrument": ref.get("intersection_rows"),
            "numerator_matches_instrument": N == ref.get("intersection_rows"),
            "star_valid_label_rows": len(star_rows),
            "star_rows_already_complete": len(star_with9),
            "gap_fixable_by_star_labels": fixable,
            "rows_needed_to_reach_floor": need,
            "fixable_covers_need": bool(fixable >= need),
            "headroom_fixable_minus_need": fixable - need,
            "residual_gap_not_star": (V - N) - fixable,
            "star_rows_present_per_factor": {f: int(bf_star.get((day, f), 0)) for f in DENSE9},
            "all_rows_present_per_factor": {f: int(bf_all.get((day, f), 0)) for f in DENSE9},
        })

    repro_ok = all(r["denominator_matches_instrument"] and r["numerator_matches_instrument"] for r in rows)
    out = {
        "question": "Of the rows the 0.90 floor drops on the 24 problem days, how many are STAR rows "
                    "that an industry-label backfill could lift?",
        "database": TARGET_DATABASE, "dense_factor_set": DENSE9, "floor": args.floor,
        "days": len(rows),
        "reproduces_the_instrument": {
            "verdict": "EXACT" if repro_ok else "MISMATCH -- decomposition below is not trustworthy",
            "denominator_identical_on": sum(1 for r in rows if r["denominator_matches_instrument"]),
            "numerator_identical_on": sum(1 for r in rows if r["numerator_matches_instrument"]),
            "days": len(rows),
        },
        "fixable_covers_need_on_all_days": all(r["fixable_covers_need"] for r in rows),
        "days_where_fixable_covers_need": sum(1 for r in rows if r["fixable_covers_need"]),
        "min_headroom": min(r["headroom_fixable_minus_need"] for r in rows),
        "max_rows_needed": max(r["rows_needed_to_reach_floor"] for r in rows),
        "min_star_rows_available": min(r["gap_fixable_by_star_labels"] for r in rows),
        "max_star_rows_available": max(r["gap_fixable_by_star_labels"] for r in rows),
        "per_day": rows,
        "database_writes": False, "weights_written": False,
    }
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_day"}, ensure_ascii=False, indent=1))
    print(f"{'day':12}{'V':>6}{'N':>6}{'need':>6}{'star':>6}{'s9':>5}{'fix':>6}{'head':>6}{'resid':>7}")
    for r in rows:
        print(f"{r['day']:12}{r['valid_label_rows']:6d}{r['intersection_rows']:6d}"
              f"{r['rows_needed_to_reach_floor']:6d}{r['star_valid_label_rows']:6d}"
              f"{r['star_rows_already_complete']:5d}{r['gap_fixable_by_star_labels']:6d}"
              f"{r['headroom_fixable_minus_need']:6d}{r['residual_gap_not_star']:7d}")


if __name__ == "__main__":
    main()
