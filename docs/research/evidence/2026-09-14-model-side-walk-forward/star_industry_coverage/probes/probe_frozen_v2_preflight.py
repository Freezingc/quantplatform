"""Pre-flight for all fifteen frozen-v2 factors: is each one actually published, dense, and STAR-complete?

The factor set is still being decided. factor_set_frozen_v2.json splits the fifteen into nine
baseline_eligible (the dense subset the evaluator's coverage floor is computed on) and six candidate_only
(sparse inputs a candidate may use but the baseline must not), and records that yeslab_398 and yeslab_405
were dropped for redundancy against yeslab_381. §AM then found that yeslab_363, which appears in the
materializer's PIT_FACTORS, has zero published rows at all -- so membership in a list is not evidence that a
factor exists.

This measures, for every one of the fifteen and on days chosen to straddle the 2023-09-25 industry-cache
cutover: how many rows factor_daily holds, how many of those are inside the registry-resolved universe, the
resulting coverage of the universe, and the STAR-only coverage. A factor that is meant to be dense and comes
back below full coverage, or a STAR coverage that jumps at the cutover, is exactly the kind of thing that
should be known before the list is fixed rather than nine days later in the evaluation stage.

SAFETY: SELECT-only against dfs://QuantResearchV2Test. No writes except --out. No weights.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd

REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
DIAG = REPO / "artifacts" / "diagnostics"
UNIVERSE_ID = "cn_liquid_a"
STAR = ("688", "689")


def sql_list(v) -> str:
    return ", ".join("'" + str(x).replace("'", "''") + "'" for x in v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set-file", type=Path, default=DIAG / "factor_set_frozen_v2.json")
    ap.add_argument("--days", default="2022-01-04,2022-06-30,2022-07-29,2023-07-05,2023-09-22,2023-09-25,2024-06-17,2024-12-20,2025-06-16,2025-12-31")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    spec = json.loads(args.set_file.read_text())
    factors = list(spec["factors"])
    eligible = set(spec.get("baseline_eligible", []))
    candidate_only = set(spec.get("candidate_only", []))
    days = args.days.split(",")
    print(f"  {len(factors)} factors, {len(days)} days", flush=True)

    sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
    from walk_forward_common import TARGET_DATABASE, current_versions, new_holder, query
    h = new_holder()
    fpins, upins = current_versions(h, "factor_daily"), current_versions(h, "universe_daily")

    rows = []
    for day in days:
        dl = day.replace("-", "."); part = day[:7]
        if part not in fpins or part not in upins:
            print(f"  {day}: unpublished partition, skipped", flush=True); continue
        fl, ul = sql_list([fpins[part]]), sql_list([upins[part]])
        ok, uni = query(h,
            f"select ts_code from loadTable('{TARGET_DATABASE}','universe_daily') "
            f"where signal_date = date('{dl}') and universe_id='{UNIVERSE_ID}' and is_member=1 "
            f"and data_version in [{ul}]")
        members = set(uni.ts_code.astype(str)) if ok and isinstance(uni, pd.DataFrame) else set()
        star_members = {c for c in members if c.startswith(STAR)}
        ok2, fac = query(h,
            f"v = select factor_id, ts_code from loadTable('{TARGET_DATABASE}','factor_daily') "
            f"where signal_date = date('{dl}') and factor_id in [{sql_list(factors)}] and data_version in [{fl}] "
            "group by factor_id, ts_code;"
            "select factor_id, count(*) as n from v group by factor_id")
        counts = dict(zip(fac.factor_id.astype(str), fac.n.astype(int))) if ok2 and isinstance(fac, pd.DataFrame) else {}
        # STAR coverage needs the per-factor code sets, so one more query restricted to STAR prefixes
        ok3, sfac = query(h,
            f"v = select factor_id, ts_code from loadTable('{TARGET_DATABASE}','factor_daily') "
            f"where signal_date = date('{dl}') and factor_id in [{sql_list(factors)}] and data_version in [{fl}] "
            "group by factor_id, ts_code;"
            "select factor_id, count(*) as n from v where ts_code like '688%' or ts_code like '689%' group by factor_id")
        star_counts = dict(zip(sfac.factor_id.astype(str), sfac.n.astype(int))) if ok3 and isinstance(sfac, pd.DataFrame) else {}
        for f in factors:
            n = int(counts.get(f, 0)); sn = int(star_counts.get(f, 0))
            rows.append({"day": day, "factor_id": f,
                         "role": ("baseline_eligible" if f in eligible else
                                  "candidate_only" if f in candidate_only else "unclassified"),
                         "universe_members": len(members), "star_universe_members": len(star_members),
                         "rows": n, "coverage_of_universe": (n / len(members)) if members else None,
                         "star_rows": sn,
                         "star_coverage": (sn / len(star_members)) if star_members else None})
        print(f"  {day}: universe {len(members)} (star {len(star_members)}); "
              + " ".join(f"{f.replace('yeslab_','')}={counts.get(f,0)}" for f in factors), flush=True)

    df = pd.DataFrame(rows)
    pre = df[df.day < "2023-09-25"]; post = df[df.day >= "2023-09-25"]
    per_factor = {}
    for f, g in df.groupby("factor_id"):
        gp = pre[pre.factor_id == f]; gq = post[post.factor_id == f]
        per_factor[f] = {
            "role": g.role.iloc[0],
            "days_measured": int(len(g)),
            "rows_min": int(g.rows.min()), "rows_max": int(g.rows.max()),
            "coverage_of_universe_min": float(g.coverage_of_universe.min()),
            "coverage_of_universe_median": float(g.coverage_of_universe.median()),
            "pre_cutover_coverage_median": float(gp.coverage_of_universe.median()) if len(gp) else None,
            "post_cutover_coverage_median": float(gq.coverage_of_universe.median()) if len(gq) else None,
            "pre_cutover_star_coverage_median": float(gp.star_coverage.median()) if len(gp) else None,
            "post_cutover_star_coverage_median": float(gq.star_coverage.median()) if len(gq) else None,
            "zero_row_days": int((g.rows == 0).sum()),
            "dense_on_every_day": bool((g.coverage_of_universe >= 0.999).all()),
        }
    out = {"question": "For each of the fifteen frozen-v2 factors, is it published, dense, and STAR-complete?",
           "set_file": str(args.set_file), "frozen_date": spec.get("frozen_date"),
           "supersedes": spec.get("supersedes"),
           "roles": {"baseline_eligible": sorted(eligible), "candidate_only": sorted(candidate_only),
                     "removed_from_v1": spec.get("removed")},
           "days": days, "per_factor": per_factor, "per_day_per_factor": rows,
           "flags": None, "database_writes": False, "weights_written": False}
    flags = []
    for f, s in per_factor.items():
        if s["zero_row_days"]:
            flags.append(f"{f}: ZERO published rows on {s['zero_row_days']} of {s['days_measured']} days")
        if s["role"] == "baseline_eligible" and not s["dense_on_every_day"]:
            flags.append(f"{f}: baseline-eligible but coverage of the universe falls below 0.999 "
                         f"(min {s['coverage_of_universe_min']:.4f})")
        pc, qc = s["pre_cutover_star_coverage_median"], s["post_cutover_star_coverage_median"]
        if pc is not None and qc is not None and qc - pc > 0.05:
            # Deliberately NOT attributed here. A STAR-only rise with non-STAR coverage flat is the §M
            # industry-cache signature and only 357/471/363 take an industry provider, so for any other
            # factor the same shape must have a different cause. Run the non-STAR control before naming one.
            takes_provider = f in {"yeslab_357", "yeslab_363", "yeslab_471"}
            flags.append(f"{f}: STAR coverage rises across the cutover ({pc:.4f} pre -> {qc:.4f} post); "
                         + ("it takes an industry provider, so this matches the §M signature, but the "
                            "non-STAR control must still be run" if takes_provider else
                            "it takes NO industry provider, so §M cannot be the cause -- run the non-STAR "
                            "control and do not attribute this to the industry cache"))
        if s["role"] == "candidate_only" and s["coverage_of_universe_median"] > 0.999:
            flags.append(f"{f}: labelled candidate_only (sparse) but is dense on every measured day "
                         f"(median coverage {s['coverage_of_universe_median']:.4f})")
        # a sparse factor whose coverage is drifting DOWN over the window matters to the factor-set decision,
        # because a candidate leaning on it would have less and less coverage in the later OOS years
        # NOTE: this must read from `s`, the per-factor summary, not from a leftover loop variable. The first
        # version of this check referenced `g` from the groupby loop above, which by then held only the LAST
        # group, so the check silently evaluated one factor and reported nothing for the other fourteen.
        pre_med, post_med = s["pre_cutover_coverage_median"], s["post_cutover_coverage_median"]
        if pre_med is not None and post_med is not None:
            drift = post_med - pre_med
            if abs(drift) > 0.05:
                flags.append(f"{f}: coverage of the universe drifts {'DOWN' if drift < 0 else 'UP'} across the "
                             f"window by {drift:+.4f} (pre-cutover median {pre_med:.4f} -> "
                             f"post-cutover {post_med:.4f})")
    out["flags"] = flags
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n=== flags ===")
    for x in flags:
        print(" ", x)
    if not flags:
        print("  none")
    t = pd.DataFrame(per_factor).T
    pd.set_option("display.width", 320)
    print("\n" + t[["role", "rows_min", "coverage_of_universe_min", "coverage_of_universe_median",
                    "pre_cutover_star_coverage_median", "post_cutover_star_coverage_median",
                    "zero_row_days", "dense_on_every_day"]].to_string())


if __name__ == "__main__":
    main()
