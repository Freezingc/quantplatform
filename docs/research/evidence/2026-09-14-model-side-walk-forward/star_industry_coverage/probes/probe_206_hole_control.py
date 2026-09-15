"""Control for the history-hole finding: do securities that DO have all nine factors also have a hole?

probe_206_absence_cause.py found that 736 of 736 residual records have at least one absent row in the
trailing 60-trading-day window. That is only meaningful against a control: if complete-window securities
also routinely have holes, the hole explains nothing. This recomputes the same statistic for a sample of
securities that are IN the dense-nine intersection on the same days, at the same published data_versions.

SAFETY: SELECT-only against dfs://QuantResearchV2Test, plus read-only local caches. No writes except --out.
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
CACHE = REPO / "artifacts" / "tushare_cache"
DENSE9 = ["yeslab_206", "yeslab_357", "yeslab_425", "yeslab_381", "yeslab_443_reg30_max20_v1",
          "yeslab_464", "yeslab_471", "yeslab_484", "yeslab_96"]
INDUSTRY = {"yeslab_357", "yeslab_471"}
STAR = ("688", "689")


def sql_list(v) -> str:
    return ", ".join("'" + str(x).replace("'", "''") + "'" for x in v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", default="2022-06-30,2022-07-29,2023-03-15,2023-09-20")
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    days = args.days.split(",")
    rng = random.Random(args.seed)

    all_dates = sorted({pd.Timestamp(p.stem.replace("daily_", "")) for p in CACHE.glob("daily_[0-9]*.json")})
    pos = {t: i for i, t in enumerate(all_dates)}
    cache_sets = {}

    def present(t):
        if t not in cache_sets:
            p = CACHE / f"daily_{t:%Y%m%d}.json"
            if not p.exists():
                cache_sets[t] = set()
            else:
                raw = json.loads(p.read_text())
                recs = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
                df = pd.DataFrame(recs); df["ts_code"] = df["ts_code"].astype(str)
                cache_sets[t] = set(df.ts_code)
        return cache_sets[t]

    def holes(codes, dt):
        i = pos[dt]
        win = all_dates[max(0, i - 59):i + 1]
        sets = [present(t) for t in win]
        return {c: sum(1 for s in sets if c not in s) for c in codes}, len(win)

    sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
    from walk_forward_common import TARGET_DATABASE, current_versions, new_holder, query
    h = new_holder()
    lpins, fpins = current_versions(h, "label_daily"), current_versions(h, "factor_daily")
    ids = sql_list(DENSE9)

    rows = []
    for day in days:
        part = f"{day[:4]}-{day[5:7]}"
        ll, fl = sql_list([lpins[part]]), sql_list([fpins[part]])
        dl = day.replace("-", ".")
        ok, lab = query(h, f"select ts_code from loadTable('{TARGET_DATABASE}','label_daily') "
                           f"where signal_date = date('{dl}') and label_valid=1 and data_version in [{ll}]")
        script = (f"fac = select ts_code, factor_id from loadTable('{TARGET_DATABASE}','factor_daily') "
                  f"where signal_date = date('{dl}') and factor_id in [{ids}] and data_version in [{fl}];"
                  "dk = select ts_code, factor_id from fac group by ts_code, factor_id;"
                  f"select ts_code from dk group by ts_code having count(*) = {len(DENSE9)}")
        ok2, comp = query(h, script)
        if not ok or not ok2:
            raise SystemExit(f"query failed {day}")
        V = set(lab.ts_code.astype(str))
        # per-security factor presence, so the gap can be split exactly as probe_206_absence_cause splits it
        ok3, fac = query(h, f"select ts_code, factor_id from loadTable('{TARGET_DATABASE}','factor_daily') "
                            f"where signal_date = date('{dl}') and factor_id in [{ids}] and data_version in [{fl}]")
        if not ok3:
            raise SystemExit(f"presence query failed {day}")
        fac["ts_code"] = fac["ts_code"].astype(str)
        have = {c: set(g.factor_id) for c, g in fac.groupby("ts_code")}
        inter = {c for c in V if len(have.get(c, set()) & set(DENSE9)) == len(DENSE9)}
        gap = V - inter
        # the residual this whole line of inquiry is about: non-STAR rows missing a NON-industry factor
        resid = {c for c in gap if not c.startswith(STAR)
                 and not ((set(DENSE9) - have.get(c, set())) <= INDUSTRY)}
        gap_other = gap - resid
        sample = rng.sample(sorted(inter), min(args.sample, len(inter)))
        dt = pd.Timestamp(day)
        hr, w = holes(sorted(resid), dt)
        ho, _ = holes(sorted(gap_other), dt)
        hi, _ = holes(sample, dt)
        rr = pd.Series(list(hr.values()), dtype=float)
        ro = pd.Series(list(ho.values()), dtype=float)
        ri = pd.Series(list(hi.values()), dtype=float)
        rows.append({
            "day": day, "window_trading_days": w,
            "valid_label_rows": len(V), "intersection_rows": len(inter),
            "gap_rows": len(gap), "residual_non_industry_rows": len(resid),
            "gap_other_rows": len(gap_other),
            "gap_other_with_a_hole": int((ro > 0).sum()),
            "gap_other_hole_median": float(ro.median()) if len(ro) else None,
            "residual_with_a_hole": int((rr > 0).sum()), "residual_n": int(len(rr)),
            "residual_without_a_hole": sorted([c for c, v in hr.items() if v == 0]),
            "residual_hole_median": float(rr.median()) if len(rr) else None,
            "residual_hole_max": float(rr.max()) if len(rr) else None,
            "control_sampled": len(sample),
            "control_with_a_hole": int((ri > 0).sum()),
            "control_hole_median": float(ri.median()) if len(ri) else None,
            "control_hole_p90": float(ri.quantile(0.9)) if len(ri) else None,
            "control_hole_max": float(ri.max()) if len(ri) else None,
            "control_share_with_a_hole": float((ri > 0).mean()) if len(ri) else None,
            "residual_share_with_a_hole": float((rr > 0).mean()) if len(rr) else None,
        })
        print(f"  {day}: residual {rows[-1]['residual_with_a_hole']}/{len(rr)} with a hole "
              f"(median {rows[-1]['residual_hole_median']}), other-gap "
              f"{rows[-1]['gap_other_with_a_hole']}/{len(ro)}, control "
              f"{rows[-1]['control_with_a_hole']}/{len(sample)} (median {rows[-1]['control_hole_median']}, "
              f"max {rows[-1]['control_hole_max']})", flush=True)

    df = pd.DataFrame(rows)
    out = {
        "question": "Is a hole in the trailing 60-trading-day window specific to the securities that lack yeslab_206?",
        "days": days, "control": "a random sample of securities that DO carry all nine dense factors, same day, same published data_versions",
        "control_sample_size": args.sample, "seed": args.seed,
        "per_day": rows,
        "totals": {
            "residual_records": int(df.residual_n.sum()),
            "residual_with_a_hole": int(df.residual_with_a_hole.sum()),
            "control_records": int(df.control_sampled.sum()),
            "control_with_a_hole": int(df.control_with_a_hole.sum()),
            "residual_share": float(df.residual_with_a_hole.sum() / df.residual_n.sum()),
            "control_share": float(df.control_with_a_hole.sum() / df.control_sampled.sum()),
            "gap_other_records": int(df.gap_other_rows.sum()),
            "gap_other_with_a_hole": int(df.gap_other_with_a_hole.sum()),
        },
        "verdict": None, "database_writes": False, "weights_written": False, "cache_writes": False,
    }
    t = out["totals"]
    out["verdict"] = (
        f"DISCRIMINATIVE -- {t['residual_share']:.1%} of residual rows have a hole in the trailing 60-trading-day "
        f"window versus {t['control_share']:.1%} of the in-intersection control "
        f"({t['residual_with_a_hole']}/{t['residual_records']} vs {t['control_with_a_hole']}/{t['control_records']}). "
        f"The {t['gap_other_records']} other gap rows on those days (STAR rows and label-fixable rows, which are "
        f"missing only the two industry factors) have {t['gap_other_with_a_hole']} holes between them."
        if t["residual_share"] - t["control_share"] > 0.25 else
        f"NOT DISCRIMINATIVE -- residual {t['residual_share']:.1%} vs control {t['control_share']:.1%}; "
        f"holes are common to both, so a history gap does not explain the absence of yeslab_206.")
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n" + out["verdict"])
    print(df[["day", "residual_n", "residual_share_with_a_hole", "residual_hole_median",
              "control_sampled", "control_share_with_a_hole", "control_hole_median",
              "control_hole_p90", "control_hole_max"]].to_string(index=False))


if __name__ == "__main__":
    main()
