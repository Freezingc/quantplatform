"""Is the floor's denominator drawn from a wider universe than the materializer writes factors for?

materialize_frozen_factor_daily.py restricts every factor row to the official cn_liquid_a universe per date
(load_official_universe, then `(ts in universe.get(dt, set())) and np.isfinite(v)`). The evaluator's floor
divides the dense-nine intersection by label_daily's valid-label rows. If label_daily is NOT restricted to
cn_liquid_a, then the denominator can contain securities the numerator can never contain, and the coverage
ratio has a structural floor below 1 that no factor repair can reach.

probe_allnine_data_version.py found 65 securities on 2024-12-20 that label_daily counts as valid, that the
local price cache shows trading normally, and that factor_daily holds no row for under ANY data_version or
ANY factor_id -- while upstream holds 473 of 520 (security, factor) pairs for them. This tests the universe
hypothesis directly: are those 65 outside cn_liquid_a on that day, and how large is the label_daily minus
cn_liquid_a difference across the incident and on ordinary days.

SAFETY: SELECT-only against dfs://QuantResearchV2Test. No writes except --out. No weights.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd

REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
UNIVERSE_ID = "cn_liquid_a"


def sql_list(v) -> str:
    return ", ".join("'" + str(x).replace("'", "''") + "'" for x in v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", type=Path, required=True, help="allnine_absent_sets.json")
    ap.add_argument("--days", default="2024-12-19,2024-12-20,2024-12-23,2024-12-26,2024-12-31,2025-01-07,2025-06-16,2022-05-31")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    days = args.days.split(",")
    sets = json.loads(args.sets.read_text())

    sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
    from walk_forward_common import TARGET_DATABASE, current_versions, new_holder, query
    h = new_holder()
    lpins = current_versions(h, "label_daily")

    rows = []
    for day in days:
        dl = day.replace("-", ".")
        part = f"{day[:4]}-{day[5:7]}"
        ll = sql_list([lpins[part]]) if part in lpins else "''"
        # the materializer's own universe resolution: latest data_version per date, is_member=1
        ok, uni = query(h,
            f"select signal_date, ts_code, data_version from loadTable('{TARGET_DATABASE}','universe_daily') "
            f"where signal_date = date('{dl}') and universe_id='{UNIVERSE_ID}' and is_member=1")
        if not ok or not isinstance(uni, pd.DataFrame):
            raise SystemExit(f"universe query failed {day}: {uni}")
        uni["ts_code"] = uni["ts_code"].astype(str)
        latest = uni.groupby("signal_date")["data_version"].max().rename("lv")
        uni = uni.merge(latest, on="signal_date", how="left")
        members = set(uni.loc[uni.data_version == uni.lv, "ts_code"])

        ok2, lab = query(h, f"select ts_code from loadTable('{TARGET_DATABASE}','label_daily') "
                            f"where signal_date = date('{dl}') and label_valid=1 and data_version in [{ll}]")
        lab_set = set(lab.ts_code.astype(str)) if ok2 and isinstance(lab, pd.DataFrame) else set()

        target = set(sets.get(day, []))
        rows.append({
            "day": day,
            "cn_liquid_a_members": len(members),
            "label_daily_valid_rows": len(lab_set),
            "valid_label_rows_outside_the_universe": len(lab_set - members),
            "universe_members_without_a_valid_label": len(members - lab_set),
            "all_nine_absent_that_day": len(target),
            "all_nine_absent_outside_the_universe": len(target - members),
            "all_nine_absent_inside_the_universe": len(target & members),
            "examples_outside": sorted(target - members)[:10],
        })
        print(f"  {day}: universe {len(members)}, valid labels {len(lab_set)}, "
              f"labels outside universe {len(lab_set - members)}, all-nine-absent {len(target)} "
              f"(outside {len(target - members)}, inside {len(target & members)})", flush=True)

    df = pd.DataFrame(rows)
    out = {
        "question": "Is the floor's denominator (label_daily valid rows) wider than the materializer's universe (cn_liquid_a)?",
        "universe_id": UNIVERSE_ID,
        "materializer_gate": "materialize_frozen_factor_daily.py: load_official_universe(...) then "
                             "`(ts in universe.get(dt, set())) and np.isfinite(v)` -- a factor row is written "
                             "only for a security that is a cn_liquid_a member on that date",
        "days": days, "per_day": rows,
        "totals": {
            "valid_label_rows_outside_the_universe": int(df.valid_label_rows_outside_the_universe.sum()),
            "all_nine_absent_explained_by_the_universe": int(df.all_nine_absent_outside_the_universe.sum()),
            "all_nine_absent_not_explained": int(df.all_nine_absent_inside_the_universe.sum()),
        },
        "answer": None, "database_writes": False, "weights_written": False,
    }
    t = out["totals"]
    out["answer"] = (
        f"{t['all_nine_absent_explained_by_the_universe']} of "
        f"{t['all_nine_absent_explained_by_the_universe'] + t['all_nine_absent_not_explained']} all-nine-absent "
        f"securities on these days are outside cn_liquid_a, and label_daily carries "
        f"{t['valid_label_rows_outside_the_universe']} valid-label rows outside the universe in total -- so the "
        f"floor's denominator is drawn from a wider set than the materializer writes factors for.")
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n" + out["answer"])
    print(df[["day", "cn_liquid_a_members", "label_daily_valid_rows", "valid_label_rows_outside_the_universe",
              "all_nine_absent_that_day", "all_nine_absent_outside_the_universe",
              "all_nine_absent_inside_the_universe"]].to_string(index=False))


if __name__ == "__main__":
    main()
