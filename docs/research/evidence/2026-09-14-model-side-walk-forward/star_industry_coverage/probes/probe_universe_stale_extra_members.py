#!/usr/bin/env python3
"""The version mis-resolution cuts both ways: what does the stale pick ADD to factor_daily?

§AJ and §AK measured one direction -- securities the newest universe contains but the version selected by
max(data_version) does not, which end up with a valid label and no factor row, depressing coverage. The
opposite direction had not been measured: on days where the stale version is LARGER (2024-12-20, 2025-01-06
and 2025-01-07), factor_daily should hold rows for securities the resolved universe no longer contains. This
counts them and checks whether they carry a valid label.

Whether those extra rows can reach a training or evaluation dataset is a separate question and is answered
from source rather than measurement: walk_forward_factor_daily.py builds each fold with
`filter_current(universe_rows, "signal_date", versions["universe_daily"])` and passes universe_rows as the
first argument to build_point_in_time_dataset, so the registry-resolved universe is the spine and a security
outside it cannot enter. This probe supplies the counts that make that argument concrete.

SAFETY: SELECT-only against dfs://QuantResearchV2Test. No writes except --out. No weights, no repository
file touched.

Usage: python probe_universe_stale_extra_members.py --versions <universe_versions_origin_20days.json> \
           --days 2024-12-20 2025-01-06 2025-01-07 --out <json>
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd

REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
UNIVERSE_ID = "cn_liquid_a"
DENSE9 = ["yeslab_206", "yeslab_357", "yeslab_425", "yeslab_381", "yeslab_443_reg30_max20_v1",
          "yeslab_464", "yeslab_471", "yeslab_484", "yeslab_96"]


def sql_list(v) -> str:
    return ", ".join("'" + str(x).replace("'", "''") + "'" for x in v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", type=Path, required=True,
                    help="probe_universe_versions_origin.py output, used to learn which version max() picks")
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--examples", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    src = json.loads(args.versions.read_text())
    by_day = {r["day"]: r for r in src["per_day"]}

    sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
    from walk_forward_common import TARGET_DATABASE, current_versions, new_holder, query
    h = new_holder()
    fpins, lpins = current_versions(h, "factor_daily"), current_versions(h, "label_daily")
    ids = sql_list(DENSE9)

    rows = []
    for day in args.days:
        r = by_day.get(day)
        if r is None:
            raise SystemExit(f"{day} is not in {args.versions}; run probe_universe_versions_origin.py for it first")
        lex_prefix, new_prefix = r["lexicographic_max"], r["newest_by_created_at"]
        dl = day.replace("-", "."); part = day[:7]
        ok, u = query(h,
            f"select data_version, ts_code from loadTable('{TARGET_DATABASE}','universe_daily') "
            f"where signal_date = date('{dl}') and universe_id='{UNIVERSE_ID}' and is_member=1")
        if not ok:
            raise SystemExit(f"universe query failed {day}: {u}")
        u["data_version"] = u.data_version.astype(str); u["ts_code"] = u.ts_code.astype(str)
        by_prefix = {v[:16]: v for v in u.data_version.unique()}
        s_lex = set(u.loc[u.data_version == by_prefix[lex_prefix], "ts_code"])
        s_new = set(u.loc[u.data_version == by_prefix[new_prefix], "ts_code"])
        extra = sorted(s_lex - s_new)
        cl = sql_list(extra) if extra else "''"
        fl = sql_list([fpins[part]]); ll = sql_list([lpins[part]])
        ok2, fac = query(h,
            f"select ts_code, count(*) as n from loadTable('{TARGET_DATABASE}','factor_daily') "
            f"where signal_date = date('{dl}') and factor_id in [{ids}] and data_version in [{fl}] "
            f"and ts_code in [{cl}] group by ts_code") if extra else (True, pd.DataFrame(columns=["ts_code", "n"]))
        ok3, lab = query(h,
            f"select ts_code from loadTable('{TARGET_DATABASE}','label_daily') "
            f"where signal_date = date('{dl}') and label_valid=1 and data_version in [{ll}] "
            f"and ts_code in [{cl}]") if extra else (True, pd.DataFrame(columns=["ts_code"]))
        if not ok2 or not ok3:
            raise SystemExit(f"factor/label query failed {day}")
        with_fac = set(fac.ts_code.astype(str)) if isinstance(fac, pd.DataFrame) else set()
        with_lab = set(lab.ts_code.astype(str)) if isinstance(lab, pd.DataFrame) else set()
        rows.append({
            "day": day, "stale_version": lex_prefix, "newest_version": new_prefix,
            "stale_members": len(s_lex), "newest_members": len(s_new),
            "stale_only_members": len(extra),
            "of_those_with_dense_nine_rows_in_factor_daily": len(with_fac),
            "of_those_with_a_valid_label": len(with_lab),
            "have_factors_but_no_valid_label": len(with_fac - with_lab),
            "examples": sorted(with_fac - with_lab)[:args.examples],
        })
        print(f"  {day}: stale-only {len(extra)}, with factor rows {len(with_fac)}, "
              f"with a valid label {len(with_lab)}", flush=True)

    df = pd.DataFrame(rows)
    out = {
        "question": "On days where the stale universe version is larger, does factor_daily hold rows for "
                    "securities the resolved universe excludes, and do they carry labels?",
        "universe_id": UNIVERSE_ID, "days": args.days, "per_day": rows,
        "totals": {"stale_only_members": int(df.stale_only_members.sum()),
                   "with_factor_rows": int(df.of_those_with_dense_nine_rows_in_factor_daily.sum()),
                   "with_a_valid_label": int(df.of_those_with_a_valid_label.sum()),
                   "have_factors_but_no_valid_label": int(df.have_factors_but_no_valid_label.sum())},
        "can_they_reach_a_dataset": {
            "answer": "no",
            "evidence": "research_v2/scripts/walk_forward_factor_daily.py resolves the universe with "
                        "filter_current(universe_rows, 'signal_date', versions['universe_daily']) -- the "
                        "published_version registry, i.e. the same newest version label_daily uses -- and "
                        "passes universe_rows as the first argument to build_point_in_time_dataset, so the "
                        "registry-resolved universe is the spine and a security outside it cannot enter",
            "verified_from_source": True,
            "not_verified": "whether build_point_in_time_dataset has any other path that could admit a row "
                            "outside the spine; only the call site was read",
        },
        "answer": None, "database_writes": False, "weights_written": False,
    }
    t = out["totals"]
    out["answer"] = (
        f"Across {len(rows)} days the stale version holds {t['stale_only_members']} members the newest does "
        f"not; {t['with_factor_rows']} of them have dense-nine rows in factor_daily and "
        f"{t['with_a_valid_label']} have a valid label, so {t['have_factors_but_no_valid_label']} published "
        f"factor rows sit outside the resolved universe. They cannot reach a dataset, because the fold "
        f"assembly uses the registry-resolved universe as its spine.")
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n" + out["answer"])
    print(df[["day", "stale_members", "newest_members", "stale_only_members",
              "of_those_with_dense_nine_rows_in_factor_daily", "of_those_with_a_valid_label"]].to_string(index=False))


if __name__ == "__main__":
    main()
