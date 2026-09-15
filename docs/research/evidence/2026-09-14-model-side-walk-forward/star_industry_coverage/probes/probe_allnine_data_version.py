"""Are the 65 securities absent from factor_daily on 2024-12-20, or merely absent from the PINNED data_version?

Every measurement so far filtered factor_daily by the data_version that current_versions() pins for that
month, which is what the evaluator reads. So "no row" could mean either that the materializer never wrote
one, or that it wrote one under a different version and the pin does not select it. The two have opposite
remedies, so this repeats the query with the version filter removed and groups by data_version.

SAFETY: SELECT-only against dfs://QuantResearchV2Test. No writes except --out.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd

REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
DENSE9 = ["yeslab_206", "yeslab_357", "yeslab_425", "yeslab_381", "yeslab_443_reg30_max20_v1",
          "yeslab_464", "yeslab_471", "yeslab_484", "yeslab_96"]


def sql_list(v) -> str:
    return ", ".join("'" + str(x).replace("'", "''") + "'" for x in v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", type=Path, required=True, help="allnine_absent_sets.json")
    ap.add_argument("--day", default="2024-12-20")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    codes = sorted(json.loads(args.sets.read_text())[args.day])
    print(f"  {len(codes)} securities on {args.day}", flush=True)

    sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
    from walk_forward_common import TARGET_DATABASE, current_versions, new_holder, query
    h = new_holder()
    pins = current_versions(h, "factor_daily")
    part = f"{args.day[:4]}-{args.day[5:7]}"
    pinned = pins.get(part)
    dl = args.day.replace("-", ".")
    cl = sql_list(codes)

    # DolphinDB rejects count(*) beside count(distinct ...) in one grouped select ("Cannot nest aggregate
    # function"), so the two are separate queries and the distinct-code count goes through a dedup subquery
    ok, allv = query(h,
        f"v = select data_version, ts_code from loadTable('{TARGET_DATABASE}','factor_daily') "
        f"where signal_date = date('{dl}') and ts_code in [{cl}] group by data_version, ts_code;"
        "select data_version, count(*) as codes from v group by data_version")
    if not ok:
        raise SystemExit(f"query failed: {allv}")
    ok2, tot = query(h,
        f"v = select data_version, ts_code from loadTable('{TARGET_DATABASE}','factor_daily') "
        f"where signal_date = date('{dl}') group by data_version, ts_code;"
        "select data_version, count(*) as codes from v group by data_version")
    ok3, byfac = query(h,
        f"v = select factor_id, ts_code from loadTable('{TARGET_DATABASE}','factor_daily') "
        f"where signal_date = date('{dl}') and ts_code in [{cl}] group by factor_id, ts_code;"
        "select factor_id, count(*) as codes from v group by factor_id")
    print(f"  pinned version for {part}: {pinned}", flush=True)

    versions = allv.to_dict("records") if isinstance(allv, pd.DataFrame) else []
    for v in versions:
        print(f"  version {str(v['data_version'])[:16]}… distinct_codes={v['codes']} "
              f"{'<-- PINNED' if v['data_version'] == pinned else ''}", flush=True)
    out = {
        "question": "Are the all-nine-absent securities missing from factor_daily, or only from the pinned data_version?",
        "day": args.day, "partition": part, "pinned_data_version": pinned,
        "securities_asked_about": len(codes),
        "versions_holding_any_of_them": versions,
        "all_versions_on_that_day": (tot.to_dict("records") if isinstance(tot, pd.DataFrame) else []),
        "by_factor_id_ignoring_version": (byfac.to_dict("records") if isinstance(byfac, pd.DataFrame) else []),
        "codes": codes,
        "answer": None, "database_writes": False, "weights_written": False,
    }
    held = {v["data_version"]: int(v["codes"]) for v in versions}
    pinned_held = held.get(pinned, 0)
    other = sum(n for v, n in held.items() if v != pinned)
    out["answer"] = (
        f"Under the pinned version {str(pinned)[:16]}… the table holds {pinned_held} of {len(codes)} securities; "
        f"across all other versions it holds {other} more. "
        + ("So the rows exist but the pin does not select them -- a versioning problem."
           if pinned_held < len(codes) and other > 0 else
           "So the rows do not exist under any version -- the materializer never wrote them for that day."))
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n" + out["answer"])
    print(json.dumps(out["by_factor_id_ignoring_version"], ensure_ascii=False))


if __name__ == "__main__":
    main()
