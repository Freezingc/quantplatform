"""Where does the 2024-12-20 gap come from: 65 securities have a valid label but no row for ANY of the nine.

probe_worst_postcutover_days.py decomposed the binding day and found a failure mode that is not the label
problem and not the suspension signature: 65 of the 113 gap rows are missing all nine dense factors at once,
while being seasoned (median 1026 trading days listed), actively trading that day (median volume 36,247 lots,
no zero-volume day) and complete in the local price cache (55 of 65 have zero absent rows in the trailing
60-day window). label_daily carries a valid label for all of them, so they are in the floor's denominator.
Neither option 1 nor option 2 can reach them, which is why that day binds both.

Two questions decide whose gap this is. First, does factor_daily hold ANY factor for those securities on
that day, or are they absent from the whole table? Second, does the upstream YesLab store hold values for
them, across five different source tables? If upstream has them and factor_daily does not, the gap is in our
materialization; if upstream lacks them too, it is a vendor production gap that the frozen chain faithfully
inherited. Also asked: does the same set of securities explain the decaying counts on 12-23, 12-24 and 12-25.

SAFETY: SELECT-only against dfs://YesLab and dfs://QuantResearchV2Test. No writes except --out.
No weights, no repository file touched.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd

REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
DENSE9 = ["yeslab_206", "yeslab_357", "yeslab_425", "yeslab_381", "yeslab_443_reg30_max20_v1",
          "yeslab_464", "yeslab_471", "yeslab_484", "yeslab_96"]
UPSTREAM = {"yeslab_96": ("factor_data3", "修正日间反转"),
            "yeslab_206": ("factor_data5", "60日成交量去均值相对偏离"),
            "yeslab_381": ("factor_data8", "GP_012"),
            "yeslab_357": ("factor_data8", "行业内收益标准化绝对值"),
            "yeslab_425": ("factor_data9", "GP_056"),
            "yeslab_464": ("factor_data10", "GP_095"),
            "yeslab_471": ("factor_data10", "AttentionSpillover"),
            "yeslab_484": ("factor_data10", "羊群效应_CSAD波动率21")}


def sql_list(v) -> str:
    return ", ".join("'" + str(x).replace("'", "''") + "'" for x in v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--absence", type=Path, required=True, help="worst_postcutover_days.json")
    ap.add_argument("--day", default="2024-12-20")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    d = json.loads(args.absence.read_text())
    day = next(x for x in d["per_day"] if x["day"] == args.day)
    R = pd.DataFrame(day["records"])
    target = sorted(R.loc[R.missing.apply(len) == len(DENSE9), "ts_code"])
    sus = sorted(R.loc[R.missing.apply(lambda m: set(m) == {"yeslab_206", "yeslab_443_reg30_max20_v1"}), "ts_code"])
    print(f"  {args.day}: all-nine-absent {len(target)}, suspension-signature {len(sus)}", flush=True)

    sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
    from walk_forward_common import TARGET_DATABASE, current_versions, new_holder, query
    h = new_holder()
    fpins = current_versions(h, "factor_daily")
    part = f"{args.day[:4]}-{args.day[5:7]}"
    fl = sql_list([fpins[part]]); dl = args.day.replace("-", ".")
    codes = sql_list(sorted(set(target) | set(sus)))

    ok, anyfac = query(h,
        f"select ts_code, factor_id from loadTable('{TARGET_DATABASE}','factor_daily') "
        f"where signal_date = date('{dl}') and ts_code in [{codes}] and data_version in [{fl}]")
    if not ok:
        raise SystemExit(f"factor_daily query failed: {anyfac}")
    anyfac["ts_code"] = anyfac["ts_code"].astype(str)
    held = anyfac.groupby("ts_code")["factor_id"].nunique().to_dict()

    up = {}
    for fid, (table, name) in UPSTREAM.items():
        ok2, u = query(h, f"select ts_code, value from loadTable('dfs://YesLab','{table}') "
                          f"where factor_name = '{name}' and trade_date = date('{dl}') and ts_code in [{codes}]")
        if not ok2:
            up[fid] = {"error": str(u)[:200]}; continue
        u["ts_code"] = u["ts_code"].astype(str)
        u["value"] = pd.to_numeric(u["value"], errors="coerce")
        got = u.dropna(subset=["value"]).groupby("ts_code")["value"].first()
        up[fid] = {"table": table, "rows_returned": int(len(u)),
                   "target_held": int(sum(1 for c in target if c in got.index)),
                   "suspension_held": int(sum(1 for c in sus if c in got.index)),
                   "target_n": len(target), "suspension_n": len(sus)}
        print(f"  upstream {fid} ({table}): target {up[fid]['target_held']}/{len(target)}, "
              f"suspension {up[fid]['suspension_held']}/{len(sus)}", flush=True)

    n_any = sum(1 for c in target if held.get(c, 0) > 0)
    out = {
        "question": "Is the all-nine-absent gap on the binding day ours or the vendor's?",
        "day": args.day,
        "all_nine_absent_securities": len(target),
        "suspension_signature_securities": len(sus),
        "factor_daily_any_factor_at_all": {
            "target_with_at_least_one_factor_row": n_any,
            "target_with_none": len(target) - n_any,
            "distinct_factor_ids_held_by_target": sorted({f for c in target for f in
                                                          anyfac.loc[anyfac.ts_code == c, "factor_id"]}),
            "note": "queried WITHOUT the dense-nine filter, so this counts any factor_id in the table",
        },
        "upstream_availability": up,
        "codes_all_nine_absent": target,
        "codes_suspension_signature": sus,
        "database_writes": False, "weights_written": False,
    }
    have_up = {k: v for k, v in up.items() if "target_held" in v}
    tot_held = sum(v["target_held"] for v in have_up.values())
    tot_n = sum(v["target_n"] for v in have_up.values())
    out["answer"] = (
        f"{len(target) - n_any} of {len(target)} all-nine-absent securities have no row for ANY factor_id in "
        f"factor_daily on {args.day}. Upstream holds {tot_held} of {tot_n} (security, factor) pairs for the same "
        f"securities across {len(have_up)} source tables, so "
        + ("the gap is in our materialization: upstream has the values and factor_daily does not."
           if tot_held > 0.5 * tot_n else
           "the gap is upstream's: the vendor store does not hold these securities on this day either, and the "
           "frozen chain faithfully inherited the hole."))
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n" + out["answer"])
    print(json.dumps(out["factor_daily_any_factor_at_all"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
