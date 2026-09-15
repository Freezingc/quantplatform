"""Is the 2024-12-20 numerator/denominator divergence a definitional difference or a universe VINTAGE skew?

§AG established that on that day 65 securities carry a valid label in label_daily but no row for any factor
in factor_daily, and that all 65 are outside the cn_liquid_a membership that
materialize_frozen_factor_daily.py resolves (latest data_version per date, is_member=1). But labels.py also
builds label_daily from a universe-members frame, so both tables descend from universe membership. That
makes a vintage skew the more parsimonious explanation than two different universe definitions: if the
universe table was republished between the two materializations, and 65 securities left the universe in the
newer version, then label_daily built on the older snapshot and factor_daily built on the newer one would
diverge by exactly those 65.

Tested here: every data_version of universe_daily for that day with its member count, whether each version
holds the 65, and the created_at of the label_daily and factor_daily partitions actually in force. If an
older universe version contains them, the divergence is a republishing-order artifact and the remedy is a
pin, not a redefinition.

SAFETY: SELECT-only against dfs://QuantResearchV2Test. No writes except --out. No weights.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd

REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
UNIVERSE_ID = "cn_liquid_a"


def recs(df) -> list:
    """to_dict("records") leaves pandas Timestamps in place, which json.dumps cannot serialize."""
    if not isinstance(df, pd.DataFrame):
        return []
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_datetime64_any_dtype(d[c]):
            d[c] = d[c].dt.strftime("%Y-%m-%d %H:%M:%S")
    return json.loads(d.to_json(orient="records"))


def sql_list(v) -> str:
    return ", ".join("'" + str(x).replace("'", "''") + "'" for x in v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", type=Path, required=True)
    ap.add_argument("--days", default="2024-12-19,2024-12-20,2024-12-23,2025-06-16")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    days = args.days.split(",")
    sets = json.loads(args.sets.read_text())

    sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
    from walk_forward_common import TARGET_DATABASE, current_versions, new_holder, query
    h = new_holder()
    lpins, fpins = current_versions(h, "label_daily"), current_versions(h, "factor_daily")

    out_days = []
    for day in days:
        dl = day.replace("-", ".")
        target = set(sets.get(day, []))
        cl = sql_list(sorted(target)) if target else "''"
        ok, ver = query(h,
            f"v = select data_version, ts_code from loadTable('{TARGET_DATABASE}','universe_daily') "
            f"where signal_date = date('{dl}') and universe_id='{UNIVERSE_ID}' and is_member=1 "
            "group by data_version, ts_code;"
            "select data_version, count(*) as members from v group by data_version")
        if not ok:
            raise SystemExit(f"version query failed {day}: {ver}")
        ok2, held = query(h,
            f"v = select data_version, ts_code from loadTable('{TARGET_DATABASE}','universe_daily') "
            f"where signal_date = date('{dl}') and universe_id='{UNIVERSE_ID}' and is_member=1 "
            f"and ts_code in [{cl}] group by data_version, ts_code;"
            "select data_version, count(*) as n from v group by data_version") if target else (True, pd.DataFrame(columns=["data_version", "n"]))
        held_map = dict(zip(held.data_version.astype(str), held.n.astype(int))) if isinstance(held, pd.DataFrame) and len(held) else {}
        versions = []
        for _, r in ver.iterrows():
            dv = str(r["data_version"])
            versions.append({"data_version": dv, "members": int(r["members"]),
                             "target_held": int(held_map.get(dv, 0)),
                             "is_the_one_the_materializer_picks": None})
        versions.sort(key=lambda x: x["data_version"])
        if versions:
            versions[-1]["is_the_one_the_materializer_picks"] = True   # max() over the version strings
        part = f"{day[:4]}-{day[5:7]}"
        ok3, meta = query(h,
            f"select data_version, min(created_at) as created_min, max(created_at) as created_max, count(*) as n "
            f"from loadTable('{TARGET_DATABASE}','label_daily') where signal_date = date('{dl}') group by data_version")
        ok4, meta2 = query(h,
            f"select data_version, min(created_at) as created_min, max(created_at) as created_max, count(*) as n "
            f"from loadTable('{TARGET_DATABASE}','factor_daily') where signal_date = date('{dl}') group by data_version")
        rec = {"day": day, "target_n": len(target),
               "universe_versions": versions,
               "label_daily_pinned_version": lpins.get(part),
               "factor_daily_pinned_version": fpins.get(part),
               "label_daily_versions_on_that_day": (recs(meta) if ok3 else []),
               "factor_daily_versions_on_that_day": (recs(meta2) if ok4 else [])}
        picks = [v for v in versions if v["is_the_one_the_materializer_picks"]]
        rec["target_held_by_the_picked_version"] = picks[0]["target_held"] if picks else None
        rec["target_held_by_any_version"] = sum(v["target_held"] for v in versions)
        rec["versions_holding_all_of_them"] = [v["data_version"][:16] for v in versions
                                               if v["target_held"] == len(target) and target]
        out_days.append(rec)
        print(f"  {day}: target {len(target)}, universe versions {len(versions)}, "
              f"picked-version holds {rec['target_held_by_the_picked_version']}, "
              f"any version holds {rec['target_held_by_any_version']}", flush=True)
        for v in versions:
            print(f"      {v['data_version'][:16]}… members={v['members']} target={v['target_held']}"
                  f"{'  <-- materializer picks this' if v['is_the_one_the_materializer_picks'] else ''}", flush=True)

    out = {"question": "Is the 2024-12-20 divergence a universe vintage skew or two different universe definitions?",
           "universe_id": UNIVERSE_ID, "days": days, "per_day": out_days,
           "answer": None, "database_writes": False, "weights_written": False}
    d20 = next((r for r in out_days if r["day"] == "2024-12-20"), None)
    if d20:
        anyv = d20["target_held_by_any_version"]; picked = d20["target_held_by_the_picked_version"]
        out["answer"] = (
            f"On 2024-12-20 the version the materializer picks holds {picked} of {d20['target_n']} target "
            f"securities while all versions together hold {anyv}. "
            + ("An older universe version DOES contain them, so this is a vintage skew: label_daily was built "
               "against an earlier universe snapshot than factor_daily, and the remedy is to pin one vintage, "
               "not to redefine the denominator."
               if anyv > picked else
               "No version of the universe contains them, so this is not a vintage skew; they were outside "
               "cn_liquid_a on that day under every published version, and label_daily's membership comes from "
               "somewhere else."))
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n" + (out["answer"] or "2024-12-20 not in the day list"))


if __name__ == "__main__":
    main()
