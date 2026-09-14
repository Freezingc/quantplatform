"""Price option 3 (governance exception: inject the upstream authoritative values) instead of assuming it.

root_cause.md records that the upstream store DOES hold finite values for yeslab_357 / yeslab_471 on the
keys the panel reports missing -- on 2023-07-05, 353 of the 354 codes, including all 318 STAR ones. That
reads as though option 3 repairs the 24 days for free. It cannot be assumed, for two reasons, and this
measures both.

CEILING. Upstream coverage is not total. On 2023-07-05 the existing probe counts 540 of 588 STAR codes
with a value for 357 and 527 of 611 for 471, and a row only becomes complete if BOTH arrive. So the
question is per gap row, per day: does upstream hold both? A gap row also has to be missing exactly those
two factors -- the 1,101 rows that already carry an industry label are missing one of the other seven and
no injection reaches them.

CONSISTENCY. Option 3 mixes two value populations into one panel: replayed values everywhere else,
upstream values on the injected keys. That is only defensible if the two agree where both exist. This
compares them numerically on post-cutover days, where the replay had labels and therefore produced
values, and reports exact-match rate, worst absolute difference and rank correlation per factor.

SAFETY: SELECT-only against dfs://YesLab and dfs://QuantResearchV2Test. No writes anywhere except
--out. No weights, no artifact mutation, no repository file touched.
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
DENSE9 = ["yeslab_206", "yeslab_357", "yeslab_381", "yeslab_425", "yeslab_443_reg30_max20_v1",
          "yeslab_464", "yeslab_471", "yeslab_484", "yeslab_96"]
UPSTREAM = {"yeslab_357": ("factor_data8", "行业内收益标准化绝对值"),
            "yeslab_471": ("factor_data10", "AttentionSpillover")}
STAR = ("688", "689")
FLOOR, BAR = 0.90, 0.95


def sql_list(v) -> str:
    return ", ".join("'" + str(x).replace("'", "''") + "'" for x in v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=Path, required=True)
    ap.add_argument("--consistency-days", nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    src = json.loads(args.days.read_text())
    days = sorted(src["days"])
    ref = src.get("per_day", {})
    cdays = sorted(args.consistency_days)

    sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
    from walk_forward_common import TARGET_DATABASE, current_versions, new_holder, query
    holder = new_holder()
    lpins, fpins = current_versions(holder, "label_daily"), current_versions(holder, "factor_daily")
    ids = sql_list(DENSE9)

    def pins(year_days, table_pins):
        parts = sorted({f"{d[:4]}-{d[5:7]}" for d in year_days})
        miss = [p for p in parts if p not in table_pins]
        if miss:
            raise SystemExit(f"unpublished partitions: {miss}")
        return sql_list(sorted({table_pins[p] for p in parts}))

    # ---- panel side: valid-label rows and how many of the nine each carries ----
    lab_f, cnt_f = [], []
    for year in sorted({d[:4] for d in days}):
        yd = [d for d in days if d.startswith(year)]
        dl = ", ".join(d.replace("-", ".") for d in yd)
        ll, fl = pins(yd, lpins), pins(yd, fpins)
        ok, lab = query(holder,
            f"select signal_date, ts_code from loadTable('{TARGET_DATABASE}','label_daily') "
            f"where signal_date in [{dl}] and label_valid=1 and data_version in [{ll}]")
        ok2, cnt = query(holder,
            f"fac = select signal_date, ts_code, factor_id from loadTable('{TARGET_DATABASE}','factor_daily') "
            f"where signal_date in [{dl}] and factor_id in [{ids}] and data_version in [{fl}];"
            "dk = select signal_date, ts_code, factor_id from fac group by signal_date, ts_code, factor_id;"
            "select signal_date, ts_code, count(*) as cnt from dk group by signal_date, ts_code")
        if not ok or not ok2:
            raise SystemExit(f"panel query failed for {year}")
        lab_f.append(lab); cnt_f.append(cnt)
        print(f"  panel {year}: {len(lab)} label rows, {len(cnt)} securities", flush=True)

    lab = pd.concat(lab_f, ignore_index=True); cnt = pd.concat(cnt_f, ignore_index=True)
    for df in (lab, cnt):
        df["day"] = pd.to_datetime(df["signal_date"]).dt.strftime("%Y-%m-%d")
        df["ts_code"] = df["ts_code"].astype(str)
    panel = lab.merge(cnt[["day", "ts_code", "cnt"]], on=["day", "ts_code"], how="left")
    panel["cnt"] = panel["cnt"].fillna(0).astype(int)

    # ---- upstream side: which keys hold a value for BOTH targets ----
    up_f = []
    for fid, (table, name) in UPSTREAM.items():
        # the consistency days are fetched too, otherwise there is nothing to compare the replay against
        all_days = sorted(set(days) | set(cdays))
        for year in sorted({d[:4] for d in all_days}):
            yd = [d for d in all_days if d.startswith(year)]
            lo, hi = min(yd).replace("-", "."), max(yd).replace("-", ".")
            ok, fr = query(holder,
                f"select ts_code, trade_date, value from loadTable('dfs://YesLab','{table}') "
                f"where factor_name = '{name}' and trade_date between date('{lo}') and date('{hi}')")
            if not ok or not isinstance(fr, pd.DataFrame):
                raise SystemExit(f"upstream query failed for {fid} {year}: {fr}")
            fr["factor_id"] = fid
            up_f.append(fr)
        print(f"  upstream {fid}: fetched {sum(len(f) for f in up_f)} rows so far", flush=True)
    up = pd.concat(up_f, ignore_index=True)
    # restrict the ceiling computation to the problem days; the extra days serve the consistency check only
    up_ceiling = up[up["day"].isin(set(days))] if False else None
    up["day"] = pd.to_datetime(up["trade_date"]).dt.strftime("%Y-%m-%d")
    up["ts_code"] = up["ts_code"].astype(str)
    up["value"] = pd.to_numeric(up["value"], errors="coerce")
    up_ceiling = up[up["day"].isin(set(days))]

    # the upstream tables can carry more than one row per key; record whether they ever disagree
    dup = up_ceiling.groupby(["factor_id", "day", "ts_code"])["value"].agg(["count", "nunique"])
    duplicate_rows = int((dup["count"] > 1).sum())
    conflicting_duplicates = int(((dup["count"] > 1) & (dup["nunique"] > 1)).sum())
    upv = up.dropna(subset=["value"]).groupby(["factor_id", "day", "ts_code"])["value"].first().reset_index()
    upv_c = upv[upv["day"].isin(set(days))]
    have = {fid: set(map(tuple, upv_c.loc[upv_c.factor_id == fid, ["day", "ts_code"]].values)) for fid in UPSTREAM}
    both = have["yeslab_357"] & have["yeslab_471"]

    panel["up_357"] = list(zip(panel["day"], panel["ts_code"]))
    panel["has_357"] = panel["up_357"].isin(set(have["yeslab_357"]))
    panel["has_471"] = panel["up_357"].isin(set(have["yeslab_471"]))
    panel["has_both"] = panel["up_357"].isin(both)
    panel["is_star"] = panel["ts_code"].str.startswith(STAR)

    rows = []
    for day, g in panel.groupby("day"):
        V = len(g); N = int((g["cnt"] == 9).sum())
        gap = g[g["cnt"] < 9]
        miss_two = gap[gap["cnt"] == 7]
        fix = miss_two[miss_two["has_both"]]
        r = ref.get(day, {})
        rows.append({
            "day": day, "valid_label_rows": V, "intersection_rows": N,
            "matches_instrument": (V == r.get("valid_label_rows") and N == r.get("intersection_rows")),
            "coverage_now": N / V,
            "gap_rows": len(gap),
            "gap_rows_missing_exactly_the_two": len(miss_two),
            "of_those_upstream_has_both": len(fix),
            "of_those_upstream_is_missing_at_least_one": len(miss_two) - len(fix),
            "gap_rows_missing_other_factors_too": len(gap) - len(miss_two),
            "fixable_star": int(fix["is_star"].sum()), "fixable_non_star": int((~fix["is_star"]).sum()),
            "coverage_option3": (N + len(fix)) / V,
            "clears_floor": ((N + len(fix)) / V) >= FLOOR,
            "clears_bar": ((N + len(fix)) / V) >= BAR,
            "rows_needed_for_floor": math.ceil(FLOOR * V) - N,
            "rows_needed_for_bar": math.ceil(BAR * V) - N,
        })

    # ---- consistency: replayed vs upstream, where both exist ----
    cons = []
    for day in cdays:
        dl = day.replace("-", ".")
        fl = pins([day], fpins)
        ok, fd = query(holder,
            f"select ts_code, factor_id, value from loadTable('{TARGET_DATABASE}','factor_daily') "
            f"where signal_date = date('{dl}') and factor_id in [{sql_list(list(UPSTREAM))}] "
            f"and data_version in [{fl}]")
        if not ok or not isinstance(fd, pd.DataFrame):
            cons.append({"day": day, "error": str(fd)[:200]}); continue
        fd["ts_code"] = fd["ts_code"].astype(str)
        fd["value"] = pd.to_numeric(fd["value"], errors="coerce")
        for fid in UPSTREAM:
            a = fd[fd.factor_id == fid].dropna(subset=["value"]).set_index("ts_code")["value"]
            b = upv[(upv.factor_id == fid) & (upv.day == day)].dropna(subset=["value"]).set_index("ts_code")["value"]
            j = pd.concat([a.rename("replayed"), b.rename("upstream")], axis=1, join="inner").dropna()
            if j.empty:
                cons.append({"day": day, "factor_id": fid, "pairs": 0}); continue
            diff = (j["replayed"] - j["upstream"]).abs()
            exact = int((diff == 0).sum())
            cons.append({
                "day": day, "factor_id": fid, "pairs": int(len(j)),
                "exact_matches": exact, "exact_match_rate": exact / len(j),
                "max_abs_difference": float(diff.max()), "mean_abs_difference": float(diff.mean()),
                "within_1e_9": int((diff <= 1e-9).sum()), "within_1e_6": int((diff <= 1e-6).sum()),
                "pearson": float(np.corrcoef(j["replayed"], j["upstream"])[0, 1]) if len(j) > 2 else None,
                "spearman": float(pd.Series(j["replayed"]).corr(pd.Series(j["upstream"]), method="spearman")),
                "replayed_rows": int(len(a)), "upstream_rows": int(len(b)),
            })
        print(f"  consistency {day}: done", flush=True)

    repro = all(r["matches_instrument"] for r in rows)
    worst = min(rows, key=lambda r: r["coverage_option3"])
    out = {
        "question": "What does option 3 actually buy, and are the injected values consistent with the replayed ones?",
        "databases_read": ["dfs://YesLab", TARGET_DATABASE],
        "floor": FLOOR, "proposed_bar": BAR, "days": len(rows),
        "reproduces_the_instrument": "EXACT" if repro else "MISMATCH -- not trustworthy",
        "upstream_duplicate_rows_per_key": duplicate_rows,
        "upstream_duplicates_that_disagree": conflicting_duplicates,
        "totals": {
            "gap_rows": sum(r["gap_rows"] for r in rows),
            "gap_rows_missing_exactly_the_two": sum(r["gap_rows_missing_exactly_the_two"] for r in rows),
            "fixable_by_option3": sum(r["of_those_upstream_has_both"] for r in rows),
            "not_fixable_upstream_missing_one": sum(r["of_those_upstream_is_missing_at_least_one"] for r in rows),
            "not_fixable_missing_other_factors": sum(r["gap_rows_missing_other_factors_too"] for r in rows),
        },
        "days_below_floor_after_option3": sum(1 for r in rows if not r["clears_floor"]),
        "days_below_bar_after_option3": sum(1 for r in rows if not r["clears_bar"]),
        "g6_after_option3": (len(rows) - sum(1 for r in rows if not r["clears_floor"])) / len(rows),
        "worst_day_after_option3": {"day": worst["day"], "coverage": worst["coverage_option3"],
                                      "clears_floor": worst["clears_floor"], "clears_bar": worst["clears_bar"]},
        "consistency_of_injected_vs_replayed_values": cons,
        "per_day": rows,
        "database_writes": False, "weights_written": False,
    }
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_day",)}, ensure_ascii=False, indent=1))
    print(f"{'day':12}{'V':>6}{'gap':>5}{'miss2':>6}{'both':>6}{'now':>9}{'opt3':>9}{'>=.95':>7}")
    for r in sorted(rows, key=lambda x: x["coverage_option3"])[:8]:
        print(f"{r['day']:12}{r['valid_label_rows']:6d}{r['gap_rows']:5d}"
              f"{r['gap_rows_missing_exactly_the_two']:6d}{r['of_those_upstream_has_both']:6d}"
              f"{r['coverage_now']:9.6f}{r['coverage_option3']:9.6f}{str(r['clears_bar']):>7}")


if __name__ == "__main__":
    main()
