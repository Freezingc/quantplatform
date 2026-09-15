"""Why is 2024-12-20 the binding day for both option 1 and option 2, and why do the worst post-cutover days cluster?

The full-window matrix in problems_found.md §AF left one day unexplained. 2024-12-20 is the worst of the 549
post-cutover days under the nine-factor baseline (3799/3912 = 0.971115), it stays the worst under option 2
(3800/3912) and under the five-factor set, and it is the only post-cutover day below 0.974. Since option 1
cannot touch it -- it is post-cutover, so the STAR labels already exist -- it is the real binding day for any
choice between those options, and the decision sheet was comparing 2022-07-29 instead.

The worst days are also not scattered: four of the five worst are 2024-12-20, 12-23, 12-24 and 12-25, and
2025-12-25, 12-26 and 12-31 are in the bottom eight. A year-end cluster points at a persistent set of
securities rather than a one-day event, so this decomposes every gap row on the cluster days and on two
control days: the exact set of missing factors, whether the security is STAR, its listing age in trading
days, and whether its trailing 60-trading-day window has an absent row (the §AC suspension signature). It
then reports how much of each day's gap the same securities explain across days.

SAFETY: SELECT-only against dfs://QuantResearchV2Test, plus read-only local caches. No writes except --out.
No weights, no repository file touched.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
CACHE = REPO / "artifacts" / "tushare_cache"
DENSE9 = ["yeslab_206", "yeslab_357", "yeslab_425", "yeslab_381", "yeslab_443_reg30_max20_v1",
          "yeslab_464", "yeslab_471", "yeslab_484", "yeslab_96"]
STAR = ("688", "689")
WINDOW = 60


def sql_list(v) -> str:
    return ", ".join("'" + str(x).replace("'", "''") + "'" for x in v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", default="2024-12-19,2024-12-20,2024-12-23,2024-12-24,2024-12-25,2024-07-30,2025-12-26,2025-06-16")
    ap.add_argument("--list-dates", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    days = args.days.split(",")

    ld = pd.DataFrame(json.loads(args.list_dates.read_text())["data"])
    ld["ts_code"] = ld["ts_code"].astype(str)
    list_date = dict(zip(ld.ts_code, pd.to_datetime(ld.list_date.astype(str), format="%Y%m%d", errors="coerce")))

    all_dates = sorted({pd.Timestamp(p.stem.replace("daily_", "")) for p in CACHE.glob("daily_[0-9]*.json")})
    tpos = {t: i for i, t in enumerate(all_dates)}
    memo: dict = {}

    def present(t):
        if t not in memo:
            p = CACHE / f"daily_{t:%Y%m%d}.json"
            if not p.exists():
                memo[t] = set()
            else:
                raw = json.loads(p.read_text())
                recs = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
                df = pd.DataFrame(recs); df["ts_code"] = df["ts_code"].astype(str)
                memo[t] = set(df.drop_duplicates("ts_code", keep="last").ts_code)
        return memo[t]

    def holes(codes, dt):
        i = tpos[dt]
        win = all_dates[max(0, i - WINDOW + 1):i + 1]
        sets = [present(t) for t in win]
        return {c: sum(1 for s in sets if c not in s) for c in codes}, len(win)

    sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
    from walk_forward_common import TARGET_DATABASE, current_versions, new_holder, query
    h = new_holder()
    lpins, fpins = current_versions(h, "label_daily"), current_versions(h, "factor_daily")
    ids = sql_list(DENSE9)

    out_days, gap_sets = [], {}
    for year in sorted({d[:4] for d in days}):
        ydays = [d for d in days if d.startswith(year)]
        parts = sorted({f"{d[:4]}-{d[5:7]}" for d in ydays})
        ll, fl = sql_list(sorted({lpins[p] for p in parts})), sql_list(sorted({fpins[p] for p in parts}))
        dl = ", ".join(d.replace("-", ".") for d in ydays)
        ok, lab = query(h, f"select signal_date, ts_code from loadTable('{TARGET_DATABASE}','label_daily') "
                           f"where signal_date in [{dl}] and label_valid=1 and data_version in [{ll}]")
        script = (f"fac = select signal_date, ts_code, factor_id from loadTable('{TARGET_DATABASE}','factor_daily') "
                  f"where signal_date in [{dl}] and factor_id in [{ids}] and data_version in [{fl}];"
                  "dk = select signal_date, ts_code, factor_id from fac group by signal_date, ts_code, factor_id;"
                  "select signal_date, ts_code, factor_id from dk")
        ok2, fac = query(h, script)
        if not ok or not ok2:
            raise SystemExit(f"query failed {year}: {lab if not ok else fac}")
        for df in (lab, fac):
            df["day"] = pd.to_datetime(df["signal_date"]).dt.strftime("%Y-%m-%d")
            df["ts_code"] = df["ts_code"].astype(str)
        lab_sets = {d: set(g.ts_code) for d, g in lab.groupby("day")}
        have = {d: {c: set(g.factor_id) for c, g in gg.groupby("ts_code")} for d, gg in fac.groupby("day")}
        print(f"  {year}: label rows {len(lab)}, factor rows {len(fac)}", flush=True)

        for day in ydays:
            V = lab_sets.get(day, set()); H = have.get(day, {})
            inter = {c for c in V if len(H.get(c, set()) & set(DENSE9)) == len(DENSE9)}
            gap = sorted(V - inter)
            gap_sets[day] = set(gap)
            dt = pd.Timestamp(day)
            hl, wlen = holes(gap, dt)
            sigs, recs = {}, []
            for c in gap:
                miss = tuple(sorted(set(DENSE9) - H.get(c, set())))
                sigs[miss] = sigs.get(miss, 0) + 1
                l = list_date.get(c)
                age = (tpos[dt] - tpos[l]) if (l is not None and l in tpos) else None
                recs.append({"ts_code": c, "missing": list(miss), "is_star": c.startswith(STAR),
                             "list_date": (str(l.date()) if l is not None else None),
                             "age_trading_days": age, "absent_rows_in_60d_window": hl.get(c),
                             "has_hole": bool(hl.get(c, 0) > 0)})
            R = pd.DataFrame(recs) if recs else pd.DataFrame(
                columns=["ts_code", "missing", "is_star", "age_trading_days", "absent_rows_in_60d_window", "has_hole"])
            miss_hist = {}
            for r in recs:
                for f in r["missing"]:
                    miss_hist[f] = miss_hist.get(f, 0) + 1
            out_days.append({
                "day": day, "window_trading_days": wlen,
                "valid_label_rows": len(V), "intersection_rows": len(inter), "gap_rows": len(gap),
                "coverage": len(inter) / len(V) if V else None,
                "gap_star_rows": int(R.is_star.sum()) if len(R) else 0,
                "signatures": {"+".join(k): v for k, v in sorted(sigs.items(), key=lambda kv: -kv[1])},
                "missing_factor_histogram": miss_hist,
                "gap_with_a_60d_hole": int(R.has_hole.sum()) if len(R) else 0,
                "gap_hole_median": float(R.absent_rows_in_60d_window.median()) if len(R) else None,
                "gap_age_trading_days_median": float(R.age_trading_days.median()) if len(R) and R.age_trading_days.notna().any() else None,
                "gap_age_trading_days_min": float(R.age_trading_days.min()) if len(R) and R.age_trading_days.notna().any() else None,
                "under_60_trading_days_listed": int((R.age_trading_days < 60).sum()) if len(R) and R.age_trading_days.notna().any() else 0,
                "records": recs,
            })
            print(f"  {day}: V={len(V)} N={len(inter)} gap={len(gap)} cov={out_days[-1]['coverage']:.6f} "
                  f"hole={out_days[-1]['gap_with_a_60d_hole']}/{len(gap)} star={out_days[-1]['gap_star_rows']} "
                  f"top_sig={list(out_days[-1]['signatures'].items())[:2]}", flush=True)

    # how much of each day's gap is shared with 2024-12-20
    anchor = "2024-12-20"
    overlap = []
    if anchor in gap_sets:
        for d, s in gap_sets.items():
            overlap.append({"day": d, "gap_rows": len(s), "shared_with_anchor": len(s & gap_sets[anchor]),
                            "share_of_anchor": (len(s & gap_sets[anchor]) / len(gap_sets[anchor]))
                            if gap_sets[anchor] else None})
    df = pd.DataFrame(out_days)
    out = {
        "question": "Why is 2024-12-20 the binding day for options 1 and 2, and why do the worst post-cutover days cluster?",
        "days": len(df), "day_list": days, "anchor_day": anchor,
        "gap_overlap_with_anchor": overlap,
        "totals": {
            "gap_rows": int(df.gap_rows.sum()),
            "gap_rows_star": int(df.gap_star_rows.sum()),
            "gap_rows_with_a_60d_hole": int(df.gap_with_a_60d_hole.sum()),
            "share_with_a_60d_hole": float(df.gap_with_a_60d_hole.sum() / df.gap_rows.sum()) if df.gap_rows.sum() else None,
            "gap_rows_under_60_trading_days_listed": int(df.under_60_trading_days_listed.sum()),
        },
        "answer": None, "per_day": out_days,
        "database_writes": False, "weights_written": False, "cache_writes": False,
    }
    t = out["totals"]
    out["answer"] = (
        f"Across the {len(df)} days the dense-nine gap is {t['gap_rows']} rows, of which "
        f"{t['gap_rows_with_a_60d_hole']} ({t['share_with_a_60d_hole']:.1%}) have an absent row in the trailing "
        f"60-trading-day window, {t['gap_rows_star']} are STAR and {t['gap_rows_under_60_trading_days_listed']} are "
        f"under 60 trading days since listing.")
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n" + out["answer"])
    print(df[["day", "valid_label_rows", "intersection_rows", "gap_rows", "coverage", "gap_star_rows",
              "gap_with_a_60d_hole", "gap_hole_median", "gap_age_trading_days_median",
              "under_60_trading_days_listed"]].to_string(index=False))
    print("\ngap overlap with 2024-12-20:")
    print(pd.DataFrame(overlap).to_string(index=False))


if __name__ == "__main__":
    main()
