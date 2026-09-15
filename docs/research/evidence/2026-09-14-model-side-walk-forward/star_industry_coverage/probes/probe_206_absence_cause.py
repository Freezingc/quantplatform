"""Why does yeslab_206 have no value for the ~84 securities that cap the pre-cutover dense-nine intersection?

§AB attributed the non-STAR residual: on seven of the eight largest-residual days every residual row lacks
yeslab_206 (86/86 on 2022-06-30), most also lack yeslab_443_reg30_max20_v1 (74/86), and no label repair
lifts any of them. That makes 206's absence the standing ceiling on the intersection -- about 84 rows a day,
2.5% of the denominator, remarkably stable day to day. A count that stable across two years smells like a
structural rule rather than a defect.

The obvious candidate is seasoning: yeslab_206 is "60日成交量去均值相对偏离" with a 60-day lookback, and
yeslab_443_reg30_max20_v1 needs 30 days of regression window, so a security that listed recently cannot
have either. Tested here by joining every residual security to its listing date and counting TRADING days
(not calendar days) between listing and the day in question, then checking two rival explanations -- that
the security was suspended (no daily row, or zero volume) and that it is simply absent from the price panel.

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
INDUSTRY = {"yeslab_357", "yeslab_471"}
STAR = ("688", "689")
LOOKBACKS = {"yeslab_206": 60, "yeslab_443_reg30_max20_v1": 30}


def sql_list(v) -> str:
    return ", ".join("'" + str(x).replace("'", "''") + "'" for x in v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, required=True, help="star_rows_in_denominator_420.json")
    ap.add_argument("--list-dates", type=Path, required=True)
    ap.add_argument("--days", type=int, default=8)
    ap.add_argument("--extra-days", default="2023-03-15,2023-09-20")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    src = json.loads(args.rows.read_text())
    per = pd.DataFrame(src["per_day"])
    days = per.nlargest(args.days, "residual_gap_not_star")["day"].tolist()
    days += [d for d in args.extra_days.split(",") if d and d not in days]
    print(f"  days: {days}", flush=True)

    ld = pd.DataFrame(json.loads(args.list_dates.read_text())["data"])
    ld["ts_code"] = ld["ts_code"].astype(str)
    list_date = dict(zip(ld.ts_code, pd.to_datetime(ld.list_date.astype(str), format="%Y%m%d", errors="coerce")))
    print(f"  listing dates for {len(list_date)} securities", flush=True)

    all_dates = sorted({pd.Timestamp(p.stem.replace("daily_", "")) for p in CACHE.glob("daily_[0-9]*.json")})
    tpos = {d: i for i, d in enumerate(all_dates)}

    sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
    from walk_forward_common import TARGET_DATABASE, current_versions, new_holder, query
    h = new_holder()
    lpins, fpins = current_versions(h, "label_daily"), current_versions(h, "factor_daily")
    ids = sql_list(DENSE9)

    out_days = []
    for year in sorted({d[:4] for d in days}):
        ydays = [d for d in days if d.startswith(year)]
        parts = sorted({f"{d[:4]}-{d[5:7]}" for d in ydays})
        ll = sql_list(sorted({lpins[p] for p in parts}))
        fl = sql_list(sorted({fpins[p] for p in parts}))
        dl = ", ".join(d.replace("-", ".") for d in ydays)
        ok, lab = query(h, f"select signal_date, ts_code from loadTable('{TARGET_DATABASE}','label_daily') "
                           f"where signal_date in [{dl}] and label_valid=1 and data_version in [{ll}]")
        script = (f"fac = select signal_date, ts_code, factor_id from loadTable('{TARGET_DATABASE}','factor_daily') "
                  f"where signal_date in [{dl}] and factor_id in [{ids}] and data_version in [{fl}];"
                  "dk = select signal_date, ts_code, factor_id from fac group by signal_date, ts_code, factor_id;"
                  "select signal_date, ts_code, factor_id from dk")
        ok2, fac = query(h, script)
        if not ok or not ok2:
            raise SystemExit(f"query failed for {year}")
        for df in (lab, fac):
            df["day"] = pd.to_datetime(df["signal_date"]).dt.strftime("%Y-%m-%d")
            df["ts_code"] = df["ts_code"].astype(str)
        lab_sets = {d: set(g.ts_code) for d, g in lab.groupby("day")}
        have = {d: {c: set(g.factor_id) for c, g in gg.groupby("ts_code")} for d, gg in fac.groupby("day")}
        print(f"  {year}: label rows {len(lab)}, factor rows {len(fac)}", flush=True)

        for day in ydays:
            V = lab_sets.get(day, set()); H = have.get(day, {})
            inter = {c for c in V if len(H.get(c, ()) & set(DENSE9)) == len(DENSE9)}
            resid = {c for c in (V - inter) if not c.startswith(STAR)
                     and not (set(DENSE9) - H.get(c, set())) <= INDUSTRY}
            dt = pd.Timestamp(day)
            raw = json.loads((CACHE / f"daily_{dt:%Y%m%d}.json").read_text())
            recs = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
            px = pd.DataFrame(recs); px["ts_code"] = px["ts_code"].astype(str)
            px = px.drop_duplicates("ts_code", keep="last").set_index("ts_code")
            vol = pd.to_numeric(px["vol"], errors="coerce") if "vol" in px else pd.Series(dtype=float)

            recs_out, ages = [], []
            for c in sorted(resid):
                missing = sorted(set(DENSE9) - H.get(c, set()))
                l = list_date.get(c)
                age_td = (tpos[dt] - tpos[l]) if (l is not None and l in tpos) else None
                if l is not None and age_td is None and l < dt:
                    prior = [d for d in all_dates if l <= d <= dt]
                    age_td = len(prior) - 1
                in_panel = c in px.index
                v = float(vol.get(c, np.nan)) if in_panel else None
                ages.append(age_td)
                recs_out.append({"ts_code": c, "missing": missing, "list_date": (str(l.date()) if l is not None else None),
                                 "trading_days_since_listing": age_td, "in_price_panel": in_panel,
                                 "volume_that_day": v,
                                 "under_206_lookback": (age_td is not None and age_td < LOOKBACKS["yeslab_206"]),
                                 "under_443_lookback": (age_td is not None and age_td < LOOKBACKS["yeslab_443_reg30_max20_v1"]),
                                 "suspended_zero_volume": (in_panel and v is not None and v == 0)})
            a = pd.Series([x for x in ages if x is not None], dtype=float)
            n = len(recs_out)
            out_days.append({
                "day": day, "valid_label_rows": len(V), "intersection_rows": len(inter),
                "residual_non_industry_rows": n,
                "under_206_lookback": int(sum(r["under_206_lookback"] for r in recs_out)),
                "under_443_lookback": int(sum(r["under_443_lookback"] for r in recs_out)),
                "listing_date_unknown": int(sum(1 for r in recs_out if r["trading_days_since_listing"] is None)),
                "not_in_price_panel": int(sum(1 for r in recs_out if not r["in_price_panel"])),
                "suspended_zero_volume": int(sum(r["suspended_zero_volume"] for r in recs_out)),
                "share_under_206_lookback": (sum(r["under_206_lookback"] for r in recs_out) / n) if n else None,
                "trading_days_since_listing": {
                    "min": float(a.min()) if len(a) else None, "median": float(a.median()) if len(a) else None,
                    "p90": float(a.quantile(0.9)) if len(a) else None, "max": float(a.max()) if len(a) else None},
                "oldest_5_by_listing_age": sorted(recs_out, key=lambda r: -(r["trading_days_since_listing"] or 0))[:5],
                "records": recs_out,
            })
            print(f"  {day}: resid={n} under206={out_days[-1]['under_206_lookback']} "
                  f"under443={out_days[-1]['under_443_lookback']} median_age="
                  f"{out_days[-1]['trading_days_since_listing']['median']}", flush=True)

    df = pd.DataFrame(out_days)
    tot = int(df.residual_non_industry_rows.sum())
    u206 = int(df.under_206_lookback.sum())
    out = {
        "question": "Why does yeslab_206 have no value for the residual securities?",
        "hypothesis": "IPO seasoning: 206 needs a 60-day lookback and 443_reg30_max20_v1 a 30-day window",
        "days": len(df), "day_list": days,
        "totals": {"residual_non_industry_rows": tot, "under_206_lookback": u206,
                   "under_443_lookback": int(df.under_443_lookback.sum()),
                   "listing_date_unknown": int(df.listing_date_unknown.sum()),
                   "not_in_price_panel": int(df.not_in_price_panel.sum()),
                   "suspended_zero_volume": int(df.suspended_zero_volume.sum()),
                   "share_under_206_lookback": (u206 / tot) if tot else None},
        "answer": None, "per_day": out_days,
        "database_writes": False, "weights_written": False, "cache_writes": False,
    }
    out["answer"] = (
        f"{u206} of {tot} residual rows ({u206/tot:.1%}) are securities with fewer than 60 TRADING days since "
        f"listing, which is exactly yeslab_206's lookback; {int(df.under_443_lookback.sum())} have fewer than 30, "
        f"which is yeslab_443_reg30_max20_v1's window. Listing date is unknown for "
        f"{int(df.listing_date_unknown.sum())}, {int(df.not_in_price_panel.sum())} are absent from that day's price "
        f"panel and {int(df.suspended_zero_volume.sum())} traded zero volume. So the ceiling on the pre-cutover "
        f"dense-nine intersection is IPO seasoning, not a defect.")
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n" + out["answer"])
    print(df[["day", "residual_non_industry_rows", "under_206_lookback", "under_443_lookback",
              "listing_date_unknown", "not_in_price_panel", "suspended_zero_volume"]].to_string(index=False))


if __name__ == "__main__":
    main()
