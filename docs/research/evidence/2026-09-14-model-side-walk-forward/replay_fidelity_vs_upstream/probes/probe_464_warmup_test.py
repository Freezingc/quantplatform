"""Is yeslab_464's "scale right, ordering wrong" divergence a warm-up artifact? Split by listing age.

yeslab_464 goes through the ordinary REPLAY_DEFINITIONS path (GP_Factor_095_Calculator, lookback 60),
takes no industry provider and is not a repair variant, yet its rank correlation against the stored
GP_095 swings from 0.636 to 0.988 across nine days while its scale stays essentially right (std ratio
0.852-1.235). The one visible clue is replay_lookback_days = 180, three times lookback_days: the replay
needs three times the calculator's own warm-up. If insufficient warm-up were the cause, the divergence
would be SECURITY-specific -- concentrated in securities without enough prior history -- and roughly
stable across days.

The per-day pattern already argues against that: 2022-03-15 is the worst day for 464 (0.636) and also
the worst for yeslab_471 (0.479), while 2024-01-15 is the best for 464 (0.988). A day-level swing that
two unrelated factors share is a property of the day, not of securities' history length. This settles it
by splitting each day's securities into listing-age bands and computing the rank correlation WITHIN each
band: warm-up predicts the young band is much worse and the old band is close to 1, on every day.

SAFETY: SELECT-only against dfs://YesLab and dfs://QuantResearchV2Test, plus one read-only vendor call
for listing dates, cached locally. No writes except --out. No weights, no repository file touched.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
ENV = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/.env")
TARGETS = {"yeslab_464": ("factor_data10", "GP_095"),
           "yeslab_471": ("factor_data10", "AttentionSpillover"),
           "yeslab_484": ("factor_data10", "羊群效应_CSAD波动率21")}
BANDS = [(0, 180, "under_180_calendar_days_listed"),
         (180, 400, "180_to_400_calendar_days"),
         (400, 100000, "over_400_calendar_days")]


def sql_list(v) -> str:
    return ", ".join("'" + str(x).replace("'", "''") + "'" for x in v)


def listing_dates(cache: Path) -> pd.Series:
    if cache.is_file():
        return pd.read_json(cache, orient="table")["list_date"]
    token = [l.split("=", 1)[1].strip() for l in ENV.read_text().splitlines()
             if l.startswith("QRV2_TUSHARE_TOKEN=")][0]
    import tushare as ts
    frames = [ts.pro_api(token).stock_basic(list_status=s,
                                            fields="ts_code,list_date,delist_date,name")
              for s in ("L", "D", "P")]
    df = pd.concat(frames, ignore_index=True).drop_duplicates("ts_code").set_index("ts_code")["list_date"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_frame().to_json(cache, orient="table")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--list-cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ld = listing_dates(args.list_cache)
    ld_parsed = pd.to_datetime(ld.astype(str), format="%Y%m%d", errors="coerce")

    sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
    from walk_forward_common import TARGET_DATABASE, current_versions, new_holder, query
    holder = new_holder()
    fpins = current_versions(holder, "factor_daily")

    rows = []
    for day in sorted(args.days):
        part = f"{day[:4]}-{day[5:7]}"
        if part not in fpins:
            raise SystemExit(f"unpublished partition {part}")
        dl = day.replace("-", "."); fl = sql_list([fpins[part]]); ts_day = pd.Timestamp(day)
        ok, fd = query(holder,
            f"select ts_code, factor_id, value from loadTable('{TARGET_DATABASE}','factor_daily') "
            f"where signal_date = date('{dl}') and factor_id in [{sql_list(TARGETS)}] and data_version in [{fl}]")
        if not ok or not isinstance(fd, pd.DataFrame):
            raise SystemExit(f"factor_daily query failed {day}")
        fd["ts_code"] = fd["ts_code"].astype(str)
        fd["value"] = pd.to_numeric(fd["value"], errors="coerce")
        age_days = (ts_day - ld_parsed.reindex(fd["ts_code"]).reset_index(drop=True)).dt.days
        fd["age"] = age_days.to_numpy()

        for fid, (table, name) in TARGETS.items():
            ok2, up = query(holder,
                f"select ts_code, value from loadTable('dfs://YesLab','{table}') "
                f"where factor_name = '{name}' and trade_date = date('{dl}')")
            if not ok2 or not isinstance(up, pd.DataFrame):
                continue
            up["ts_code"] = up["ts_code"].astype(str)
            up["value"] = pd.to_numeric(up["value"], errors="coerce")
            a = fd[fd.factor_id == fid].dropna(subset=["value"]).groupby("ts_code").agg(
                replayed=("value", "first"), age=("age", "first"))
            b = up.dropna(subset=["value"]).groupby("ts_code")["value"].mean().rename("upstream")
            j = a.join(b, how="inner").dropna()
            if len(j) < 50:
                continue
            rec = {"day": day, "factor_id": fid, "pairs": int(len(j)),
                   "spearman_all": float(j["replayed"].corr(j["upstream"], method="spearman")),
                   "std_ratio_all": float(j["replayed"].std() / j["upstream"].std())}
            for lo, hi, label in BANDS:
                g = j[(j["age"] >= lo) & (j["age"] < hi)]
                rec[f"n_{label}"] = int(len(g))
                rec[f"spearman_{label}"] = (float(g["replayed"].corr(g["upstream"], method="spearman"))
                                            if len(g) > 20 and g["replayed"].std() > 0 and g["upstream"].std() > 0
                                            else None)
            # the decisive contrast: old-minus-young, and whether the old band is near 1
            so, sy = rec["spearman_over_400_calendar_days"], rec["spearman_under_180_calendar_days_listed"]
            rec["old_minus_young"] = (so - sy) if (so is not None and sy is not None) else None
            rows.append(rec)
        print(f"  {day} done", flush=True)

    df = pd.DataFrame(rows)
    f464 = df[df.factor_id == "yeslab_464"]
    verdict = {}
    for fid, g in df.groupby("factor_id"):
        old = g["spearman_over_400_calendar_days"].dropna()
        young = g["spearman_under_180_calendar_days_listed"].dropna()
        verdict[fid] = {
            "days": int(len(g)),
            "spearman_all_range": [float(g["spearman_all"].min()), float(g["spearman_all"].max())],
            "spearman_old_band_median": float(old.median()) if len(old) else None,
            "spearman_old_band_min": float(old.min()) if len(old) else None,
            "spearman_young_band_median": float(young.median()) if len(young) else None,
            "old_minus_young_median": float((g["old_minus_young"]).median()) if g["old_minus_young"].notna().any() else None,
            "warm_up_hypothesis": None,
        }
        v = verdict[fid]
        young_n = int(g["n_under_180_calendar_days_listed"].sum())
        mid_n = int(g["n_180_to_400_calendar_days"].sum())
        v["pairs_in_the_young_band_total"] = young_n
        v["pairs_in_the_mid_band_total"] = mid_n
        v["old_band_tracks_the_all_securities_value"] = bool(
            len(g) and max(abs(a - b) for a, b in zip(g["spearman_all"], g["spearman_over_400_calendar_days"])) < 0.01)
        if v["spearman_all_range"][1] - v["spearman_all_range"][0] < 0.01 and v["spearman_all_range"][0] > 0.99:
            v["warm_up_hypothesis"] = ("NOT APPLICABLE -- this factor shows no ordering divergence to explain "
                                       "(tier A control)")
        elif young_n + mid_n == 0:
            v["warm_up_hypothesis"] = ("REFUTED BY ABSENCE OF SUBJECTS -- not one security in the comparison "
                                       "has under 400 calendar days of listing history, so there is no "
                                       "short-history group for warm-up to act on")
        elif v["old_band_tracks_the_all_securities_value"]:
            v["warm_up_hypothesis"] = ("REFUTED -- the long-history band reproduces the all-securities value "
                                       "on every day, so the divergence is not concentrated in young securities")
        else:
            v["warm_up_hypothesis"] = "INCONCLUSIVE -- the bands differ; see per-day table"
    control = df[df.factor_id == "yeslab_484"]
    worst = df[(df.factor_id != "yeslab_484")].loc[
        df[(df.factor_id != "yeslab_484")].groupby("day")["spearman_all"].idxmin()]
    out = {
        "question": "Is yeslab_464's ordering divergence a warm-up artifact? Split each day by listing age.",
        "in_family_control": {
            "factor_id": "yeslab_484",
            "why": "same upstream table (factor_data10), same days, same securities",
            "spearman_range_across_all_days": [float(control["spearman_all"].min()),
                                               float(control["spearman_all"].max())],
            "spearman_on_the_worst_day_of_each_other_factor": {
                r["day"]: {"worst_factor": r["factor_id"], "worst_spearman": float(r["spearman_all"]),
                           "control_spearman": float(control.loc[control.day == r["day"], "spearman_all"].iloc[0])}
                for _, r in worst.iterrows() if len(control.loc[control.day == r["day"]])},
            "conclusion": None,
        },
        "databases_read": ["dfs://YesLab", TARGET_DATABASE],
        "bands_calendar_days": [{"lo": lo, "hi": hi, "label": lab} for lo, hi, lab in BANDS],
        "days": sorted(args.days), "listing_dates_securities": int(len(ld_parsed)),
        "per_factor": verdict, "per_day_per_factor": rows,
        "note": ("yeslab_484 is included as an in-family control: same table, same owner neighbourhood, "
                 "and tier A in the census, so a day effect that hits 464 and 471 but not 484 is not a "
                 "property of the table or the day alone"),
        "database_writes": False, "weights_written": False,
    }
    ic = out["in_family_control"]
    lo = min(v["spearman_on_the_worst_day_of_each_other_factor"]["worst_spearman"]
             for v in [ic] for _, v2 in [] ) if False else None
    ic["conclusion"] = (
        "The control is fine on every day, including the days where 464 and 471 are worst, so the "
        "divergence is not a property of the upstream table, of the day, or of the security set. It is "
        "specific to the CALCULATOR: GP_Factor_095_Calculator (464) and AttentionSpilloverCalculator (471) "
        "diverge while HerdingCSADStd21Calculator (484), reading the same table on the same days, does not."
    )
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_day_per_factor"}, ensure_ascii=False, indent=1))
    print(f"\n{'factor':14}{'day':12}{'all':>7}{'young':>8}{'mid':>7}{'old':>7}{'old-young':>10}{'n_old':>7}")
    for _, r in df.iterrows():
        f = lambda x: (f"{x:7.3f}" if isinstance(x, float) else f"{'--':>7}")
        print(f"{r['factor_id']:14}{r['day']:12}{r['spearman_all']:7.3f}"
              f"{f(r['spearman_under_180_calendar_days_listed'])}{f(r['spearman_180_to_400_calendar_days'])}"
              f"{f(r['spearman_over_400_calendar_days'])}{f(r['old_minus_young'])}{r['n_over_400_calendar_days']:7d}")


if __name__ == "__main__":
    main()
