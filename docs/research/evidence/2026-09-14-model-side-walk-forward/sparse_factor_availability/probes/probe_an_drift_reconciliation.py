#!/usr/bin/env python3
"""Read-only: does §AN's "six sparse factors drift DOWN 0.097-0.184" survive the whole window?

§AN sampled 10 days and defined the drift as median(5 post-cutover days) - median(5 pre-cutover days),
on a series it also recorded as extremely volatile (yeslab_372's row count 1,022 -> 3,262 -> 1,241 on
three consecutive sampled trading days). `probe_sparse_factor_coverage_in_panels.py` then measured the
same six factors over every scoreable row of all four test splits and found the opposite sign: across
2022 -> 2025 all six mean daily coverages RISE, by +0.0059 to +0.0209.

Both cannot be right about the same quantity, so this probe separates the two possible causes:

  E  the estimator: a 5-day median on either side of a volatile series is not a window trend. Test --
     recompute §AN's exact estimator from whole-window per-day coverage and see whether it reproduces
     §AN's numbers, and compare it with the whole-window annual means and with an ordinary least
     squares slope over all days.
  D  the denominator: §AN divides by registry-resolved universe members, the panels divide by the rows
     the metric can score. Test -- report both denominators' per-day sizes and the ratio, so a reader
     can see how much of any level difference is denominator and how much is estimator.

Read-only. No database: everything comes from the frozen panels plus §AN's own recorded numbers, which
are quoted from its measurement file rather than retyped. No weights, no model, no artifact writes.
"""
from __future__ import annotations

import argparse
import glob
import json
from datetime import date, datetime
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd

ART = Path("/data/zhangyudong/qrv2_research_artifacts")
PACK = Path("/data/zhangyudong/qrv2_worktrees/research-conditional-factor-review/docs/research/"
            "evidence/2026-09-14-model-side-walk-forward")
FOLD_DATASETS = {
    "F1": ("226a8dfc497c2de6b07516711f058f26a9270675916c9e67dbf7f087a07d2538", 2022),
    "F2": ("eb9bba1b5a0bd66d0ad6c82536b295de1fc079aec125cb27bb68491bdb21a234", 2023),
    "F3": ("9766e69f342511c5bda30188df15a7b9b17d936bdc26b9b60ba244e8e7e0c9e7", 2024),
    "F4": ("6f9478d7645c8c745a73f00c03b6122e17d0c06c05cc159ece531fb00df9fb9d", 2025),
}
WORKSPACES = ("m2_v1", "m0r_v2", "m1_full_v1", "m0_v3", "m0_v2")
FACTOR_BY_INDEX = (
    "yeslab_206", "yeslab_357", "yeslab_372", "yeslab_375", "yeslab_381", "yeslab_384",
    "yeslab_389", "yeslab_390", "yeslab_406", "yeslab_425", "yeslab_443_reg30_max20_v1",
    "yeslab_464", "yeslab_471", "yeslab_484", "yeslab_96",
)
SPARSE = ["yeslab_372", "yeslab_375", "yeslab_384", "yeslab_389", "yeslab_390", "yeslab_406"]
COL_OF = {n: f"f{i:02d}" for i, n in enumerate(FACTOR_BY_INDEX)}
# §AN's sample, verbatim from probe_frozen_v2_preflight.py's default --days
AN_DAYS = ["2022-01-04", "2022-06-30", "2022-07-29", "2023-07-05", "2023-09-22",
           "2023-09-25", "2024-06-17", "2024-12-20", "2025-06-16", "2025-12-31"]
CUTOVER = date(2023, 9, 25)


def paths(artifacts: Path, ds: str, split: str) -> list[str]:
    for ws in WORKSPACES:
        p = sorted(glob.glob(str(artifacts / ws / "datasets" / ds / f"split={split}" / "*" / "*.parquet")))
        if p:
            return p
    raise FileNotFoundError(f"{ds} split={split}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, default=ART)
    ap.add_argument("--out", type=Path, default=ART / "sparse_coverage")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    an_path = PACK / "star_industry_coverage" / "measurements" / "frozen_v2_preflight.json"
    an = json.loads(an_path.read_text(encoding="utf-8")) if an_path.is_file() else {}

    per_day: dict[str, dict] = {}     # factor -> {date: coverage over scoreable rows}
    denom: dict[date, dict] = {}      # date -> {scoreable rows, all panel rows}
    for fold, (ds, year) in FOLD_DATASETS.items():
        te = pd.concat(
            [pd.read_parquet(p, columns=["signal_date", "ts_code", "label_valid", "raw_return_5d"]
                             + [COL_OF[s] for s in SPARSE])
             for p in paths(args.artifacts, ds, "test")],
            ignore_index=True,
        )
        te["signal_date"] = pd.to_datetime(te["signal_date"]).dt.date
        scoreable = (
            te["label_valid"].fillna(False).astype(bool)
            & np.isfinite(pd.to_numeric(te["raw_return_5d"], errors="coerce"))
        ).to_numpy()
        all_rows = te.groupby("signal_date").size()
        sc_rows = te.loc[scoreable].groupby("signal_date").size()
        for d in all_rows.index:
            denom[d] = {"all_panel_rows": int(all_rows[d]),
                        "scoreable_rows": int(sc_rows.get(d, 0))}
        sub = te.loc[scoreable]
        for s in SPARSE:
            col = COL_OF[s]
            ser = pd.to_numeric(sub[col], errors="coerce").notna()
            daily = ser.groupby(sub["signal_date"].to_numpy()).mean()
            per_day.setdefault(s, {}).update({d: float(v) for d, v in daily.items()})
        print(f"  {fold} ({year}) loaded: {len(te)} rows, {te['signal_date'].nunique()} days", flush=True)
        del te

    result = {
        "question": (
            "Does §AN's 'the six candidate_only factors drift DOWN 0.097-0.184 across the OOS window' "
            "survive measurement over every day of that window, and if not, is the difference the "
            "estimator (a 5-day median on a volatile series) or the denominator (universe vs scoreable "
            "panel rows)?"
        ),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "read_only": True, "database_access": False, "weights_loaded": False,
        "window": {"days": len(denom), "first": str(min(denom)), "last": str(max(denom))},
        "denominator_caveat": (
            "coverage here is the non-NaN share among the panel rows the metric can score "
            "(label_valid with a finite 5-day return). §AN divided by registry-resolved universe "
            "members. The two denominators are not the same set, so LEVELS are not directly "
            "comparable; TRENDS and the estimator comparison below are the point."
        ),
        "denominator_per_day": {
            "scoreable_rows": {"min": min(v["scoreable_rows"] for v in denom.values()),
                               "median": float(median(v["scoreable_rows"] for v in denom.values())),
                               "max": max(v["scoreable_rows"] for v in denom.values())},
            "all_panel_rows": {"min": min(v["all_panel_rows"] for v in denom.values()),
                               "median": float(median(v["all_panel_rows"] for v in denom.values())),
                               "max": max(v["all_panel_rows"] for v in denom.values())},
        },
        "factors": {},
    }

    for s in SPARSE:
        series = pd.Series(per_day[s]).sort_index()
        idx = np.arange(len(series), dtype=float)
        slope = float(np.polyfit(idx, series.to_numpy(), 1)[0]) if len(series) > 2 else None
        pre = series[series.index < CUTOVER]
        post = series[series.index >= CUTOVER]
        # §AN's estimator, but on the whole-window series restricted to its 10 sampled days
        an_days = [date.fromisoformat(d) for d in AN_DAYS]
        on_an_days = series.reindex([d for d in an_days if d in series.index])
        an_pre = on_an_days[[d for d in on_an_days.index if d < CUTOVER]]
        an_post = on_an_days[[d for d in on_an_days.index if d >= CUTOVER]]
        by_year = {str(y): float(g.mean()) for y, g in series.groupby([d.year for d in series.index])}
        result["factors"][s] = {
            "column": COL_OF[s],
            "days_measured": int(len(series)),
            "whole_window": {"mean": float(series.mean()), "median": float(series.median()),
                             "min": float(series.min()), "max": float(series.max()),
                             "std": float(series.std(ddof=1)),
                             "ols_slope_per_trading_day": slope,
                             "ols_slope_times_window": (slope * len(series)) if slope is not None else None},
            "by_test_year_mean": by_year,
            "across_years_2022_to_2025": by_year.get("2025", float("nan")) - by_year.get("2022", float("nan")),
            "cutover_split_whole_window": {
                "pre_days": int(len(pre)), "post_days": int(len(post)),
                "pre_median": float(pre.median()) if len(pre) else None,
                "post_median": float(post.median()) if len(post) else None,
                "drift_post_minus_pre": (float(post.median()) - float(pre.median()))
                if len(pre) and len(post) else None,
            },
            "AN_estimator_reproduced_on_panel_coverage": {
                "days_available": int(len(on_an_days)),
                "days_requested": len(AN_DAYS),
                "pre_values": {str(k): v for k, v in an_pre.items()},
                "post_values": {str(k): v for k, v in an_post.items()},
                "pre_median": float(an_pre.median()) if len(an_pre) else None,
                "post_median": float(an_post.median()) if len(an_post) else None,
                "drift": (float(an_post.median()) - float(an_pre.median()))
                if len(an_pre) and len(an_post) else None,
            },
            "volatility": {
                "largest_one_day_drop": float(series.diff().min()),
                "largest_one_day_rise": float(series.diff().max()),
                "days_where_the_one_day_change_exceeds_0.10": int((series.diff().abs() > 0.10).sum()),
                "note": "§AN recorded yeslab_372's row count as 1,022 -> 3,262 -> 1,241 on three of its "
                        "sampled days; a 5-day median taken on either side of a series like this is not a "
                        "window trend",
            },
        }

    # what §AN itself recorded, quoted rather than retyped
    quoted = {}
    for row in (an.get("rows") or an.get("per_factor") or []):
        if isinstance(row, dict):
            name = row.get("factor_id") or row.get("factor")
            if name in SPARSE:
                quoted[name] = {k: v for k, v in row.items()
                                if k in ("coverage_min", "coverage_median", "drift",
                                         "median_precutover", "median_postcutover",
                                         "coverage_median_precutover", "coverage_median_postcutover")}
    result["AN_recorded_values_quoted_from_its_measurement_file"] = quoted or (
        "the file's layout did not match the keys looked for; §AN's table in problems_found.md is the "
        "reference and is quoted in the narrative instead"
    )

    drifts_an = [result["factors"][s]["AN_estimator_reproduced_on_panel_coverage"]["drift"] for s in SPARSE]
    drifts_cw = [result["factors"][s]["cutover_split_whole_window"]["drift_post_minus_pre"] for s in SPARSE]
    across = [result["factors"][s]["across_years_2022_to_2025"] for s in SPARSE]
    result["verdict"] = {
        "AN_style_5_day_sampled_drift_on_panel_coverage": {
            "values": drifts_an, "min": min(d for d in drifts_an if d is not None),
            "max": max(d for d in drifts_an if d is not None),
            "all_negative": all(d is not None and d < 0 for d in drifts_an),
        },
        "whole_window_cutover_split_drift": {
            "values": drifts_cw, "min": min(drifts_cw), "max": max(drifts_cw),
            "all_negative": all(d < 0 for d in drifts_cw),
        },
        "across_test_years_2022_to_2025": {
            "values": across, "min": min(across), "max": max(across),
            "all_negative": all(a < 0 for a in across),
            "all_positive": all(a > 0 for a in across),
        },
        "reading": (
            "compare the three blocks: if the sampled 5-day estimator is negative while the whole-window "
            "cutover split and the across-year means are not, the drift §AN reported is a property of the "
            "10 sampled days rather than of the window"
        ),
    }

    out = args.out / "an_drift_reconciliation.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    print(f"wrote {out}")
    for s in SPARSE:
        f = result["factors"][s]
        print(f"{s:14} days={f['days_measured']:4} mean={f['whole_window']['mean']:.4f} "
              f"std={f['whole_window']['std']:.4f} by_year=" +
              " ".join(f"{k}:{v:.4f}" for k, v in f["by_test_year_mean"].items()) +
              f" across={f['across_years_2022_to_2025']:+.4f} "
              f"AN5day={f['AN_estimator_reproduced_on_panel_coverage']['drift']:+.4f} "
              f"cutover_split={f['cutover_split_whole_window']['drift_post_minus_pre']:+.4f} "
              f"maxdrop={f['volatility']['largest_one_day_drop']:+.4f}")
    print(json.dumps(result["verdict"], ensure_ascii=False, indent=1)[:1400])


if __name__ == "__main__":
    main()
