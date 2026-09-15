#!/usr/bin/env python3
"""Read-only: does the re-implemented compute_gp074_variant reproduce the frozen yeslab_443 column?

§BJ.1 measured that yeslab_443_reg30_max20_v1 depends on a division by a variance of order 1e-14, so
beta reaches 1e9-1e15 and the regression term is 39% of the demeaned target at the median cell. That
measurement re-implemented repair_alpha443_window_scan.py:40-52 line by line rather than importing it,
and §BJ.6 recorded the obvious gap: nothing yet shows the re-implementation computes the same column the
materializer wrote. This closes that gap.

It recomputes the variant from the canonical Tushare cache and compares it, per OOS day, against the
frozen panel's f10 column (which walk_forward_factor_daily.py:75-91 maps to
yeslab_443_reg30_max20_v1) on the rows both sides have. Two numbers matter: the per-day Spearman, which
should be near 1 if the two are the same quantity, and the exact-equality rate, which says whether they
are the same floats or merely the same ordering -- and given a 1e14 amplification of rounding residue,
"same ordering but not same floats" is a real possible outcome that would itself be worth recording.

A caveat that must travel with the result: the materializer warmed up from its own market_start, this
probe warms up from 2021-10-01, so the earliest reported days have less history than the frozen column
had. The comparison therefore starts at 2022-01-04, where both sides have at least 50 trading days.

Read-only: the cache and the frozen parquet panels are opened for reading; no repository module is
imported or executed; no database is touched; nothing is written into any artifact directory.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from datetime import date, datetime
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd
from scipy import stats

ART = Path("/data/zhangyudong/qrv2_research_artifacts")
CACHE = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2/artifacts/tushare_cache")
FOLD_DATASETS = {
    "F1": ("226a8dfc497c2de6b07516711f058f26a9270675916c9e67dbf7f087a07d2538", 2022),
    "F2": ("eb9bba1b5a0bd66d0ad6c82536b295de1fc079aec125cb27bb68491bdb21a234", 2023),
    "F3": ("9766e69f342511c5bda30188df15a7b9b17d936bdc26b9b60ba244e8e7e0c9e7", 2024),
    "F4": ("6f9478d7645c8c745a73f00c03b6122e17d0c06c05cc159ece531fb00df9fb9d", 2025),
}
WORKSPACES = ("m2_v1", "m0r_v2", "m1_full_v1", "m0_v3", "m0_v2")
REG_WINDOW, MAX_WINDOW = 30, 20
FROZEN_COL = "f10"   # walk_forward_factor_daily.py:75-91 -> yeslab_443_reg30_max20_v1


def load_market(start: date, end: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    days = sorted(
        datetime.strptime(m.group(1), "%Y%m%d").date()
        for p in CACHE.glob("daily_*.json")
        if (m := re.fullmatch(r"daily_(\d{8})\.json", p.name))
    )
    days = [d for d in days if start <= d <= end]
    amt, pct = {}, {}
    for i, d in enumerate(days):
        with open(CACHE / f"daily_{d:%Y%m%d}.json", encoding="utf-8") as fh:
            frame = pd.DataFrame(json.load(fh)["data"])
        frame["ts_code"] = frame["ts_code"].astype(str)
        amt[d] = pd.to_numeric(frame.set_index("ts_code")["amount"], errors="coerce")
        pct[d] = pd.to_numeric(frame.set_index("ts_code")["pct_chg"], errors="coerce")
        if i % 250 == 0:
            print(f"    loaded {i}/{len(days)}", flush=True)
    return pd.DataFrame(amt).T.sort_index(), pd.DataFrame(pct).T.sort_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ART / "alpha443")
    ap.add_argument("--start", default="2021-10-01")
    ap.add_argument("--daily-cache", type=Path,
                    default=ART / "alpha443" / "alpha443_match_daily_rows.csv",
                    help="where to persist the per-day comparison rows; the summary block "
                         "crashed once after nine minutes of compute, so the expensive part "
                         "is now written to disk the moment it exists")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    amount_all, pct_all = load_market(date.fromisoformat(args.start), date(2025, 12, 31))

    def variant(amount: pd.DataFrame, pct_chg: pd.DataFrame) -> pd.DataFrame:
        x = amount.rank(axis=1, pct=True).mean(axis=1)
        y = amount * pct_chg
        mx = x.rolling(REG_WINDOW, min_periods=REG_WINDOW).mean()
        varx = x.rolling(REG_WINDOW, min_periods=REG_WINDOW).var().replace(0.0, np.nan)
        my = y.rolling(REG_WINDOW, min_periods=REG_WINDOW).mean()
        cov = (y.mul(x, axis=0).rolling(REG_WINDOW, min_periods=REG_WINDOW).mean()
               .sub(my.mul(mx, axis=0), axis=0))
        beta = cov.div(varx, axis=0)
        residual = y.sub(my, axis=0).sub(beta.mul(x.sub(mx, axis=0), axis=0))
        return residual.rolling(MAX_WINDOW, min_periods=MAX_WINDOW).max()

    # normalize_full_tushare_market (audit_formula_replay_tushare.py:295-309) keeps only .SH/.SZ codes,
    # and fetch_full_tushare_shsz_market is what the materializer called -- so the frozen column was
    # computed on the SH/SZ cross-section. x is a CROSS-SECTIONAL statistic, so the code set is an input.
    shsz = [c for c in amount_all.columns if str(c).endswith((".SH", ".SZ"))]
    other = [c for c in amount_all.columns if not str(c).endswith((".SH", ".SZ"))]
    print(f"  codes: {len(amount_all.columns)} total, {len(shsz)} SH/SZ, {len(other)} other "
          f"({sorted({str(c).split('.')[-1] for c in other})})", flush=True)
    factor = variant(amount_all.loc[:, shsz], pct_all.loc[:, shsz])
    factor_all_codes = variant(amount_all, pct_all)
    xa = amount_all.rank(axis=1, pct=True).mean(axis=1)
    xs = amount_all.loc[:, shsz].rank(axis=1, pct=True).mean(axis=1)
    print(f"  x on all codes: mean {xa.mean():.10f} std {xa.std():.3e} | "
          f"x on SH/SZ: mean {xs.mean():.10f} std {xs.std():.3e} | "
          f"median |difference| {float((xa - xs).abs().median()):.3e}", flush=True)
    VARIANTS = {"shsz_only_as_the_materializer_did": factor, "all_codes_including_bj": factor_all_codes}

    rows = []
    panels = {}
    for fold, (ds, year) in FOLD_DATASETS.items():
        for ws in WORKSPACES:
            paths = sorted(glob.glob(str(ART / ws / "datasets" / ds / "split=test" / "*" / "*.parquet")))
            if paths:
                break
        pan = pd.concat([pd.read_parquet(p, columns=["signal_date", "ts_code", FROZEN_COL])
                         for p in paths], ignore_index=True)
        pan["signal_date"] = pd.to_datetime(pan["signal_date"]).dt.date
        pan["frozen"] = pd.to_numeric(pan[FROZEN_COL], errors="coerce")
        # only that day's own rows: a rectangular year-wide grid would count "code absent from this
        # day's cross-section" as "factor missing", which is a different thing (and was a bug in the
        # first version of this probe -- see the withdrawn nan_agreement block)
        pan["frozen"] = pd.to_numeric(pan[FROZEN_COL], errors="coerce")
        panels[fold] = (pan, year)

    for vname, fac in VARIANTS.items():
        vrows = []
        for fold, (pan, year) in panels.items():
            for d, g in pan.groupby("signal_date"):
                if d not in fac.index:
                    continue
                a = fac.loc[d]
                codes = [c for c in g["ts_code"] if c in a.index]
                if len(codes) < 100:
                    continue
                av = a[codes]
                bv = g.set_index("ts_code").loc[codes, "frozen"]
                # capture the NaN shares BEFORE the both-mask: two lines down `av, bv = av[both], bv[both]`
                # makes both series non-NaN by construction, so any NaN share read off them afterwards is
                # vacuous. The first run of this probe reported median_recomputed_nan_share = 0.0 for
                # exactly that reason -- a measurement artifact, not a property of the factor.
                recomputed_nan_share_prefilter = float(av.isna().mean())
                frozen_nan_share_on_compared_codes = float(bv.isna().mean())
                both = av.notna() & bv.notna()
                if int(both.sum()) < 100:
                    continue
                av, bv = av[both], bv[both]
                rho = stats.spearmanr(av, bv).statistic
                vrows.append({
                    "variant": vname, "fold": fold, "year": year, "signal_date": d,
                    "n": int(both.sum()),
                    "spearman": float(rho) if np.isfinite(rho) else None,
                    "exact_equal_share": float((av.to_numpy() == bv.to_numpy()).mean()),
                    "within_1e_9_relative_share": float(
                        ((av - bv).abs() <= 1e-9 * bv.abs().clip(lower=1e-12)).mean()),
                    "within_1e_6_relative_share": float(
                        ((av - bv).abs() <= 1e-6 * bv.abs().clip(lower=1e-12)).mean()),
                    "median_relative_difference": float(
                        ((av - bv).abs() / bv.abs().replace(0, np.nan)).median()),
                    "frozen_nan_share_of_that_days_own_rows": float(
                        pd.to_numeric(g[FROZEN_COL], errors="coerce").isna().mean()),
                    "recomputed_nan_share_of_that_days_own_codes": float(av.isna().mean()),
                    "recomputed_nan_share_prefilter": recomputed_nan_share_prefilter,
                    "frozen_nan_share_on_compared_codes": frozen_nan_share_on_compared_codes,
                })
            print(f"  {vname} {fold} ({year}) done", flush=True)
        rows.extend(vrows)

    all_daily = pd.DataFrame(rows).dropna(subset=["spearman"])
    args.daily_cache.parent.mkdir(parents=True, exist_ok=True)
    all_daily.to_csv(args.daily_cache, index=False)
    print(f"  wrote {len(all_daily)} per-day rows to {args.daily_cache}", flush=True)
    per_variant = {}
    for vname, g in all_daily.groupby("variant"):
        s = g["spearman"]
        per_variant[vname] = {
            "days": int(len(g)),
            "spearman_median": float(s.median()), "spearman_min": float(s.min()),
            "spearman_max": float(s.max()),
            "share_of_days_above_0.9999": float((s > 0.9999).mean()),
            "share_of_days_above_0.999": float((s > 0.999).mean()),
            "exact_equal_share_median": float(g["exact_equal_share"].median()),
            "exact_equal_share_max": float(g["exact_equal_share"].max()),
            "days_all_cells_bit_equal": int((g["exact_equal_share"] == 1.0).sum()),
            "within_1e_9_share_median": float(g["within_1e_9_relative_share"].median()),
            "within_1e_6_share_median": float(g["within_1e_6_relative_share"].median()),
            "median_relative_difference_median": float(g["median_relative_difference"].median()),
            "frozen_nan_share_median": float(g["frozen_nan_share_of_that_days_own_rows"].median()),
            "recomputed_nan_share_median_VACUOUS_zero_by_construction": float(
                g["recomputed_nan_share_of_that_days_own_codes"].median()),
            "recomputed_nan_share_prefilter_median": float(
                g["recomputed_nan_share_prefilter"].median()),
            "recomputed_nan_share_prefilter_max": float(
                g["recomputed_nan_share_prefilter"].max()),
            "frozen_nan_share_on_compared_codes_median": float(
                g["frozen_nan_share_on_compared_codes"].median()),
            "days_where_frozen_has_nan_but_recompute_does_not": int(
                ((g["frozen_nan_share_on_compared_codes"] > 0)
                 & (g["recomputed_nan_share_prefilter"] == 0)).sum()),
        }
        print(f"  VARIANT {vname}: " + json.dumps(per_variant[vname], indent=None), flush=True)

    daily = all_daily[all_daily["variant"] == "shsz_only_as_the_materializer_did"] \
        .sort_values("signal_date").reset_index(drop=True)
    sp = daily["spearman"]
    result = {
        "question": (
            "Does the re-implemented compute_gp074_variant reproduce the frozen "
            "yeslab_443_reg30_max20_v1 column, and if so to what precision?"
        ),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "read_only": True, "database_access": False, "repository_module_executed": False,
        "frozen_column": {"panel_field": FROZEN_COL, "factor_id": "yeslab_443_reg30_max20_v1"},
        "warmup_caveat": (
            "this probe warms up from 2021-10-01 while the materializer used its own market_start, so the "
            "comparison starts at 2022-01-04 where both sides have at least 50 trading days"
        ),
        "days_compared": int(len(daily)),
        "per_variant": per_variant,
        "rows_compared_total": int(daily["n"].sum()),
        "spearman": {"median": float(sp.median()), "p05": float(sp.quantile(0.05)),
                     "min": float(sp.min()), "max": float(sp.max()),
                     "share_of_days_above_0.999": float((sp > 0.999).mean()),
                     "share_of_days_above_0.9999": float((sp > 0.9999).mean()),
                     "days_below_0.99": int((sp < 0.99).sum())},
        "exact_equality": {
            "median_share_of_cells_bit_equal": float(daily["exact_equal_share"].median()),
            "min_share": float(daily["exact_equal_share"].min()),
            "max_share": float(daily["exact_equal_share"].max()),
            "days_where_every_compared_cell_is_bit_equal": int((daily["exact_equal_share"] == 1.0).sum()),
        },
        "relative_closeness": {
            "median_of_the_median_relative_difference": float(daily["median_relative_difference"].median()),
            "max_of_the_median_relative_difference": float(daily["median_relative_difference"].max()),
            "median_share_within_1e_9": float(daily["within_1e_9_relative_share"].median()),
            "median_share_within_1e_6": float(daily["within_1e_6_relative_share"].median()),
        },
        "nan_agreement": {
            "median_frozen_nan_share": float(
                daily["frozen_nan_share_of_that_days_own_rows"].median()),
            "median_recomputed_nan_share_VACUOUS_zero_by_construction": float(
                daily["recomputed_nan_share_of_that_days_own_codes"].median()),
            "median_recomputed_nan_share_prefilter": float(
                daily["recomputed_nan_share_prefilter"].median()),
            "max_recomputed_nan_share_prefilter": float(
                daily["recomputed_nan_share_prefilter"].max()),
            "median_frozen_nan_share_on_compared_codes": float(
                daily["frozen_nan_share_on_compared_codes"].median()),
            "days_where_frozen_has_nan_but_recompute_does_not": int(
                ((daily["frozen_nan_share_on_compared_codes"] > 0)
                 & (daily["recomputed_nan_share_prefilter"] == 0)).sum()),
        },
        "per_year": {str(y): {"days": int((daily["year"] == y).sum()),
                              "spearman_median": float(sp[daily["year"] == y].median()),
                              "spearman_min": float(sp[daily["year"] == y].min()),
                              "exact_equal_median": float(
                                  daily.loc[daily["year"] == y, "exact_equal_share"].median())}
                     for y in sorted(set(daily["year"]))},
        "worst_ten_days": [{"date": str(r["signal_date"]), "spearman": r["spearman"], "n": r["n"],
                            "exact_equal_share": r["exact_equal_share"]}
                           for r in daily.nsmallest(10, "spearman").to_dict("records")],
        "reading": (
            "a median Spearman at 1.0 with a bit-equality share at 1.0 means §BJ.1 measured the column "
            "the models actually consume. A median Spearman at 1.0 with a low bit-equality share means "
            "the two are the same ordering but not the same floats, which for a 1e14 amplification of "
            "rounding residue is itself the finding: the column is order-stable but not bit-stable "
            "across environments."
        ),
    }
    out = args.out / "alpha443_matches_frozen_v2.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps({k: result[k] for k in ("days_compared", "rows_compared_total", "spearman",
                                             "exact_equality", "relative_closeness", "nan_agreement",
                                             "per_year")}, ensure_ascii=False, indent=1))
    print("worst:", ", ".join(f"{w['date']}={w['spearman']:.5f}" for w in result["worst_ten_days"][:6]))


if __name__ == "__main__":
    main()
