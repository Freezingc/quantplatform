#!/usr/bin/env python3
"""Read-only: how much of yeslab_443_reg30_max20_v1 is the regression term, and is beta well conditioned?

§BI.2 read compute_gp074_variant and noticed that its regressor
    x = amount.rank(axis=1, pct=True).mean(axis=1)
is the mean of a cross-sectional percentile rank, which is close to 0.5 on every day by construction.
So varx is the variance of a nearly constant series and beta = cov / varx is a ratio of two near-zero
quantities. The source comment beside it says the covariance approximation is "negligible for
screening", and this variant was then frozen into the model input set. §BI.2 recorded the concern as an
observation and explicitly did not measure it. This measures it.

It recomputes the same expression from the canonical Tushare cache the materializer used and reports:
the distribution of x and of varx; the distribution of beta; and, decisively, the ratio of the
regression term |beta * (x - mx)| to the demeaned target |y - my| -- if that ratio is tiny the factor is
effectively rolling_max(y - my) and the ill-conditioning is harmless, and if it is not, the frozen
column is materially a product of an unstable division.

Read-only: the cache is opened for reading, the repository function is re-implemented here line by line
rather than imported so that nothing in the main tree is executed with side effects, no database is
touched, no calculator is run, nothing is written into any artifact directory.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import date, datetime
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd

ART = Path("/data/zhangyudong/qrv2_research_artifacts")
CACHE = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2/artifacts/tushare_cache")
REG_WINDOW, MAX_WINDOW = 30, 20


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
            rows = json.load(fh)["data"]
        frame = pd.DataFrame(rows)
        frame["ts_code"] = frame["ts_code"].astype(str)
        amt[d] = pd.to_numeric(frame.set_index("ts_code")["amount"], errors="coerce")
        pct[d] = pd.to_numeric(frame.set_index("ts_code")["pct_chg"], errors="coerce")
        if i % 200 == 0:
            print(f"    loaded {i}/{len(days)}", flush=True)
    return pd.DataFrame(amt).T.sort_index(), pd.DataFrame(pct).T.sort_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ART / "alpha443")
    ap.add_argument("--start", default="2021-10-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--report-from", default="2022-01-04")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    amount, pct_chg = load_market(date.fromisoformat(args.start), date.fromisoformat(args.end))
    print(f"  market panel {amount.shape}", flush=True)

    # compute_gp074_variant, line by line, with the intermediates kept
    x = amount.rank(axis=1, pct=True).mean(axis=1)
    y = amount * pct_chg
    mx = x.rolling(REG_WINDOW, min_periods=REG_WINDOW).mean()
    varx = x.rolling(REG_WINDOW, min_periods=REG_WINDOW).var().replace(0.0, np.nan)
    my = y.rolling(REG_WINDOW, min_periods=REG_WINDOW).mean()
    cov = y.mul(x, axis=0).rolling(REG_WINDOW, min_periods=REG_WINDOW).mean().sub(my.mul(mx, axis=0), axis=0)
    beta = cov.div(varx, axis=0)
    demeaned = y.sub(my, axis=0)
    regression_term = beta.mul(x.sub(mx, axis=0), axis=0)
    residual = demeaned.sub(regression_term)
    factor = residual.rolling(MAX_WINDOW, min_periods=MAX_WINDOW).max()

    report_from = pd.Timestamp(args.report_from).date()
    in_window = pd.Series([d >= report_from for d in factor.index], index=factor.index)

    def describe(name: str, values) -> dict:
        v = pd.Series(np.asarray(values, dtype=float)).dropna()
        if v.empty:
            return {"name": name, "n": 0}
        return {"name": name, "n": int(len(v)), "mean": float(v.mean()), "std": float(v.std(ddof=1)),
                "min": float(v.min()), "p01": float(v.quantile(0.01)), "p05": float(v.quantile(0.05)),
                "median": float(v.median()), "p95": float(v.quantile(0.95)),
                "p99": float(v.quantile(0.99)), "max": float(v.max()),
                "abs_median": float(v.abs().median()), "abs_p99": float(v.abs().quantile(0.99))}

    ratio = (regression_term.abs() / demeaned.abs().replace(0.0, np.nan)).where(in_window, other=np.nan)
    ratio_flat = ratio.to_numpy().ravel()
    ratio_flat = ratio_flat[np.isfinite(ratio_flat)]

    xs = x.where(in_window, other=np.nan).dropna()
    varx_s = varx.where(in_window, other=np.nan).dropna()
    beta_flat = beta.where(in_window, other=np.nan).to_numpy().ravel()
    beta_flat = beta_flat[np.isfinite(beta_flat)]

    result = {
        "question": (
            "In compute_gp074_variant, the regressor x is a cross-sectional pct-rank mean and is "
            "therefore near 0.5 every day. How ill-conditioned is beta = cov/varx, and how much of the "
            "frozen yeslab_443_reg30_max20_v1 column does the regression term actually contribute?"
        ),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "read_only": True, "database_access": False, "calculator_executed": False,
        "reimplemented_from": "research_v2/scripts/repair_alpha443_window_scan.py:40-52",
        "market_source": str(CACHE),
        "windows": {"reg_window": REG_WINDOW, "max_window": MAX_WINDOW,
                    "loaded": [args.start, args.end], "reported_from": args.report_from},
        "panel_shape": list(amount.shape),
        "x_the_regressor": {
            "distribution": describe("x", xs),
            "reading": "the mean of a cross-sectional percentile rank; 0.5 by construction up to ties",
            "deviation_from_one_half": {"max_abs": float((xs - 0.5).abs().max()),
                                        "median_abs": float((xs - 0.5).abs().median())},
        },
        "varx": {"distribution": describe("varx", varx_s),
                 "share_of_days_where_varx_is_nan": float(varx.where(in_window).isna().sum()
                                                          / max(int(in_window.sum()), 1))},
        "beta": {"distribution": describe("beta", beta_flat),
                 "share_abs_above_1e6": float((np.abs(beta_flat) > 1e6).mean()),
                 "share_abs_above_1e9": float((np.abs(beta_flat) > 1e9).mean())},
        "regression_term_vs_demeaned_target": {
            "ratio_abs_regression_over_abs_demeaned": describe("ratio", ratio_flat),
            "share_of_cells_where_the_ratio_exceeds_0.01": float((ratio_flat > 0.01).mean()),
            "share_of_cells_where_the_ratio_exceeds_0.10": float((ratio_flat > 0.10).mean()),
            "share_of_cells_where_the_ratio_exceeds_1.00": float((ratio_flat > 1.00).mean()),
            "reading": (
                "if the median ratio is far below 1 the regression term is a small correction and the "
                "frozen column is effectively rolling_max(y - my); if it is near or above 1 the column is "
                "materially a product of the cov/varx division"
            ),
        },
        "the_frozen_column_itself": {
            "non_nan_cells": int(factor.where(in_window).notna().sum().sum()),
            "distribution": describe("factor", factor.where(in_window).to_numpy().ravel()[
                np.isfinite(factor.where(in_window).to_numpy().ravel())]),
            "spearman_against_the_no_regression_counterfactual": None,
        },
    }
    # the decisive comparison: the same column with the regression term removed
    from scipy import stats as st
    counter = demeaned.rolling(MAX_WINDOW, min_periods=MAX_WINDOW).max()
    a = factor.where(in_window)
    b = counter.where(in_window)
    per_day = []
    for d in a.index:
        va, vb = a.loc[d], b.loc[d]
        both = va.notna() & vb.notna()
        if int(both.sum()) >= 100:
            rho = st.spearmanr(va[both], vb[both]).statistic
            if np.isfinite(rho):
                per_day.append(float(rho))
    result["the_frozen_column_itself"]["spearman_against_the_no_regression_counterfactual"] = {
        "what": (
            "per-day Spearman between compute_gp074_variant as written and the same expression with the "
            "beta*(x-mx) term dropped, i.e. rolling_max(y - my)"
        ),
        "days": len(per_day),
        "median": float(median(per_day)) if per_day else None,
        "min": min(per_day) if per_day else None,
        "max": max(per_day) if per_day else None,
        "share_of_days_above_0.999": float(np.mean([r > 0.999 for r in per_day])) if per_day else None,
    }

    out = args.out / "alpha443_beta_scale.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    print(f"wrote {out}")
    for k in ("x_the_regressor", "varx", "beta", "regression_term_vs_demeaned_target"):
        print(f"\n== {k}"); print(json.dumps(result[k], ensure_ascii=False, indent=1)[:1400])
    print("\n== counterfactual"); print(json.dumps(
        result["the_frozen_column_itself"]["spearman_against_the_no_regression_counterfactual"],
        ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
