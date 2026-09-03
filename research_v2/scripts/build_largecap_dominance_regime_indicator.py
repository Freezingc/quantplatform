"""Build and validate the large-cap-dominance regime indicator (2016-2025).

Regime trigger proposed for the 425/443/437/431/396/442 regime monitors and
for regime-aware model weighting (brief 4s/4u): the 60-trading-day return
spread between CSI300 and CSI500 (large caps minus mid caps). This script
builds the daily series from cached Tushare index data, evaluates candidate
thresholds against the documented pre-training weakness periods, and emits
the indicator artifact (daily CSV + summary JSON). Read-only research.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE = REPO_ROOT / "research_v2" / "artifacts" / "tushare_cache"

# Documented weak periods to validate against (month granularity)
WEAK_MONTHS = {
    # 437/431 2017 collapse: weak months from brief 4u monthly decomposition
    "2017": ["2017-01", "2017-04", "2017-05", "2017-07", "2017-10", "2017-11"],
    # 425 2018 weak months (brief 4o) + 443 2018 weak months
    "2018": ["2018-01", "2018-02", "2018-03", "2018-05", "2018-06", "2018-09", "2018-10"],
}
NORMAL_YEARS = ["2019", "2020", "2021", "2022", "2023", "2024", "2025"]


def load_index(name: str, files: list[str]) -> pd.Series:
    frames = []
    for fn in files:
        df = pd.read_csv(CACHE / fn)
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        frames.append(df.set_index("trade_date")["close"])
    s = pd.concat(frames).sort_index()
    return s[~s.index.duplicated(keep="last")]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    csi300 = load_index("csi300", ["index_daily_csi300_20160101_20191231.csv", "index_daily_000300_SH_20190701_20251231.csv"])
    csi500 = load_index("csi500", ["index_daily_csi500_20160101_20191231.csv", "index_daily_csi500_20200101_20251231.csv"])
    px = pd.DataFrame({"csi300": csi300, "csi500": csi500}).dropna()
    spread = (px["csi300"].pct_change(60) - px["csi500"].pct_change(60)).rename("spread_60d")
    spread = spread.dropna()
    print(f"series: {spread.index.min().date()} -> {spread.index.max().date()}, {len(spread)} days")

    spread_df = spread.to_frame()
    spread_df["month"] = spread_df.index.to_period("M").astype(str)
    monthly_share = {}
    for (month, grp) in spread_df.groupby("month"):
        monthly_share[month] = float((grp["spread_60d"] >= 0.10).mean())

    report = {}
    for thr in (0.05, 0.08, 0.10, 0.12):
        on = spread >= thr
        yearly = {}
        for year in sorted(set(spread.index.year)):
            sub = on[on.index.year == year]
            yearly[str(year)] = round(float(sub.mean()), 4)
        weak_share = []
        for year, months in WEAK_MONTHS.items():
            for m in months:
                if m in monthly_share:
                    pass
        # weak-month hit rate: share of documented weak months where >=40% of days were ON at threshold
        month_on = spread_df.assign(on=on.values).groupby("month")["on"].mean()
        hits, total = 0, 0
        for year, months in WEAK_MONTHS.items():
            for m in months:
                if m in month_on.index:
                    total += 1
                    if month_on[m] >= 0.40:
                        hits += 1
        normal_shares = [yearly[y] for y in NORMAL_YEARS if y in yearly]
        report[f"threshold_{thr}"] = {
            "threshold": thr,
            "weak_month_hit_rate_40pct": f"{hits}/{total}",
            "yearly_on_share": yearly,
            "normal_years_mean_on_share": round(float(np.mean(normal_shares)), 4) if normal_shares else None,
        }
        print(f"thr={thr}: weak-month hits {hits}/{total}; normal-year mean ON {report[f'threshold_{thr}']['normal_years_mean_on_share']}")
        print("   yearly:", yearly)

    threshold = 0.10
    on = spread >= threshold
    month_on = spread_df.assign(on=on.values).groupby("month")["on"].mean()
    weak_month_detail = {}
    for year, months in WEAK_MONTHS.items():
        for m in months:
            if m in month_on.index:
                weak_month_detail[m] = round(float(month_on[m]), 3)
    daily = pd.DataFrame({"trade_date": spread.index.strftime("%Y-%m-%d"), "spread_60d": spread.values, "regime_on": on.values.astype(int)})
    daily.to_csv(args.output / "largecap_dominance_regime_daily.csv", index=False)
    payload = {
        "scope": "LARGE_CAP_DOMINANCE_REGIME_INDICATOR_V1",
        "definition": "spread_60d = CSI300 60-trading-day return - CSI500 60-trading-day return; regime ON when spread_60d >= 0.10",
        "data_source": "Tushare index_daily 000300.SH / 000905.SH (cached)",
        "window": f"{spread.index.min().date()}/{spread.index.max().date()}",
        "chosen_threshold": threshold,
        "threshold_screen": report,
        "validation_note": "weak months from brief 4u (437/431 2017 collapse monthly decomposition) and 4o (425/443 2018 monthly decomposition)",
        "weak_month_on_share_at_threshold": weak_month_detail,
        "interpretation": "threshold 0.10 concentrates ON-days in the two collapse regimes (2017: 21%, 2018: 10%) plus early-2021 core-asset dynamics (10.7%), with near-zero activation in 2022/2023/2025 - high specificity as a regime ALERT. Weak-month day-coverage is partial (factor weakness extends beyond peak-spread days, consistent with the documented weak daily IC-spread correlation), so the indicator is a monitor/downweight trigger, not a complete weak-day filter.",
        "usage": "monitor trigger for 425/443 (admission condition, brief 4s) and 437/431/396/442 (formal-set regime note, brief 4u); regime-aware weighting input for model stage",
        "database_writes": False,
        "locked_test_read": False,
    }
    (args.output / "largecap_dominance_regime_indicator_v1.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output}/largecap_dominance_regime_indicator_v1.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
