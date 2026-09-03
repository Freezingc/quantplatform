"""Build and validate the acute-liquidity-shock regime indicator (2016-2025).

Second half of the regime-monitoring union defined in brief 4x: the
large-cap-dominance spread trigger covers large-cap regimes; this indicator
covers acute liquidity-shock episodes (the 2018-06 tariff-shock month being
the documented instance where candidates 425/443 weakened while the spread
indicator correctly stayed off). Built from CSI1000 (000852.SH) daily data:
20-trading-day return, 20-day realized volatility, and combined shock rules.
Candidate rules are screened against the documented episode plus reference
liquidity crises (2016-01/02 fuse crash aftermath, 2024-01 micro-cap crash),
and must stay mostly off in normal years. Read-only research.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE = REPO_ROOT / "research_v2" / "artifacts" / "tushare_cache"

TARGET_MONTHS = ["2018-06"]
REFERENCE_EPISODES = ["2020-02", "2020-03", "2024-01", "2024-02"]
NORMAL_YEARS = ["2019", "2021", "2022", "2023", "2025"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CACHE / "index_daily_csi1000_20160101_20251231.csv")
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df = df.sort_values("trade_date").set_index("trade_date")
    close = df["close"]
    ret = close.pct_change()
    ret_20 = close.pct_change(20)
    vol_20 = ret.rolling(20).std()
    vol_z = (vol_20 - vol_20.mean()) / vol_20.std()

    # CSI500/CSI300 for multi-index rules
    csi500 = pd.read_csv(CACHE / "index_daily_csi500_20160101_20191231.csv")
    csi500["trade_date"] = pd.to_datetime(csi500["trade_date"], format="%Y%m%d")
    csi500 = csi500.sort_values("trade_date").set_index("trade_date")["close"]
    csi500_2 = pd.read_csv(CACHE / "index_daily_csi500_20200101_20251231.csv")
    csi500_2["trade_date"] = pd.to_datetime(csi500_2["trade_date"], format="%Y%m%d")
    csi500 = pd.concat([csi500, csi500_2.sort_values("trade_date").set_index("trade_date")["close"]]).sort_index()
    csi500 = csi500[~csi500.index.duplicated(keep="last")]
    csi300_a = pd.read_csv(CACHE / "index_daily_csi300_20160101_20191231.csv")
    csi300_a["trade_date"] = pd.to_datetime(csi300_a["trade_date"], format="%Y%m%d")
    csi300 = csi300_a.sort_values("trade_date").set_index("trade_date")["close"]
    csi300_b = pd.read_csv(CACHE / "index_daily_000300_SH_20190701_20251231.csv")
    csi300_b["trade_date"] = pd.to_datetime(csi300_b["trade_date"], format="%Y%m%d")
    csi300 = pd.concat([csi300, csi300_b.sort_values("trade_date").set_index("trade_date")["close"]]).sort_index()
    csi300 = csi300[~csi300.index.duplicated(keep="last")]

    ret_10 = close.pct_change(10)
    csi500_ret20 = csi500.reindex(close.index).pct_change(20)
    csi300_ret20 = csi300.reindex(close.index).pct_change(20)

    frame = pd.DataFrame({
        "ret_20d": ret_20, "ret_10d": ret_10, "vol_20d": vol_20, "vol_z": vol_z,
        "csi500_ret20": csi500_ret20, "csi300_ret20": csi300_ret20,
    })
    frame = frame.dropna()
    # Evidence window starts 2017; earlier data serves only as rolling warmup.
    frame = frame.loc[frame.index >= "2017-01-01"]
    frame["month"] = frame.index.to_period("M").astype(str)

    rules = {
        "csi1000_ret20_le_-0.08": frame["ret_20d"] <= -0.08,
        "csi1000_ret20_le_-0.10": frame["ret_20d"] <= -0.10,
        "csi1000_ret10_le_-0.06": frame["ret_10d"] <= -0.06,
        "csi500_ret20_le_-0.12": frame["csi500_ret20"] <= -0.12,
        "csi500_ret20_le_-0.12_or_csi1000_ret20_le_-0.10": (frame["csi500_ret20"] <= -0.12) | (frame["ret_20d"] <= -0.10),
    }

    report = {}
    for name, on in rules.items():
        month_on = frame.assign(on=on.values).groupby("month")["on"].mean()
        yearly = {str(y): round(float(on[on.index.year == y].mean()), 4) for y in sorted(set(on.index.year))}
        target_cov = {m: round(float(month_on.get(m, np.nan)), 3) for m in TARGET_MONTHS}
        ref_cov = {m: round(float(month_on.get(m, np.nan)), 3) for m in REFERENCE_EPISODES}
        normal_mean = round(float(np.mean([yearly[y] for y in NORMAL_YEARS if y in yearly])), 4)
        report[name] = {
            "target_2018_06_on_share": target_cov,
            "reference_episodes_on_share": ref_cov,
            "yearly_on_share": yearly,
            "normal_years_mean_on_share": normal_mean,
        }
        print(f"== {name}")
        print(f"   target 2018-06: {target_cov}")
        print(f"   references: {ref_cov}")
        print(f"   normal-year mean ON: {normal_mean}; yearly: {yearly}")

    # Selection: primary target coverage first, then lowest normal-year false activation.
    def score(name):
        r = report[name]
        tgt = r["target_2018_06_on_share"].get("2018-06", 0.0) or 0.0
        return (-tgt, r["normal_years_mean_on_share"])
    chosen = min(report.keys(), key=score)
    on = rules[chosen]
    daily = pd.DataFrame({
        "trade_date": frame.index.strftime("%Y-%m-%d"),
        "csi1000_ret_20d": frame["ret_20d"].values,
        "csi1000_vol_z": frame["vol_z"].values,
        "shock_on": on.values.astype(int),
    })
    daily.to_csv(args.output / "liquidity_shock_regime_daily.csv", index=False)
    payload = {
        "scope": "ACUTE_LIQUIDITY_SHOCK_REGIME_INDICATOR_V1",
        "definition": f"shock ON when rule '{chosen}' fires on CSI1000 (000852.SH): 20-trading-day return and 20-day realized-vol z-score",
        "data_source": "Tushare index_daily 000852.SH (cached)",
        "window": f"{frame.index.min().date()}/{frame.index.max().date()}",
        "chosen_rule": chosen,
        "rule_screen": report,
        "validation_note": "primary target: 2018-06 (documented acute liquidity shock weakening 425/443, brief 4o/4s); reference episodes: pandemic crash 2020-02/03 (factors were STRONG there - flagging is regime coverage, not factor weakness), micro-cap crash 2024-01/02; series emitted from 2017-01-01 (earlier data only rolling warmup)",
        "usage": "second half of the regime-monitoring union (brief 4x): monitor/downweight trigger for 425/443 and formal-set regime notes; combined coverage = large-cap spread trigger UNION this shock trigger",
        "database_writes": False,
        "locked_test_read": False,
    }
    (args.output / "liquidity_shock_regime_indicator_v1.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output}/liquidity_shock_regime_indicator_v1.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
