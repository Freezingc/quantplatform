"""Monthly breakdown of the 2018-WATCH candidates (425, 443) across 2018 and 2019.

Consumes the per-candidate daily IC CSVs produced by research_historical_window_eval.py
(--dump-daily) and aggregates per-month signed mean IC, day count, and t-statistic
(mean/std*sqrt(n)) for both universe variants. Identifies weak months (monthly mean IC
< 0.02 or t < 0) so the decision pack can pinpoint where the bear-regime softness lives.

Read-only research artifact; no database writes, no locked-test reads.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DIAG = REPO_ROOT / "research_v2" / "artifacts" / "diagnostics"
DEFAULT_ROOT = DIAG / "watch_monthly_425_443_v1"

FACTORS = ("yeslab_425", "yeslab_443")
YEARS = ("2018", "2019")


def monthly_stats(daily: pd.DataFrame, ic_col: str) -> list[dict]:
    daily = daily.dropna(subset=[ic_col]).copy()
    daily["month"] = pd.to_datetime(daily["signal_date"]).dt.strftime("%Y-%m")
    rows = []
    for month, grp in daily.groupby("month"):
        ic = grp[ic_col].to_numpy(dtype=float)
        mean = float(ic.mean())
        std = float(ic.std())
        t = float(mean / std * np.sqrt(len(ic))) if std > 0 else None
        rows.append({
            "month": month,
            "days": int(len(ic)),
            "mean_ic": round(mean, 6),
            "t_stat": round(t, 3) if t is not None else None,
            "weak": bool(mean < 0.02 or (t is not None and t < 0)),
        })
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    output = args.output or (args.root / "watch_monthly_breakdown.json")

    payload: dict[str, object] = {"scope": "WATCH_MONTHLY_BREAKDOWN_V1", "factors": {}}
    for fid in FACTORS:
        factor_block: dict[str, object] = {}
        for year in YEARS:
            path = args.root / year / f"daily_ic_{fid}.csv"
            if not path.is_file():
                continue
            daily = pd.read_csv(path)
            year_block = {}
            for variant, col in (("full_market", "ic_full"), ("adv20_liquid", "ic_liquid")):
                stats = monthly_stats(daily, col)
                if not stats:
                    continue
                weak_months = [r["month"] for r in stats if r["weak"]]
                year_block[variant] = {
                    "months": stats,
                    "weak_months": weak_months,
                    "weak_month_count": len(weak_months),
                }
            factor_block[year] = year_block
        payload["factors"][fid] = factor_block

    for fid, block in payload["factors"].items():
        print(f"=== {fid} ===")
        for year in YEARS:
            if year not in block:
                continue
            for variant in ("full_market", "adv20_liquid"):
                if variant not in block[year]:
                    continue
                vb = block[year][variant]
                line = " ".join(
                    f"{m['month'][-2:]}:{m['mean_ic']:+.3f}(t={m['t_stat']})" for m in vb["months"]
                )
                print(f"  {year} {variant:13s} weak={vb['weak_month_count']}/12 | {line}")
    payload["database_writes"] = False
    payload["locked_test_read"] = False
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
