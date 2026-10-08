#!/usr/bin/env python3
"""Index benchmarks measured on exactly the strategy's dates and holding windows.

The user's standard is the SSE Composite (000001.SH). Two measurements are produced
so the comparison is honest rather than flattering either side:

  * buy_and_hold -- index close-to-close over calendar years and over the whole
    2022-2025 window. This is what "the index returned X" normally means.
  * cycle_matched -- for every signal date T the backtest used, the index return
    open(T+6)/open(T+1) - 1, aggregated into the SAME 192 non-overlapping cycles and
    annualised at 48.6 cycles/year. This is the only version directly comparable to
    the strategy table, because it uses identical holding windows.

Signal dates and the T+1/T+6 offsets come from the frozen TEST panels and the
project's own trading calendar, not from a re-derived calendar.

Style note: the strategy's universe is liquidity-screened and equal-weighted, so it
holds a lot of small caps, while the SSE Composite is capitalisation-weighted and
dominated by large state-owned issuers. Supplementary indices are therefore pulled
alongside it (whole market, CSI 1000, CSI 2000) so the reader can separate model
skill from style exposure.

Read-only database access; writes only --out.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ARTIFACTS = Path("/data/zhangyudong/qrv2_research_artifacts")
WORKSPACES = {"M0R": "m0r_v2", "M1": "m1_full_v1", "M2": "m2_v1", "M0-refined": "m0_v3"}
FOLDS = ("F1", "F2", "F3", "F4")
FOLD_YEAR = {"F1": 2022, "F2": 2023, "F3": 2024, "F4": 2025}
CYCLES_PER_YEAR = 243 / 5
INDICES = {
    "000001.SH": "上证综指 SSE Composite (requested standard)",
    "000985.CSI": "中证全指 CSI All Share (whole market, cap-weighted)",
    "000852.SH": "中证1000 CSI 1000 (mid/small)",
    "932000.CSI": "中证2000 CSI 2000 (micro/small)",
}
HOLDING_DAYS = 5


def signal_dates_by_fold(artifacts: Path) -> dict[str, list[date]]:
    """The distinct TEST signal dates each fold actually scored, in order."""
    out: dict[str, list[date]] = {}
    ws = artifacts / WORKSPACES["M0R"]
    for fold in FOLDS:
        for manifest in sorted((ws / "predictions").glob("*/manifest.json")):
            meta = json.loads(manifest.read_text())["metadata"]
            if meta["fold_id"] == fold and meta["kind"] == "candidate":
                scores = pd.read_parquet(manifest.parent / "scores.parquet")
                dates = pd.to_datetime(scores["signal_date"]).dt.date.unique()
                out[fold] = sorted(dates.tolist())
    return out


def load_index_prices() -> pd.DataFrame:
    import sys
    sys.path.insert(0, "/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2/scripts")
    from walk_forward_common import new_holder, query

    holder = new_holder()
    codes = "', '".join(INDICES)
    _, frame = query(
        holder,
        "select ts_code, trade_date, open, close from "
        "loadTable('dfs://YesLab','index_daily_data') "
        f"where ts_code in ['{codes}'] and trade_date >= 2021.01.01 and trade_date <= 2026.03.31",
    )
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
    for column in ("open", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def trading_calendar() -> list[date]:
    import sys
    sys.path.insert(0, "/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2/scripts")
    from walk_forward_common import load_calendar, new_holder

    return list(load_calendar(new_holder()).dates)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "index_benchmark.json",
    )
    args = parser.parse_args()

    dates_by_fold = signal_dates_by_fold(args.artifacts)
    calendar = trading_calendar()
    position = {day: index for index, day in enumerate(calendar)}

    prices = load_index_prices()
    by_index: dict[str, dict[date, tuple[float, float]]] = {}
    for code, group in prices.groupby("ts_code"):
        by_index[str(code)] = {
            row.trade_date: (row.open, row.close)
            for row in group.itertuples(index=False)
        }

    result: dict[str, object] = {
        "config": {
            "holding_days": HOLDING_DAYS,
            "cycles_per_year": CYCLES_PER_YEAR,
            "cycle_matched_definition": "index open(T+6)/open(T+1)-1, same 192 non-overlapping cycles as the strategy",
            "buy_and_hold_definition": "index close-to-close, calendar years and the full 2022-2025 window",
            "source": "dfs://YesLab index_daily_data",
        },
        "signal_dates_per_fold": {f: len(d) for f, d in dates_by_fold.items()},
        "indices": {},
    }

    for code, label in INDICES.items():
        series = by_index.get(code)
        if not series:
            continue
        entry: dict[str, object] = {"label": label}

        # --- buy and hold: calendar-year and full-window close returns
        yearly: dict[str, float] = {}
        for fold, year in FOLD_YEAR.items():
            days = sorted(d for d in series if d.year == year)
            if days:
                yearly[str(year)] = float(series[days[-1]][1] / series[days[0]][1] - 1)
        window = sorted(d for d in series if 2022 <= d.year <= 2025)
        total = float(series[window[-1]][1] / series[window[0]][1] - 1)
        closes = np.array([series[d][1] for d in window], dtype=float)
        peak = np.maximum.accumulate(closes)
        entry["buy_and_hold"] = {
            "yearly_close_return": yearly,
            "total_close_return_2022_2025": total,
            "max_drawdown_on_closes": float((closes / peak - 1.0).min()),
            "first_close": float(closes[0]),
            "last_close": float(closes[-1]),
            "first_date": str(window[0]),
            "last_date": str(window[-1]),
        }

        # --- cycle matched: identical windows to the strategy
        cycle_returns: list[float] = []
        per_year: dict[str, list[float]] = {str(y): [] for y in FOLD_YEAR.values()}
        missing = 0
        for fold, signals in dates_by_fold.items():
            period_rets: list[float] = []
            for signal in signals:
                try:
                    entry_day = calendar[position[signal] + 1]
                    exit_day = calendar[position[signal] + 6]
                except (KeyError, IndexError):
                    missing += 1
                    continue
                entry_price = series.get(entry_day, (float("nan"), float("nan")))[0]
                exit_price = series.get(exit_day, (float("nan"), float("nan")))[0]
                if not np.isfinite(entry_price) or not np.isfinite(exit_price) or entry_price <= 0:
                    missing += 1
                    continue
                ret = float(exit_price / entry_price - 1)
                period_rets.append(ret)
                per_year[str(FOLD_YEAR[fold])].append(ret)
            full = (len(period_rets) // HOLDING_DAYS) * HOLDING_DAYS
            for start in range(0, full, HOLDING_DAYS):
                cycle_returns.append(float(np.mean(period_rets[start:start + HOLDING_DAYS])))
        arr = np.array(cycle_returns, dtype=float)
        equity = np.cumprod(1.0 + arr)
        peak = np.maximum.accumulate(equity)
        std = float(arr.std(ddof=1))
        entry["cycle_matched"] = {
            "cycles": int(len(arr)),
            "signal_dates_matched": int(len(arr) * HOLDING_DAYS),
            "signal_dates_missing_index_price": missing,
            "mean_per_cycle": float(arr.mean()),
            "annualized_return": float(arr.mean()) * CYCLES_PER_YEAR,
            "annualized_vol": std * float(np.sqrt(CYCLES_PER_YEAR)),
            "sharpe": float(arr.mean() / std * np.sqrt(CYCLES_PER_YEAR)) if std > 0 else None,
            "win_rate_per_cycle": float((arr > 0).mean()),
            "max_drawdown": float((equity / peak - 1.0).min()),
            "total_return": float(equity[-1] - 1.0),
            "yearly_annualized": {
                year: float(np.mean(vals) * CYCLES_PER_YEAR)
                for year, vals in sorted(per_year.items())
                if vals
            },
        }
        result["indices"][code] = entry

    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")
    for code, entry in result["indices"].items():
        cm = entry["cycle_matched"]
        bh = entry["buy_and_hold"]
        print(
            f"  {code:<12} cycle-matched ann {cm['annualized_return']*100:>7.2f}%  "
            f"buy&hold 2022-2025 {bh['total_close_return_2022_2025']*100:>7.2f}%  "
            f"cycles {cm['cycles']}  missing {cm['signal_dates_missing_index_price']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
