#!/usr/bin/env python3
"""Re-price the trend-filter overlay the way a real book would trade it.

The published overlay numbers in `2026-10-08-optimization-levers.md` averaged the five
daily exposures inside each holding cycle into ONE scalar and charged the cash switch once
per cycle. That convention has two consequences, and `verify_portfolio_policy_replay.py`
isolated both:

  1. it hides intra-cycle flips. The cycle-level view saw ~19.7 switches/year for MA10;
     at daily granularity the same rule flips ~40.6/year. Each flip is a real liquidation
     and a real re-entry, so the published run charged roughly half the true switch cost.
  2. mean(return) x mean(exposure) is not mean(return x exposure). A staggered book holds
     five overlapping tranches, and each tranche gets its own exposure decided at its own
     entry, so the daily product is the correct accounting.

This script re-measures the whole overlay sweep under daily accounting with the correct
one-leg switch cost, so the recommendation rests on the tradeable number rather than the
convenient one. It reuses `research_v2.portfolio`, i.e. the exact code the daily CLI runs.

Read-only over the signed artifacts plus index queries; writes only --out.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research_v2" / "src"))

from research_v2.portfolio.engine import replay, summarise, to_cycles  # noqa: E402
from research_v2.portfolio.policy import PortfolioPolicy, load_portfolio_policy  # noqa: E402
from research_v2.portfolio.sources import (  # noqa: E402
    DEFAULT_ARTIFACTS,
    FrozenArtifactSource,
    load_index_closes,
)

INDICES = {"932000.CSI": "中证2000", "000985.CSI": "中证全指", "000001.SH": "上证综指"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path,
                        default=REPO / "research_v2" / "configs" / "portfolio" / "default.yaml")
    parser.add_argument("--index-cache", type=Path, default=Path("/tmp/qrv2_index_closes.parquet"))
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent / "exposure_daily_results.json")
    args = parser.parse_args()

    base = load_portfolio_policy(args.policy)
    src = FrozenArtifactSource(artifacts=DEFAULT_ARTIFACTS, with_labels=True)
    src.load(base.ensemble_members)
    members = list(base.ensemble_members)
    dates = src.signal_dates(members)
    sbm = {m: src.scores(m) for m in members}
    common: dict[str, Any] = dict(
        folds_by_date=src.folds_by_date(), returns=src.returns(),
        universe=src.universe(), dates=dates,
    )

    feeds: dict[str, list[tuple[Any, float]]] = {}
    for code in INDICES:
        feeds[code] = load_index_closes(code, start=dates[0] - timedelta(days=400),
                                        end=dates[-1], cache=args.index_cache)
        print(f"index {code}: {len(feeds[code])} closes from {feeds[code][0][0]}", flush=True)

    out: dict[str, Any] = {"config": {
        "members": members, "signal_dates": len(dates), "cost_bp": base.round_trip_bp,
        "switch_cost_factor": base.switch_cost_factor,
        "accounting": "daily: each tranche scaled by its own entry-day exposure",
        "note": ("cycle-level averaging hides intra-cycle flips; MA10 flips ~19.7/yr at cycle "
                 "level but ~40.6/yr daily, so the published overlay understated switch cost"),
    }, "results": {}}

    def run(n_exit: int, ma: int, code: str | None, switch_factor: float | None) -> dict[str, Any]:
        overrides: dict[str, Any] = {"n_exit": n_exit, "trend_ma_window": ma}
        if code is not None:
            overrides["trend_index_code"] = code
        if switch_factor is not None:
            overrides["switch_cost_factor"] = switch_factor
        pol = PortfolioPolicy(**{**base.__dict__, **overrides})
        dec = replay(sbm, policy=pol, index_closes=(feeds[code] if code else ()), **common)
        cycles = to_cycles(dec, pol, exposure="none" if code is None else "daily")
        s = summarise(cycles)
        flips = sum(1 for a, b in zip(dec, dec[1:]) if abs(a.exposure - b.exposure) > 1e-9)
        flat = sum(1 for d in dec if d.exposure <= 0)
        gross_flat = sum(d.gross_return or 0.0 for d in dec if d.exposure <= 0)
        gross_all = sum(d.gross_return or 0.0 for d in dec)
        return {
            "n_exit": n_exit, "ma": ma, "index": code,
            "switch_cost_factor": pol.switch_cost_factor,
            "cycles": s.cycles, "net_annualized": s.net_annualized,
            "gross_annualized": s.gross_annualized, "net_sharpe": s.net_sharpe,
            "net_vol": s.net_vol, "max_drawdown": s.max_drawdown,
            "total_return": s.total_return, "mean_turnover": s.mean_turnover,
            "flips": flips, "flips_per_year": flips / len(dec) * 243,
            "flat_days": flat, "flat_share": flat / len(dec),
            "gross_share_sat_out": (gross_flat / gross_all) if gross_all else None,
            "dd_peak": str(s.drawdown_peak), "dd_trough": str(s.drawdown_trough),
        }

    print("\n=== unfiltered reference (no overlay) ===", flush=True)
    for n_exit in (60, 150):
        r = run(n_exit, base.trend_ma_window, None, None)
        out["results"][f"none_exit{n_exit}"] = r
        print(f"  exit{n_exit:<4} 年化 {r['net_annualized']*100:>6.2f}%  Sharpe {r['net_sharpe']:.3f}  "
              f"回撤 {r['max_drawdown']*100:>6.2f}%  累计 {r['total_return']*100:>6.2f}%", flush=True)

    print("\n=== daily accounting, correct one-leg switch cost ===", flush=True)
    for code in ("932000.CSI", "000985.CSI"):
        for ma in (5, 10, 20, 30, 60, 120):
            for n_exit in (60, 150):
                r = run(n_exit, ma, code, None)
                out["results"][f"{code}_ma{ma}_exit{n_exit}"] = r
                print(f"  {INDICES[code]:<9} MA{ma:<4} exit{n_exit:<4} 年化 {r['net_annualized']*100:>6.2f}%  "
                      f"Sharpe {r['net_sharpe']:.3f}  回撤 {r['max_drawdown']*100:>6.2f}%  "
                      f"累计 {r['total_return']*100:>6.2f}%  切换 {r['flips_per_year']:>5.1f}/年  "
                      f"空仓 {r['flat_share']*100:>4.1f}%  踏空毛利 {r['gross_share_sat_out']*100:>4.1f}%",
                      flush=True)

    print("\n=== same rules, research switch convention (1.0) for comparison ===", flush=True)
    for ma in (10, 20, 60):
        r = run(150, ma, "932000.CSI", 1.0)
        out["results"][f"researchcost_932000.CSI_ma{ma}_exit150"] = r
        print(f"  MA{ma:<4} exit150 switch1.0  年化 {r['net_annualized']*100:>6.2f}%  "
              f"Sharpe {r['net_sharpe']:.3f}  回撤 {r['max_drawdown']*100:>6.2f}%", flush=True)

    args.out.write_text(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
