#!/usr/bin/env python3
"""Sweep the PARTIAL exposure floor under daily accounting.

`exposure_daily_reaccount.py` showed that a full 0/1 trend filter reliably halves the
drawdown but costs 7-10pp of annual return and leaves Sharpe roughly unchanged. That makes
it a risk-appetite dial rather than an optimisation. A partial floor is the obvious middle
setting: stay half invested when the index is below its MA instead of flat. It should keep
more of the return, halve the size of every switch, and still cut the drawdown -- but by how
much is a measurement.

Daily accounting and the correct one-leg switch cost throughout, via `research_v2.portfolio`
(the same code the daily CLI runs). Read-only; writes only --out.
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

INDICES = {"932000.CSI": "中证2000", "000985.CSI": "中证全指"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path,
                        default=REPO / "research_v2" / "configs" / "portfolio" / "default.yaml")
    parser.add_argument("--n-exit", type=int, default=150)
    parser.add_argument("--index-cache", type=Path, default=Path("/tmp/qrv2_index_closes.parquet"))
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent / "exposure_floor_results.json")
    args = parser.parse_args()

    base = load_portfolio_policy(args.policy)
    src = FrozenArtifactSource(artifacts=DEFAULT_ARTIFACTS, with_labels=True)
    src.load(base.ensemble_members)
    members = list(base.ensemble_members)
    dates = src.signal_dates(members)
    sbm = {m: src.scores(m) for m in members}
    common: dict[str, Any] = dict(folds_by_date=src.folds_by_date(), returns=src.returns(),
                                  universe=src.universe(), dates=dates)
    feeds = {c: load_index_closes(c, start=dates[0] - timedelta(days=400), end=dates[-1],
                                 cache=args.index_cache) for c in INDICES}

    results: dict[str, Any] = {"config": {
        "members": members, "n_exit": args.n_exit, "n_enter": base.n_enter,
        "cost_bp": base.round_trip_bp, "switch_cost_factor": base.switch_cost_factor,
        "accounting": "daily, one-leg switch cost", "signal_dates": len(dates),
    }, "results": {}}

    def run(code: str | None, ma: int, floor: float) -> dict[str, Any]:
        overrides: dict[str, Any] = {"n_exit": args.n_exit, "trend_ma_window": ma}
        if code is not None:
            overrides.update(trend_index_code=code, trend_exposure_floor=floor)
        pol = PortfolioPolicy(**{**base.__dict__, **overrides})
        dec = replay(sbm, policy=pol, index_closes=(feeds[code] if code else ()), **common)
        s = summarise(to_cycles(dec, pol, exposure="none" if code is None else "daily"))
        mean_exposure = sum(d.exposure for d in dec) / len(dec)
        flips = sum(abs(b.exposure - a.exposure) for a, b in zip(dec, dec[1:]))
        return {"index": code, "ma": ma, "floor": floor, "cycles": s.cycles,
                "net_annualized": s.net_annualized, "net_sharpe": s.net_sharpe,
                "max_drawdown": s.max_drawdown, "total_return": s.total_return,
                "mean_turnover": s.mean_turnover, "mean_exposure": mean_exposure,
                "switch_units_per_year": flips / len(dec) * 243}

    print(f"=== n_enter={base.n_enter} n_exit={args.n_exit}, daily accounting ===", flush=True)
    r = run(None, base.trend_ma_window, 0.0)
    results["results"]["none"] = r
    print(f"  {'无过滤':<22} 年化 {r['net_annualized']*100:>6.2f}%  Sharpe {r['net_sharpe']:.3f}  "
          f"回撤 {r['max_drawdown']*100:>6.2f}%  累计 {r['total_return']*100:>6.2f}%", flush=True)
    base_sharpe, base_ret, base_dd = r["net_sharpe"], r["net_annualized"], r["max_drawdown"]

    for code, name in INDICES.items():
        for ma in (10, 20, 30):
            for floor in (0.0, 0.25, 0.5, 0.75):
                r = run(code, ma, floor)
                results["results"][f"{code}_ma{ma}_floor{floor}"] = r
                print(f"  {name:<9} MA{ma:<3} floor{floor:<5} 年化 {r['net_annualized']*100:>6.2f}%  "
                      f"Sharpe {r['net_sharpe']:.3f}  回撤 {r['max_drawdown']*100:>6.2f}%  "
                      f"累计 {r['total_return']*100:>6.2f}%  平均仓位 {r['mean_exposure']*100:>5.1f}%  "
                      f"| ΔSharpe {r['net_sharpe']-base_sharpe:>+.3f}  "
                      f"Δ年化 {(r['net_annualized']-base_ret)*100:>+6.2f}pp  "
                      f"Δ回撤 {(r['max_drawdown']-base_dd)*100:>+6.2f}pp", flush=True)

    best = max((k for k in results["results"] if k != "none"),
               key=lambda k: results["results"][k]["net_sharpe"])
    results["best_by_sharpe"] = best
    results["reference_unfiltered"] = {"net_annualized": base_ret, "net_sharpe": base_sharpe,
                                       "max_drawdown": base_dd}
    print(f"\nbest by Sharpe: {best} -> {results['results'][best]['net_sharpe']:.3f} "
          f"(unfiltered {base_sharpe:.3f})")
    args.out.write_text(json.dumps(results, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
