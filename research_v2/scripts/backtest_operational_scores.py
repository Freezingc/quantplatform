"""Backtest the operational scores over a window the published pipeline does not cover.

WHAT THIS IS FOR

The owner asked to see how the strategy behaves on the newest dates before using it, so this
converts scores into realised five-trading-day returns and runs them through the platform's own
portfolio engine (``replay``/``to_cycles``/``summarise``) with the same policy object the published
backtest used. Nothing here re-implements a portfolio rule.

READ THE RESULT AS A LOCKED-SEGMENT READ

Signal dates from 2026-01-01 belong to the locked FINAL test. This script measures them, which the
locked-test runbook does not authorise, so the output is an operational check and must not be used
to select, tune, retrain, or otherwise choose anything: once a performance number for the locked
segment has informed a decision, that segment is no longer an out-of-sample test. The sidecar
records this explicitly.

The forward return is the platform's own convention, T+1 open to T+6 open on adjusted opens, and
``--verify-labels`` recomputes it for an overlapping window that still has published labels so the
two can be compared before any of it is trusted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "research_v2" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "research_v2" / "src"))

from materialize_operational_factors import (  # noqa: E402
    as_plain_date,
    query_open_dates,
)

from research_v2.pit.calendar import (  # noqa: E402
    TradingCalendar,
    TradingCalendarBoundaryError,
)
from research_v2.portfolio.engine import replay, summarise, to_cycles  # noqa: E402
from research_v2.portfolio.policy import PortfolioPolicy  # noqa: E402

#: The engine's cycle length, mirroring the frozen forward-open label horizon.
HOLDING_DAYS = 5
_PRICE_COLUMNS = ["trade_date", "ts_code", "adjusted_open"]


def with_adjusted_open(market: pd.DataFrame) -> pd.DataFrame:
    """Add the platform's ``adjusted_open`` (raw open scaled by the adjustment factor)."""
    required = {"open", "adj_factor"}
    missing = sorted(required.difference(market.columns))
    if missing:
        raise ValueError(f"market frame is missing columns: {missing}")
    frame = market.copy(deep=True)
    frame["adjusted_open"] = frame["open"].astype(float) * frame["adj_factor"].astype(float)
    return frame


def forward_returns(
    market: pd.DataFrame,
    calendar: TradingCalendar,
    signal_dates: Iterable[date],
) -> dict[date, dict[str, float]]:
    """Realised ``T+1 open -> T+6 open`` return per signal date, from adjusted opens.

    Names missing either endpoint are dropped rather than imputed: an unmeasurable trade is not a
    zero return.
    """
    missing = sorted(set(_PRICE_COLUMNS).difference(market.columns))
    if missing:
        raise ValueError(f"market frame is missing columns: {missing}")
    prices = {
        (as_plain_date(row.trade_date), str(row.ts_code)): float(row.adjusted_open)
        for row in market.itertuples(index=False)
        if pd.notna(row.adjusted_open)
    }
    codes = sorted({code for _, code in prices})
    out: dict[date, dict[str, float]] = {}
    for signal_date in signal_dates:
        day = as_plain_date(signal_date)
        try:
            entry_date = calendar.shift(day, 1)
            exit_date = calendar.shift(day, 6)
        except TradingCalendarBoundaryError:
            out[day] = {}
            continue
        realised: dict[str, float] = {}
        for code in codes:
            entry_price = prices.get((entry_date, code))
            exit_price = prices.get((exit_date, code))
            if entry_price is None or exit_price is None or entry_price <= 0 or exit_price <= 0:
                continue
            realised[code] = exit_price / entry_price - 1.0
        out[day] = realised
    return out


def measurable_signal_dates(
    calendar: TradingCalendar, signal_dates: Sequence[date], *, priced_through: date
) -> tuple[date, ...]:
    """Signals whose six-day-forward exit has already priced, in date order."""
    measurable: list[date] = []
    for signal_date in sorted(as_plain_date(value) for value in signal_dates):
        try:
            exit_date = calendar.shift(signal_date, 6)
        except TradingCalendarBoundaryError:
            continue
        if exit_date <= priced_through:
            measurable.append(signal_date)
    return tuple(measurable)


def parse_policies(text: str) -> tuple[PortfolioPolicy, ...]:
    """``"10:50,30:150"`` -> policies; the buffer width is validated by the policy itself."""
    policies: list[PortfolioPolicy] = []
    for chunk in str(text).split(","):
        piece = chunk.strip()
        if not piece:
            continue
        enter, separator, exit_ = piece.partition(":")
        if not separator:
            raise ValueError(f"policy spec must be enter:exit, got {piece!r}")
        policies.append(PortfolioPolicy(n_enter=int(enter), n_exit=int(exit_)))
    if not policies:
        raise ValueError("at least one policy spec is required")
    return tuple(policies)


def _scores_by_member(
    frame: pd.DataFrame, members: Mapping[str, str]
) -> dict[str, dict[date, dict[str, float]]]:
    out: dict[str, dict[date, dict[str, float]]] = {}
    for member, column in members.items():
        if column not in frame.columns:
            raise ValueError(f"scores are missing the {column} column")
        per_date: dict[date, dict[str, float]] = {}
        for day, group in frame.groupby("signal_date", sort=True):
            per_date[as_plain_date(day)] = dict(
                zip(
                    group["ts_code"].astype(str),
                    group[column].astype(float),
                    strict=True,
                )
            )
        out[member] = per_date
    return out


def _policy_summary(cycles, benchmark_cycles) -> dict[str, object]:
    summary = summarise(cycles)
    benchmark = summarise(benchmark_cycles)
    return {
        "cycles": summary.cycles,
        "total_return": summary.total_return,
        "net_annualized": summary.net_annualized,
        "gross_annualized": summary.gross_annualized,
        "net_sharpe": summary.net_sharpe,
        "net_vol": summary.net_vol,
        "max_drawdown": summary.max_drawdown,
        "mean_turnover": summary.mean_turnover,
        "mean_names": summary.mean_names,
        "benchmark_total_return": benchmark.total_return,
        "benchmark_max_drawdown": benchmark.max_drawdown,
        "excess_total_return": summary.total_return - benchmark.total_return,
        "cycle_returns": [cycle.net_return for cycle in cycles],
        "cycle_win_rate": (
            sum(1 for cycle in cycles if cycle.net_return > 0) / len(cycles)
            if cycles
            else float("nan")
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--universe", type=Path, required=True, help="rankable membership frame")
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True, help="JSON result path")
    parser.add_argument("--policies", type=str, default="10:50,20:100,30:150,50:250,100:400")
    parser.add_argument("--index-code", type=str, default="932000.CSI")
    parser.add_argument("--verify-start", type=date.fromisoformat, default=date(2026, 7, 1))
    parser.add_argument("--verify-end", type=date.fromisoformat, default=date(2026, 7, 31))
    args = parser.parse_args(argv)

    import audit_formula_replay_tushare as tushare_replay
    import materialize_frozen_factor_daily as frozen_mod
    import materialize_operational_factors as operational

    from research_v2.portfolio.sources import load_index_closes

    scores = pd.read_parquet(args.scores)
    scores["signal_date"] = [as_plain_date(value) for value in scores["signal_date"]]
    scores["ts_code"] = scores["ts_code"].astype(str)
    scores = scores.loc[
        scores["signal_date"].map(lambda day: args.start <= day <= args.end)
    ]
    if scores.empty:
        raise SystemExit(f"no scores inside {args.start} .. {args.end}")
    universe_frame = pd.read_parquet(args.universe)
    universe_map = operational.universe_map_from_frame(universe_frame, args.start, args.end)

    client = tushare_replay._connect()
    try:
        calendar = _load_calendar(client, max(args.end, args.verify_end) + timedelta(days=20))
    finally:
        client.close()
    priced_through = max(
        day for day in calendar.dates if day <= max(args.end, args.verify_end)
    )
    measurable = measurable_signal_dates(
        calendar, tuple(sorted(set(scores["signal_date"]))), priced_through=priced_through
    )
    if not measurable:
        raise SystemExit("no signal date has a priced forward return yet")

    first = min(min(measurable), args.verify_start) - timedelta(days=10)
    last = max(calendar.shift(measurable[-1], 6), args.verify_end)
    client = tushare_replay._connect()
    try:
        open_dates = query_open_dates(client, first, last)
    finally:
        client.close()
    market = tushare_replay.fetch_full_tushare_shsz_market(
        open_dates,
        token=os.getenv("QRV2_TUSHARE_TOKEN", ""),
        cache_dir=frozen_mod.CACHE,
        include_daily_basic=False,
    )
    market = with_adjusted_open(
        market.assign(
            trade_date=pd.to_datetime(market["trade_date"]),
            ts_code=market["ts_code"].astype(str),
        )
    )
    returns = forward_returns(market, calendar, measurable)
    print(f"measurable signal dates: {len(measurable)} ({measurable[0]} .. {measurable[-1]})")

    verification = _verify_labels(market, calendar, args)
    print(f"label verification: {verification}")

    scores_by_member = _scores_by_member(
        scores, {"M0R": "m0r-final-v1_percentile", "M1": "m1-final-v1_percentile"}
    )
    # The backtest convention: only names whose forward return is known can be ranked, because
    # everything else is unmeasurable. The published backtest restricts the same way.
    rankable = {
        day: frozenset(
            code for code in universe_map.get(day, set()) if code in returns.get(day, {})
        )
        for day in measurable
    }
    index_closes = load_index_closes(
        args.index_code,
        start=measurable[0] - timedelta(days=30),
        end=measurable[-1],
        cache=None,
    )

    results: dict[str, object] = {}
    benchmark_cycles = None
    for policy in parse_policies(args.policies):
        decisions = replay(
            scores_by_member,
            policy=policy,
            index_closes=index_closes,
            returns=returns,
            dates=measurable,
            universe=rankable,
            reset_book_at_fold_boundary=False,
        )
        cycles = to_cycles(decisions, policy)
        if benchmark_cycles is None:
            benchmark_policy = PortfolioPolicy(
                n_enter=1_000_000,
                n_exit=1_000_000,
                ensemble_members=("ALL",),
                round_trip_bp=0.0,
            )
            benchmark_decisions = replay(
                {"ALL": {day: {code: 0.0 for code in rankable[day]} for day in measurable}},
                policy=benchmark_policy,
                index_closes=(),
                returns=returns,
                dates=measurable,
                universe=rankable,
                reset_book_at_fold_boundary=False,
            )
            benchmark_cycles = to_cycles(benchmark_decisions, benchmark_policy)
        key = f"enter{policy.n_enter}_exit{policy.n_exit}"
        results[key] = _policy_summary(cycles, benchmark_cycles)

    payload = {
        "scope": "OPERATIONAL_BACKTEST_V1",
        "warning": (
            "reads the locked 2026 segment's realised outcomes; an operational check only, and "
            "must not be used to select, tune or retrain anything"
        ),
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "measurable_signal_dates": [day.isoformat() for day in measurable],
        "holding_days": HOLDING_DAYS,
        "forward_return": "adjusted_open(T+6) / adjusted_open(T+1) - 1",
        "index_code": args.index_code,
        "label_verification": verification,
        "policies": results,
        "scores": str(args.scores),
        "scores_sha256": hashlib.sha256(args.scores.read_bytes()).hexdigest(),
        "universe": str(args.universe),
        "universe_sha256": hashlib.sha256(args.universe.read_bytes()).hexdigest(),
        "reporting_convention": (
            "cycle returns are the arithmetic mean of their five daily returns, compounded across "
            "cycles; with a two-month window the annualised figures are not meaningful"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for key, result in results.items():
        print(
            f"{key:>18}: total {result['total_return']:+.4%} "
            f"| bench {result['benchmark_total_return']:+.4%} "
            f"| excess {result['excess_total_return']:+.4%} "
            f"| maxDD {result['max_drawdown']:+.2%} "
            f"| turnover {result['mean_turnover']:.3f} | names {result['mean_names']:.0f} "
            f"| win {result['cycle_win_rate']:.2f}",
            flush=True,
        )
    print(f"wrote {args.output}", flush=True)
    return 0


def _load_calendar(client: object, end: date) -> TradingCalendar:
    frame = client.run(
        "select cal_date from loadTable('dfs://YesLab','trade_calendar') "
        "where exchange='SSE' and is_open=1 "
        f"and cal_date between 1990.12.19 and {end:%Y.%m.%d} order by cal_date"
    )
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise RuntimeError("trade calendar query returned no rows")
    return TradingCalendar.from_dates(
        [as_plain_date(value) for value in pd.to_datetime(frame["cal_date"]).dropna()]
    )


def _verify_labels(
    market: pd.DataFrame, calendar: TradingCalendar, args: argparse.Namespace
) -> dict[str, float | int]:
    """Compare this script's forward returns against published labels on an overlapping window."""
    import materialize_operational_factors as operational
    import walk_forward_common as wfc

    from research_v2.portfolio.sources import load_index_closes  # noqa: F401

    frame = market.loc[:, ["trade_date", "ts_code", "adjusted_open"]]
    signals = tuple(calendar.window(args.verify_start, args.verify_end))
    computed = forward_returns(frame, calendar, signals)
    rows = [
        {"signal_date": day, "ts_code": code, "value": value}
        for day, realised in computed.items()
        for code, value in realised.items()
    ]
    holder = wfc.new_holder()
    _, published = wfc.query(
        holder,
        "select signal_date, ts_code, raw_return from "
        f"loadTable('{wfc.TARGET_DATABASE}','label_daily') "
        f"where signal_date between {args.verify_start:%Y.%m.%d} and {args.verify_end:%Y.%m.%d}",
    )
    if published is None or published.empty:
        return {"matched_rows": 0}
    published = published.rename(columns={"raw_return": "value"})
    published["signal_date"] = [
        operational.as_plain_date(value) for value in published["signal_date"]
    ]
    return operational.compare_against_reference(
        pd.DataFrame(rows), published, tolerance=1e-9, relative_tolerance=1e-9
    )


if __name__ == "__main__":
    raise SystemExit(main())
