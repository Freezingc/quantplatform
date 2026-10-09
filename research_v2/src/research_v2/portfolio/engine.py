"""Replay engine: one code path for a single daily decision and for a full backtest.

This exists as a module rather than as logic inside the CLI so that a daily run and a
historical replay cannot disagree: both call `replay()`, and a single-date answer is
simply the last element of a replay ending on that date. The holding buffer is
path-dependent -- today's book depends on yesterday's -- so "what do I hold today" cannot
be answered without replaying from a warmup start in any case.

Aggregation follows the research backtest's conventions so the two are cross-checkable:

  * five consecutive signal dates form one non-overlapping holding cycle;
  * cycles are grouped PER MODEL FOLD and never straddle a fold boundary, because the
    model that produced the scores changes there (192 cycles = 4 folds x 48);
  * annualisation is 243/5 = 48.6 cycles per year;
  * Sharpe uses a zero risk-free rate, so it overstates excess Sharpe by roughly 0.07-0.10.

`to_cycles` takes an explicit ``exposure`` mode because there are two defensible ways to
apply the overlay, and hiding the choice inside the code is how a backtest stops matching
production:

  * ``"none"``   -- no overlay; the book is always fully invested.
  * ``"daily"``  -- each day's return is scaled by that day's own exposure and charged that
                   day's own cost, including the cash switch. This is what actually
                   happens when a book is traded, and it is what the CLI reports.
  * ``"cycle"``  -- the five daily exposures are averaged into one cycle-level scalar and
                   the switch is charged once per cycle. This is what the published
                   research numbers in `docs/research/2026-10-08-optimization-levers.md`
                   were computed with.

The two filtered modes differ slightly, and `verify_portfolio_policy_replay.py` reports the
difference rather than papering over it. With ``"none"`` they coincide, which is what makes
the unfiltered cross-check an exact one.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Literal

from research_v2.portfolio.policy import (
    PortfolioPolicy,
    combine_scores,
    cost_of,
    position_weights,
    rank_descending,
    select_holdings,
    trend_exposure,
    turnover,
)

HOLDING_DAYS = 5
CYCLES_PER_YEAR = 243 / HOLDING_DAYS
ExposureMode = Literal["none", "daily", "cycle"]


@dataclass(frozen=True)
class DailyDecision:
    """One signal date's complete, tradeable output."""

    signal_date: date
    fold: str | None
    exposure: float
    index_close: float | None
    index_ma: float | None
    holdings: tuple[str, ...]
    statuses: Mapping[str, str]
    buys: tuple[str, ...]
    sells: tuple[str, ...]
    turnover: float
    cost: float
    weights: Mapping[str, float]
    candidates: int
    gross_return: float | None = None


@dataclass(frozen=True)
class Cycle:
    start: date
    end: date
    fold: str | None
    gross_return: float
    net_return: float
    turnover: float
    names: float


@dataclass(frozen=True)
class PerformanceSummary:
    cycles: int
    gross_annualized: float
    net_annualized: float
    net_sharpe: float
    net_vol: float
    total_return: float
    max_drawdown: float
    mean_turnover: float
    mean_names: float
    drawdown_peak: date | None
    drawdown_trough: date | None


def replay(
    scores_by_member: Mapping[str, Mapping[date, Mapping[str, float]]],
    *,
    policy: PortfolioPolicy,
    folds_by_date: Mapping[date, str] | None = None,
    index_closes: Sequence[tuple[date, float]] = (),
    returns: Mapping[date, Mapping[str, float]] | None = None,
    dates: Iterable[date] | None = None,
    universe: Mapping[date, frozenset[str]] | None = None,
    reset_book_at_fold_boundary: bool = True,
) -> list[DailyDecision]:
    """Walk the signal dates forward, applying the buffer and the exposure overlay.

    ``index_closes`` must be oldest-first. Only closes up to and including the signal date
    are ever handed to `trend_exposure`, so the overlay uses no future information.

    ``reset_book_at_fold_boundary`` reproduces the measured convention: each annual fold is
    a DIFFERENT trained model, so the published backtest rebuilt each fold's book from an
    empty position instead of inheriting the previous fold's holdings. Inheriting them
    lowers turnover at the four boundaries and silently disagrees with every published
    number. A live run has one model and no folds, so the flag is inert there. The exposure
    switch cost is deliberately NOT reset at the boundary: the cash position is a property
    of the market, not of the model.

    ``universe`` optionally restricts which names may be ranked on each date. It exists
    because a backtest and a live run do not have the same eligible set, and letting that
    difference hide inside the code is how a replay stops matching its own published
    numbers. In the backtest the only rankable names are those whose realised forward
    return is known (``label_valid``), since everything else is unmeasurable; on a live
    date there are no labels at all, so no restriction applies and ``universe`` is None.
    """
    if dates is None:
        found = (
            set.intersection(*(set(scores_by_member.get(m, {})) for m in policy.ensemble_members))
            if policy.ensemble_members
            else set()
        )
        ordered = sorted(found)
    else:
        ordered = sorted(dates)

    close_by_date = dict(index_closes)
    close_dates = [d for d, _ in index_closes]
    position = {d: i for i, d in enumerate(close_dates)}

    held: list[str] = []
    previous_exposure: float | None = None
    previous_fold: object = object()
    out: list[DailyDecision] = []
    for day in ordered:
        fold = (folds_by_date or {}).get(day)
        if reset_book_at_fold_boundary and fold != previous_fold:
            held = []
        previous_fold = fold
        per_member = {m: scores_by_member.get(m, {}).get(day, {}) for m in policy.ensemble_members}
        combined = combine_scores(per_member, policy)
        allowed = universe.get(day) if universe is not None else None
        if allowed is not None:
            combined = {c: v for c, v in combined.items() if c in allowed}
        ranked = rank_descending(combined)
        previous = list(held)
        held = select_holdings(ranked, previous, policy)

        cut = position.get(day)
        history = [close_by_date[d] for d in close_dates[: cut + 1]] if cut is not None else []
        exposure = trend_exposure(history, policy)
        window = policy.trend_ma_window
        trailing = history[-window:] if len(history) >= window else []
        index_ma = sum(trailing) / len(trailing) if trailing else None

        day_turnover = turnover(held, previous)
        cost = cost_of(held, previous, exposure, previous_exposure, policy)
        previous_exposure = exposure

        kept = set(previous)
        gross: float | None = None
        if returns is not None and held:
            day_returns = returns.get(day, {})
            observed = [day_returns[c] for c in held if c in day_returns]
            if observed:
                gross = sum(observed) / len(observed)

        out.append(
            DailyDecision(
                signal_date=day,
                fold=fold,
                exposure=exposure,
                index_close=close_by_date.get(day),
                index_ma=index_ma,
                holdings=tuple(held),
                statuses={c: ("HELD" if c in kept else "NEW") for c in held},
                buys=tuple(c for c in held if c not in kept),
                sells=tuple(c for c in previous if c not in set(held)),
                turnover=day_turnover,
                cost=cost,
                weights=position_weights(held, exposure),
                candidates=len(combined),
                gross_return=gross,
            )
        )
    return out


def to_cycles(
    decisions: Sequence[DailyDecision],
    policy: PortfolioPolicy,
    exposure: ExposureMode = "none",
) -> list[Cycle]:
    """Group into non-overlapping 5-day cycles, restarting at every fold boundary."""
    if exposure not in ("none", "daily", "cycle"):
        raise ValueError(f"unknown exposure mode {exposure!r}")
    scored = [d for d in decisions if d.gross_return is not None]
    bp_rate = policy.round_trip_bp / 10_000.0

    groups = _groups(scored)
    # one scalar per cycle, averaged over its five days, with the switch charged once
    cycle_scale: list[float] = []
    cycle_switch: list[float] = []
    if exposure == "cycle":
        previous: float | None = None
        for group in groups:
            scale = sum(d.exposure for d in group) / len(group)
            cycle_scale.append(scale)
            cycle_switch.append(0.0 if previous is None else abs(scale - previous))
            previous = scale

    cycles: list[Cycle] = []
    for index, group in enumerate(groups):
        gross_raw = sum(d.gross_return for d in group) / len(group)  # type: ignore[misc]
        turn_raw = sum(d.turnover for d in group) / len(group)
        names = float(sum(len(d.holdings) for d in group) / len(group))
        if exposure == "none":
            gross, turn = gross_raw, turn_raw
        elif exposure == "daily":
            # each day scaled by its own exposure and charged its own cost, switch included
            daily_net = [
                (d.gross_return or 0.0) * d.exposure - d.cost  # type: ignore[operator]
                for d in group
            ]
            gross = sum((d.gross_return or 0.0) * d.exposure for d in group) / len(group)  # type: ignore[misc]
            net = sum(daily_net) / len(daily_net)
            turn = (gross - net) / bp_rate if bp_rate > 0 else turn_raw
            cycles.append(Cycle(group[0].signal_date, group[-1].signal_date, group[0].fold,
                                gross, net, turn, names))
            continue
        else:
            gross = gross_raw * cycle_scale[index]
            turn = turn_raw * cycle_scale[index] + cycle_switch[index] * policy.switch_cost_factor
        net = gross - turn * bp_rate
        cycles.append(Cycle(group[0].signal_date, group[-1].signal_date, group[0].fold,
                            gross, net, turn, names))
    return cycles


def _groups(scored: Sequence[DailyDecision]) -> list[list[DailyDecision]]:
    """Five-day groups, restarting at every fold boundary so no cycle straddles one."""
    out: list[list[DailyDecision]] = []
    current_fold: object = object()
    bucket: list[DailyDecision] = []
    for d in scored:
        if d.fold != current_fold:
            current_fold = d.fold
            bucket = []
        bucket.append(d)
        if len(bucket) == HOLDING_DAYS:
            out.append(bucket)
            bucket = []
    return out


def summarise(cycles: Sequence[Cycle]) -> PerformanceSummary:
    """Annualise, Sharpe and drawdown, using the research backtest's conventions."""
    if not cycles:
        nan = float("nan")
        return PerformanceSummary(0, nan, nan, nan, nan, nan, nan, nan, nan, None, None)
    net = [c.net_return for c in cycles]
    gross = [c.gross_return for c in cycles]
    level, equity = 1.0, []
    for value in net:
        level *= 1.0 + value
        equity.append(level)
    peak, worst, worst_index = equity[0], 0.0, 0
    for i, value in enumerate(equity):
        peak = max(peak, value)
        drawdown = value / peak - 1.0
        if drawdown < worst:
            worst, worst_index = drawdown, i
    peak_index = max(range(worst_index + 1), key=lambda i: equity[i]) if worst_index else 0
    mean = sum(net) / len(net)
    variance = sum((v - mean) ** 2 for v in net) / (len(net) - 1) if len(net) > 1 else 0.0
    std = math.sqrt(variance)
    return PerformanceSummary(
        cycles=len(net),
        gross_annualized=(sum(gross) / len(gross)) * CYCLES_PER_YEAR,
        net_annualized=mean * CYCLES_PER_YEAR,
        net_sharpe=(mean / std * math.sqrt(CYCLES_PER_YEAR)) if std > 0 else float("nan"),
        net_vol=std * math.sqrt(CYCLES_PER_YEAR),
        total_return=equity[-1] - 1.0,
        max_drawdown=worst,
        mean_turnover=sum(c.turnover for c in cycles) / len(cycles),
        mean_names=sum(c.names for c in cycles) / len(cycles),
        drawdown_peak=cycles[peak_index].start,
        drawdown_trough=cycles[worst_index].end,
    )
