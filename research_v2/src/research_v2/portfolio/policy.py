"""Typed portfolio policy and the pure decision rules that implement it.

This layer sits DOWNSTREAM of `research_v2.inference`. That boundary is deliberate:
the daily inference DAG is documented as "without label or performance inputs" and its
manifest builder rejects any target-like column, so a market-timing overlay must not be
pushed into it. Inference produces scores; this module turns scores into holdings.

Both levers encoded here were measured on the frozen 2022-2025 TEST predictions before
being written down, in `docs/research/2026-10-08-optimization-levers.md`:

  * HOLDING BUFFER -- keeping a name until its rank falls below ``n_exit`` instead of
    ``n_enter`` cut turnover from 0.537 to 0.209 and RAISED net annualised return from
    19.01% to 21.29%, so it is a pure net gain rather than a trade-off.
  * TREND EXPOSURE -- going to cash when the style index closes below its own trailing
    moving average cut max drawdown from -24.15% to -6.18% and lifted Sharpe from 1.009
    to 1.392, at a cost of about 2.6pp of annual return.

Two things that were measured and REFUTED are encoded as hard errors rather than options,
so nobody can configure them by accident:

  * volatility targeting -- a 12% target cut annual return from 20.07% to 8.11% while
    moving max drawdown only from -25.17% to -25.10%, because this drawdown is a
    sustained grind and not a volatility spike, so a rolling-vol rule never fires;
  * z-score ensemble averaging -- Sharpe 0.458 against 0.994 for equal-weight raw score,
    because the members' score dispersions differ, so standardising hands the weight to
    whichever member is narrowest and the weight jumps around every day.

The decision rules here are pure: no I/O, no database, no clock. Everything is a function
of the ranked names, the previous holdings, and the index closes handed in, so a daily
run and a full historical replay execute literally the same code path.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

#: Equal-weight average of the members' raw scores. The only validated weighting.
WEIGHTING_EQUAL_RAW_SCORE = "equal_raw_score"

#: Refuted alternatives, kept so the error message can say why they are refused.
_REFUTED_WEIGHTINGS = {
    "zscore": (
        "z-score ensemble averaging is refused: measured Sharpe 0.458 against 0.994 for "
        "equal-weight raw score, because member score dispersions differ so standardising "
        "hands the weight to the narrowest member"
    ),
    "zscore_mean": "z-score ensemble averaging is refused; see WEIGHTING_EQUAL_RAW_SCORE",
    "vol_target": (
        "volatility targeting is refused: a 12% target cut annual return from 20.07% to "
        "8.11% while moving max drawdown only from -25.17% to -25.10%"
    ),
    "vol_targeting": "volatility targeting is refused; see the module docstring",
}


@dataclass(frozen=True)
class PortfolioPolicy:
    """One complete, validated portfolio policy."""

    n_enter: int = 30
    n_exit: int = 150
    ensemble_members: tuple[str, ...] = ("M0R", "M1")
    ensemble_weighting: str = WEIGHTING_EQUAL_RAW_SCORE
    trend_index_code: str = "932000.CSI"
    trend_ma_window: int = 10
    trend_exposure_floor: float = 0.0
    round_trip_bp: float = 15.0
    switch_cost_factor: float = 0.5

    def __post_init__(self) -> None:
        for name in ("n_enter", "n_exit", "trend_ma_window"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        for name in ("trend_exposure_floor", "round_trip_bp", "switch_cost_factor"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.n_enter < 1:
            raise ValueError("n_enter must be at least 1")
        # a buffer narrower than the book is not a buffer, it is just a smaller top-N,
        # and silently accepting it would make the measured turnover reduction unreachable
        if self.n_exit < self.n_enter:
            raise ValueError(f"n_exit ({self.n_exit}) must be >= n_enter ({self.n_enter})")
        if not isinstance(self.ensemble_members, tuple) or not self.ensemble_members:
            raise ValueError("ensemble_members must be a non-empty tuple")
        if any(not isinstance(m, str) or not m for m in self.ensemble_members):
            raise ValueError("ensemble_members must be non-empty strings")
        if len(set(self.ensemble_members)) != len(self.ensemble_members):
            raise ValueError("ensemble_members must not contain duplicates")
        if self.ensemble_weighting != WEIGHTING_EQUAL_RAW_SCORE:
            reason = _REFUTED_WEIGHTINGS.get(self.ensemble_weighting)
            raise ValueError(
                reason
                or f"unsupported ensemble_weighting {self.ensemble_weighting!r}; "
                f"only {WEIGHTING_EQUAL_RAW_SCORE!r} is validated"
            )
        if not isinstance(self.trend_index_code, str) or not self.trend_index_code:
            raise ValueError("trend_index_code must be a non-empty string")
        if self.trend_ma_window < 2:
            raise ValueError("trend_ma_window must be at least 2")
        if not 0.0 <= self.trend_exposure_floor <= 1.0:
            raise ValueError("trend_exposure_floor must be in [0, 1]")
        if self.round_trip_bp < 0:
            raise ValueError("round_trip_bp must be non-negative")
        # Going invested->cash is a SELL and cash->invested is a BUY, i.e. one leg of a
        # round trip, so the correct factor is 0.5. The research backtest charged 1.0; it
        # is kept reachable so those published numbers stay reproducible, but 0.5 is the
        # default because 1.0 overstates the overlay's cost by a factor of two.
        if not 0.0 <= self.switch_cost_factor <= 1.0:
            raise ValueError("switch_cost_factor must be in [0, 1]")


DEFAULT_PORTFOLIO_POLICY = PortfolioPolicy()


def _mapping(value: object, where: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be a mapping")
    return value


def load_portfolio_policy(path: Path) -> PortfolioPolicy:
    """Load one complete portfolio policy, rejecting unknown or malformed keys."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"unable to load portfolio policy: {path}") from error
    if not isinstance(raw, Mapping):
        raise ValueError("portfolio policy has an invalid top-level schema")
    expected = {"selection", "ensemble", "exposure", "cost"}
    if set(raw) != expected:
        raise ValueError(
            f"portfolio policy must have exactly the top-level keys {sorted(expected)}, "
            f"got {sorted(raw)}"
        )

    selection = _mapping(raw.get("selection"), "selection")
    _require_exact_keys(selection, {"n_enter", "n_exit"}, "selection")
    ensemble = _mapping(raw.get("ensemble"), "ensemble")
    _require_exact_keys(ensemble, {"members", "weighting"}, "ensemble")
    exposure = _mapping(raw.get("exposure"), "exposure")
    _require_exact_keys(exposure, {"index_code", "ma_window", "floor"}, "exposure")
    cost = _mapping(raw.get("cost"), "cost")
    _require_exact_keys(cost, {"round_trip_bp", "switch_cost_factor"}, "cost")

    members = ensemble.get("members")
    if not isinstance(members, Sequence) or isinstance(members, (str, bytes)):
        raise ValueError("ensemble.members must be a list of model names")
    return PortfolioPolicy(
        n_enter=_as_int(selection.get("n_enter"), "selection.n_enter"),
        n_exit=_as_int(selection.get("n_exit"), "selection.n_exit"),
        ensemble_members=tuple(str(m) for m in members),
        ensemble_weighting=str(ensemble.get("weighting")),
        trend_index_code=str(exposure.get("index_code")),
        trend_ma_window=_as_int(exposure.get("ma_window"), "exposure.ma_window"),
        trend_exposure_floor=float(exposure.get("floor")),  # type: ignore[arg-type]
        round_trip_bp=float(cost.get("round_trip_bp")),  # type: ignore[arg-type]
        switch_cost_factor=float(cost.get("switch_cost_factor")),  # type: ignore[arg-type]
    )


def _require_exact_keys(mapping: Mapping[str, object], expected: set[str], where: str) -> None:
    if set(mapping) != expected:
        raise ValueError(
            f"{where} must have exactly the keys {sorted(expected)}, got {sorted(mapping)}")


def _as_int(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where} must be an integer")
    return value


# --------------------------------------------------------------------- pure rules
def combine_scores(
    per_member: Mapping[str, Mapping[str, float]],
    policy: PortfolioPolicy,
) -> dict[str, float]:
    """Equal-weight average of the members' raw scores, over names every member scored.

    The intersection is intentional: averaging over names one member is missing would
    silently weight the remaining members more heavily for those rows, which is how the
    M2 coverage gap used to distort cross-model comparisons.
    """
    if policy.ensemble_weighting != WEIGHTING_EQUAL_RAW_SCORE:  # pragma: no cover - guarded
        raise ValueError(f"unsupported weighting {policy.ensemble_weighting!r}")
    members = list(policy.ensemble_members)
    if not members:
        return {}
    common = set.intersection(*(set(per_member.get(m, {})) for m in members)) if members else set()
    return {
        code: float(sum(per_member[m][code] for m in members) / len(members))
        for code in sorted(common)
    }


def rank_descending(scores: Mapping[str, float]) -> list[str]:
    """Names ordered by score descending, ties broken by code for determinism."""
    return [code for code, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def select_holdings(
    ranked: Sequence[str],
    previous: Sequence[str],
    policy: PortfolioPolicy,
) -> list[str]:
    """Apply the holding buffer: keep incumbents until they fall out of the top ``n_exit``.

    ``previous`` order is preserved for the kept names and vacated slots are filled from
    the top of today's ranking. That ordering is part of the measured behaviour, so it is
    reproduced exactly rather than re-sorted by today's score.
    """
    held_window = set(ranked[: policy.n_exit])
    keep = [code for code in previous if code in held_window]
    fill = [code for code in ranked[: max(policy.n_enter, policy.n_exit)] if code not in set(keep)]
    return (keep + fill)[: policy.n_enter]


def turnover(new: Sequence[str], previous: Sequence[str]) -> float:
    """Fraction of the book replaced. A first-ever build is a full 1.0 turnover."""
    if not new:
        return 0.0
    if not previous:
        return 1.0
    return 1.0 - len(set(new) & set(previous)) / len(new)


def trend_exposure(closes: Sequence[float], policy: PortfolioPolicy) -> float:
    """1.0 when the last close is above its own trailing MA, else the floor.

    ``closes`` must be ordered oldest-first and end on the signal date. Only that date and
    earlier are used, so the value is knowable at the close of the signal date and the
    trade happens at the next open: no lookahead. Returns 1.0 (fully invested) when there
    is not yet enough history, so a cold start degrades to the unfiltered strategy rather
    than to an arbitrary cash position.
    """
    window = policy.trend_ma_window
    if len(closes) < window:
        return 1.0
    trailing = closes[-window:]
    if any(not math.isfinite(c) for c in trailing):
        return 1.0
    average = sum(trailing) / window
    if average <= 0:
        return 1.0
    return 1.0 if closes[-1] > average else policy.trend_exposure_floor


def position_weights(holdings: Sequence[str], exposure: float) -> dict[str, float]:
    """Equal weight scaled by the exposure overlay. Cash is the remainder."""
    if not holdings or exposure <= 0:
        return {}
    weight = exposure / len(holdings)
    return {code: weight for code in holdings}


def cost_of(new: Sequence[str], previous: Sequence[str], exposure: float,
            previous_exposure: float | None, policy: PortfolioPolicy) -> float:
    """Round-trip cost for one day, including the cash switch charged explicitly.

    Going to cash and back is a real trade, so the change in exposure is charged instead of
    being waved through -- but at ``switch_cost_factor`` of a round trip, because a switch
    is one leg (a sell, or a buy) and not both.
    """
    switched = 0.0 if previous_exposure is None else abs(exposure - previous_exposure)
    name_cost = turnover(new, previous) * exposure
    switch_cost = switched * policy.switch_cost_factor
    return (name_cost + switch_cost) * policy.round_trip_bp / 10_000.0
