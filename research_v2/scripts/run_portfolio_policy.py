#!/usr/bin/env python3
"""Turn published model scores into one day's tradeable book.

This is the layer downstream of inference. `research_v2.inference` produces scores and
deliberately knows nothing about labels or market performance; this script consumes those
scores and applies the two portfolio levers that were measured on the frozen 2022-2025
TEST predictions:

  * a holding buffer (keep a name until it falls out of the top ``n_exit``, not ``n_enter``)
  * a trailing trend exposure overlay (go to cash when the style index closes below its own
    moving average)

Both come from `research_v2/configs/portfolio/default.yaml`, and both are implemented in
`research_v2.portfolio`, so a single-date run and a full historical replay execute the same
code. `verify_portfolio_policy_replay.py` proves that replay reproduces the published
backtest.

Because the buffer is path-dependent, answering one date requires replaying from the start
of that date's model fold (or from --replay-from). This is not a workaround: it is the only
way the daily answer can equal what the backtest measured.

Read-only. Writes nothing but --out.

Examples:
    # what do I hold on this date, and what changed since the previous signal date?
    python research_v2/scripts/run_portfolio_policy.py --date 2025-06-13

    # same, but write the book to CSV and show more names
    python research_v2/scripts/run_portfolio_policy.py --date 2025-06-13 --out book.csv

    # the daily decision series over a window (used for review and for the replay check)
    python research_v2/scripts/run_portfolio_policy.py --date 2025-06-13 --replay-from 2025-01-01
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "research_v2" / "src"))

from research_v2.portfolio.engine import replay  # noqa: E402
from research_v2.portfolio.policy import load_portfolio_policy  # noqa: E402
from research_v2.portfolio.sources import (  # noqa: E402
    DEFAULT_ARTIFACTS,
    FrozenArtifactSource,
    fold_for_date,
    load_index_closes,
)

DEFAULT_POLICY = REPO_ROOT / "research_v2" / "configs" / "portfolio" / "default.yaml"
DEFAULT_INDEX_CACHE = Path("/tmp/qrv2_index_closes.parquet")

_LIVE_SOURCE_HINT = (
    "source 'prediction_daily' serves dates the frozen artifacts do not cover (2026 "
    "onward). It needs published prediction_daily batches, which need factors for those "
    "dates: factor_daily currently ends 2025-12-31, so 2026 must be materialised into "
    "dfs://QuantResearchV2 first (research_v2/scripts/materialize_frozen_factor_daily.py), "
    "then scored through research_v2.inference. Nothing in this script writes to a database."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--date", required=True, help="signal date, YYYY-MM-DD")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--source", choices=("artifacts", "prediction_daily"), default="artifacts")
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument(
        "--replay-from",
        type=str,
        default=None,
        help="warmup start for the path-dependent buffer; defaults to the first signal "
             "date of the fold containing --date",
    )
    parser.add_argument("--index-code", default=None, help="override the policy's trend index")
    parser.add_argument("--index-cache", type=Path, default=DEFAULT_INDEX_CACHE)
    parser.add_argument("--no-index", action="store_true",
                        help="skip the DB read and report the book fully invested")
    parser.add_argument("--live-universe", action="store_true",
                        help="rank every scored name instead of only the label-valid ones. "
                             "The default reproduces the backtest, which can only rank names "
                             "whose realised forward return is known; a live 2026 date has no "
                             "labels and should use this flag.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    target = date.fromisoformat(args.date)
    policy = load_portfolio_policy(args.policy)

    if args.source == "prediction_daily":
        raise SystemExit(_LIVE_SOURCE_HINT)

    source = FrozenArtifactSource(artifacts=args.artifacts, with_labels=True)
    source.load(policy.ensemble_members)
    members = list(policy.ensemble_members)
    all_dates = source.signal_dates(members)
    if target not in all_dates:
        fold = fold_for_date(target)
        raise SystemExit(
            f"{target} is not a scored signal date for every member {members}. "
            + (
                f"It falls in fold {fold}, whose scored dates run "
                f"{min(d for d in all_dates if d.year == target.year)} .. "
                f"{max(d for d in all_dates if d.year == target.year)}."
                if fold else "It is outside the signed TEST folds (2022-2025); " + _LIVE_SOURCE_HINT
            )
        )

    if args.replay_from:
        start = date.fromisoformat(args.replay_from)
    else:
        # replay from the first date of the fold containing the target, with an empty book,
        # exactly as the backtest does; the buffer is path-dependent so this is not optional
        fold = fold_for_date(target)
        start = min(d for d in all_dates if fold_for_date(d) == fold)
    dates = [d for d in all_dates if start <= d <= target]

    index_code = args.index_code or policy.trend_index_code
    closes: list[tuple[date, float]] = []
    index_note = "index feed unavailable; reporting fully invested"
    if not args.no_index:
        try:
            closes = load_index_closes(
                index_code, start=start - timedelta(days=policy.trend_ma_window * 3 + 40),
                end=target, cache=args.index_cache,
            )
            index_note = f"{index_code}, {len(closes)} closes loaded"
        except Exception as error:  # noqa: BLE001 - a missing feed must not block the book
            index_note = f"{index_code} feed FAILED ({error}); reporting fully invested"

    scores_by_member = {m: source.scores(m) for m in members}
    universe = None if args.live_universe else source.universe()
    decisions = replay(
        scores_by_member,
        policy=policy,
        folds_by_date=source.folds_by_date(),
        index_closes=closes,
        returns=source.returns(),
        dates=dates,
        universe=universe,
    )
    today = decisions[-1]
    previous = decisions[-2] if len(decisions) > 1 else None

    returns = source.returns() or {}
    day_returns = returns.get(target, {})

    if not args.quiet:
        _print_decision(today, previous, policy, source, members, index_note, index_code,
                        day_returns, len(dates), start, universe is not None)

    if args.out is not None:
        rows = []
        for code in today.holdings:
            rows.append({
                "signal_date": target.isoformat(),
                "ts_code": code,
                "status": today.statuses[code],
                "weight": today.weights.get(code, 0.0),
                "exposure": today.exposure,
                "rank_in_book": today.holdings.index(code) + 1,
                "realized_5d_pct": (round(day_returns[code] * 100, 4)
                                    if code in day_returns else None),
            })
        frame = pd.DataFrame(rows)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.out, index=False)
        if not args.quiet:
            print(f"\nwrote {args.out}")
    return 0


def _print_decision(today, previous, policy, source, members, index_note, index_code,
                    day_returns, replay_days, start, eligibility_applied) -> None:
    fold = today.fold or "?"
    print(f"=== {today.signal_date} (fold {fold}) ===")
    print("score source: signed frozen TEST prediction artifacts")
    for entry in source.provenance:
        if entry.member in members and entry.fold == today.fold:
            print(f"  {entry.member:<11} {entry.workspace:<12} model_id={entry.model_id[:16]} "
                  f"prediction={entry.prediction_id[:12]} rows={entry.rows}")
    print(f"ensemble: equal-weight raw score over {'+'.join(members)}, "
          f"{today.candidates} names scored by every member")
    print(f"buffer replay: {replay_days} signal dates from {start} (path-dependent)")
    print("eligible universe: " + ("label-valid names only (backtest-consistent)"
          if eligibility_applied else "every scored name (live mode)"))

    # the message must follow the actual exposure number, not the binary above/below test:
    # with a partial floor the index being below its MA means "hold LESS", and telling the
    # operator to hold cash when the policy says hold half is how a tool gets traded wrong
    close = f"{today.index_close:.2f}" if today.index_close else "n/a"
    ma = f"{today.index_ma:.2f}" if today.index_ma else "n/a"
    above = today.index_close is not None and today.index_ma is not None \
        and today.index_close > today.index_ma
    if above:
        state = f"ABOVE MA -> fully invested ({today.exposure:.0%})"
    elif today.exposure <= 0.0:
        state = f"AT/BELOW MA -> FLAT ({today.exposure:.0%})"
    else:
        state = f"AT/BELOW MA -> reduced to the floor ({today.exposure:.0%})"
    print(f"\nexposure overlay: {index_note}")
    print(f"  {index_code} close {close}  MA{policy.trend_ma_window} {ma}  {state}")
    print(f"  EXPOSURE = {today.exposure:.2f}  "
          f"(floor {policy.trend_exposure_floor:.2f}, so below-MA days hold "
          f"{policy.trend_exposure_floor:.0%}, not cash)"
          if policy.trend_exposure_floor > 0 else
          f"  EXPOSURE = {today.exposure:.2f}  (floor 0.00, so below-MA days hold cash)")
    if not above:
        if today.exposure <= 0.0:
            print("  -> the overlay is FLAT: hold cash today. The names below are the book "
                  "you re-enter when the index closes back above its MA.")
        else:
            print(f"  -> the overlay is REDUCED: hold the names below at "
                  f"{today.exposure:.0%} of normal size, i.e. "
                  f"{today.exposure / len(today.holdings) * 100:.2f}% each, and keep "
                  f"{(1 - today.exposure) * 100:.0f}% in cash. Scale back up to "
                  f"{1.0 / len(today.holdings) * 100:.2f}% each when the index closes "
                  f"above its MA.")

    per_name = today.exposure / len(today.holdings) if today.holdings else 0.0
    print(f"\nbook: {len(today.holdings)} names x {per_name * 100:.2f}% "
          f"(= {today.exposure * 100:.0f}% invested, {(1 - today.exposure) * 100:.0f}% cash)")
    header = f"{'#':>3} {'code':<11} {'status':<6} {'weight':>7} {'realized_5d':>12}"
    print(header)
    for i, code in enumerate(today.holdings, 1):
        realized = day_returns.get(code)
        text = f"{realized * 100:+.3f}%" if realized is not None else "n/a"
        print(f"{i:>3} {code:<11} {today.statuses[code]:<6} "
              f"{today.weights.get(code, 0.0) * 100:>6.2f}% {text:>12}")

    if previous is not None:
        print(f"\nrebalance vs {previous.signal_date}: "
              f"{len(today.buys)} buys, {len(today.sells)} sells, "
              f"turnover {today.turnover:.3f}, cost {today.cost * 10_000:.1f}bp")
        if today.buys:
            print(f"  BUY : {' '.join(today.buys)}")
        if today.sells:
            print(f"  SELL: {' '.join(today.sells)}")
    else:
        print(f"\nfirst day of the replay: full build, turnover {today.turnover:.3f}, "
              f"cost {today.cost * 10_000:.1f}bp")

    if day_returns:
        held = [day_returns[c] for c in today.holdings if c in day_returns]
        if held:
            mean = sum(held) / len(held)
            print(f"\nrealised 5-day return of this book (equal weight, gross): "
                  f"{mean * 100:+.3f}%  n={len(held)}  "
                  f"win rate {sum(1 for r in held if r > 0) / len(held):.1%}")
            print(f"  after the {policy.trend_exposure_floor:.0%}-floor overlay at "
                  f"exposure {today.exposure:.2f}: {mean * today.exposure * 100 + 0.0:.3f}%")


if __name__ == "__main__":
    raise SystemExit(main())
