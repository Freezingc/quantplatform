#!/usr/bin/env python3
"""Prove the portfolio policy layer reproduces the published research backtest.

`research_v2.portfolio` is an INDEPENDENT re-implementation of the buffer, the ranking, the
exposure overlay and the performance maths that produced
`docs/research/2026-10-08-optimization-levers.md`. It shares no code with the research
scripts. That is deliberate: two implementations agreeing to 1e-12 is a much stronger check
on both than one importing the other, and it is the only thing standing between "the CLI
prints a book" and "the CLI prints the book that was actually measured".

Three assertions, in increasing order of how much they have to get right:

  1. UNFILTERED, EXACT. With no overlay the two implementations have no semantic freedom
     left, so annualised return, Sharpe, drawdown, cumulative return, turnover, cycle count
     and the drawdown window dates must all match the published JSON bit for bit. This is
     the check that validates the buffer and the ranking.
  2. FILTERED AT CYCLE LEVEL, EXACT. The published overlay averaged the five daily
     exposures into one cycle scalar and charged the cash switch once per cycle.
     ``exposure="cycle"`` reproduces that convention, so it must also match exactly. This
     is the check that validates the overlay and its cost model.
  3. FILTERED AT DAILY LEVEL, REPORTED. ``exposure="daily"`` scales each day by its own
     exposure and charges each day its own cost, which is what a real book does and what
     the CLI reports. It is NOT expected to equal the published number, and the difference
     is printed rather than hidden, with the sign explained.

Exits non-zero if either exact check fails. Read-only; writes nothing.
"""

from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "research_v2" / "src"))
sys.path.insert(0, str(REPO_ROOT / "research_v2" / "scripts"))

from research_v2.portfolio.engine import CYCLES_PER_YEAR, replay, summarise, to_cycles  # noqa: E402
from research_v2.portfolio.policy import PortfolioPolicy, load_portfolio_policy  # noqa: E402
from research_v2.portfolio.sources import (  # noqa: E402
    DEFAULT_ARTIFACTS,
    FrozenArtifactSource,
    load_index_closes,
)

PUBLISHED = (REPO_ROOT / "docs" / "research" / "2026-10-08-model-backtest"
             / "gate_clean_ensemble_results.json")
INDEX_CACHE = Path("/tmp/qrv2_index_closes.parquet")
EXACT_TOLERANCE = 1e-12
#: the daily-vs-cycle overlay conventions differ by a second-order term, so this is a
#: sanity bound on the reported difference, not a claim that they should agree
DAILY_SANITY_BOUND = 0.02


def main() -> int:
    policy = load_portfolio_policy(
        REPO_ROOT / "research_v2" / "configs" / "portfolio" / "default.yaml")
    published = json.loads(PUBLISHED.read_text())
    member_set = "M0R+M1 (gate-clean)"
    if member_set not in published["member_sets"]:
        raise SystemExit(f"{member_set} missing from {PUBLISHED}")
    entry = published["member_sets"][member_set]

    print(f"policy      : {policy}")
    print(f"published   : {PUBLISHED.name} [{member_set}]")
    print(f"conventions : {CYCLES_PER_YEAR:.1f} cycles/year, holding 5 signal dates, "
          f"cost {policy.round_trip_bp:.0f}bp round trip, Sharpe rf=0\n")

    source = FrozenArtifactSource(artifacts=DEFAULT_ARTIFACTS, with_labels=True)
    source.load(policy.ensemble_members)
    members = list(policy.ensemble_members)
    all_dates = source.signal_dates(members)
    # the moving average must already be warm on the first signal date, exactly as it was
    # in the research run, which pulled index history from 2021. Starting the feed at the
    # first signal date would leave the overlay blind -- and a blind overlay silently
    # reports "fully invested", which is the failure mode this script exists to catch.
    warmup = all_dates[0] - timedelta(days=policy.trend_ma_window * 5 + 300)
    closes = load_index_closes(policy.trend_index_code, start=warmup, end=all_dates[-1],
                               cache=INDEX_CACHE)
    covered = sum(1 for d, _ in closes if d >= all_dates[0])
    print(f"index feed  : {policy.trend_index_code}, {len(closes)} closes from {closes[0][0]}, "
          f"{covered} of them on or after the first signal date {all_dates[0]}")
    if covered < len(all_dates):
        raise SystemExit(
            f"index feed covers {covered} of {len(all_dates)} signal dates; refusing to "
            f"compare, because missing dates degrade to fully invested and would make the "
            f"overlay look like a no-op"
        )

    scores_by_member = {m: source.scores(m) for m in members}
    common = dict(
        policy=policy,
        folds_by_date=source.folds_by_date(),
        index_closes=closes,
        returns=source.returns(),
        universe=source.universe(),
        dates=all_dates,
    )

    failures: list[str] = []
    checked = 0

    def compare(label: str, got: float, want: float, *, exact: bool) -> None:
        nonlocal checked
        checked += 1
        delta = got - want
        ok = abs(delta) <= EXACT_TOLERANCE if exact else abs(delta) <= DAILY_SANITY_BOUND
        flag = "EXACT" if exact else "delta"
        status = "ok " if ok else "FAIL"
        print(f"  [{status}] {label:<34} engine {got:>12.9f}  published {want:>12.9f}  "
              f"{flag} {delta:>+.3e}")
        if not ok:
            failures.append(f"{label}: engine {got!r} vs published {want!r} (delta {delta:+.3e})")

    RESEARCH_MA = 10  # the window the published overlay was measured with
    for variant_key, n_exit, overlay in (
        ("top30_exit150", 150, RESEARCH_MA),
        ("top30_exit60", 60, RESEARCH_MA),
    ):
        published_variant = entry["variants"][variant_key]
        # The published overlay was measured with CSI 2000, MA10, floor 0.0 and a FULL
        # round trip charged per exposure switch. Those settings are pinned here on purpose
        # rather than inherited from default.yaml: tuning the shipped policy must not break
        # this check, and the check is about the published numbers, not about current taste.
        research_policy = PortfolioPolicy(
            n_enter=policy.n_enter, n_exit=n_exit, ensemble_members=policy.ensemble_members,
            ensemble_weighting=policy.ensemble_weighting, trend_index_code="932000.CSI",
            trend_ma_window=10, trend_exposure_floor=0.0, round_trip_bp=policy.round_trip_bp,
            switch_cost_factor=1.0,
        )
        # identical rule, but with the correct one-leg switch cost, to isolate how much of
        # the gap is the accounting convention and how much is the cost rate
        corrected_policy = PortfolioPolicy(**{**research_policy.__dict__,
                                              "switch_cost_factor": 0.5})
        this_policy = research_policy
        decisions = replay(scores_by_member, **{**common, "policy": this_policy})
        print(f"=== {variant_key}: {len(all_dates)} signal dates replayed, "
              f"{len(decisions)} decisions ===")

        # ---- 1. unfiltered, must be exact
        cycles = to_cycles(decisions, this_policy, exposure="none")
        got = summarise(cycles)
        want = published_variant["none"]
        print(" unfiltered (exact):")
        compare("cycles", float(got.cycles), float(want["cycles"]), exact=True)
        compare("gross_annualized", got.gross_annualized, want["gross_annualized"], exact=True)
        compare("net_annualized", got.net_annualized, want["net_annualized"], exact=True)
        compare("net_sharpe", got.net_sharpe, want["net_sharpe"], exact=True)
        compare("net_vol", got.net_vol, want["net_vol"], exact=True)
        compare("max_drawdown", got.max_drawdown, want["max_drawdown"], exact=True)
        compare("total_return", got.total_return, want["total_return"], exact=True)
        compare("mean_turnover", got.mean_turnover, want["mean_turnover"], exact=True)
        compare("mean_names", got.mean_names, want["mean_names"], exact=True)
        if str(got.drawdown_peak) != str(want["dd_peak_date"]):
            failures.append(f"{variant_key} drawdown peak: "
                            f"{got.drawdown_peak} vs {want['dd_peak_date']}")
            print(f"  [FAIL] drawdown peak date  engine {got.drawdown_peak}  "
                  f"published {want['dd_peak_date']}")
        else:
            checked += 1
            print(f"  [ok ] {'drawdown window':<34} {got.drawdown_peak} .. {got.drawdown_trough}")
        if str(got.drawdown_trough) != str(want["dd_trough_date"]):
            failures.append(f"{variant_key} drawdown trough: "
                            f"{got.drawdown_trough} vs {want['dd_trough_date']}")

        # ---- 2. filtered at cycle level, must be exact
        want_filtered = published_variant[f"ma{overlay}"]
        cyc = summarise(to_cycles(decisions, this_policy, exposure="cycle"))
        print(f" filtered MA{overlay}, cycle-level convention (exact):")
        compare("net_annualized", cyc.net_annualized, want_filtered["net_annualized"], exact=True)
        compare("net_sharpe", cyc.net_sharpe, want_filtered["net_sharpe"], exact=True)
        compare("max_drawdown", cyc.max_drawdown, want_filtered["max_drawdown"], exact=True)
        compare("total_return", cyc.total_return, want_filtered["total_return"], exact=True)

        # ---- 3. filtered at daily level under the SHIPPED cost convention.
        # Not asserted against the published number: the two differ for two honest reasons,
        # both decomposed below. (a) convention -- cycle-level averaging multiplies the mean
        # return by the mean exposure, whereas a real book scales each day separately, and
        # (b) cost -- a 5-day average hides intra-cycle flips, so the published run charged
        # far fewer switches than a daily book actually makes.
        daily_research_cost = summarise(to_cycles(decisions, research_policy, exposure="daily"))
        daily_decisions = replay(scores_by_member, **{**common, "policy": corrected_policy})
        daily = summarise(to_cycles(daily_decisions, corrected_policy, exposure="daily"))
        print(f" filtered MA{overlay}, DAILY-level convention, one-leg switch cost 0.5:")
        # The daily-level figures are NOT expected to match: they are the correction. What
        # is asserted instead is the DIRECTION and a sane magnitude, which is a stronger
        # claim than a tolerance -- cycle-level averaging must overstate the overlay, so
        # daily accounting must come out lower on return, Sharpe and cumulative, and it
        # must not come out lower by an absurd amount.
        directional = (
            ("net_annualized", daily.net_annualized, want_filtered["net_annualized"], -0.12, -0.02),
            ("net_sharpe", daily.net_sharpe, want_filtered["net_sharpe"], -0.70, -0.10),
            ("total_return", daily.total_return, want_filtered["total_return"], -0.60, -0.10),
        )
        for name, value, ref, lo, hi in directional:
            checked += 1
            delta = value - ref
            ok = lo <= delta <= hi
            print(f"  [{'ok ' if ok else 'FAIL'}] {name:<34} engine {value:>12.9f}  "
                  f"published {ref:>12.9f}  delta {delta:>+.3e}  expected in [{lo}, {hi}]")
            if not ok:
                failures.append(
                    f"{variant_key} daily {name}: delta {delta:+.4f} outside [{lo}, {hi}]")
        # drawdown is the one quantity the overlay genuinely improves, and daily accounting
        # makes it modestly worse than the cycle-level figure claimed, but it must still be
        # far better than the unfiltered book
        # drawdowns are negative, so compare MAGNITUDES. The overlay must still more than
        # halve the drawdown under daily accounting, and daily accounting must make it
        # somewhat worse than the cycle-level figure claimed -- never better, because the
        # cycle-level convention only ever flattered the overlay.
        checked += 1
        unfiltered_dd = published_variant["none"]["max_drawdown"]
        published_dd = want_filtered["max_drawdown"]
        ok = (abs(daily.max_drawdown) < abs(unfiltered_dd) / 2
              and abs(published_dd) <= abs(daily.max_drawdown))
        print(f"  [{'ok ' if ok else 'FAIL'}] {'max_drawdown magnitudes':<34} "
              f"daily {abs(daily.max_drawdown):.6f}  published {abs(published_dd):.6f}  "
              f"unfiltered {abs(unfiltered_dd):.6f} (need daily < half of unfiltered, "
              f"and published <= daily)")
        if not ok:
            failures.append(f"{variant_key}: drawdown ordering broken")
        # the correction must be dominated by the accounting convention, not by the cost
        # rate; if that ever inverts, one of the two implementations has changed meaning
        convention_effect = abs(daily_research_cost.net_annualized
                                - want_filtered["net_annualized"])
        cost_effect = abs(daily.net_annualized - daily_research_cost.net_annualized)
        checked += 1
        if convention_effect <= cost_effect:
            failures.append(
                f"{variant_key}: accounting convention ({convention_effect:.4f}) is no longer "
                f"the dominant term over the switch-cost rate ({cost_effect:.4f})")
            print(f"  [FAIL] convention effect {convention_effect:.4f} should dominate "
                  f"cost-rate effect {cost_effect:.4f}")
        else:
            print(f"  [ok ] {'convention effect dominates':<34} "
                  f"{convention_effect:.4f} vs cost-rate {cost_effect:.4f}")
        print("   decomposition of the gap to the published cycle-level number:")
        print(f"     published (cycle-level, switch cost 1.0)  "
              f"net {want_filtered['net_annualized']:.6f}  "
              f"Sharpe {want_filtered['net_sharpe']:.4f}")
        print(f"     daily-level, switch cost 1.0            "
              f"net {daily_research_cost.net_annualized:.6f}  "
              f"Sharpe {daily_research_cost.net_sharpe:.4f}   <- convention effect only")
        print(f"     daily-level, switch cost 0.5            "
              f"net {daily.net_annualized:.6f}  Sharpe {daily.net_sharpe:.4f}"
              f"   <- convention + corrected cost")
        # a list zipped with its own tail is deliberately one shorter, so strict=False
        flips = sum(1 for a, b in zip(daily_decisions, daily_decisions[1:], strict=False)
                    if abs(a.exposure - b.exposure) > 1e-9)
        per_year = flips / len(daily_decisions) * CYCLES_PER_YEAR * 5
        print(f"     exposure flips observed: {flips} over {len(daily_decisions)} signal dates "
              f"(~{per_year:.1f}/year at daily granularity)")
        print()

    print(f"compared {checked} quantities")
    if failures:
        print(f"\nFAILED with {len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nOK: the policy layer reproduces the published backtest to machine precision")
    print("    under both the unfiltered and the cycle-level overlay conventions.")
    print("    Its daily-level accounting -- the one the CLI actually trades -- is lower,")
    print("    and the gap is dominated by the accounting convention rather than by the")
    print("    switch-cost rate, which is the finding recorded in the research report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
