"""Pre-flight gate for the frozen-set baseline walk-forward (reconstruction plan Task 3.1).

The frozen baseline becomes the new trust root, and G1-G6 do not measure cross-section
width: all six frozen test years pass G6 at 100% valid days, so a baseline fitted on
12-35% of the pairs would clear every model gate while carrying a defect only these
gates can see. This script therefore runs them in dependency order and refuses to clear
the run unless each passes or carries an explicit, reasoned waiver.

Order matters: data completeness first (a missing year would make every later number
meaningless), then governance, then per-factor coverage on the consumed source, then the
per-fold joint cross-section that the baselines actually train and score on, and finally
proof that the run's own report can be serialised -- the report is written at the very end
of a roughly fifteen-hour run, so a defect there is paid for in full before it is seen.

Usage:
  preflight_frozen_baseline.py
  preflight_frozen_baseline.py --waive admission_consistency --waiver-reason "<decision ref>"
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

GATE_ORDER: tuple[str, ...] = (
    "backfill_completeness",
    "admission_consistency",
    "factor_coverage_screen",
    "fold_coverage",
    "report_path",
)

# The C6.2 screen must cover every materialized year. The first two windows span
# 1,464 + 1,212 = 2,676 trading days while 2014-2025 holds 2,919, i.e. 2025 was never
# screened at all -- and the frozen factor set is *defined* by this screen, so an
# unscreened year means the set's own classification was unverified for it. Measured
# 2026-09-06: all 17 factors present on 243/243 days, day_ratio 1.0000, dups 0,
# PASS 11/17 with the same six ELIGIBILITY_FAIL, so the classification is stable across
# all three windows.
DEFAULT_SCREEN_WINDOWS: tuple[tuple[str, str], ...] = (
    ("2014-01-01", "2019-12-31"),
    ("2020-01-01", "2024-12-31"),
    ("2025-01-01", "2025-12-31"),
)


@dataclass(frozen=True)
class GateResult:
    name: str
    exit_code: int
    waived: bool
    waiver_reason: str

    @property
    def passed(self) -> bool:
        return self.exit_code == 0

    @property
    def blocking(self) -> bool:
        return not self.passed and not self.waived


@dataclass(frozen=True)
class PreflightVerdict:
    cleared: bool
    results: tuple[GateResult, ...]
    blocking: tuple[str, ...]
    waived: tuple[str, ...]
    failures: tuple[str, ...]
    unused_waivers: tuple[str, ...] = ()


def aggregate(
    results: tuple[GateResult, ...], waivers: dict[str, str] | None = None
) -> PreflightVerdict:
    """Pure verdict. A waiver must name a real gate and carry a non-empty reason."""
    waivers = dict(waivers or {})
    unknown = sorted(set(waivers) - set(GATE_ORDER))
    empty = sorted(name for name, reason in waivers.items() if not reason.strip())

    failures: list[str] = []
    if unknown:
        failures.append(f"waiver names unknown gate(s): {unknown}")
    if empty:
        failures.append(f"waiver without a reason: {empty}")

    resolved: list[GateResult] = []
    for result in results:
        reason = waivers.get(result.name, "")
        resolved.append(
            GateResult(
                name=result.name,
                exit_code=result.exit_code,
                waived=bool(reason.strip()) and result.name in waivers,
                waiver_reason=reason,
            )
        )

    seen = [r.name for r in resolved]
    missing = [name for name in GATE_ORDER if name not in seen]
    if missing:
        failures.append(f"gate(s) did not run: {missing}")

    blocking = tuple(r.name for r in resolved if r.blocking)
    waived = tuple(r.name for r in resolved if r.waived and not r.passed)
    # A waiver whose gate now passes is stale. Surface it so waivers do not accumulate
    # and silently mask a future failure of the same gate.
    unused = tuple(
        r.name for r in resolved if r.name in waivers and waivers[r.name].strip() and r.passed
    )
    for result in resolved:
        if result.blocking:
            failures.append(f"{result.name} FAILED (exit {result.exit_code}) and is not waived")
    return PreflightVerdict(
        cleared=not failures and not blocking,
        results=tuple(resolved),
        blocking=blocking,
        waived=waived,
        failures=tuple(failures),
        unused_waivers=unused,
    )


def _run(argv_prefix: list[str]) -> int:
    """Run one gate script in-process and return its exit code."""
    import runpy

    script = argv_prefix[0]
    argv = argv_prefix[1:]
    saved = sys.argv
    sys.argv = [script, *argv]
    try:
        runpy.run_path(str(Path(__file__).resolve().parent / script), run_name="__main__")
    except SystemExit as exit_request:  # the gate scripts raise SystemExit(main())
        code = exit_request.code
        return int(code) if isinstance(code, int) else (0 if code is None else 1)
    finally:
        sys.argv = saved
    return 0


def run_gates(
    *,
    years: str = "2014..2025",
    screen_windows: tuple[tuple[str, str], ...] = DEFAULT_SCREEN_WINDOWS,
    runner: Callable[[list[str]], int] = _run,
) -> tuple[GateResult, ...]:
    """Execute the five gates in order, stopping early only on a data-completeness failure."""
    results: list[GateResult] = []

    code = runner(["verify_factor_daily_backfill.py", "--years", years])
    results.append(GateResult("backfill_completeness", code, False, ""))
    if code != 0:
        # Later gates would report on incomplete data and mislead; stop here.
        for name in GATE_ORDER[1:]:
            results.append(GateResult(name, 1, False, ""))
        return tuple(results)

    code = runner(["verify_admission_consistency.py"])
    results.append(GateResult("admission_consistency", code, False, ""))

    worst = 0
    for start, end in screen_windows:
        worst = max(worst, runner([
            "screen_factor_daily_coverage.py", "--start", start, "--end", end,
        ]))
    results.append(GateResult("factor_coverage_screen", worst, False, ""))

    code = runner(["verify_frozen_fold_coverage.py"])
    results.append(GateResult("fold_coverage", code, False, ""))

    code = runner(["verify_frozen_report_path.py"])
    results.append(GateResult("report_path", code, False, ""))
    return tuple(results)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--waive", action="append", default=[], choices=list(GATE_ORDER),
        help="gate to waive despite failure (repeatable); requires --waiver-reason",
    )
    parser.add_argument(
        "--waiver-reason", default="",
        help="decision reference justifying every --waive (recorded in the output)",
    )
    parser.add_argument("--years", default="2014..2025")
    args = parser.parse_args(argv)

    waivers = {name: args.waiver_reason for name in args.waive}
    if args.waive and not args.waiver_reason.strip():
        print("--waive requires a non-empty --waiver-reason naming the decision")
        return 2

    print(f"running {len(GATE_ORDER)} gates in order: {', '.join(GATE_ORDER)}")
    results = run_gates(years=args.years)
    verdict = aggregate(results, waivers)

    print()
    for result in verdict.results:
        state = "PASS" if result.passed else ("WAIVED" if result.waived else "FAIL")
        note = f" ({result.waiver_reason})" if result.waived and not result.passed else ""
        print(f"  {result.name:26} exit={result.exit_code} {state}{note}")
    for failure in verdict.failures:
        print(f"  BLOCKER: {failure}")
    for name in verdict.unused_waivers:
        print(f"  NOTE {name} passed; its waiver is stale and should be removed")
    print(f"blocking: {list(verdict.blocking) or 'none'}; waived: {list(verdict.waived) or 'none'}")
    print("FROZEN BASELINE PRE-FLIGHT:", "CLEARED" if verdict.cleared else "REFUSED")
    if not verdict.cleared:
        print(
            "Task 3.1 must not start: its output becomes the trust root that Task 4.1 "
            "points the consistency verifier at and Task 5.1 consumes as --baseline-report."
        )
    return 0 if verdict.cleared else 1


if __name__ == "__main__":
    raise SystemExit(main())
