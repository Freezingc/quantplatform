"""Prove the frozen report can be written before paying for the run that writes it.

Task 3.1 is a roughly fifteen-hour walk-forward whose report is serialised at the very
end. That report now carries two blocks -- ``environment`` and ``assembly_coverage`` --
which had never been exercised on the real frozen path. A defect in either would surface
after the whole run had already been paid for, and ``json.dumps`` is exactly the kind of
place where that happens: numpy scalars from a real frame raise ``TypeError`` where a
hand-built fixture does not, and a zero-row segment makes ``survival_ratio`` NaN, which
Python's json happily emits as the invalid literal ``NaN`` and would corrupt the archived
trust root.

So this verifier assembles real frozen segments through the same extractor the runner
uses, measures them with the same library function, builds the environment block by
calling the runner's own ``describe_run_environment``, and fails closed unless the result
both round-trips through JSON and covers every requested (fold, role).

It defaults to the two cheap segments. Pass ``--roles train,validation,test`` for the
full check, which also exercises the multi-million-row train frame.

Usage:
  verify_frozen_report_path.py                              # F01 validation + test
  verify_frozen_report_path.py --roles train,validation,test
  verify_frozen_report_path.py --fold F04 --roles test
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REQUIRED_ROLES: tuple[str, ...] = ("train", "validation", "test")
DEFAULT_ROLES: tuple[str, ...] = ("validation", "test")


def report_block_problems(
    environment: Mapping[str, Any], coverage: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    """Fail-closed check that both report blocks can actually be archived.

    ``allow_nan=False`` matters: the default would emit ``NaN`` for a zero-row segment's
    survival ratio, producing a file that is not valid JSON and cannot be compared
    byte-for-byte against a reference later.
    """
    problems: list[str] = []
    if not environment.get("factor_ids"):
        problems.append(
            "environment.factor_ids is empty; the report would not say which factor "
            "set produced it"
        )
    for name, block in (("environment", environment), ("assembly_coverage", coverage)):
        try:
            json.dumps(block, allow_nan=False)
        except (TypeError, ValueError) as error:
            problems.append(f"{name} is not JSON serialisable: {error}")
    for fold_id, roles in coverage.items():
        for role, summary in roles.items():
            ratio = summary.get("survival_ratio")
            if ratio is None:
                problems.append(f"{fold_id}/{role}: survival_ratio is absent")
            elif not math.isfinite(float(ratio)):
                problems.append(f"{fold_id}/{role}: survival_ratio is not finite ({ratio})")
    return problems


def coverage_completeness_problems(
    coverage: Mapping[str, Mapping[str, Any]],
    fold_ids: Sequence[str],
    roles: Sequence[str] = REQUIRED_ROLES,
) -> list[str]:
    """Every requested (fold, role) must have been measured, not silently skipped."""
    problems: list[str] = []
    for fold_id in fold_ids:
        present = coverage.get(fold_id)
        if present is None:
            problems.append(f"{fold_id}: no coverage recorded")
            continue
        for role in roles:
            if role not in present:
                problems.append(f"{fold_id}: missing coverage for the {role} segment")
    return problems


def collect_fold_coverage(
    holder: Any,
    *,
    fold_id: str,
    roles: Sequence[str],
    factor_source: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], tuple[str, ...]]:
    """Assemble one fold's requested segments and summarise their coverage.

    Returns ``(environment, coverage, factor_ids)``. This is the seam that makes the rest of
    the verifier testable: everything above it -- the two report blocks, the completeness
    rule, the serialisation and the exit code -- is pure or near-pure, and everything below
    it needs a database, a resolver and the real extractor. ``verify_factor_daily_backfill``
    draws the same line around ``collect_year_state`` for the same reason; before this split
    the work was inline in ``main()``, which imported eight collaborators inside the function
    body and left nothing for a test to replace.

    The prints stay here rather than moving to the caller because they report per-segment
    measurements as each expensive frame is assembled and released, and holding all of them
    until the end would keep every frame alive at once.
    """
    import run_real_walk_forward as driver
    from walk_forward_common import (
        current_versions,
        factor_members_for,
        load_calendar,
    )
    from walk_forward_factor_daily import extract_fn_for, factor_source_plan

    from research_v2.datasets.coverage import summarize_frame_coverage
    from research_v2.datasets.splits import SplitResolver

    resolver = SplitResolver(load_calendar(holder))
    factor_ids, version_tables = factor_source_plan(factor_source)
    factor_members = factor_members_for(factor_ids)
    versions = {table: current_versions(holder, table) for table in version_tables}
    extract = extract_fn_for(factor_source)
    feature_order = tuple(member.feature_name for member in factor_members)
    print(
        f"factor source: {factor_source} ({len(factor_ids)} factors); "
        f"fold {fold_id}; roles {list(roles)}",
        flush=True,
    )

    coverage: dict[str, dict[str, Any]] = {fold_id: {}}
    for role in roles:
        frame = extract(holder, resolver, fold_id, role, factor_members, versions)
        summary = summarize_frame_coverage(frame, feature_order)
        coverage[fold_id][role] = summary.to_json()
        print(
            f"  {fold_id}/{role:11} rows={summary.rows:>9} complete={summary.complete_rows:>9} "
            f"survival={summary.survival_ratio:.4f} days={summary.days:>5} "
            f"<30={summary.days_below_floor:>3} empty={summary.empty_days:>2} "
            f"median={summary.median_width:>5} min={summary.min_width:>5}",
            flush=True,
        )
        del frame

    environment = driver.describe_run_environment(factor_source, factor_ids)
    return environment, coverage, factor_ids


def build_report(
    environment: Mapping[str, Any],
    coverage: Mapping[str, Mapping[str, Any]],
    *,
    fold_id: str,
    roles: Sequence[str],
) -> tuple[list[str], str]:
    """Return every problem with the report, plus the text that would be archived.

    Pure: no database, no file writes, no printing. Both halves of the verdict live here --
    the problems that make the door fail and the serialised payload whose length the door
    reports -- so a test can exercise the archived shape directly instead of only through a
    fifteen-hour run.

    ``allow_nan=False`` is the reason this function exists separately from
    :func:`report_block_problems`. That helper checks each block on its own; this serialises
    the *combined* payload exactly as the runner archives it, which is the artefact that has
    to round-trip. A zero-row segment makes ``survival_ratio`` NaN and Python's json emits
    the bare token ``NaN`` by default, which is not valid JSON and cannot be compared
    byte-for-byte against a reference later.
    """
    problems = report_block_problems(environment, coverage)
    problems += coverage_completeness_problems(coverage, (fold_id,), roles)

    # Serialise the way the runner does, so the archived shape itself is what is tested.
    payload = {"environment": environment, "assembly_coverage": {
        "minimum_cross_section": 30,
        "measured_on": (
            "assembled frames, after the universe and label joins and before the "
            "dropna the baselines train through"
        ),
        "folds": coverage,
    }}
    try:
        text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    except (TypeError, ValueError) as error:
        problems.append(f"the combined report payload is not JSON serialisable: {error}")
        text = ""
    return problems, text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", default="F01")
    parser.add_argument(
        "--roles", default=",".join(DEFAULT_ROLES),
        help="comma-separated segments to assemble and measure; train is the expensive one",
    )
    parser.add_argument("--factor-source", default="frozen")
    args = parser.parse_args(argv)

    roles = tuple(role.strip() for role in args.roles.split(",") if role.strip())
    unknown = sorted(set(roles) - set(REQUIRED_ROLES))
    if unknown:
        raise SystemExit(f"unknown roles: {unknown}; choose from {list(REQUIRED_ROLES)}")
    if not roles:
        raise SystemExit("no roles requested")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from walk_forward_common import new_holder

    fold_id = args.fold.strip().upper()
    holder = new_holder()
    environment, coverage, _factor_ids = collect_fold_coverage(
        holder, fold_id=fold_id, roles=roles, factor_source=args.factor_source
    )
    problems, text = build_report(environment, coverage, fold_id=fold_id, roles=roles)

    for problem in problems:
        print(f"  PROBLEM: {problem}")
    if not problems:
        print(
            f"  report path OK: {len(text)} bytes of serialisable environment + "
            f"assembly_coverage for {fold_id} {list(roles)}"
        )
    print("FROZEN REPORT PATH:", "FAIL" if problems else "PASS")
    return 1 if problems else 0

if __name__ == "__main__":
    raise SystemExit(main())
