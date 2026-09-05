"""Verify the baseline OOS report satisfies the candidate-training contract.

Fail-closed schema gate run after a baseline walk-forward re-run and before
any M1/M2/M3 full candidate run: the extended selection fields
(``invalid_day_count``, ``unselected_reasons``), the G1-G6 gate block, the
data-version snapshot, and every fold-matched baseline prediction artifact
must be present. With the basic schema intact, a deep round-trip through the
candidate driver's ``reconstruct_baseline_record`` is attempted.

Usage:
  verify_baseline_report_candidate_schema.py \
      --report /data/zhangyudong/qrv2_research_artifacts/baseline_oos_report.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

EXPECTED_FOLDS: tuple[str, ...] = ("F01", "F02", "F03", "F04", "F05", "F06")
EXPECTED_GATES: tuple[str, ...] = ("G1", "G2", "G3", "G4", "G5", "G6")
SCORE_FIELDS: tuple[str, ...] = (
    "baseline_id",
    "mean_rank_ic",
    "valid_day_count",
    "invalid_day_count",
)


def check_report(report: dict[str, Any]) -> list[str]:
    """Return human-readable schema problems (empty list == contract satisfied)."""
    problems: list[str] = []
    for key in ("data_versions", "baseline_oos_artifacts", "oos", "selections"):
        if key not in report:
            problems.append(f"missing top-level key: {key}")

    oos = report.get("oos")
    if isinstance(oos, dict):
        if set(oos.get("gates", {})) != set(EXPECTED_GATES):
            problems.append("oos.gates does not contain exactly G1..G6")
        if "all_passed" not in oos:
            problems.append("oos.all_passed missing")

    selections = cast(list[Any], report.get("selections", []))
    fold_ids = sorted(str(entry.get("fold_id")) for entry in selections)
    if fold_ids != list(EXPECTED_FOLDS):
        problems.append(f"selections do not cover exactly F01..F06 (got {fold_ids})")

    for entry_raw in selections:
        entry = cast(dict[str, Any], entry_raw)
        fold_id = str(entry.get("fold_id"))
        if "unselected_reasons" not in entry:
            problems.append(f"{fold_id}: missing unselected_reasons")
        for score_raw in cast(list[Any], entry.get("validation_scores", [])):
            score = cast(dict[str, Any], score_raw)
            for field in SCORE_FIELDS:
                if field not in score:
                    problems.append(f"{fold_id}: validation_scores missing {field}")

    artifacts = cast(dict[str, Any], report.get("baseline_oos_artifacts", {}))
    for fold_id in EXPECTED_FOLDS:
        target = artifacts.get(fold_id)
        if not target or not Path(str(target)).exists():
            problems.append(f"{fold_id}: baseline OOS artifact missing on disk")
    return problems


def reconstruct_round_trip(report: dict[str, Any]) -> list[str]:
    """Deep check: every selection entry must survive the driver reconstruction."""
    driver_path = Path(__file__).resolve().parent / "run_real_candidate_training.py"
    sys.path.insert(0, str(driver_path.parent))
    spec = importlib.util.spec_from_file_location("run_real_candidate_training", driver_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("run_real_candidate_training", module)
    spec.loader.exec_module(module)
    problems: list[str] = []
    for entry_raw in cast(list[Any], report.get("selections", [])):
        entry = cast(dict[str, Any], entry_raw)
        fold_id = str(entry.get("fold_id"))
        try:
            module.reconstruct_baseline_record(entry)
        except Exception as exc:  # noqa: BLE001 - fail closed on any reconstruction error
            problems.append(f"{fold_id}: reconstruct_baseline_record failed: {exc}")
    return problems


def run(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--skip-deep-check",
        action="store_true",
        help="skip the driver reconstruction round-trip (schema-only gate)",
    )
    args = parser.parse_args(list(argv))
    report = cast(dict[str, Any], json.loads(args.report.read_text(encoding="utf-8")))
    problems = check_report(report)
    if not problems and not args.skip_deep_check:
        problems.extend(reconstruct_round_trip(report))
    if problems:
        print(f"SCHEMA FAIL ({len(problems)} problems) for {args.report}")
        for item in problems:
            print(" -", item)
        return 1
    mode_note = " (schema-only)" if args.skip_deep_check else ""
    print(f"SCHEMA PASS{mode_note} for {args.report}")
    return 0


def main() -> None:
    raise SystemExit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
