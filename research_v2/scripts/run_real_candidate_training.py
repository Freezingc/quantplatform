"""Real-data candidate model training driver (M1 -> M2 -> M3).

Task 2 skeleton: enforces the E-plan execution order and the fold-matched
baseline artifact contract; dataset assembly reuses walk_forward_common so
candidates consume exactly the same PIT datasets as the B0-B3 baselines.
Training/search wiring lands in Tasks 3-4 of
docs/superpowers/plans/2026-09-04-real-candidate-model-training-plan.md.

Usage:
  run_real_candidate_training.py --family M1 --baseline-artifacts <dir> --report <path>
  run_real_candidate_training.py --family M1 --smoke --report <path>
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

ORDINARY_FOLDS: tuple[str, ...] = ("F01", "F02", "F03", "F04", "F05", "F06")
MODEL_FAMILIES: tuple[str, ...] = ("M1", "M2", "M3")
FAMILY_ORDER_PREREQUISITES: dict[str, tuple[str, ...]] = {
    "M1": (),
    "M2": ("M1",),
    "M3": ("M1", "M2"),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=list(MODEL_FAMILIES))
    parser.add_argument(
        "--baseline-artifacts",
        type=Path,
        default=None,
        help="directory with fold-matched baseline OOS parquet artifacts (F01..F06)",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--folds",
        default=",".join(ORDINARY_FOLDS),
        help="comma-separated subset of ordinary folds (default all six)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def load_baseline_predictions(directory: Path) -> dict[str, pd.DataFrame]:
    """Load every ordinary fold's winner-baseline TEST predictions, fail-closed."""
    frames: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for fold_id in ORDINARY_FOLDS:
        target = directory / f"{fold_id}.parquet"
        if not target.exists():
            missing.append(fold_id)
            continue
        frames[fold_id] = pd.read_parquet(target)
    if missing:
        raise ValueError(
            f"missing fold-matched baseline artifacts for: {', '.join(missing)}"
        )
    return frames


def validate_family_order(
    family: str, prior_families_passed: Mapping[str, object] | tuple[str, ...]
) -> None:
    """Refuse to run a family before its E-plan predecessors are evidenced."""
    if family not in FAMILY_ORDER_PREREQUISITES:
        raise ValueError(f"unknown model family: {family}")
    prerequisites = FAMILY_ORDER_PREREQUISITES[family]
    missing = [prior for prior in prerequisites if prior not in prior_families_passed]
    if missing:
        raise ValueError(
            f"family {family} requires prior passed families {missing} "
            f"(E-plan execution order M1 -> M2 -> M3)"
        )


def main() -> None:
    args = parse_args()
    if not args.smoke:
        if args.baseline_artifacts is None:
            raise SystemExit("--baseline-artifacts is required for full runs")
        load_baseline_predictions(args.baseline_artifacts)
    # Training/search wiring lands in Tasks 3-4 of the candidate plan.
    raise SystemExit(
        "candidate training backend is not wired yet (Tasks 3-4 of "
        "2026-09-04-real-candidate-model-training-plan.md); refusing to run"
    )


if __name__ == "__main__":
    main()
