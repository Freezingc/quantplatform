"""Real-data candidate model training driver (M0, M0R, M1 -> M2 -> M3).

M0 and M0R are two carriers for the same LightGBM candidate slot and are alternatives, not
stages: M0 is the frozen LambdaRank recipe, M0R regresses the per-day return percentile rank.
Neither is a prerequisite for the other, and the E-plan ordering M1 -> M2 -> M3 is unchanged.
M0R exists because the LambdaRank carrier caps out below the G4 bar; see
docs/handoffs/2026-09-08-m0-attribution-and-carrier-findings.md.

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
import contextlib
import hashlib
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal, cast

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from walk_forward_common import (
    current_versions,
    factor_members_for,
    load_calendar,
    new_holder,
)
from walk_forward_factor_daily import (
    FACTOR_SOURCES,
    extract_fn_for,
    factor_source_plan,
)

from research_v2.baselines.lgbm_ranker import LGBMHyperparameters, LGBMRankerBaseline
from research_v2.baselines.lgbm_rankreg import LGBMRankRegBaseline
from research_v2.baselines.orchestration import (
    BaselineSelectionRecord,
    BaselineValidationScore,
)
from research_v2.datasets.manifest import build_split_dataset_manifest
from research_v2.datasets.preprocessing import FoldPreprocessor
from research_v2.datasets.splits import SplitResolver
from research_v2.datasets.storage import DatasetArtifactStore
from research_v2.evaluation.model_evaluator import evaluate_frozen_oos
from research_v2.evaluation.model_walk_forward import (
    MODEL_FAMILIES,
    ModelOosFold,
    evaluate_model_walk_forward,
)
from research_v2.evaluation.rank_ic import compute_daily_rank_ic, summarize_rank_ic
from research_v2.evaluation.selection_evidence import (
    FrozenPredictionStore,
    SelectionEvidenceStore,
)
from research_v2.inference.artifact_loader import (
    load_bundle_for_evaluation,
    model_config_fields,
)
from research_v2.inference.torch_scoring import (
    concat_date_scores as _concat_date_scores,
)
from research_v2.inference.torch_scoring import (  # noqa: F401 - script-module compatibility alias
    mean_seed_percentile_scores as _mean_seed_percentile_scores,
)
from research_v2.inference.torch_scoring import (
    percentile_frame as _percentile_frame,
)
from research_v2.inference.torch_scoring import (
    prepare_frame as _prepare_frame,
)
from research_v2.models.base import ModelSignature
from research_v2.models.residual_mlp import ResidualMLP, ResidualMLPConfig
from research_v2.models.set_attention import SetAttention, SetAttentionConfig
from research_v2.models.temporal import (
    TemporalGRU,
    TemporalGRUConfig,
    TemporalTCN,
    TemporalTCNConfig,
)
from research_v2.registry.promotion_evidence import (
    PROMOTION_EVIDENCE_FILENAME,
    PromotionEvidenceError,
    _prediction_ids_by_fold_and_kind,
    emit_promotion_evidence,
)
from research_v2.training.artifacts import CheckpointStore
from research_v2.training.batches import TemporalTrainingBatch, TrainingBatch
from research_v2.training.cross_section_search import (
    CrossSectionConfigTask,
    CrossSectionSeedTask,
    run_cross_section_config_task,
    run_cross_section_seed_task,
)
from research_v2.training.ensemble import REQUIRED_SEEDS, build_seed_ensemble
from research_v2.training.model_bundle import ModelArtifactBundleStore
from research_v2.training.parallel_search import (
    SearchTask,
    available_memory_gb,
    resident_memory_gb,
    run_parallel_search,
    search_granularity,
    worker_memory_budget,
)
from research_v2.training.runner import run_validation_governed_training
from research_v2.training.search_resume import SearchResume, torch_runtime_bindings
from research_v2.training.seeds import configure_reproducibility, configure_torch_threads
from research_v2.training.static_batches import build_static_training_batches
from research_v2.training.temporal_batches import build_temporal_training_batches
from research_v2.training.temporal_search import (
    HISTORY_LENGTH,
    MINIMUM_VALID_HISTORY_LENGTH,
    TemporalConfigTask,
    TemporalFoldSpec,
    TemporalSeedTask,
    run_temporal_config_task,
    run_temporal_seed_task,
    temporal_test_panel_batches,
)
from research_v2.training.trainer import DatasetRole, DatasetView, TrainerConfig

ORDINARY_FOLDS: tuple[str, ...] = ("F1", "F2", "F3", "F4")
# MODEL_FAMILIES is imported from research_v2.evaluation.model_walk_forward, which is the
# evaluator's admission list. Do not redeclare it here: a family this driver accepts but the
# evaluator does not trains for an hour and then fails at the final gate step.
# A deviation from the E-plan order is a plan-level decision, not a convenience flag.
# The reason has to be long enough to state what was authorized and why, so an empty or
# token string cannot be used to smuggle past the guard.
MINIMUM_DEVIATION_REASON_LENGTH = 24

FAMILY_ORDER_PREREQUISITES: dict[str, tuple[str, ...]] = {
    "M0": (),
    # M0R is an alternative CARRIER for the same LightGBM candidate slot, not a later stage
    # of the E-plan, so like M0 it has no predecessors. M1/M2/M3 ordering is untouched.
    "M0R": (),
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
        help=(
            "directory with fold-matched baseline OOS parquet artifacts, one <fold>.parquet "
            "per ordinary fold (F1..F4 under the current contract)"
        ),
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--folds",
        default=",".join(ORDINARY_FOLDS),
        help=(
            "comma-separated subset of ordinary folds "
            "(default: every ordinary fold the contract defines)"
        ),
    )
    parser.add_argument("--baseline-report", type=Path, default=None)
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument(
        "--grid",
        choices=("base", "refined"),
        default="base",
        help=(
            "M0 config grid: base (24-capacity) or refined (24 with lambda_l2 sweep). "
            "Ignored by M0R, whose grid is pinned to the single config its +0.009149 was "
            "measured at, so that number carries no validation-selection bias"
        ),
    )
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help=(
            "maximum spawned workers for M1/M3 validation config search; "
            "defaults to one, which preserves the legacy serial path"
        ),
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=None,
        help=(
            "explicit Torch intra-op and inter-op thread count for a new execution lineage; "
            "required when --parallel-workers exceeds one"
        ),
    )
    parser.add_argument(
        "--worker-memory-gb",
        type=float,
        default=None,
        help=(
            "gibibytes each spawned worker is expected to hold, checked against MemAvailable "
            "before the run starts. A temporal worker holds the fold's window panels and a static "
            "one the batches it was handed, so this is a property of the family and the fold, not "
            "a guess: measure it once and pass it, or a run that does not fit is killed by the OOM "
            "picker hours in instead of refused at startup"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "adopt a fold's persisted per-config search records instead of retraining them. "
            "Records are written by every full run; adopting them is opt-in, and a record is "
            "refused (and the config trained) unless its datasets, factor order, effective "
            "budget, family, code version, config and runtime all still match this run"
        ),
    )
    parser.add_argument(
        "--factor-source",
        choices=FACTOR_SOURCES,
        default="legacy",
        help=(
            "legacy: factor_observation_clean + the 14 walk-forward ids "
            "(bit-identical to the criterion-303 contract); frozen: factor_daily "
            "+ FROZEN_FACTOR_SET_V1. Must match the baseline report's source."
        ),
    )
    parser.add_argument(
        "--prior-report",
        action="append",
        default=[],
        help=(
            "FAMILY=PATH evidence that a prior family passed full walk-forward "
            "(repeatable; enforces the E-plan execution order)"
        ),
    )
    parser.add_argument(
        "--prior-report-failed",
        action="append",
        default=[],
        help=(
            "FAMILY=PATH evidence that a prerequisite family ran full walk-forward and "
            "did NOT pass all gates (repeatable). Only meaningful together with "
            "--family-order-deviation; the report's identity and gate statuses are "
            "recorded in the run report so the deviation is auditable."
        ),
    )
    parser.add_argument(
        "--family-order-deviation",
        default=None,
        help=(
            "Owner-authorized reason for running a family whose E-plan prerequisites "
            "did not all pass. Refused when empty or shorter than "
            f"{MINIMUM_DEVIATION_REASON_LENGTH} characters, and refused unless every "
            "missing prerequisite is also supplied via --prior-report-failed."
        ),
    )
    parser.add_argument(
        "--family-order-deviation-owner",
        default=None,
        help="Name of the owner authorizing --family-order-deviation (required with it).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


@dataclass(frozen=True)
class _ParallelExecution:
    workers: int
    torch_threads: int | None


def _parallel_execution(args: argparse.Namespace) -> _ParallelExecution:
    """Validate explicit parallel search settings without perturbing legacy runs."""
    workers = args.parallel_workers
    threads = args.torch_threads
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise SystemExit("parallel-workers must be a positive integer")
    if threads is not None and (
        isinstance(threads, bool) or not isinstance(threads, int) or threads < 1
    ):
        raise SystemExit("torch-threads must be a positive integer")
    if workers > 1 and threads is None:
        raise SystemExit("parallel-workers above one requires explicit --torch-threads")
    if workers > 1 and args.family not in {"M1", "M2", "M3"}:
        raise SystemExit(
            "parallel-workers is supported only for the M1, M2 and M3 torch config searches; "
            "the LightGBM carriers fit in minutes and already pin their own thread count"
        )
    return _ParallelExecution(workers=workers, torch_threads=threads)


def _check_worker_memory(args: argparse.Namespace, execution: _ParallelExecution) -> None:
    """Refuse a worker count that cannot fit in this host's memory, before hours are spent.

    The failure being prevented is the expensive kind. A worker that does not fit is not refused at
    startup, it is picked off by the kernel's OOM killer hours into a multi-day run, and the
    operator is left to restart it with --resume. Declaring the footprint turns that into an
    arithmetic check at second zero, and the refusal states the largest worker count that does fit
    because that is always the next question.

    Undeclared is allowed -- this is a new precondition and an existing invocation must not start
    failing -- but then the margin is printed rather than checked, so the number is at least in the
    transcript before the run commits to it.
    """
    if execution.workers <= 1:
        return
    available = available_memory_gb()
    declared = getattr(args, "worker_memory_gb", None)
    # The parent is counted too, and measured rather than assumed: in a temporal run it holds the
    # prepared train, validation and test frames and all three batch sets, so it can outweigh a
    # worker. Checking once at startup would measure it before any of that was loaded, which is why
    # the torch loops call this again per fold, after assembly.
    parent_gb = resident_memory_gb() or 0.0
    if declared is None:
        if available is not None:
            print(
                f"parallel workers={execution.workers} with no declared footprint: "
                f"MemAvailable={available:.1f} GiB, this parent holds {parent_gb:.1f} GiB, i.e. "
                f"{max(0.0, available - parent_gb) / execution.workers:.1f} GiB per worker. "
                "Pass --worker-memory-gb to make that a checked precondition.",
                flush=True,
            )
        return
    if available is None:
        raise SystemExit(
            "--worker-memory-gb was given but MemAvailable could not be read, so the request "
            "cannot be checked; drop the flag to run without the check"
        )
    budget = worker_memory_budget(
        workers=execution.workers,
        per_worker_gb=float(declared),
        available_gb=available,
        parent_gb=parent_gb,
    )
    if not budget.fits:
        raise SystemExit(
            f"{budget.workers} workers x {budget.per_worker_gb:.1f} GiB, plus this parent's "
            f"{budget.parent_gb:.1f} GiB, is {budget.required_gb:.1f} GiB, but MemAvailable is "
            f"{budget.available_gb:.1f} GiB and the run may commit 90% of it. The largest safe "
            f"worker count is "
            f"{budget.safe_workers}."
        )
    print(
        f"worker memory budget: {budget.workers} x {budget.per_worker_gb:.1f} GiB + parent "
        f"{budget.parent_gb:.1f} GiB = {budget.required_gb:.1f} GiB of "
        f"{budget.available_gb:.1f} GiB available",
        flush=True,
    )


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


def missing_prerequisites(
    family: str, prior_families_passed: Mapping[str, object] | tuple[str, ...]
) -> list[str]:
    """Return the E-plan prerequisites of ``family`` that have no passed evidence."""
    if family not in FAMILY_ORDER_PREREQUISITES:
        raise ValueError(f"unknown model family: {family}")
    return [
        prior
        for prior in FAMILY_ORDER_PREREQUISITES[family]
        if prior not in prior_families_passed
    ]


def validate_family_order(
    family: str,
    prior_families_passed: Mapping[str, object] | tuple[str, ...],
    *,
    acknowledged_failed_priors: Sequence[Mapping[str, object]] = (),
    deviation_reason: str | None = None,
    deviation_owner: str | None = None,
) -> None:
    """Refuse to run a family before its E-plan predecessors are evidenced.

    The guard is fail-closed by default. It can be deviated from only with an explicit,
    recorded owner authorization: a reason of at least
    ``MINIMUM_DEVIATION_REASON_LENGTH`` characters, an owner name, and -- for every
    missing prerequisite -- the actual full-mode report that failed, supplied through
    ``acknowledged_failed_priors``. A deviation that is not needed is refused, so the
    audit record can never claim an authorization the run did not depend on.
    """
    missing = missing_prerequisites(family, prior_families_passed)
    reason = (deviation_reason or "").strip()
    owner = (deviation_owner or "").strip()
    if not missing:
        if reason or owner or acknowledged_failed_priors:
            raise ValueError(
                f"family {family} needs no order deviation; drop --family-order-deviation, "
                "--family-order-deviation-owner and --prior-report-failed"
            )
        return
    if not reason:
        raise ValueError(_order_violation_message(family, missing))
    if len(reason) < MINIMUM_DEVIATION_REASON_LENGTH:
        raise ValueError(
            f"--family-order-deviation must state the authorization in at least "
            f"{MINIMUM_DEVIATION_REASON_LENGTH} characters, got {len(reason)}: {reason!r}"
        )
    if not owner:
        raise ValueError(
            "--family-order-deviation-owner is required to record who authorized "
            f"deviating from the E-plan order for {family}"
        )
    acknowledged = {str(entry.get("family")) for entry in acknowledged_failed_priors}
    unevidenced = [prior for prior in missing if prior not in acknowledged]
    if unevidenced:
        raise ValueError(
            f"{_order_violation_message(family, missing)}; a deviation additionally requires "
            f"the failing full-mode report for {unevidenced} via --prior-report-failed "
            "FAMILY=PATH, so the deviation is auditable"
        )


def _order_violation_message(family: str, missing: Sequence[str]) -> str:
    return (
        f"family {family} requires prior passed families {list(missing)} "
        f"(E-plan execution order M1 -> M2 -> M3)"
    )


def prior_families_from_failed_reports(
    entries: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    """Resolve acknowledged-but-failed prerequisite reports, fail-closed in both directions.

    Each entry must be ``FAMILY=PATH`` pointing at a full-mode run of FAMILY whose
    ``oos.all_passed`` is NOT true. A report that did pass is refused here: it belongs in
    ``--prior-report``, and accepting it as "failed" would let a caller hide which
    prerequisite was actually missing.
    """
    records: list[dict[str, Any]] = []
    for entry in entries:
        family, separator, raw_path = entry.partition("=")
        if not separator or not raw_path:
            raise SystemExit(f"--prior-report-failed expects FAMILY=PATH, got: {entry}")
        family = family.strip().upper()
        if family not in MODEL_FAMILIES:
            raise SystemExit(f"--prior-report-failed references unknown family: {family}")
        report_path = Path(raw_path)
        if not report_path.exists():
            raise SystemExit(f"--prior-report-failed path does not exist: {report_path}")
        raw_bytes = report_path.read_bytes()
        report = json.loads(raw_bytes.decode("utf-8"))
        if report.get("family") != family:
            raise SystemExit(
                f"--prior-report-failed family mismatch for {family}: {report_path}"
            )
        if report.get("mode") != "full":
            raise SystemExit(
                f"--prior-report-failed for {family} is not a full run: {report_path}"
            )
        oos = report.get("oos")
        if isinstance(oos, dict) and oos.get("all_passed") is True:
            raise SystemExit(
                f"--prior-report-failed for {family} actually passed all gates; supply it "
                f"via --prior-report instead: {report_path}"
            )
        gates = oos.get("gates") if isinstance(oos, dict) else None
        records.append(
            {
                "family": family,
                "report_path": str(report_path),
                "report_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "all_passed": bool(isinstance(oos, dict) and oos.get("all_passed") is True),
                "failed_gates": sorted(
                    str(name)
                    for name, result in (gates or {}).items()
                    if isinstance(result, dict) and result.get("status") != "PASS"
                ),
                "observed_gates": {
                    str(name): result.get("observed_value")
                    for name, result in sorted((gates or {}).items())
                    if isinstance(result, dict)
                },
            }
        )
    return tuple(records)


def family_order_deviation_record(
    family: str,
    prior_families_passed: tuple[str, ...],
    acknowledged_failed_priors: Sequence[Mapping[str, object]],
    reason: str,
    owner: str,
) -> dict[str, Any]:
    """Render the audit block persisted into the run report for an order deviation."""
    return {
        "family": family,
        "deviated_from_e_plan_order": True,
        "reason": reason.strip(),
        "authorized_by": owner.strip(),
        "prerequisites": list(FAMILY_ORDER_PREREQUISITES[family]),
        "prerequisites_with_passed_evidence": list(prior_families_passed),
        "prerequisites_acknowledged_as_failed": [
            str(entry.get("family")) for entry in acknowledged_failed_priors
        ],
        "failed_prior_reports": [dict(entry) for entry in acknowledged_failed_priors],
    }


def prior_families_from_reports(entries: Sequence[str]) -> tuple[str, ...]:
    """Resolve passed prior families from full-run reports, fail-closed.

    Every entry must be ``FAMILY=PATH`` where the report at PATH is a
    full-mode run of FAMILY with ``oos.all_passed`` true; anything else is
    refused so a later family can never start on unverified evidence.
    """
    passed: list[str] = []
    for entry in entries:
        family, separator, raw_path = entry.partition("=")
        if not separator or not raw_path:
            raise SystemExit(f"--prior-report expects FAMILY=PATH, got: {entry}")
        family = family.strip().upper()
        if family not in MODEL_FAMILIES:
            raise SystemExit(f"--prior-report references unknown family: {family}")
        report_path = Path(raw_path)
        if not report_path.exists():
            raise SystemExit(f"--prior-report path does not exist: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("family") != family:
            raise SystemExit(
                f"--prior-report family mismatch for {family}: {report_path}"
            )
        if report.get("mode") != "full":
            raise SystemExit(f"--prior-report for {family} is not a full run: {report_path}")
        oos = report.get("oos")
        if not isinstance(oos, dict) or oos.get("all_passed") is not True:
            raise SystemExit(
                f"--prior-report for {family} did not pass all walk-forward gates: {report_path}"
            )
        passed.append(family)
    return tuple(passed)

SMOKE_FACTOR_COUNT = 15
SMOKE_STOCK_COUNT = 40
SMOKE_TRAIN_DAYS = (date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4))
SMOKE_VALIDATION_DAY = date(2024, 1, 5)
SMOKE_TEST_DAYS = (
    date(2024, 1, 8),
    date(2024, 1, 9),
    date(2024, 1, 10),
    date(2024, 1, 11),
    date(2024, 1, 12),
    date(2024, 1, 15),
    date(2024, 1, 16),
    date(2024, 1, 17),
)


def _smoke_codes() -> tuple[str, ...]:
    return tuple(f"s{index:02d}" for index in range(SMOKE_STOCK_COUNT))


def build_smoke_frames(factor_count: int) -> dict[str, Any]:
    """Synthetic frames for the M1 smoke run (orchestration evidence only)."""
    alpha = np.random.default_rng(21).standard_normal(SMOKE_STOCK_COUNT)
    noise = np.random.default_rng(22)

    def _frame(days: Sequence[date], with_target: bool) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for day in days:
            values = noise.standard_normal((SMOKE_STOCK_COUNT, factor_count))
            values[:, 0] = alpha
            targets = alpha + noise.standard_normal(SMOKE_STOCK_COUNT) * 0.5
            for index, code in enumerate(_smoke_codes()):
                row: dict[str, object] = {
                    "signal_date": day,
                    "ts_code": code,
                    "label_valid": True,
                }
                if with_target:
                    row["raw_return_5d"] = float(targets[index])
                for factor_index in range(factor_count):
                    row[f"f{factor_index:02d}"] = float(values[index, factor_index])
                rows.append(row)
        return pd.DataFrame(rows)

    train_frame = _frame(SMOKE_TRAIN_DAYS, with_target=True)
    validation_frame = _frame([SMOKE_VALIDATION_DAY], with_target=True)
    test_frame = _frame(SMOKE_TEST_DAYS, with_target=False)

    baseline_rows = []
    for day in SMOKE_TEST_DAYS:
        for index, code in enumerate(_smoke_codes()):
            baseline_rows.append(
                {"signal_date": day, "ts_code": code, "score": float(alpha[index])}
            )
    return {
        "train": train_frame,
        "validation": validation_frame,
        "test": test_frame,
        "baseline_predictions": pd.DataFrame(baseline_rows),
        "labels": test_frame.loc[:, ["signal_date", "ts_code"]].assign(
            raw_return_5d=np.random.default_rng(24)
            .standard_normal(len(test_frame))
            + np.tile(alpha, len(SMOKE_TEST_DAYS)),
            label_valid=True,
        ),
    }


def _validation_rank_ic(predictions: pd.DataFrame, labels: pd.DataFrame) -> float:
    merged = predictions.merge(
        labels.loc[:, ["signal_date", "ts_code", "raw_return_5d", "label_valid"]],
        on=["signal_date", "ts_code"],
        how="inner",
    )
    daily = compute_daily_rank_ic(
        merged.loc[:, ["signal_date", "ts_code", "percentile_rank"]].rename(
            columns={"percentile_rank": "score"}
        ),
        merged.loc[:, ["signal_date", "ts_code", "raw_return_5d", "label_valid"]],
        minimum_cross_section=30,
    )
    return float(summarize_rank_ic(daily).mean_rank_ic)


def _preprocessing_artifact(preprocessor: FoldPreprocessor) -> dict[str, object]:
    """Persist the fitted preprocessing state beside its hash, not the hash alone.

    Recording only ``state_hash`` -- which is what these four bundle writes did -- left a torch
    bundle that could not be served: putting a new frame into the space the weights were trained
    in needs the fitted winsor bounds, median and IQR per factor, and a hash cannot be inverted
    into them. The loader restores from this payload and re-derives the hash from what it
    rebuilt, so the two cannot drift apart. The LightGBM bundles are unaffected and keep
    ``{"state_hash": "identity"}``, because a tree ensemble is invariant to the monotone
    per-feature transforms and genuinely has no state.
    """
    return {
        "state_hash": preprocessor.state_hash,
        "state_payload": preprocessor.state_payload(),
    }


def _predict_validation_percentiles(
    model: torch.nn.Module, validation_batches: Mapping[date, TrainingBatch]
) -> pd.DataFrame:
    return _percentile_frame(model, validation_batches)


def _train_cross_section_smoke(
    workspace: Path,
    *,
    family: str,
    family_name: str,
    model_code_version: str,
    config_factory: Any,
    model_factory: Any,
) -> dict[str, Any]:
    """Shared end-to-end cross-section orchestration for M1/M3 smoke runs."""
    factor_count = SMOKE_FACTOR_COUNT
    frames = build_smoke_frames(factor_count)
    factor_order = tuple(f"f{index:02d}" for index in range(factor_count))
    preprocessor = FoldPreprocessor(factor_order=factor_order)
    preprocessor.fit(frames["train"].loc[:, list(factor_order)], role="TRAIN")
    train_batches = build_static_training_batches(
        _prepare_frame(frames["train"], preprocessor, factor_order), factor_order=factor_order
    )
    validation_batches = build_static_training_batches(
        _prepare_frame(frames["validation"], preprocessor, factor_order),
        factor_order=factor_order,
    )
    config = config_factory()
    config_payload = asdict(config)
    checkpoint_store = CheckpointStore(workspace / "checkpoints")
    seed_checkpoints: dict[int, Path] = {}
    seed_validation_predictions: dict[int, pd.DataFrame] = {}
    seed_rank_ic: dict[int, float] = {}
    prototype = model_factory()
    for seed in REQUIRED_SEEDS:
        # Seed BEFORE constructing the model. run_validation_governed_training seeds too, but it
        # runs after model_factory(), so the initial weights were being drawn from whatever the
        # global RNG held on entry. For the first seed that is the process's entropy-seeded state,
        # i.e. different on every run; for later seeds it is the state the previous iteration's
        # configure_reproducibility left behind, i.e. reproducible. That asymmetry was measured on
        # the M1 smoke: seeds 20260802 and 20260803 gave identical validation RankIC across four
        # runs (0.828893 / 0.830957) while 20260801 ranged over 0.7289-0.8527. One irreproducible
        # seed out of three is enough to make the ensemble, the config selected on it, and every
        # downstream gate number irreproducible too.
        configure_reproducibility(seed)
        model = model_factory()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
        run = run_validation_governed_training(
            model,
            optimizer,
            train_batches=train_batches,
            validation_batches=validation_batches,
            train_view=DatasetView(role=DatasetRole.TRAIN, dataset_id="smoke-train"),
            validation_view=DatasetView(
                role=DatasetRole.VALIDATION, dataset_id="smoke-validation"
            ),
            trainer_config=TrainerConfig(max_epochs=8, early_stopping_patience=8),
            seed=seed,
            checkpoint_store=checkpoint_store,
            checkpoint_prefix=f"smoke-{family.lower()}-{seed}",
        )
        seed_checkpoints[seed] = (
            workspace / "checkpoints" / run.checkpoint_ids[-1] / "checkpoint.pt"
        )
        seed_validation_predictions[seed] = _predict_validation_percentiles(
            model, validation_batches
        )
        seed_rank_ic[seed] = _validation_rank_ic(
            seed_validation_predictions[seed], frames["validation"]
        )
    ensemble = build_seed_ensemble(seed_validation_predictions, validation_rank_ic=seed_rank_ic)
    bundle_store = ModelArtifactBundleStore(workspace / "models")
    model_id = f"smoke-{family.lower()}"
    bundle_store.write(
        model_id,
        model_card=torch_model_card(
            card_family=family_name,
            dataset_id="smoke-train",
            model_code_version=model_code_version,
        ),
        config=config_payload,
        signature=asdict(prototype.artifact_signature()),
        feature_order=factor_order,
        preprocessing=_preprocessing_artifact(preprocessor),
        seed_checkpoints=seed_checkpoints,
        validation_predictions=ensemble.predictions.rename(
            columns={"ensemble_score": "percentile_rank"}
        ),
    )
    loaded = load_bundle_for_evaluation(
        bundle_store,
        model_id,
        expected_dataset_id="smoke-train",
        expected_feature_order=factor_order,
        expected_preprocessor_hash=preprocessor.state_hash,
        device=torch.device("cpu"),
    )
    test_batches = build_static_training_batches(
        frames["test"].assign(raw_return_5d=0.0), factor_order=factor_order
    )
    candidate_predictions = _concat_date_scores(loaded.models, test_batches)
    evaluation = evaluate_frozen_oos(
        candidate_predictions,
        frames["labels"],
        baseline_predictions=frames["baseline_predictions"],
        role="TEST",
        minimum_cross_section=30,
    )
    return {
        "family": family,
        "mode": "smoke",
        "seed_validation_rank_ic": seed_rank_ic,
        "worst_seed": ensemble.worst_seed,
        "oos": {
            "valid_day_count": evaluation.summary.valid_day_count,
            "mean_rank_ic": evaluation.summary.mean_rank_ic,
            "hac_t_statistic": evaluation.hac.t_statistic,
            "baseline_mean_rank_ic": evaluation.baseline_mean_rank_ic,
            "delta_rank_ic": evaluation.delta_rank_ic,
            "gates": {
                gate: result.status.value
                for gate, result in evaluation.gate_report.results.items()
            },
        },
        "note": "smoke orchestration evidence only; never model passing evidence",
    }


def train_m1_smoke(workspace: Path) -> dict[str, Any]:
    """End-to-end M1 (ResidualMLP) orchestration on synthetic frames."""
    return _train_cross_section_smoke(
        workspace,
        family="M1",
        family_name=M1_FAMILY_NAME,
        model_code_version=M1_MODEL_CODE_VERSION,
        config_factory=lambda: ResidualMLPConfig(
            raw_factor_count=SMOKE_FACTOR_COUNT, hidden_dim=16, num_blocks=1, dropout=0.0
        ),
        model_factory=lambda: ResidualMLP(
            ResidualMLPConfig(
                raw_factor_count=SMOKE_FACTOR_COUNT, hidden_dim=16, num_blocks=1, dropout=0.0
            )
        ),
    )


def train_m3_smoke(workspace: Path) -> dict[str, Any]:
    """End-to-end M3 (SetAttention) orchestration on synthetic frames."""
    return _train_cross_section_smoke(
        workspace,
        family="M3",
        family_name=M3_FAMILY_NAME,
        model_code_version=M3_MODEL_CODE_VERSION,
        config_factory=lambda: SetAttentionConfig(
            raw_factor_count=SMOKE_FACTOR_COUNT,
            hidden_dim=16,
            num_heads=2,
            inducing_points=4,
            num_blocks=1,
            dropout=0.0,
        ),
        model_factory=lambda: SetAttention(
            SetAttentionConfig(
                raw_factor_count=SMOKE_FACTOR_COUNT,
                hidden_dim=16,
                num_heads=2,
                inducing_points=4,
                num_blocks=1,
                dropout=0.0,
            )
        ),
    )


def publish_fold_dataset(
    store: DatasetArtifactStore,
    fold_id: str,
    frames: Mapping[str, pd.DataFrame],
    versions: Mapping[str, Mapping[str, str]],
    factor_order: tuple[str, ...],
) -> str:
    """Publish one fold's train/validation/test splits; content-addressed idempotent."""
    metadata = {
        "fold_id": fold_id,
        "factor_order": list(factor_order),
        "data_versions": {table: dict(parts) for table, parts in versions.items()},
    }
    manifest = build_split_dataset_manifest(
        dict(frames),
        business_keys=("signal_date", "ts_code"),
        metadata=metadata,
    )
    with contextlib.suppress(FileExistsError):
        # identical content already published: same dataset_id by construction
        store.write(manifest, dict(frames), preprocessing={"assembler": "walk_forward_common"})
    return manifest.dataset_id


def reconstruct_baseline_record(selection_entry: Mapping[str, Any]) -> BaselineSelectionRecord:
    """Rebuild the immutable baseline selection record from the report JSON."""
    return BaselineSelectionRecord(
        fold_id=str(selection_entry["fold_id"]),
        selected_baseline_id=str(selection_entry["selected_baseline_id"]),
        validation_scores=tuple(
            BaselineValidationScore(
                baseline_id=str(score["baseline_id"]),
                mean_rank_ic=float(score["mean_rank_ic"]),
                valid_day_count=int(score["valid_day_count"]),
                invalid_day_count=int(score["invalid_day_count"]),
            )
            for score in selection_entry["validation_scores"]
        ),
        unselected_reasons=tuple(
            (str(reason[0]), str(reason[1]))
            for reason in selection_entry["unselected_reasons"]
        ),
    )


def issue_baseline_selection_evidence(
    baseline_report: Mapping[str, Any],
    dataset_ids: Mapping[str, str],
    selection_store: SelectionEvidenceStore,
) -> dict[str, str]:
    """Sign per-fold baseline selection evidence from the baseline report (idempotent)."""
    evidence_ids: dict[str, str] = {}
    for entry in baseline_report["selections"]:
        fold_id = str(entry["fold_id"])
        record = reconstruct_baseline_record(entry)
        evidence = selection_store.write_baseline_selection(
            record=record,
            test_dataset_id=dataset_ids[fold_id],
        )
        evidence_ids[fold_id] = evidence.evidence_id
    return evidence_ids


SMOKE_TEMPORAL_HISTORY_LENGTH = 20
SMOKE_TEMPORAL_TOTAL_DAYS = 30
SMOKE_TEMPORAL_TRAIN_TARGET_INDICES = (15, 16, 17, 18, 19)
SMOKE_TEMPORAL_VALIDATION_INDEX = 20
SMOKE_TEMPORAL_TEST_INDICES = (21, 22, 23, 24, 25, 26, 27, 28)


def _smoke_temporal_dates() -> tuple[date, ...]:
    base = date(2024, 1, 1)
    days: list[date] = []
    current = base
    while len(days) < SMOKE_TEMPORAL_TOTAL_DAYS:
        if current.weekday() < 5:
            days.append(current)
        current = date.fromordinal(current.toordinal() + 1)
    return tuple(days)


def build_smoke_temporal_frames(factor_count: int) -> dict[str, Any]:
    """Synthetic panel with causal history for the M2 temporal smoke run."""
    alpha = np.random.default_rng(31).standard_normal(SMOKE_STOCK_COUNT)
    noise = np.random.default_rng(32)
    days = _smoke_temporal_dates()
    train_days = tuple(days[i] for i in SMOKE_TEMPORAL_TRAIN_TARGET_INDICES)
    validation_day = days[SMOKE_TEMPORAL_VALIDATION_INDEX]
    test_days = tuple(days[i] for i in SMOKE_TEMPORAL_TEST_INDICES)

    rows: list[dict[str, object]] = []
    for day in days:
        values = noise.standard_normal((SMOKE_STOCK_COUNT, factor_count)) * 0.1
        values[:, 0] = alpha
        is_target = day in train_days or day == validation_day
        targets = alpha + noise.standard_normal(SMOKE_STOCK_COUNT) * 0.5
        for index, code in enumerate(_smoke_codes()):
            row: dict[str, object] = {
                "signal_date": day,
                "ts_code": code,
                "raw_return_5d": float(targets[index]) if is_target else 0.0,
                "label_valid": is_target,
            }
            for factor_index in range(factor_count):
                row[f"f{factor_index:02d}"] = float(values[index, factor_index])
            rows.append(row)
    frame = pd.DataFrame(rows)

    test_rows = frame.loc[frame["signal_date"].isin(test_days)].copy()
    test_rows["raw_return_5d"] = 0.0
    test_rows["label_valid"] = True

    baseline_rows = []
    for day in test_days:
        for index, code in enumerate(_smoke_codes()):
            baseline_rows.append(
                {"signal_date": day, "ts_code": code, "score": float(alpha[index])}
            )
    test_generator = np.random.default_rng(34)
    labels = test_rows.loc[:, ["signal_date", "ts_code"]].copy()
    label_returns = np.concatenate(
        [
            alpha + test_generator.standard_normal(SMOKE_STOCK_COUNT) * 0.5
            for _ in test_days
        ]
    )
    labels["raw_return_5d"] = label_returns
    labels["label_valid"] = True
    return {
        "train": frame.loc[frame["signal_date"].isin(train_days)].copy(),
        "validation": frame.loc[frame["signal_date"] == validation_day].copy(),
        "test": test_rows.reset_index(drop=True),
        "history": frame,
        "baseline_predictions": pd.DataFrame(baseline_rows),
        "labels": labels,
    }


def _predict_temporal_validation_percentiles(
    model: torch.nn.Module, validation_batches: Mapping[date, TemporalTrainingBatch]
) -> pd.DataFrame:
    return _percentile_frame(model, validation_batches)


def train_m2_smoke(workspace: Path) -> dict[str, Any]:
    """End-to-end M2 (TemporalGRU) orchestration on synthetic frames."""
    factor_count = 6
    frames = build_smoke_temporal_frames(factor_count)
    factor_order = tuple(f"f{index:02d}" for index in range(factor_count))
    preprocessor = FoldPreprocessor(factor_order=factor_order)
    preprocessor.fit(frames["train"].loc[:, list(factor_order)], role="TRAIN")

    def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
        return _prepare_frame(frame, preprocessor, factor_order)

    history = frames["history"].copy()
    for factor_name in factor_order:
        history[factor_name] = preprocessor.transform(
            history.loc[:, list(factor_order)]
        ).values[factor_name].to_numpy()
    train_with_history = pd.concat(
        [
            history.loc[
                history["signal_date"] < frames["train"]["signal_date"].min()
            ],
            _prepare(frames["train"]),
        ],
        ignore_index=True,
    )
    train_batches = build_temporal_training_batches(
        train_with_history,
        factor_order=factor_order,
        history_length=SMOKE_TEMPORAL_HISTORY_LENGTH,
        minimum_valid_history_length=15,
    )
    validation_with_history = pd.concat(
        [history.loc[history["signal_date"] < frames["validation"]["signal_date"].min()],
         _prepare(frames["validation"])],
        ignore_index=True,
    )
    validation_batches = build_temporal_training_batches(
        validation_with_history,
        factor_order=factor_order,
        history_length=SMOKE_TEMPORAL_HISTORY_LENGTH,
        minimum_valid_history_length=15,
    )
    config = TemporalGRUConfig(raw_factor_count=factor_count, hidden_dim=16, dropout=0.0)
    config_payload = {"raw_factor_count": factor_count, "hidden_dim": 16, "dropout": 0.0}
    checkpoint_store = CheckpointStore(workspace / "checkpoints")
    seed_checkpoints: dict[int, Path] = {}
    seed_validation_predictions: dict[int, pd.DataFrame] = {}
    seed_rank_ic: dict[int, float] = {}
    prototype = TemporalGRU(config)
    for seed in REQUIRED_SEEDS:
        configure_reproducibility(seed)  # seed before init; see _train_cross_section_smoke
        model = TemporalGRU(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
        run = run_validation_governed_training(
            model,
            optimizer,
            train_batches=train_batches,
            validation_batches=validation_batches,
            train_view=DatasetView(role=DatasetRole.TRAIN, dataset_id="smoke-train"),
            validation_view=DatasetView(role=DatasetRole.VALIDATION, dataset_id="smoke-validation"),
            trainer_config=TrainerConfig(max_epochs=8, early_stopping_patience=8),
            seed=seed,
            checkpoint_store=checkpoint_store,
            checkpoint_prefix=f"smoke-m2-{seed}",
        )
        seed_checkpoints[seed] = (
            workspace / "checkpoints" / run.checkpoint_ids[-1] / "checkpoint.pt"
        )
        seed_validation_predictions[seed] = _predict_temporal_validation_percentiles(
            model, validation_batches
        )
        seed_rank_ic[seed] = _validation_rank_ic(
            seed_validation_predictions[seed], frames["validation"]
        )
    ensemble = build_seed_ensemble(seed_validation_predictions, validation_rank_ic=seed_rank_ic)
    bundle_store = ModelArtifactBundleStore(workspace / "models")
    model_id = "smoke-m2"
    bundle_store.write(
        model_id,
        model_card=torch_model_card(
            card_family=m2_card_family(M2_SMOKE_FAMILY_CHOICE),
            dataset_id="smoke-train",
            model_code_version=m2_model_code_version(M2_SMOKE_FAMILY_CHOICE),
        ),
        config=config_payload,
        signature=asdict(prototype.artifact_signature()),
        feature_order=factor_order,
        preprocessing=_preprocessing_artifact(preprocessor),
        seed_checkpoints=seed_checkpoints,
        validation_predictions=ensemble.predictions.rename(
            columns={"ensemble_score": "percentile_rank"}
        ),
    )
    loaded = load_bundle_for_evaluation(
        bundle_store,
        model_id,
        expected_dataset_id="smoke-train",
        expected_feature_order=factor_order,
        expected_preprocessor_hash=preprocessor.state_hash,
        device=torch.device("cpu"),
    )
    test_dates = set(frames["test"]["signal_date"])
    test_panel = pd.concat(
        [
            history.loc[~history["signal_date"].isin(test_dates)],
            _prepare(frames["test"]),
        ],
        ignore_index=True,
    )
    test_batches = build_temporal_training_batches(
        test_panel,
        factor_order=factor_order,
        history_length=SMOKE_TEMPORAL_HISTORY_LENGTH,
        minimum_valid_history_length=15,
    )
    candidate_predictions = _concat_date_scores(loaded.models, test_batches)
    evaluation = evaluate_frozen_oos(
        candidate_predictions,
        frames["labels"],
        baseline_predictions=frames["baseline_predictions"],
        role="TEST",
        minimum_cross_section=30,
    )
    return {
        "family": "M2",
        "mode": "smoke",
        "seed_validation_rank_ic": seed_rank_ic,
        "worst_seed": ensemble.worst_seed,
        "oos": {
            "valid_day_count": evaluation.summary.valid_day_count,
            "mean_rank_ic": evaluation.summary.mean_rank_ic,
            "hac_t_statistic": evaluation.hac.t_statistic,
            "baseline_mean_rank_ic": evaluation.baseline_mean_rank_ic,
            "delta_rank_ic": evaluation.delta_rank_ic,
            "gates": {
                gate: result.status.value
                for gate, result in evaluation.gate_report.results.items()
            },
        },
        "note": "smoke orchestration evidence only; never model passing evidence",
    }


M0_NUM_BOOST_ROUND = 500
M0_EARLY_STOPPING_PATIENCE = 50
# Threads per fit for the M0 search. deterministic=True is kept, so a fixed thread
# count is reproducible (pinned by a determinism test); raising it is what makes a
# 300-fit search on ~2.3M rows viable, since num_threads=1 measured >15 min for a
# single fit. This is a candidate-family performance knob, NOT the B3 contract.
M0_NUM_THREADS = 16
M0_SMOKE_NUM_BOOST_ROUND = 64
M0_FAMILY_NAME = "LGBM_RANKER"
M0_MODEL_CODE_VERSION = "lgbm-ranker-1"

# M0R: the rank-regression carrier. Same frozen-v2 factor set, same folds, same three-seed
# percentile-mean ensemble, same bundle and evidence machinery as M0 -- only the objective,
# the target and the early-stopping metric differ. It exists because the LambdaRank carrier
# caps out below the G4 bar: at one fixed config over the chain's 3,417,963 paired TEST rows
# M0 reaches delta +0.00109 and M0R reaches +0.00915, and with no validation search at all
# M0R passes every gate (G1 0.094376, G2 12.905, G3 1.00, G4 +0.009149, G5 0.2791, G6 0.9752)
# where M0 needed a 24-config per-fold search to reach +0.004202 and still failed G4.
# Mechanism and the negative results: docs/handoffs/
# 2026-09-08-m0-attribution-and-carrier-findings.md. M0 and the B0-B3 baseline contract are
# deliberately left bit-identical, so no existing bundle digest or evidence is disturbed.
M0R_FAMILY_NAME = "LGBM_RANKREG"
M0R_MODEL_CODE_VERSION = "lgbm-rankreg-1"
M0R_MODEL_ID_PREFIX = "m0r"
M0R_SMOKE_NUM_BOOST_ROUND = 64

# The torch carriers' card identities, derived from the model classes' own ``model_family`` where
# there is one to derive from. Four blockers in this chain were the same mistake: a model-card field
# written one way here and expected another way by the walk-forward evaluator or the artifact
# loader, each found by a run that had already trained for days. M2 has no constant because its
# card is named by the winning configuration -- see ``m2_card_family``.
M1_FAMILY_NAME = ResidualMLP.model_family
M1_MODEL_CODE_VERSION = "residual-mlp-1"
M3_FAMILY_NAME = SetAttention.model_family
M3_MODEL_CODE_VERSION = "set-attention-1"
M2_SMOKE_FAMILY_CHOICE = "GRU"

LightGBMCarrier = Literal["lambdarank", "rankreg"]
AnyLightGBMBaseline = LGBMRankerBaseline | LGBMRankRegBaseline


def m0_config_grid(max_configs: int | None = None) -> list[dict[str, Any]]:
    """M0 (LightGBM LambdaRank) search space: 24 configs, within the 30-trial E7 budget.

    feature_fraction is fixed below 1 so the three REQUIRED_SEEDS train genuinely
    different boosters and the seed ensemble measures stochastic-training variance
    (pinned in tests/unit/baselines/test_lgbm_ranker_baseline.py); with the fully
    deterministic B3 recipe the three seeds would be three copies of one model and
    the seed ensemble would be vacuous.
    """
    grid: list[dict[str, Any]] = [
        {
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "min_child_samples": min_child_samples,
            "feature_fraction": 0.85,
        }
        for num_leaves in (15, 31, 63)
        for max_depth in (4, 6)
        for learning_rate in (0.03, 0.05)
        for min_child_samples in (50, 200)
    ]
    return grid if max_configs is None else grid[:max_configs]


def m0_refined_config_grid(max_configs: int | None = None) -> list[dict[str, Any]]:
    """Refined M0 search: L2 regularization around the winning capacity region.

    The base 24-config grid found the best at delta +0.00364 (below the +0.005 G4 bar),
    with F1 needing num_leaves=63 and the strong folds at num_leaves=15. This refined
    grid holds that winning capacity region and varies lambda_l2 (the standard lever for
    weak, noisy signals like the sparse factors'), which the base grid never touched.
    24 configs, within the 30-trial E7 budget.
    """
    grid: list[dict[str, Any]] = [
        {
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "min_child_samples": 200,
            "feature_fraction": 0.85,
            "lambda_l2": lambda_l2,
        }
        for num_leaves in (15, 63)
        for max_depth in (4, 6)
        for learning_rate in (0.03, 0.05)
        for lambda_l2 in (0.0, 1.0, 10.0)
    ]
    return grid if max_configs is None else grid[:max_configs]


# The config every M0R number in the attribution handoff was measured at: num_leaves 15,
# max_depth 4, lr 0.05, min_child_samples 200, feature_fraction 0.85, lambda_l2 0. It is the
# modal winner of the M0 refined grid (3 of 4 folds), reused unchanged so the two carriers are
# compared at one point in hyperparameter space rather than each at its own searched optimum.
M0R_MEASURED_CONFIG: dict[str, Any] = {
    "num_leaves": 15,
    "max_depth": 4,
    "learning_rate": 0.05,
    "min_child_samples": 200,
    "feature_fraction": 0.85,
    "lambda_l2": 0.0,
}


def m0r_config_grid(max_configs: int | None = None) -> list[dict[str, Any]]:
    """M0R search space: one config, deliberately.

    A 24-config search is what makes M0's +0.004202 optimistic -- it is the maximum over 24
    draws selected on a single validation year per fold, and the same protocol measured on a
    different factor subset picked a config with the better validation score and the worse
    test delta (F1 dense9: val 0.06141 -> test +0.00243, against +0.00389 for an untuned
    config). M0R's +0.00915 was measured with no search at all, so pinning the grid to that
    one config is what keeps the production run's number free of validation-selection bias
    and directly comparable to the measurement that motivated the carrier. It is 1 trial
    against the E7 budget of 30, so it is inside the budget by construction rather than by
    trimming. A tuned M0R search is a legitimate follow-up; it just has to be reported as
    the biased quantity that it is.
    """
    grid = [dict(M0R_MEASURED_CONFIG)]
    return grid if max_configs is None else grid[:max_configs]


def _m0_grid(args: argparse.Namespace, max_configs: int | None) -> list[dict[str, Any]]:
    grid_fn = (
        m0_refined_config_grid
        if getattr(args, "grid", "base") == "refined"
        else m0_config_grid
    )
    return grid_fn(max_configs)


def _m0r_grid(args: argparse.Namespace, max_configs: int | None) -> list[dict[str, Any]]:
    del args  # --grid selects between M0's two LambdaRank grids; M0R has one pinned config
    return m0r_config_grid(max_configs)


def _m0_hyperparameters(config_payload: Mapping[str, Any]) -> LGBMHyperparameters:
    return LGBMHyperparameters(
        num_leaves=int(config_payload["num_leaves"]),
        max_depth=int(config_payload["max_depth"]),
        learning_rate=float(config_payload["learning_rate"]),
        min_child_samples=int(config_payload["min_child_samples"]),
        feature_fraction=float(config_payload.get("feature_fraction", 1.0)),
        num_threads=M0_NUM_THREADS,
        lambda_l2=float(config_payload.get("lambda_l2", 0.0)),
    )


def _m0_fit(
    config_payload: Mapping[str, Any],
    factor_order: tuple[str, ...],
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame | None,
    *,
    seed: int,
    num_boost_round: int = M0_NUM_BOOST_ROUND,
) -> LGBMRankerBaseline:
    """Fit one seeded LambdaRank booster on TRAIN, early-stopped on VALIDATION.

    ``validation_frame=None`` trains the full ``num_boost_round`` without early
    stopping. The smoke uses that: on the trivially separable synthetic frames a
    ranker early-stops at a one-tree model, whose handful of distinct scores then
    trips the OOS score-health gate -- a degeneracy of the data, not of the
    machinery. The early-stopping branch itself is covered by the reviewed
    baseline's own tests; full runs always pass a validation frame.
    """
    keys = ["signal_date", "ts_code"]
    evaluation_features = None
    evaluation_labels = None
    if validation_frame is not None:
        evaluation_features = validation_frame.loc[:, keys + list(factor_order)]
        evaluation_labels = validation_frame.loc[:, keys + ["raw_return_5d", "label_valid"]]
    return LGBMRankerBaseline.fit(
        train_frame.loc[:, keys + list(factor_order)],
        train_frame.loc[:, keys + ["raw_return_5d", "label_valid"]],
        factor_order=factor_order,
        role="TRAIN",
        num_boost_round=num_boost_round,
        minimum_cross_section=30,
        hyperparameters=_m0_hyperparameters(config_payload),
        evaluation_features=evaluation_features,
        evaluation_labels=evaluation_labels,
        early_stopping_patience=M0_EARLY_STOPPING_PATIENCE,
        seed=seed,
        allow_nan=True,
    )


def _m0r_fit(
    config_payload: Mapping[str, Any],
    factor_order: tuple[str, ...],
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame | None,
    *,
    seed: int,
    num_boost_round: int = M0_NUM_BOOST_ROUND,
) -> LGBMRankRegBaseline:
    """Fit one seeded rank-regression booster on TRAIN, early-stopped on VALIDATION RankIC.

    The mirror of ``_m0_fit`` for the M0R carrier: same rows, same config schema, same seed
    contract, but the target is the per-day percentile rank of ``raw_return_5d`` under
    ``objective=regression`` and early stopping watches the daily RankIC feval instead of
    LambdaRank's default ndcg@1..5. ``validation_frame=None`` trains the full
    ``num_boost_round`` with no early stopping, which is what the smoke uses.
    """
    keys = ["signal_date", "ts_code"]
    evaluation_features = None
    evaluation_labels = None
    if validation_frame is not None:
        evaluation_features = validation_frame.loc[:, keys + list(factor_order)]
        evaluation_labels = validation_frame.loc[:, keys + ["raw_return_5d", "label_valid"]]
    return LGBMRankRegBaseline.fit(
        train_frame.loc[:, keys + list(factor_order)],
        train_frame.loc[:, keys + ["raw_return_5d", "label_valid"]],
        factor_order=factor_order,
        role="TRAIN",
        num_boost_round=num_boost_round,
        minimum_cross_section=30,
        hyperparameters=_m0_hyperparameters(config_payload),
        evaluation_features=evaluation_features,
        evaluation_labels=evaluation_labels,
        early_stopping_patience=M0_EARLY_STOPPING_PATIENCE,
        seed=seed,
        allow_nan=True,
    )


@dataclass(frozen=True)
class _LightGBMFamilySpec:
    """Everything that differs between the two LightGBM candidate carriers.

    Both carriers share the assembly, publication, per-fold search, three-seed ensemble,
    bundle, signed-evidence and G1-G6 paths; only these fields and the fit function vary.
    Keeping the difference in one frozen value is what stops the two families from drifting
    into two copies of a 200-line pipeline.
    """

    family: str
    family_name: str
    model_code_version: str
    model_id_prefix: str
    carrier: LightGBMCarrier
    fit: Callable[..., AnyLightGBMBaseline]
    smoke_num_boost_round: int
    grid: Callable[[argparse.Namespace, int | None], list[dict[str, Any]]]


M0_SPEC = _LightGBMFamilySpec(
    family="M0",
    family_name=M0_FAMILY_NAME,
    model_code_version=M0_MODEL_CODE_VERSION,
    model_id_prefix="m0",
    carrier="lambdarank",
    fit=_m0_fit,
    smoke_num_boost_round=M0_SMOKE_NUM_BOOST_ROUND,
    grid=_m0_grid,
)
M0R_SPEC = _LightGBMFamilySpec(
    family="M0R",
    family_name=M0R_FAMILY_NAME,
    model_code_version=M0R_MODEL_CODE_VERSION,
    model_id_prefix=M0R_MODEL_ID_PREFIX,
    carrier="rankreg",
    fit=_m0r_fit,
    smoke_num_boost_round=M0R_SMOKE_NUM_BOOST_ROUND,
    grid=_m0r_grid,
)


def _m0_percentile_frame(
    model: AnyLightGBMBaseline,
    frame: pd.DataFrame,
    factor_order: tuple[str, ...],
) -> pd.DataFrame:
    """Per-date percentile ranks of one booster -- the predictions_to_frame contract."""
    keys = ["signal_date", "ts_code"]
    scores = model.predict(frame.loc[:, keys + list(factor_order)])
    frame_out = frame.loc[:, keys].copy()
    frame_out["percentile_rank"] = scores.to_numpy(dtype=float)
    frame_out["percentile_rank"] = frame_out.groupby("signal_date")[
        "percentile_rank"
    ].rank(pct=True)
    return frame_out


def _m0_seed_mean_scores(
    models_by_seed: Mapping[int, AnyLightGBMBaseline],
    frame: pd.DataFrame,
    factor_order: tuple[str, ...],
) -> pd.DataFrame:
    """Seed-mean percentile scores averaged on business keys, never positionally."""
    keys = ["signal_date", "ts_code"]
    merged: pd.DataFrame | None = None
    rank_columns: list[str] = []
    for seed in REQUIRED_SEEDS:
        column = f"seed_rank_{seed}"
        percentile = _m0_percentile_frame(models_by_seed[seed], frame, factor_order).rename(
            columns={"percentile_rank": column}
        )
        merged = (
            percentile
            if merged is None
            else merged.merge(percentile, on=keys, validate="one_to_one")
        )
        rank_columns.append(column)
    if merged is None:
        raise ValueError("at least one seed model is required")
    merged["score"] = merged.loc[:, rank_columns].mean(axis=1)
    return merged.loc[:, [*keys, "score"]]


def _m0_save_booster(model: AnyLightGBMBaseline, path: Path) -> None:
    """Persist one fitted booster; the frozen dataclass stores the booster as ``object``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cast("lgb.Booster", model.booster).save_model(str(path))


def _m0_signature(
    config_payload: Mapping[str, Any],
    factor_count: int,
    *,
    family_name: str = M0_FAMILY_NAME,
) -> dict[str, object]:
    config_hash = hashlib.sha256(
        json.dumps(dict(config_payload), separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    return asdict(
        ModelSignature(
            model_family=family_name,
            raw_factor_count=factor_count,
            config_hash=config_hash,
        )
    )


def _m0_load_booster(
    bundle_store: ModelArtifactBundleStore,
    model_id: str,
    seed: int,
    *,
    carrier: LightGBMCarrier = "lambdarank",
) -> AnyLightGBMBaseline:
    """Load one seeded booster back from the verified bundle, never from training memory."""
    if not bundle_store.verify(model_id):
        raise ValueError(f"model bundle failed verification: {model_id}")
    root = bundle_store.bundle_path(model_id)
    feature_order = json.loads((root / "feature_order.json").read_text(encoding="utf-8"))
    booster = lgb.Booster(model_file=str(root / f"seed-{seed}.pt"))
    loaded_order = tuple(str(name) for name in feature_order["feature_order"])
    loaded_rounds = int(booster.num_trees())
    if carrier == "rankreg":
        return LGBMRankRegBaseline(
            factor_order=loaded_order, booster=booster, num_boost_round=loaded_rounds
        )
    return LGBMRankerBaseline(
        factor_order=loaded_order,
        booster=booster,
        num_boost_round=loaded_rounds,
        # M0 is the candidate family that consumes the six sparse factors, so the
        # frozen boosters must keep consuming NaN natively after being read back
        # from the bundle (the default False would raise on the first NaN at TEST).
        allow_nan=True,
    )


def _m0_write_bundle(
    model_store: ModelArtifactBundleStore,
    model_id: str,
    *,
    dataset_id: str,
    config_payload: Mapping[str, Any],
    factor_order: tuple[str, ...],
    seed_checkpoints: Mapping[int, Path],
    ensemble_predictions: pd.DataFrame,
    family_name: str = M0_FAMILY_NAME,
    model_code_version: str = M0_MODEL_CODE_VERSION,
) -> None:
    model_store.write(
        model_id,
        model_card={
            "family": family_name,
            "dataset_id": dataset_id,
            "model_status": "WALK_FORWARD_PASSED",
            "model_code_version": model_code_version,
        },
        config=dict(config_payload, raw_factor_count=len(factor_order)),
        signature=_m0_signature(
            config_payload, len(factor_order), family_name=family_name
        ),
        feature_order=factor_order,
        preprocessing={"state_hash": "identity"},
        seed_checkpoints=seed_checkpoints,
        validation_predictions=ensemble_predictions,
    )


def _train_lightgbm_smoke(workspace: Path, spec: _LightGBMFamilySpec) -> dict[str, Any]:
    """Orchestration smoke for either LightGBM carrier: ensemble + bundle + frozen-gate path."""
    frames = build_smoke_frames(SMOKE_FACTOR_COUNT)
    factor_order = tuple(f"f{index:02d}" for index in range(SMOKE_FACTOR_COUNT))
    config_payload = m0_config_grid()[0]
    seed_checkpoints: dict[int, Path] = {}
    seed_validation_predictions: dict[int, pd.DataFrame] = {}
    seed_rank_ic: dict[int, float] = {}
    for seed in REQUIRED_SEEDS:
        model = spec.fit(
            config_payload,
            factor_order,
            frames["train"],
            None,
            seed=seed,
            num_boost_round=spec.smoke_num_boost_round,
        )
        checkpoint_path = (
            workspace / "checkpoints" / f"smoke-{spec.model_id_prefix}-{seed}" / "booster.txt"
        )
        _m0_save_booster(model, checkpoint_path)
        seed_checkpoints[seed] = checkpoint_path
        predictions = _m0_percentile_frame(model, frames["validation"], factor_order)
        seed_validation_predictions[seed] = predictions
        seed_rank_ic[seed] = _validation_rank_ic(predictions, frames["validation"])
    ensemble = build_seed_ensemble(seed_validation_predictions, validation_rank_ic=seed_rank_ic)
    bundle_store = ModelArtifactBundleStore(workspace / "models")
    model_id = f"smoke-{spec.model_id_prefix}"
    _m0_write_bundle(
        bundle_store,
        model_id,
        dataset_id="smoke-train",
        config_payload=config_payload,
        factor_order=factor_order,
        seed_checkpoints=seed_checkpoints,
        ensemble_predictions=ensemble.predictions.rename(
            columns={"ensemble_score": "percentile_rank"}
        ),
        family_name=spec.family_name,
        model_code_version=spec.model_code_version,
    )
    loaded = {
        seed: _m0_load_booster(bundle_store, model_id, seed, carrier=spec.carrier)
        for seed in REQUIRED_SEEDS
    }
    candidate_predictions = _m0_seed_mean_scores(loaded, frames["test"], factor_order)
    evaluation = evaluate_frozen_oos(
        candidate_predictions,
        frames["labels"],
        baseline_predictions=frames["baseline_predictions"],
        role="TEST",
        minimum_cross_section=30,
    )
    return {
        "family": spec.family,
        "mode": "smoke",
        "seed_validation_rank_ic": seed_rank_ic,
        "worst_seed": ensemble.worst_seed,
        "oos": {
            "valid_day_count": evaluation.summary.valid_day_count,
            "mean_rank_ic": evaluation.summary.mean_rank_ic,
            "hac_t_statistic": evaluation.hac.t_statistic,
            "baseline_mean_rank_ic": evaluation.baseline_mean_rank_ic,
            "delta_rank_ic": evaluation.delta_rank_ic,
            "gates": {
                gate: result.status.value
                for gate, result in evaluation.gate_report.results.items()
            },
        },
        "note": "smoke orchestration evidence only; never model passing evidence",
    }


def train_m0_smoke(workspace: Path) -> dict[str, Any]:
    """M0 orchestration smoke: the same ensemble + bundle + frozen-gate path as M1/M3."""
    return _train_lightgbm_smoke(workspace, M0_SPEC)


def train_m0r_smoke(workspace: Path) -> dict[str, Any]:
    """M0R orchestration smoke: the rank-regression carrier through the same path."""
    return _train_lightgbm_smoke(workspace, M0R_SPEC)


def _search_resume(
    args: argparse.Namespace,
    workspace: Path,
    *,
    fold_id: str,
    family: str,
    model_code_version: str,
    train_dataset_id: str,
    validation_dataset_id: str,
    factor_order: tuple[str, ...],
    runtime: Mapping[str, object],
) -> SearchResume:
    """One fold's search-record handle, logging into the transcript the run already writes.

    The search loop had no resume at any granularity: ``_existing_evidence_id`` makes the evidence
    writes idempotent but nothing skips the search, so an interruption at fold 3 config 20 restarted
    at fold 1 config 0 seed 1 -- on a multi-day run, losing all of it. Records are persisted by
    every full run and adopted only under ``--resume``, so the default path is unchanged.
    """
    return SearchResume(
        workspace=workspace,
        fold_id=fold_id,
        family=family,
        model_code_version=model_code_version,
        train_dataset_id=train_dataset_id,
        validation_dataset_id=validation_dataset_id,
        factor_order=factor_order,
        runtime=runtime,
        enabled=bool(getattr(args, "resume", False)),
        log=lambda message: print(message, flush=True),
    )


@dataclass(frozen=True)
class _AdoptedFold:
    """One already-frozen fold's outcome, reconstructed from artifacts this run re-verified."""

    report_entry: dict[str, Any]
    oos_fold: ModelOosFold


@dataclass(frozen=True)
class _FoldAdoption:
    """The adoption decision plus, when it is a refusal, the reason -- never both."""

    fold: _AdoptedFold | None = None
    reason: str = ""


def _adopt_completed_fold(
    args: argparse.Namespace,
    workspace: Path,
    model_store: ModelArtifactBundleStore,
    selection_store: SelectionEvidenceStore,
    resume: SearchResume,
    *,
    fold_id: str,
    model_id: str,
    dataset_id: str,
    train_dataset_id: str,
    validation_dataset_id: str,
    factor_order: tuple[str, ...],
    configs: Sequence[Mapping[str, Any]],
    budget_for: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> _AdoptedFold | None:
    """Skip a fold whose artifacts are already frozen in this workspace, or refuse before fitting.

    A bundle is immutable and the prediction index refuses a second candidate for one fold, so a
    fold that already finished cannot be retrained in place: without this the run would redo the
    fold's three final fits -- hours -- and then die on the bundle write. Adoption is therefore not
    an optimisation layered on top of resume, it is the only way ``--resume`` can cross a fold
    boundary at all.

    Adoption still requires ``--resume``. Without it a pre-existing bundle is refused before any
    fitting too, because the outcome is the same crash either way and an operator deserves the
    reason at the start of the run rather than hours into it.

    Nothing here is trusted. The bundle is re-hashed, the signed model-selection evidence is loaded
    through its verifier and its ``model_bundle_digest`` is compared to that re-hash, its three
    dataset ids and its baseline evidence id are compared to this run's, the bundle's own card and
    feature order are compared to this fold's dataset and factor order, both signed prediction
    artifacts must be indexed for this fold, and every config's search record must be adoptable
    under the same bindings the search itself uses. The report entry is then REBUILT from those
    records -- including ``best_config`` via the same ``max`` over grid order the fresh search uses
    -- so a resumed run's report is the report a fresh run would have written, not an approximation
    of it. Adopting skips recomputation only; the walk-forward evaluation still loads and verifies
    the bundle, the selections and the predictions itself, so no check is bypassed.
    """
    if not model_store.bundle_path(model_id).exists():
        return None
    if not bool(getattr(args, "resume", False)):
        raise SystemExit(
            f"fold {fold_id} already wrote bundle {model_id} into this workspace, and a bundle is "
            "immutable, so it cannot be retrained here. Pass --resume to adopt it if its artifacts "
            "verify, or point --workspace at a fresh directory to start over."
        )
    adoption = _fold_adoption(
        workspace,
        model_store,
        selection_store,
        resume,
        fold_id=fold_id,
        model_id=model_id,
        dataset_id=dataset_id,
        train_dataset_id=train_dataset_id,
        validation_dataset_id=validation_dataset_id,
        factor_order=factor_order,
        configs=configs,
        budget_for=budget_for,
    )
    if adoption.fold is None:
        raise SystemExit(
            f"fold {fold_id} already wrote bundle {model_id} into this workspace, and a bundle is "
            f"immutable, so it cannot be retrained here, and --resume cannot adopt it either: "
            f"{adoption.reason}. Point --workspace at a fresh directory to start over."
        )
    return adoption.fold


def _fold_adoption(
    workspace: Path,
    model_store: ModelArtifactBundleStore,
    selection_store: SelectionEvidenceStore,
    resume: SearchResume,
    *,
    fold_id: str,
    model_id: str,
    dataset_id: str,
    train_dataset_id: str,
    validation_dataset_id: str,
    factor_order: tuple[str, ...],
    configs: Sequence[Mapping[str, Any]],
    budget_for: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> _FoldAdoption:
    """Every check adoption rests on, as a reason rather than an exception, so each is testable."""
    if not model_store.verify(model_id):
        return _FoldAdoption(reason="its bundle does not re-verify")
    bundle_root = model_store.bundle_path(model_id)
    card = json.loads((bundle_root / "model_card.json").read_text(encoding="utf-8"))
    recorded_dataset = card.get("dataset_id")
    if recorded_dataset not in (dataset_id, f"{dataset_id}:train"):
        return _FoldAdoption(
            reason=(
                f"its bundle records dataset_id {recorded_dataset!r}, which is neither this "
                f"fold's {dataset_id!r} nor its train split"
            )
        )
    order_document = json.loads((bundle_root / "feature_order.json").read_text(encoding="utf-8"))
    recorded_order = order_document.get("feature_order")
    if not isinstance(recorded_order, list) or tuple(recorded_order) != tuple(factor_order):
        return _FoldAdoption(reason="its bundle was fitted on a different factor order")
    baseline_selection_id = _existing_evidence_id(
        workspace / "selections", selection_store, "baseline", fold_id, dataset_id
    )
    model_selection_id = _existing_evidence_id(
        workspace / "selections", selection_store, "model", fold_id, dataset_id
    )
    if baseline_selection_id is None or model_selection_id is None:
        return _FoldAdoption(reason="its signed selection evidence is missing")
    evidence = selection_store.load_model_selection(model_selection_id)
    if evidence.model_artifact_id != model_id:
        return _FoldAdoption(
            reason=f"its signed evidence names {evidence.model_artifact_id}, not {model_id}"
        )
    digest = model_store.bundle_digest(model_id)
    if evidence.model_bundle_digest != digest:
        return _FoldAdoption(
            reason=(
                "its signed evidence binds bundle digest "
                f"{evidence.model_bundle_digest}, but the bundle on disk re-hashes to {digest}"
            )
        )
    if (
        evidence.test_dataset_id != dataset_id
        or evidence.training_dataset_id != train_dataset_id
        or evidence.validation_dataset_id != validation_dataset_id
    ):
        return _FoldAdoption(reason="its signed evidence binds different dataset ids")
    if evidence.baseline_selection_id != baseline_selection_id:
        return _FoldAdoption(reason="its signed evidence names a different baseline selection")
    prediction_ids = _prediction_ids_by_fold_and_kind(workspace)
    candidate_id = prediction_ids.get((fold_id, "candidate"))
    best_simple_id = prediction_ids.get((fold_id, "best_simple"))
    if candidate_id is None or best_simple_id is None:
        return _FoldAdoption(reason="its signed prediction artifacts are not indexed")
    search_results: list[dict[str, Any]] = []
    for config_index, config_payload in enumerate(configs):
        lookup = resume.adopt(config_index, config_payload, budget=budget_for(config_payload))
        if lookup.record is None:
            return _FoldAdoption(
                reason=(
                    f"config {config_index} has no adoptable search record ({lookup.reason}), so "
                    "the search that selected this bundle cannot be reconstructed"
                )
            )
        search_results.append(lookup.record)
    best = max(search_results, key=lambda item: item["ensemble_validation_rank_ic"])
    return _FoldAdoption(
        fold=_AdoptedFold(
            report_entry={
                "fold_id": fold_id,
                "dataset_id": dataset_id,
                "model_id": model_id,
                "best_config": best["config"],
                "search_results": search_results,
                "baseline_selection_id": baseline_selection_id,
                "model_selection_id": model_selection_id,
            },
            oos_fold=ModelOosFold(
                model_selection_id=model_selection_id,
                candidate_prediction_id=candidate_id,
                best_simple_prediction_id=best_simple_id,
            ),
        )
    )


def _torch_search_budget(max_epochs: int) -> dict[str, Any]:
    """The torch fit's length and stopping rule, which the config payload does not carry."""
    return {
        "max_epochs": max_epochs,
        "early_stopping_patience": TORCH_EARLY_STOPPING_PATIENCE,
    }


def _temporal_search_budget(max_epochs: int) -> dict[str, Any]:
    """M2's budget: the fit length plus the window contract its batches are built under.

    The window is part of the budget and not of the config payload, and it changes results as much
    as the epoch count does -- a 20-day window and a 10-day one are different models. Leaving it
    out would let a record produced under one window be adopted by a run using another, which is
    the same class of vacuous binding the effective-hyperparameter entry exists to prevent. Both
    values are imported from the module the worker builds batches with, so the serial loop, the
    spawned worker and the record cannot disagree about them.
    """
    return dict(
        _torch_search_budget(max_epochs),
        history_length=HISTORY_LENGTH,
        minimum_valid_history_length=MINIMUM_VALID_HISTORY_LENGTH,
    )


def _search_granularity(unadopted_count: int, workers: int) -> Literal["config", "seed"]:
    """Which task granularity to dispatch -- or either one, when there is nothing to dispatch.

    A fold whose search records are all adoptable but whose bundle was never written resumes with
    zero unadopted configs, and that is exactly where a long run is most likely to have been
    interrupted: after the search, during the final fits. ``search_granularity`` refuses a zero
    config count, correctly, because a makespan over no work is meaningless -- but the caller still
    has to reach ``_ordered_search_results`` and then the final fits. Both branches build an empty
    task list for an empty grid and ``run_parallel_search`` returns immediately, so the label
    returned here does not matter; what matters is that this does not raise.
    """
    if unadopted_count < 1:
        return "config"
    return search_granularity(
        config_count=unadopted_count,
        workers=workers,
        seeds_per_config=len(REQUIRED_SEEDS),
    )


def _record_persister(
    resume: SearchResume,
    configs: Sequence[Mapping[str, Any]],
    budget: Mapping[str, Any],
) -> Callable[[int, dict[str, object]], None]:
    """Persist each configuration's search record the moment its result lands.

    Persisting after the round instead would mean a worker failure discards every configuration that
    finished, because ``run_parallel_search`` raises as soon as one child dies and results collected
    only at the end are lost with it. On a 24-configuration temporal round that is twenty-odd hours
    of fits, and ``--resume`` would have nothing to adopt on the restart -- the exact situation
    resume exists for.
    """

    def persist(index: int, result: dict[str, object]) -> None:
        resume.persist(index, configs[index], cast(dict[str, Any], result), budget=budget)

    return persist


def _seed_result_collector(
    resume: SearchResume,
    configs: Sequence[Mapping[str, Any]],
    budget: Mapping[str, Any],
    validation_labels: pd.DataFrame,
    fits_by_config: dict[int, dict[int, Mapping[str, object]]],
    records: dict[int, dict[str, Any]],
) -> Callable[[int, dict[str, object]], None]:
    """Aggregate seed-level results per configuration and persist each one as soon as it is whole.

    A seed-level round cannot persist on arrival the way a config-level one can, because the record
    is the config's three-seed ensemble and no single seed result contains it. So the record is
    written at the moment the third seed of a configuration lands -- which is what keeps a narrow
    grid's round from being all-or-nothing, and lets the post-round loop read finished records
    instead of recomputing them.
    """

    def collect(index: int, fitted: dict[str, object]) -> None:
        config_index = index // len(REQUIRED_SEEDS)
        seed = int(cast(int, fitted["seed"]))
        fits = fits_by_config.setdefault(config_index, {})
        fits[seed] = fitted
        if set(fits) != set(REQUIRED_SEEDS):
            return
        record = _ensemble_from_seed_fits(fits, configs[config_index], validation_labels)
        records[config_index] = record
        resume.persist(config_index, configs[config_index], record, budget=budget)

    return collect


def _lightgbm_search_budget(config_payload: Mapping[str, Any]) -> dict[str, Any]:
    """The effective LightGBM budget: the round/patience constants plus the fitted hyperparameters.

    ``_m0_hyperparameters`` fills in library-side defaults the config payload never names
    (``bagging_fraction``, ``bagging_freq``, ``feature_fraction``, ``lambda_l2``, ``num_threads``),
    and ``model_code_version`` is a hand-maintained label that does not move when those defaults do.
    Binding the effective values is what stops a record produced under different defaults from being
    adopted as if it were this run's result.
    """
    return {
        "num_boost_round": M0_NUM_BOOST_ROUND,
        "early_stopping_patience": M0_EARLY_STOPPING_PATIENCE,
        "minimum_cross_section": 30,
        "hyperparameters": asdict(_m0_hyperparameters(config_payload)),
    }


def _run_full_lightgbm_family(
    args: argparse.Namespace, spec: _LightGBMFamilySpec
) -> dict[str, Any]:
    """Full real-data pipeline shared by the M0 (LambdaRank) and M0R (rank-regression) carriers.

    Same assembly -> publication -> per-fold constrained search -> final bundles ->
    signed evidence chain -> frozen TEST predictions -> G1-G6 walk-forward gates
    as the torch cross-section families; the torch loop and batch machinery are
    replaced by the date-grouped LambdaRank fit from the reviewed B3 baseline.
    LightGBM has no FoldPreprocessor step: the tree ensemble is invariant to the
    monotone per-feature transforms it applies, and the bundle records
    preprocessing={"state_hash": "identity"} so the contract stays explicit.
    """
    context = setup_full_run_context(args)
    workspace = context.workspace
    dataset_store = context.dataset_store
    model_store = context.model_store
    selection_store = context.selection_store
    prediction_store = context.prediction_store
    baseline_report = context.baseline_report
    baseline_predictions = context.baseline_predictions
    holder = context.holder
    resolver = context.resolver
    versions = context.versions
    factor_members = context.factor_members
    factor_order = context.factor_order
    extract = context.extract
    configs = spec.grid(args, args.max_configs)

    oos_folds: dict[str, ModelOosFold] = {}
    fold_reports: list[dict[str, Any]] = []

    for fold_id in ORDINARY_FOLDS:
        train = extract(holder, resolver, fold_id, "train", factor_members, versions)
        validation = extract(holder, resolver, fold_id, "validation", factor_members, versions)
        test = extract(holder, resolver, fold_id, "test", factor_members, versions)
        frames = {"train": train, "validation": validation, "test": test}
        dataset_id = publish_fold_dataset(dataset_store, fold_id, frames, versions, factor_order)
        train_dataset_id = f"{dataset_id}:train"
        validation_dataset_id = f"{dataset_id}:validation"

        entry = next(e for e in baseline_report["selections"] if e["fold_id"] == fold_id)
        baseline_evidence_id = _existing_evidence_id(
            workspace / "selections", selection_store, "baseline", fold_id, dataset_id
        )
        if baseline_evidence_id is None:
            baseline_evidence = selection_store.write_baseline_selection(
                record=reconstruct_baseline_record(entry),
                test_dataset_id=dataset_id,
            )
            baseline_evidence_id = baseline_evidence.evidence_id

        search_results: list[dict[str, Any]] = []
        model_id = f"{spec.model_id_prefix}-{fold_id.lower()}"
        resume = _search_resume(
            args,
            workspace,
            fold_id=fold_id,
            family=spec.family_name,
            model_code_version=spec.model_code_version,
            train_dataset_id=train_dataset_id,
            validation_dataset_id=validation_dataset_id,
            factor_order=factor_order,
            runtime={"lightgbm_version": lgb.__version__},
        )
        adopted_fold = _adopt_completed_fold(
            args,
            workspace,
            model_store,
            selection_store,
            resume,
            fold_id=fold_id,
            model_id=model_id,
            dataset_id=dataset_id,
            train_dataset_id=train_dataset_id,
            validation_dataset_id=validation_dataset_id,
            factor_order=factor_order,
            configs=configs,
            budget_for=_lightgbm_search_budget,
        )
        if adopted_fold is not None:
            fold_reports.append(adopted_fold.report_entry)
            oos_folds[fold_id] = adopted_fold.oos_fold
            print(f"  {fold_id} adopted from verified artifacts: model={model_id}", flush=True)
            continue
        for config_index, config_payload in enumerate(configs):
            budget = _lightgbm_search_budget(config_payload)
            adopted = resume.adopt(config_index, config_payload, budget=budget)
            if adopted.record is not None:
                search_results.append(adopted.record)
                continue
            seed_validation_predictions: dict[int, pd.DataFrame] = {}
            seed_rank_ic: dict[int, float] = {}
            for seed in REQUIRED_SEEDS:
                model = spec.fit(config_payload, factor_order, train, validation, seed=seed)
                predictions = _m0_percentile_frame(model, validation, factor_order)
                seed_validation_predictions[seed] = predictions
                seed_rank_ic[seed] = _validation_rank_ic(predictions, validation)
            ensemble = build_seed_ensemble(
                seed_validation_predictions, validation_rank_ic=seed_rank_ic
            )
            ensemble_predictions = ensemble.predictions.rename(
                columns={"ensemble_score": "percentile_rank"}
            )
            ensemble_rank_ic = _validation_rank_ic(ensemble_predictions, validation)
            result: dict[str, Any] = {
                "config": config_payload,
                "ensemble_validation_rank_ic": ensemble_rank_ic,
                "seed_validation_rank_ic": seed_rank_ic,
                "worst_seed": ensemble.worst_seed,
            }
            search_results.append(result)
            resume.persist(config_index, config_payload, result, budget=budget)
            print(
                f"  {fold_id} config {config_index + 1}/{len(configs)}: "
                f"ensemble validation RankIC={ensemble_rank_ic:.4f}",
                flush=True,
            )
        best = max(search_results, key=lambda item: item["ensemble_validation_rank_ic"])
        best_config = best["config"]

        seed_checkpoints: dict[int, Path] = {}
        seed_validation_predictions = {}
        seed_rank_ic = {}
        for seed in REQUIRED_SEEDS:
            model = spec.fit(best_config, factor_order, train, validation, seed=seed)
            checkpoint_path = (
                workspace
                / "checkpoints"
                / f"{spec.model_id_prefix}-{fold_id.lower()}-final-{seed}"
                / "booster.txt"
            )
            _m0_save_booster(model, checkpoint_path)
            seed_checkpoints[seed] = checkpoint_path
            predictions = _m0_percentile_frame(model, validation, factor_order)
            seed_validation_predictions[seed] = predictions
            seed_rank_ic[seed] = _validation_rank_ic(predictions, validation)
        ensemble = build_seed_ensemble(
            seed_validation_predictions, validation_rank_ic=seed_rank_ic
        )
        _m0_write_bundle(
            model_store,
            model_id,
            dataset_id=train_dataset_id,
            config_payload=best_config,
            factor_order=factor_order,
            seed_checkpoints=seed_checkpoints,
            ensemble_predictions=ensemble.predictions.rename(
                columns={"ensemble_score": "percentile_rank"}
            ),
            family_name=spec.family_name,
            model_code_version=spec.model_code_version,
        )

        model_evidence_id = _existing_evidence_id(
            workspace / "selections", selection_store, "model", fold_id, dataset_id
        )
        if model_evidence_id is None:
            model_evidence = selection_store.write_model_selection(
                fold_id=fold_id,
                model_family=spec.family,
                model_artifact_id=model_id,
                model_bundle_digest=model_store.bundle_digest(model_id),
                training_dataset_id=train_dataset_id,
                validation_dataset_id=validation_dataset_id,
                test_dataset_id=dataset_id,
                baseline_selection_id=baseline_evidence_id,
                validation_rank_ic=float(best["ensemble_validation_rank_ic"]),
            )
            model_evidence_id = model_evidence.evidence_id

        loaded = {
            seed: _m0_load_booster(model_store, model_id, seed, carrier=spec.carrier)
            for seed in REQUIRED_SEEDS
        }
        candidate_scores = _m0_seed_mean_scores(loaded, test, factor_order)
        candidate_artifact = prediction_store.write(
            selection_id=model_evidence_id,
            kind="candidate",
            fold_id=fold_id,
            dataset_id=dataset_id,
            model_id=model_id,
            scores=candidate_scores,
            selections=selection_store,
        )
        baseline_scores = baseline_predictions[fold_id].loc[
            :, ["signal_date", "ts_code", "score"]
        ]
        baseline_artifact = prediction_store.write(
            selection_id=baseline_evidence_id,
            kind="best_simple",
            fold_id=fold_id,
            dataset_id=dataset_id,
            model_id=str(entry["selected_baseline_id"]),
            scores=baseline_scores,
            selections=selection_store,
        )
        oos_folds[fold_id] = ModelOosFold(
            model_selection_id=model_evidence_id,
            candidate_prediction_id=candidate_artifact.prediction_id,
            best_simple_prediction_id=baseline_artifact.prediction_id,
        )
        fold_reports.append(
            {
                "fold_id": fold_id,
                "dataset_id": dataset_id,
                "model_id": model_id,
                "best_config": best_config,
                "search_results": search_results,
                "baseline_selection_id": baseline_evidence_id,
                "model_selection_id": model_evidence_id,
            }
        )
        print(f"  {fold_id} complete: model={model_id}", flush=True)

    walk_forward = evaluate_model_walk_forward(
        oos_folds,
        dataset_store=dataset_store,
        model_store=model_store,
        selection_root=workspace / "selections",
        prediction_root=workspace / "predictions",
        minimum_cross_section=30,
    )
    evaluation = walk_forward.evaluation
    return {
        "family": spec.family,
        "mode": "full",
        "folds": fold_reports,
        "oos": {
            "valid_day_count": evaluation.summary.valid_day_count,
            "mean_rank_ic": evaluation.summary.mean_rank_ic,
            "icir": evaluation.summary.icir,
            "hac_t_statistic": evaluation.hac.t_statistic,
            "baseline_mean_rank_ic": evaluation.baseline_mean_rank_ic,
            "delta_rank_ic": evaluation.delta_rank_ic,
            "yearly_mean_rank_ic": {
                str(year): value for year, value in evaluation.yearly_mean_rank_ic.items()
            },
            "gates": {
                gate: {
                    "status": gate_result.status.value,
                    "observed_value": gate_result.observed_value,
                    "threshold": gate_result.threshold,
                }
                for gate, gate_result in sorted(evaluation.gate_report.results.items())
            },
            "all_passed": evaluation.gate_report.all_passed,
        },
    }


def run_full_m0(args: argparse.Namespace) -> dict[str, Any]:
    """Full real-data M0 (LightGBM LambdaRank) pipeline."""
    return _run_full_lightgbm_family(args, M0_SPEC)


def run_full_m0r(args: argparse.Namespace) -> dict[str, Any]:
    """Full real-data M0R (LightGBM rank-regression) pipeline.

    Identical to M0 in every respect the evidence chain cares about -- same frozen-v2 factor
    set, same F1-F4 folds, same published datasets, same 24-config search budget, same
    three-seed percentile-mean ensemble, same signed selection/model/prediction bundles, same
    G1-G6 gates against the same best_simple baseline -- and different only in the carrier.
    That is the point: the two reports are directly comparable fold by fold.
    """
    return _run_full_lightgbm_family(args, M0R_SPEC)


def m1_config_grid(max_configs: int | None = None) -> list[dict[str, float | int]]:
    """E7-budget M1 search space (3x3x3 = 27 <= 30 configs)."""
    grid: list[dict[str, float | int]] = [
        {"hidden_dim": hidden, "num_blocks": blocks, "dropout": dropout}
        for hidden in (64, 128, 256)
        for blocks in (1, 2, 3)
        for dropout in (0.0, 0.1, 0.2)
    ]
    return grid if max_configs is None else grid[:max_configs]


def _existing_evidence_id(
    store_root: Path,
    verifier: SelectionEvidenceStore,
    kind: str,
    fold_id: str,
    test_dataset_id: str,
) -> str | None:
    """Locate already-signed evidence for one fold (content-addressed idempotence)."""
    directory = store_root / kind
    if not directory.exists():
        return None
    for candidate in sorted(directory.glob("*.json")):
        evidence_id = candidate.stem
        try:
            if kind == "baseline":
                baseline_evidence = verifier.load_baseline_selection(evidence_id)
                if (
                    baseline_evidence.fold_id == fold_id
                    and baseline_evidence.test_dataset_id == test_dataset_id
                ):
                    return evidence_id
            else:
                model_evidence = verifier.load_model_selection(evidence_id)
                if (
                    model_evidence.fold_id == fold_id
                    and model_evidence.test_dataset_id == test_dataset_id
                ):
                    return evidence_id
        except ValueError:
            continue
    return None


TORCH_EARLY_STOPPING_PATIENCE = 10
"""Validation patience for every torch fit, search and final alike.

A single constant because the resume budget binds it: a per-call literal would let the persisted
record claim a stopping rule the code no longer uses, and adoption would then be comparing a number
nobody reads.
"""


def _train_cross_section_seed(
    config_payload: Mapping[str, Any],
    factor_order: tuple[str, ...],
    train_batches: Mapping[date, TrainingBatch],
    validation_batches: Mapping[date, TrainingBatch],
    *,
    seed: int,
    checkpoint_store: CheckpointStore,
    checkpoint_prefix: str,
    train_dataset_id: str,
    validation_dataset_id: str,
    max_epochs: int,
    model_factory: Any,
    learning_rate: float = 0.003,
) -> tuple[torch.nn.Module, Path]:
    # Seed before init, not only before training: see _train_cross_section_smoke.
    configure_reproducibility(seed)
    model = model_factory(config_payload, factor_order)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    run = run_validation_governed_training(
        model,
        optimizer,
        train_batches=train_batches,
        validation_batches=validation_batches,
        train_view=DatasetView(role=DatasetRole.TRAIN, dataset_id=train_dataset_id),
        validation_view=DatasetView(
            role=DatasetRole.VALIDATION, dataset_id=validation_dataset_id
        ),
        trainer_config=TrainerConfig(
            max_epochs=max_epochs, early_stopping_patience=TORCH_EARLY_STOPPING_PATIENCE
        ),
        seed=seed,
        checkpoint_store=checkpoint_store,
        checkpoint_prefix=checkpoint_prefix,
    )
    return model, checkpoint_store._root / run.checkpoint_ids[-1] / "checkpoint.pt"


def _ordered_search_results(
    configs: Sequence[Mapping[str, Any]], completed: Mapping[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Restore canonical grid order before the existing stable best-config selection."""
    expected_indexes = set(range(len(configs)))
    if set(completed) != expected_indexes:
        raise ValueError("parallel search did not produce every configuration result")
    ordered: list[dict[str, Any]] = []
    for config_index, config_payload in enumerate(configs):
        result = completed[config_index]
        if result.get("config") != dict(config_payload):
            raise ValueError("parallel search result does not match its configuration")
        ordered.append(result)
    return ordered


def _ensemble_from_seed_fits(
    fits: Mapping[int, Mapping[str, object]],
    config_payload: Mapping[str, Any],
    validation_labels: pd.DataFrame,
) -> dict[str, Any]:
    """Assemble one config's search result from seed fits that ran in other processes.

    The shape returned is the serial loop's shape exactly, because it becomes both a search record
    and a report entry, and a resumed run has to be able to adopt a record written by either
    granularity interchangeably. Refusing an incomplete seed set rather than ensembling whatever
    arrived is the scheduler's rule applied one level down: selecting from a partial grid is
    invalid, and so is selecting on two seeds out of three.
    """
    if set(fits) != set(REQUIRED_SEEDS):
        raise ValueError(
            f"seed-level search returned seeds {sorted(fits)}, expected all of "
            f"{sorted(REQUIRED_SEEDS)}"
        )
    predictions = {seed: cast(pd.DataFrame, fits[seed]["predictions"]) for seed in REQUIRED_SEEDS}
    rank_ic = {seed: cast(float, fits[seed]["validation_rank_ic"]) for seed in REQUIRED_SEEDS}
    ensemble = build_seed_ensemble(predictions, validation_rank_ic=rank_ic)
    ensemble_predictions = ensemble.predictions.rename(
        columns={"ensemble_score": "percentile_rank"}
    )
    return {
        "config": dict(config_payload),
        "ensemble_validation_rank_ic": _validation_rank_ic(ensemble_predictions, validation_labels),
        "seed_validation_rank_ic": rank_ic,
        "worst_seed": ensemble.worst_seed,
    }


def _constant_budget(
    budget: Mapping[str, Any],
) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    """A per-config budget function for families whose budget does not depend on the config.

    The torch families' budget is the epoch and stopping contract, identical for every config, so
    the callable ignores its argument. Written as a named factory rather than a lambda with a
    default argument because mypy cannot infer the type of that lambda, and an unannotated callable
    at a binding site is how a check stops being one.
    """

    def budget_for(_config: Mapping[str, Any]) -> Mapping[str, Any]:
        return budget

    return budget_for


@dataclass(frozen=True)
class _FullRunContext:
    workspace: Path
    dataset_store: DatasetArtifactStore
    model_store: ModelArtifactBundleStore
    checkpoint_store: CheckpointStore
    selection_store: SelectionEvidenceStore
    prediction_store: FrozenPredictionStore
    baseline_report: dict[str, Any]
    baseline_predictions: dict[str, pd.DataFrame]
    holder: dict[str, Any]
    resolver: SplitResolver
    versions: dict[str, dict[str, str]]
    factor_members: tuple[Any, ...]
    factor_order: tuple[str, ...]
    extract: Callable[..., Any]


def setup_full_run_context(args: argparse.Namespace) -> _FullRunContext:
    """Validate trust-root inputs and build every shared store/dataset dependency."""
    secret = os.environ.get("QRV2_SELECTION_EVIDENCE_HMAC_KEY", "")
    if not secret:
        raise SystemExit(
            "full runs require QRV2_SELECTION_EVIDENCE_HMAC_KEY (signed evidence chain)"
        )
    workspace: Path = args.workspace
    holder = new_holder()
    calendar = load_calendar(holder)
    resolver = SplitResolver(calendar)
    factor_ids, version_tables = factor_source_plan(args.factor_source)
    versions = {table: current_versions(holder, table) for table in version_tables}
    factor_members = factor_members_for(factor_ids)
    factor_order = tuple(member.feature_name for member in factor_members)
    print(
        f"factor source: {args.factor_source} "
        f"({len(factor_ids)} factors, tables={','.join(version_tables)})",
        flush=True,
    )
    return _FullRunContext(
        workspace=workspace,
        dataset_store=DatasetArtifactStore(workspace / "datasets"),
        model_store=ModelArtifactBundleStore(workspace / "models"),
        checkpoint_store=CheckpointStore(workspace / "checkpoints"),
        selection_store=SelectionEvidenceStore(
            workspace / "selections", signing_key=secret.encode("utf-8")
        ),
        prediction_store=FrozenPredictionStore(
            workspace / "predictions", signing_key=secret.encode("utf-8")
        ),
        baseline_report=json.loads(Path(args.baseline_report).read_text(encoding="utf-8")),
        baseline_predictions=load_baseline_predictions(Path(args.baseline_artifacts)),
        holder=holder,
        resolver=resolver,
        versions=versions,
        factor_members=factor_members,
        factor_order=factor_order,
        extract=extract_fn_for(args.factor_source),
    )


def _run_full_cross_section(
    args: argparse.Namespace,
    *,
    family: str,
    family_name: str,
    model_code_version: str,
    configs: list[dict[str, Any]],
    model_factory: Any,
) -> dict[str, Any]:
    """Full real-data cross-sectional pipeline (M1/M3 shared).

    assembly -> publication -> per-fold constrained search -> final bundles ->
    signed evidence chain -> frozen TEST predictions -> G1-G6 walk-forward gates.
    """
    context = setup_full_run_context(args)
    workspace = context.workspace
    dataset_store = context.dataset_store
    model_store = context.model_store
    checkpoint_store = context.checkpoint_store
    selection_store = context.selection_store
    prediction_store = context.prediction_store
    baseline_report = context.baseline_report
    baseline_predictions = context.baseline_predictions
    holder = context.holder
    resolver = context.resolver
    versions = context.versions
    factor_members = context.factor_members
    factor_order = context.factor_order
    extract = context.extract

    max_epochs = int(args.max_epochs)
    oos_folds: dict[str, ModelOosFold] = {}
    fold_reports: list[dict[str, Any]] = []

    for fold_id in ORDINARY_FOLDS:
        train = extract(holder, resolver, fold_id, "train", factor_members, versions)
        validation = extract(holder, resolver, fold_id, "validation", factor_members, versions)
        test = extract(holder, resolver, fold_id, "test", factor_members, versions)
        frames = {"train": train, "validation": validation, "test": test}
        dataset_id = publish_fold_dataset(dataset_store, fold_id, frames, versions, factor_order)
        train_dataset_id = f"{dataset_id}:train"
        validation_dataset_id = f"{dataset_id}:validation"

        entry = next(e for e in baseline_report["selections"] if e["fold_id"] == fold_id)
        baseline_evidence_id = _existing_evidence_id(
            workspace / "selections", selection_store, "baseline", fold_id, dataset_id
        )
        if baseline_evidence_id is None:
            baseline_evidence = selection_store.write_baseline_selection(
                record=reconstruct_baseline_record(entry),
                test_dataset_id=dataset_id,
            )
            baseline_evidence_id = baseline_evidence.evidence_id

        # Constrained config search: per config, three-seed validation ensemble RankIC.
        search_results: list[dict[str, Any]] = []
        model_id = f"{family.lower()}-{fold_id.lower()}"
        resume = _search_resume(
            args,
            workspace,
            fold_id=fold_id,
            family=family_name,
            model_code_version=model_code_version,
            train_dataset_id=train_dataset_id,
            validation_dataset_id=validation_dataset_id,
            factor_order=factor_order,
            runtime=torch_runtime_bindings(),
        )
        torch_budget = _torch_search_budget(max_epochs)
        # Adoption is decided BEFORE this fold's inputs are built. A fold that is already frozen
        # needs neither a fitted preprocessor nor a single batch, and --resume is the planned way to
        # cross a fold boundary on a multi-day run rather than an exceptional event, so building
        # first meant paying minutes per adopted fold for work that the continue below discards.
        # Everything adoption reads -- the model id, the three dataset ids, the factor order, the
        # grid, the budget, and the baseline evidence written above -- is already in hand here.
        adopted_fold = _adopt_completed_fold(
            args,
            workspace,
            model_store,
            selection_store,
            resume,
            fold_id=fold_id,
            model_id=model_id,
            dataset_id=dataset_id,
            train_dataset_id=train_dataset_id,
            validation_dataset_id=validation_dataset_id,
            factor_order=factor_order,
            configs=configs,
            budget_for=_constant_budget(torch_budget),
        )
        if adopted_fold is not None:
            fold_reports.append(adopted_fold.report_entry)
            oos_folds[fold_id] = adopted_fold.oos_fold
            print(f"  {fold_id} adopted from verified artifacts: model={model_id}", flush=True)
            continue

        preprocessor = FoldPreprocessor(factor_order=factor_order)
        preprocessor.fit(train.loc[:, list(factor_order)], role="TRAIN")
        train_batches = build_static_training_batches(
            _prepare_frame(train, preprocessor, factor_order), factor_order=factor_order
        )
        validation_batches = build_static_training_batches(
            _prepare_frame(validation, preprocessor, factor_order), factor_order=factor_order
        )
        execution = _parallel_execution(args)
        _check_worker_memory(args, execution)
        configs_to_run: Sequence[tuple[int, dict[str, Any]]] = tuple(enumerate(configs))
        if execution.workers > 1:
            if execution.torch_threads is None:
                raise AssertionError("parallel execution validation did not require torch threads")
            completed: dict[int, dict[str, Any]] = {}
            unadopted: list[tuple[int, dict[str, Any]]] = []
            for config_index, config_payload in enumerate(configs):
                adopted = resume.adopt(config_index, config_payload, budget=torch_budget)
                if adopted.record is not None:
                    completed[config_index] = adopted.record
                    continue
                unadopted.append((config_index, config_payload))
            # Chosen by makespan, not by whether the workers look busy: a config task is three
            # fits that cannot be split once dispatched, so a grid that overruns the worker count
            # by one config pays a whole extra round of three.
            granularity = _search_granularity(len(unadopted), execution.workers)
            if granularity == "config":
                pending_tasks = [
                    SearchTask(
                        index=config_index,
                        payload={
                            "task": CrossSectionConfigTask(
                                family=cast(Literal["M1", "M3"], family),
                                config_index=config_index,
                                config_payload=config_payload,
                                factor_order=factor_order,
                                train_batches=train_batches,
                                validation_batches=validation_batches,
                                validation_labels=validation,
                                checkpoint_root=checkpoint_store._root,
                                checkpoint_prefix=(
                                    f"{family.lower()}-{fold_id.lower()}-c{config_index}"
                                ),
                                train_dataset_id=train_dataset_id,
                                validation_dataset_id=validation_dataset_id,
                                max_epochs=max_epochs,
                                early_stopping_patience=TORCH_EARLY_STOPPING_PATIENCE,
                                torch_threads=execution.torch_threads,
                            )
                        },
                    )
                    for config_index, config_payload in unadopted
                ]
                for config_index, parallel_result in run_parallel_search(
                    pending_tasks,
                    workers=execution.workers,
                    run_task=run_cross_section_config_task,
                    on_result=_record_persister(resume, configs, torch_budget),
                ):
                    completed[config_index] = cast(dict[str, Any], parallel_result)
            else:
                # A grid narrower than the worker count cannot fill them at config granularity, and
                # that is the shape of every zero-tuning comparison run -- one config, chosen so the
                # number carries no validation-selection bias. Splitting the unit of work to
                # (config, seed) is what turns six serial fits per fold into two parallel rounds.
                seed_tasks = [
                    SearchTask(
                        index=config_index * len(REQUIRED_SEEDS) + slot,
                        payload={
                            "task": CrossSectionSeedTask(
                                family=cast(Literal["M1", "M3"], family),
                                config_payload=config_payload,
                                factor_order=factor_order,
                                seed=seed,
                                train_batches=train_batches,
                                validation_batches=validation_batches,
                                validation_labels=validation,
                                checkpoint_root=checkpoint_store._root,
                                checkpoint_prefix=(
                                    f"{family.lower()}-{fold_id.lower()}"
                                    f"-c{config_index}-{seed}"
                                ),
                                train_dataset_id=train_dataset_id,
                                validation_dataset_id=validation_dataset_id,
                                max_epochs=max_epochs,
                                early_stopping_patience=TORCH_EARLY_STOPPING_PATIENCE,
                                torch_threads=execution.torch_threads,
                            )
                        },
                    )
                    for config_index, config_payload in unadopted
                    for slot, seed in enumerate(REQUIRED_SEEDS)
                ]
                fits_by_config: dict[int, dict[int, Mapping[str, object]]] = {}
                seed_records: dict[int, dict[str, Any]] = {}
                run_parallel_search(
                    seed_tasks,
                    workers=execution.workers,
                    run_task=run_cross_section_seed_task,
                    on_result=_seed_result_collector(
                        resume, configs, torch_budget, validation, fits_by_config, seed_records
                    ),
                )
                for config_index, _config in unadopted:
                    completed[config_index] = seed_records[config_index]
            search_results = _ordered_search_results(configs, completed)
            for config_index, _ in unadopted:
                result = search_results[config_index]
                print(
                    f"  {fold_id} config {config_index + 1}/{len(configs)}: "
                    f"ensemble validation RankIC={result['ensemble_validation_rank_ic']:.4f}",
                    flush=True,
                )
            configs_to_run = ()

        for config_index, config_payload in configs_to_run:
            adopted = resume.adopt(config_index, config_payload, budget=torch_budget)
            if adopted.record is not None:
                search_results.append(adopted.record)
                continue
            seed_validation_predictions: dict[int, pd.DataFrame] = {}
            seed_rank_ic: dict[int, float] = {}
            for seed in REQUIRED_SEEDS:
                model, _ = _train_cross_section_seed(
                    config_payload,
                    factor_order,
                    train_batches,
                    validation_batches,
                    seed=seed,
                    checkpoint_store=checkpoint_store,
                    checkpoint_prefix=f"{family.lower()}-{fold_id.lower()}-c{config_index}-{seed}",
                    train_dataset_id=train_dataset_id,
                    validation_dataset_id=validation_dataset_id,
                    max_epochs=max_epochs,
                    model_factory=model_factory,
                    learning_rate=float(config_payload.get("learning_rate", 0.003)),
                )
                predictions = _predict_validation_percentiles(model, validation_batches)
                seed_validation_predictions[seed] = predictions
                seed_rank_ic[seed] = _validation_rank_ic(predictions, validation)
            ensemble = build_seed_ensemble(
                seed_validation_predictions, validation_rank_ic=seed_rank_ic
            )
            ensemble_predictions = ensemble.predictions.rename(
                columns={"ensemble_score": "percentile_rank"}
            )
            ensemble_rank_ic = _validation_rank_ic(ensemble_predictions, validation)
            serial_result: dict[str, Any] = {
                "config": config_payload,
                "ensemble_validation_rank_ic": ensemble_rank_ic,
                "seed_validation_rank_ic": seed_rank_ic,
                "worst_seed": ensemble.worst_seed,
            }
            search_results.append(serial_result)
            resume.persist(config_index, config_payload, serial_result, budget=torch_budget)
            print(
                f"  {fold_id} config {config_index + 1}/{len(configs)}: "
                f"ensemble validation RankIC={ensemble_rank_ic:.4f}",
                flush=True,
            )
        best = max(search_results, key=lambda item: item["ensemble_validation_rank_ic"])
        best_config = best["config"]

        # Final three-seed training on the selected config -> immutable bundle.
        seed_checkpoints: dict[int, Path] = {}
        seed_validation_predictions = {}
        seed_rank_ic = {}
        if execution.workers > 1:
            if execution.torch_threads is None:
                raise AssertionError("parallel execution validation did not require torch threads")
            # The three final fits are independent, and on a fold whose grid already saturated the
            # workers they are the only serial part left. Same fit, same checkpoint layout, same
            # ensemble: the workers hand back the checkpoint paths, which is what the bundle write
            # consumes, so the parent freezes weights it did not train itself.
            final_tasks = [
                SearchTask(
                    index=seed,
                    payload={
                        "task": CrossSectionSeedTask(
                            family=cast(Literal["M1", "M3"], family),
                            config_payload=best_config,
                            factor_order=factor_order,
                            seed=seed,
                            train_batches=train_batches,
                            validation_batches=validation_batches,
                            validation_labels=validation,
                            checkpoint_root=checkpoint_store._root,
                            checkpoint_prefix=f"{family.lower()}-{fold_id.lower()}-final-{seed}",
                            train_dataset_id=train_dataset_id,
                            validation_dataset_id=validation_dataset_id,
                            max_epochs=max_epochs,
                            early_stopping_patience=TORCH_EARLY_STOPPING_PATIENCE,
                            torch_threads=execution.torch_threads,
                        )
                    },
                )
                for seed in REQUIRED_SEEDS
            ]
            for seed, fitted in run_parallel_search(
                final_tasks,
                workers=execution.workers,
                run_task=run_cross_section_seed_task,
            ):
                seed_checkpoints[seed] = Path(str(fitted["checkpoint_path"]))
                seed_validation_predictions[seed] = cast(
                    pd.DataFrame, fitted["predictions"]
                )
                seed_rank_ic[seed] = cast(float, fitted["validation_rank_ic"])
        else:
            for seed in REQUIRED_SEEDS:
                model, checkpoint_path = _train_cross_section_seed(
                    best_config,
                    factor_order,
                    train_batches,
                    validation_batches,
                    seed=seed,
                    checkpoint_store=checkpoint_store,
                    checkpoint_prefix=f"{family.lower()}-{fold_id.lower()}-final-{seed}",
                    train_dataset_id=train_dataset_id,
                    validation_dataset_id=validation_dataset_id,
                    max_epochs=max_epochs,
                    model_factory=model_factory,
                    learning_rate=float(best_config.get("learning_rate", 0.003)),
                )
                seed_checkpoints[seed] = checkpoint_path
                predictions = _predict_validation_percentiles(model, validation_batches)
                seed_validation_predictions[seed] = predictions
                seed_rank_ic[seed] = _validation_rank_ic(predictions, validation)
        ensemble = build_seed_ensemble(
            seed_validation_predictions, validation_rank_ic=seed_rank_ic
        )
        prototype = model_factory(best_config, factor_order)
        model_store.write(
            model_id,
            model_card=torch_model_card(
                card_family=family_name,
                dataset_id=dataset_id,
                model_code_version=model_code_version,
            ),
            config=torch_bundle_config(
                best_config, card_family=family_name, factor_count=len(factor_order)
            ),
            signature=asdict(prototype.artifact_signature()),
            feature_order=factor_order,
            preprocessing=_preprocessing_artifact(preprocessor),
            seed_checkpoints=seed_checkpoints,
            validation_predictions=ensemble.predictions.rename(
                columns={"ensemble_score": "percentile_rank"}
            ),
        )

        # Signed model selection evidence bound to the baseline evidence.
        model_evidence_id = _existing_evidence_id(
            workspace / "selections", selection_store, "model", fold_id, dataset_id
        )
        if model_evidence_id is None:
            model_evidence = selection_store.write_model_selection(
                fold_id=fold_id,
                model_family=family,
                model_artifact_id=model_id,
                model_bundle_digest=model_store.bundle_digest(model_id),
                training_dataset_id=train_dataset_id,
                validation_dataset_id=validation_dataset_id,
                test_dataset_id=dataset_id,
                baseline_selection_id=baseline_evidence_id,
                validation_rank_ic=float(best["ensemble_validation_rank_ic"]),
            )
            model_evidence_id = model_evidence.evidence_id

        # Frozen TEST predictions: candidate (verified bundle) + best_simple (baseline).
        loaded = load_bundle_for_evaluation(
            model_store,
            model_id,
            expected_dataset_id=dataset_id,
            expected_feature_order=factor_order,
            expected_preprocessor_hash=preprocessor.state_hash,
            device=torch.device("cpu"),
        )
        test_batches = build_static_training_batches(
            _prepare_frame(test.assign(raw_return_5d=0.0), preprocessor, factor_order),
            factor_order=factor_order,
        )
        candidate_scores = _concat_date_scores(loaded.models, test_batches)
        candidate_artifact = prediction_store.write(
            selection_id=model_evidence_id,
            kind="candidate",
            fold_id=fold_id,
            dataset_id=dataset_id,
            model_id=model_id,
            scores=candidate_scores,
            selections=selection_store,
        )
        baseline_scores = baseline_predictions[fold_id].loc[
            :, ["signal_date", "ts_code", "score"]
        ]
        baseline_artifact = prediction_store.write(
            selection_id=baseline_evidence_id,
            kind="best_simple",
            fold_id=fold_id,
            dataset_id=dataset_id,
            model_id=str(entry["selected_baseline_id"]),
            scores=baseline_scores,
            selections=selection_store,
        )
        oos_folds[fold_id] = ModelOosFold(
            model_selection_id=model_evidence_id,
            candidate_prediction_id=candidate_artifact.prediction_id,
            best_simple_prediction_id=baseline_artifact.prediction_id,
        )
        fold_reports.append(
            {
                "fold_id": fold_id,
                "dataset_id": dataset_id,
                "model_id": model_id,
                "best_config": best_config,
                "search_results": search_results,
                "baseline_selection_id": baseline_evidence_id,
                "model_selection_id": model_evidence_id,
            }
        )
        print(f"  {fold_id} complete: model={model_id}", flush=True)

    walk_forward = evaluate_model_walk_forward(
        oos_folds,
        dataset_store=dataset_store,
        model_store=model_store,
        selection_root=workspace / "selections",
        prediction_root=workspace / "predictions",
        minimum_cross_section=30,
    )
    evaluation = walk_forward.evaluation
    return {
        "family": family,
        "mode": "full",
        "folds": fold_reports,
        "oos": {
            "valid_day_count": evaluation.summary.valid_day_count,
            "mean_rank_ic": evaluation.summary.mean_rank_ic,
            "icir": evaluation.summary.icir,
            "hac_t_statistic": evaluation.hac.t_statistic,
            "baseline_mean_rank_ic": evaluation.baseline_mean_rank_ic,
            "delta_rank_ic": evaluation.delta_rank_ic,
            "yearly_mean_rank_ic": {
                str(year): value for year, value in evaluation.yearly_mean_rank_ic.items()
            },
            "gates": {
                gate: {
                    "status": gate_result.status.value,
                    "observed_value": gate_result.observed_value,
                    "threshold": gate_result.threshold,
                }
                for gate, gate_result in sorted(evaluation.gate_report.results.items())
            },
            "all_passed": evaluation.gate_report.all_passed,
        },
    }


def _m1_model_factory(config_payload: Mapping[str, Any], factor_order: tuple[str, ...]) -> Any:
    return ResidualMLP(
        ResidualMLPConfig(
            raw_factor_count=len(factor_order),
            hidden_dim=int(config_payload["hidden_dim"]),
            num_blocks=int(config_payload["num_blocks"]),
            dropout=float(config_payload["dropout"]),
        )
    )


def _m3_model_factory(config_payload: Mapping[str, Any], factor_order: tuple[str, ...]) -> Any:
    return SetAttention(
        SetAttentionConfig(
            raw_factor_count=len(factor_order),
            hidden_dim=int(config_payload["hidden_dim"]),
            num_heads=int(config_payload["num_heads"]),
            inducing_points=int(config_payload["inducing_points"]),
            num_blocks=int(config_payload["num_blocks"]),
            dropout=float(config_payload["dropout"]),
        )
    )


def m3_config_grid(max_configs: int | None = None) -> list[dict[str, Any]]:
    """Budget-compliant M3 subspace: architecture grid with dropout/lr fixed.

    The yaml space has 64 combinations (above the E7 budget of 30); the
    documented reduction fixes dropout=0.1 and learning_rate=0.0003 and
    searches hidden_dim x num_heads x inducing_points x num_blocks (16).
    """
    grid: list[dict[str, Any]] = [
        {
            "hidden_dim": hidden,
            "num_heads": heads,
            "inducing_points": points,
            "num_blocks": blocks,
            "dropout": 0.1,
            "learning_rate": 0.0003,
        }
        for hidden in (64, 128)
        for heads in (2, 4)
        for points in (32, 64)
        for blocks in (1, 2)
    ]
    return grid if max_configs is None else grid[:max_configs]


def run_full_m1(args: argparse.Namespace) -> dict[str, Any]:
    """Full real-data M1 (ResidualMLP) pipeline."""
    return _run_full_cross_section(
        args,
        family="M1",
        family_name=M1_FAMILY_NAME,
        model_code_version=M1_MODEL_CODE_VERSION,
        configs=m1_config_grid(args.max_configs),
        model_factory=_m1_model_factory,
    )


def run_full_m3(args: argparse.Namespace) -> dict[str, Any]:
    """Full real-data M3 (SetAttention) pipeline."""
    return _run_full_cross_section(
        args,
        family="M3",
        family_name=M3_FAMILY_NAME,
        model_code_version=M3_MODEL_CODE_VERSION,
        configs=m3_config_grid(args.max_configs),
        model_factory=_m3_model_factory,
    )


def m2_config_grid(max_configs: int | None = None) -> list[dict[str, Any]]:
    """M2 search space: family x hidden_dim x dropout x learning_rate (24 <= 30)."""
    grid: list[dict[str, Any]] = [
        {"family": family, "hidden_dim": hidden, "dropout": dropout, "learning_rate": lr}
        for family in ("GRU", "TCN")
        for hidden in (64, 128)
        for dropout in (0.0, 0.1, 0.2)
        for lr in (0.0003, 0.001)
    ]
    return grid if max_configs is None else grid[:max_configs]


def _m2_model_factory(config_payload: Mapping[str, Any], factor_order: tuple[str, ...]) -> Any:
    if config_payload["family"] == "GRU":
        return TemporalGRU(
            TemporalGRUConfig(
                raw_factor_count=len(factor_order),
                hidden_dim=int(config_payload["hidden_dim"]),
                dropout=float(config_payload["dropout"]),
            )
        )
    return TemporalTCN(
        TemporalTCNConfig(
            raw_factor_count=len(factor_order),
            hidden_dim=int(config_payload["hidden_dim"]),
            dropout=float(config_payload["dropout"]),
        )
    )


def torch_model_card(
    *, card_family: str, dataset_id: str, model_code_version: str
) -> dict[str, str]:
    """The model card every torch carrier writes, built in exactly one place.

    ``dataset_id`` here is the BARE dataset id, which is what the reproduction audit then loads the
    TEST split with; the LightGBM carriers write ``<dataset>:train`` instead and the walk-forward
    evaluator accepts either, because both are legitimate and it checks membership rather than
    equality. What no consumer can accept is a field spelled one way here and expected another way
    there, which is what killed both real M1 runs and would have killed every M2 run.

    ``test_every_card_the_driver_can_write_is_accepted_downstream`` hands this function's output to
    the evaluator's card-family allowlist and the loader's code-version map, so the writer and its
    acceptors are checked against each other through the writer's own code instead of through a
    fixture restating what the writer is believed to do.
    """
    return {
        "family": card_family,
        "dataset_id": dataset_id,
        "model_status": "WALK_FORWARD_PASSED",
        "model_code_version": model_code_version,
    }


def m2_card_family(winning_family: object) -> str:
    """M2's card family is named by the WINNING configuration, not by the carrier.

    The grid searches GRU and TCN alike, so neither is knowable per fold before the search runs. The
    config hash binds the choice; this turns it into the name the bundle carries, and the
    evaluator's M2 allowlist has to accept every value it can return.
    """
    return f"TEMPORAL_{winning_family}"


def m2_model_code_version(winning_family: object) -> str:
    """The code version paired with :func:`m2_card_family`, from the same winning choice."""
    return f"temporal-{str(winning_family).lower()}-1"


def torch_bundle_config(
    config_payload: Mapping[str, Any], *, card_family: str, factor_count: int
) -> dict[str, Any]:
    """Exactly the fields that reconstruct the model, and none the search also happened to carry.

    ``artifact_loader._require_fields`` refuses a bundle whose config does not match the
    architecture's own config fields EXACTLY -- "model config has unsupported or missing fields" --
    and both full-run writers passed ``dict(best_config, raw_factor_count=...)`` straight through.
    M2's grid carries ``family`` (already the card's family) and ``learning_rate`` (an optimiser
    setting); M3's grid carries ``learning_rate``. Verified before this change: M1 loads, while
    M2/GRU, M2/TCN and M3 are all refused, so each of those runs would have trained for days, frozen
    and signed a bundle, and then found it unloadable by the evaluator, the reproduction audit and
    serving alike.

    M1 was clean by accident -- its grid holds only architecture fields -- which is why nothing
    caught this: M1 is the only torch family that has run, and both smoke paths build their config
    from ``asdict`` of the config dataclass or spell the fields by hand, so neither could reach it.

    Nothing is lost by filtering. The full winning payload stays in the fold's search record and in
    the report's ``best_config``, which is where ``learning_rate`` and the architecture choice
    belong; a bundle's config is only what reconstructs the weights. The field set is asked of the
    loader rather than restated here, so the writer and the acceptor cannot drift.
    """
    architecture_fields = model_config_fields(card_family) - {"raw_factor_count"}
    missing = sorted(architecture_fields.difference(config_payload))
    if missing:
        raise ValueError(
            f"the winning {card_family} config carries no {missing}, so no bundle could be "
            "reconstructed from it"
        )
    config: dict[str, Any] = {name: config_payload[name] for name in sorted(architecture_fields)}
    config["raw_factor_count"] = factor_count
    return config


def torch_card_conventions() -> tuple[tuple[str, str, str], ...]:
    """Every ``(carrier, card family, model_code_version)`` the torch writers can produce.

    Derived, not listed: M2's entries come from the grid actually searched, so an architecture that
    becomes searchable becomes checked here automatically instead of being trainable but unloadable.
    This is the list the tests compare against the evaluator's card-family allowlist and the
    loader's code-version map -- the comparison whose absence let ``{"M2", "TEMPORAL"}`` sit in the
    allowlist while this driver wrote ``TEMPORAL_GRU`` and ``TEMPORAL_TCN``.
    """
    return (
        ("M1", M1_FAMILY_NAME, M1_MODEL_CODE_VERSION),
        ("M3", M3_FAMILY_NAME, M3_MODEL_CODE_VERSION),
        *(
            ("M2", m2_card_family(choice), m2_model_code_version(choice))
            for choice in sorted({str(config["family"]) for config in m2_config_grid(None)})
        ),
    )


def _temporal_panel_batches(
    panel: pd.DataFrame,
    factor_order: tuple[str, ...],
    allowed_dates: set[date],
) -> dict[date, TemporalTrainingBatch]:
    batches = build_temporal_training_batches(
        panel,
        factor_order=factor_order,
        history_length=HISTORY_LENGTH,
        minimum_valid_history_length=MINIMUM_VALID_HISTORY_LENGTH,
    )
    return {day: batch for day, batch in batches.items() if day in allowed_dates}


def run_full_m2(args: argparse.Namespace) -> dict[str, Any]:
    """Full real-data M2 (GRU/TCN temporal) pipeline with the signed evidence chain."""
    context = setup_full_run_context(args)
    workspace = context.workspace
    dataset_store = context.dataset_store
    model_store = context.model_store
    checkpoint_store = context.checkpoint_store
    selection_store = context.selection_store
    prediction_store = context.prediction_store
    baseline_report = context.baseline_report
    baseline_predictions = context.baseline_predictions
    holder = context.holder
    resolver = context.resolver
    versions = context.versions
    factor_members = context.factor_members
    factor_order = context.factor_order
    extract = context.extract

    configs = m2_config_grid(args.max_configs)
    max_epochs = int(args.max_epochs)
    oos_folds: dict[str, ModelOosFold] = {}
    fold_reports: list[dict[str, Any]] = []

    for fold_id in ORDINARY_FOLDS:
        train = extract(holder, resolver, fold_id, "train", factor_members, versions)
        validation = extract(holder, resolver, fold_id, "validation", factor_members, versions)
        test = extract(holder, resolver, fold_id, "test", factor_members, versions)
        frames = {"train": train, "validation": validation, "test": test}
        dataset_id = publish_fold_dataset(dataset_store, fold_id, frames, versions, factor_order)
        train_dataset_id = f"{dataset_id}:train"
        validation_dataset_id = f"{dataset_id}:validation"

        entry = next(e for e in baseline_report["selections"] if e["fold_id"] == fold_id)
        baseline_evidence_id = _existing_evidence_id(
            workspace / "selections", selection_store, "baseline", fold_id, dataset_id
        )
        if baseline_evidence_id is None:
            baseline_evidence = selection_store.write_baseline_selection(
                record=reconstruct_baseline_record(entry),
                test_dataset_id=dataset_id,
            )
            baseline_evidence_id = baseline_evidence.evidence_id

        search_results: list[dict[str, Any]] = []
        # M2's card family and code version are derived from the WINNING config (TEMPORAL_GRU or
        # TEMPORAL_TCN), so neither is knowable per fold before the search runs. The config payload
        # carries the family choice and the config hash binds it; these two labels only keep M2's
        # records separate from every other family's.
        model_id = f"m2-{fold_id.lower()}"
        resume = _search_resume(
            args,
            workspace,
            fold_id=fold_id,
            family="M2",
            model_code_version="m2-temporal-search",
            train_dataset_id=train_dataset_id,
            validation_dataset_id=validation_dataset_id,
            factor_order=factor_order,
            runtime=torch_runtime_bindings(),
        )
        torch_budget = _temporal_search_budget(max_epochs)
        # Adoption is decided BEFORE this fold's panels are built, and for M2 that ordering is worth
        # more than anywhere else: an adopted fold would otherwise reconstruct the train panel, the
        # train+validation panel and the TEST panel -- measured on the real F1 fold at 400 s for the
        # first two together and 552 s for the TEST one, about sixteen minutes -- and then discard
        # all three at the continue below. Adoption needs none of them: the window contract it binds
        # comes from _temporal_search_budget, which reads the frozen constants, not a panel.
        adopted_fold = _adopt_completed_fold(
            args,
            workspace,
            model_store,
            selection_store,
            resume,
            fold_id=fold_id,
            model_id=model_id,
            dataset_id=dataset_id,
            train_dataset_id=train_dataset_id,
            validation_dataset_id=validation_dataset_id,
            factor_order=factor_order,
            configs=configs,
            budget_for=_constant_budget(torch_budget),
        )
        if adopted_fold is not None:
            fold_reports.append(adopted_fold.report_entry)
            oos_folds[fold_id] = adopted_fold.oos_fold
            print(f"  {fold_id} adopted from verified artifacts: model={model_id}", flush=True)
            continue

        preprocessor = FoldPreprocessor(factor_order=factor_order)
        preprocessor.fit(train.loc[:, list(factor_order)], role="TRAIN")
        prepared_train = _prepare_frame(train, preprocessor, factor_order)
        prepared_validation = _prepare_frame(validation, preprocessor, factor_order)
        prepared_test = _prepare_frame(test, preprocessor, factor_order)

        train_dates = set(pd.to_datetime(train["signal_date"]).dt.date)
        validation_dates = set(pd.to_datetime(validation["signal_date"]).dt.date)
        test_dates = set(pd.to_datetime(test["signal_date"]).dt.date)

        train_batches = _temporal_panel_batches(prepared_train, factor_order, train_dates)
        validation_panel = pd.concat(
            [prepared_train, prepared_validation], ignore_index=True
        )
        validation_batches = _temporal_panel_batches(
            validation_panel, factor_order, validation_dates
        )
        # The same composition the reproduction audit rebuilds from the published artifacts, so
        # the audit and this run cannot disagree about what the TEST panel was.
        test_batches = temporal_test_panel_batches(
            prepared_train,
            prepared_validation,
            prepared_test,
            factor_order,
            test_dates,
        )
        execution = _parallel_execution(args)
        _check_worker_memory(args, execution)
        configs_to_run: Sequence[tuple[int, dict[str, Any]]] = tuple(enumerate(configs))
        fold_spec: TemporalFoldSpec | None = None
        if execution.workers > 1:
            if execution.torch_threads is None:
                raise AssertionError("parallel execution validation did not require torch threads")
            # The task carries identifiers, not batches: an M2 batch holds a 20-day window per row
            # and the validation panel is train+validation concatenated, so each worker rebuilds
            # the fold's panels once from the published dataset artifact and reuses them for every
            # configuration it is handed. temporal_fold_batches is the tested equivalence point.
            fold_spec = TemporalFoldSpec(
                workspace=workspace,
                fold_id=fold_id,
                dataset_id=dataset_id,
                factor_order=factor_order,
                preprocessing={
                    "state_hash": preprocessor.state_hash,
                    "state_payload": preprocessor.state_payload(),
                },
                torch_threads=execution.torch_threads,
            )
            completed: dict[int, dict[str, Any]] = {}
            unadopted: list[tuple[int, dict[str, Any]]] = []
            for config_index, config_payload in enumerate(configs):
                adopted = resume.adopt(config_index, config_payload, budget=torch_budget)
                if adopted.record is not None:
                    completed[config_index] = adopted.record
                    continue
                unadopted.append((config_index, config_payload))
            granularity = _search_granularity(len(unadopted), execution.workers)
            if granularity == "config":
                pending_tasks = [
                    SearchTask(
                        index=config_index,
                        payload={
                            "task": TemporalConfigTask(
                                config_index=config_index,
                                config_payload=config_payload,
                                fold=fold_spec,
                                train_dataset_id=train_dataset_id,
                                validation_dataset_id=validation_dataset_id,
                                checkpoint_root=checkpoint_store._root,
                                checkpoint_prefix=f"m2-{fold_id.lower()}-c{config_index}",
                                max_epochs=max_epochs,
                                early_stopping_patience=TORCH_EARLY_STOPPING_PATIENCE,
                            )
                        },
                    )
                    for config_index, config_payload in unadopted
                ]
                for config_index, parallel_result in run_parallel_search(
                    pending_tasks,
                    workers=execution.workers,
                    run_task=run_temporal_config_task,
                    on_result=_record_persister(resume, configs, torch_budget),
                ):
                    completed[config_index] = cast(dict[str, Any], parallel_result)
            else:
                # A grid narrower than the worker count cannot fill them at config granularity.
                # For M2 that case is not hypothetical: each worker rebuilds the fold's temporal
                # panels once and reuses them, so splitting to (config, seed) costs one extra
                # reconstruction per worker and turns six serial fits per fold into two rounds.
                seed_tasks = [
                    SearchTask(
                        index=config_index * len(REQUIRED_SEEDS) + slot,
                        payload={
                            "task": TemporalSeedTask(
                                config_payload=config_payload,
                                seed=seed,
                                fold=fold_spec,
                                train_dataset_id=train_dataset_id,
                                validation_dataset_id=validation_dataset_id,
                                checkpoint_root=checkpoint_store._root,
                                checkpoint_prefix=(
                                    f"m2-{fold_id.lower()}-c{config_index}-{seed}"
                                ),
                                max_epochs=max_epochs,
                                early_stopping_patience=TORCH_EARLY_STOPPING_PATIENCE,
                            )
                        },
                    )
                    for config_index, config_payload in unadopted
                    for slot, seed in enumerate(REQUIRED_SEEDS)
                ]
                fits_by_config: dict[int, dict[int, Mapping[str, object]]] = {}
                seed_records: dict[int, dict[str, Any]] = {}
                run_parallel_search(
                    seed_tasks,
                    workers=execution.workers,
                    run_task=run_temporal_seed_task,
                    on_result=_seed_result_collector(
                        resume, configs, torch_budget, validation, fits_by_config, seed_records
                    ),
                )
                for config_index, _config in unadopted:
                    completed[config_index] = seed_records[config_index]
            search_results = _ordered_search_results(configs, completed)
            for config_index, _ in unadopted:
                result = search_results[config_index]
                print(
                    f"  {fold_id} config {config_index + 1}/{len(configs)}: "
                    f"ensemble validation RankIC={result['ensemble_validation_rank_ic']:.4f}",
                    flush=True,
                )
            configs_to_run = ()

        for config_index, config_payload in configs_to_run:
            adopted = resume.adopt(config_index, config_payload, budget=torch_budget)
            if adopted.record is not None:
                search_results.append(adopted.record)
                continue
            seed_validation_predictions: dict[int, pd.DataFrame] = {}
            seed_rank_ic: dict[int, float] = {}
            for seed in REQUIRED_SEEDS:
                configure_reproducibility(seed)  # seed before init; see the smoke helper
                model = _m2_model_factory(config_payload, factor_order)
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=float(config_payload["learning_rate"])
                )
                run = run_validation_governed_training(
                    model,
                    optimizer,
                    train_batches=train_batches,
                    validation_batches=validation_batches,
                    train_view=DatasetView(role=DatasetRole.TRAIN, dataset_id=train_dataset_id),
                    validation_view=DatasetView(
                        role=DatasetRole.VALIDATION, dataset_id=validation_dataset_id
                    ),
                    trainer_config=TrainerConfig(
                        max_epochs=max_epochs,
                        early_stopping_patience=TORCH_EARLY_STOPPING_PATIENCE,
                    ),
                    seed=seed,
                    checkpoint_store=checkpoint_store,
                    checkpoint_prefix=(
                        f"m2-{fold_id.lower()}-c{config_index}-{seed}"
                    ),
                )
                predictions = _predict_temporal_validation_percentiles(
                    model, validation_batches
                )
                seed_validation_predictions[seed] = predictions
                seed_rank_ic[seed] = _validation_rank_ic(predictions, validation)
            ensemble = build_seed_ensemble(
                seed_validation_predictions, validation_rank_ic=seed_rank_ic
            )
            ensemble_predictions = ensemble.predictions.rename(
                columns={"ensemble_score": "percentile_rank"}
            )
            ensemble_rank_ic = _validation_rank_ic(ensemble_predictions, validation)
            serial_result: dict[str, Any] = {
                "config": config_payload,
                "ensemble_validation_rank_ic": ensemble_rank_ic,
                "seed_validation_rank_ic": seed_rank_ic,
                "worst_seed": ensemble.worst_seed,
            }
            search_results.append(serial_result)
            resume.persist(config_index, config_payload, serial_result, budget=torch_budget)
            print(
                f"  {fold_id} config {config_index + 1}/{len(configs)}: "
                f"ensemble validation RankIC={ensemble_rank_ic:.4f}",
                flush=True,
            )
        best = max(search_results, key=lambda item: item["ensemble_validation_rank_ic"])
        best_config = best["config"]

        seed_checkpoints: dict[int, Path] = {}
        seed_validation_predictions = {}
        seed_rank_ic = {}
        if execution.workers > 1:
            if execution.torch_threads is None or fold_spec is None:
                raise AssertionError("parallel execution validation did not require torch threads")
            # For a fold whose 24-config grid already saturated the workers, these three fits are
            # the only serial part left, and at M2's measured ~8.6h per fit they are half the fold.
            final_tasks = [
                SearchTask(
                    index=seed,
                    payload={
                        "task": TemporalSeedTask(
                            config_payload=best_config,
                            seed=seed,
                            fold=fold_spec,
                            train_dataset_id=train_dataset_id,
                            validation_dataset_id=validation_dataset_id,
                            checkpoint_root=checkpoint_store._root,
                            checkpoint_prefix=f"m2-{fold_id.lower()}-final-{seed}",
                            max_epochs=max_epochs,
                            early_stopping_patience=TORCH_EARLY_STOPPING_PATIENCE,
                        )
                    },
                )
                for seed in REQUIRED_SEEDS
            ]
            for seed, fitted in run_parallel_search(
                final_tasks,
                workers=execution.workers,
                run_task=run_temporal_seed_task,
            ):
                seed_checkpoints[seed] = Path(str(fitted["checkpoint_path"]))
                seed_validation_predictions[seed] = cast(pd.DataFrame, fitted["predictions"])
                seed_rank_ic[seed] = cast(float, fitted["validation_rank_ic"])
        else:
            for seed in REQUIRED_SEEDS:
                configure_reproducibility(seed)  # seed before init; see the smoke helper
                model = _m2_model_factory(best_config, factor_order)
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=float(best_config["learning_rate"])
                )
                run = run_validation_governed_training(
                    model,
                    optimizer,
                    train_batches=train_batches,
                    validation_batches=validation_batches,
                    train_view=DatasetView(role=DatasetRole.TRAIN, dataset_id=train_dataset_id),
                    validation_view=DatasetView(
                        role=DatasetRole.VALIDATION, dataset_id=validation_dataset_id
                    ),
                    trainer_config=TrainerConfig(
                        max_epochs=max_epochs,
                        early_stopping_patience=TORCH_EARLY_STOPPING_PATIENCE,
                    ),
                    seed=seed,
                    checkpoint_store=checkpoint_store,
                    checkpoint_prefix=f"m2-{fold_id.lower()}-final-{seed}",
                )
                seed_checkpoints[seed] = (
                    workspace / "checkpoints" / run.checkpoint_ids[-1] / "checkpoint.pt"
                )
                predictions = _predict_temporal_validation_percentiles(model, validation_batches)
                seed_validation_predictions[seed] = predictions
                seed_rank_ic[seed] = _validation_rank_ic(predictions, validation)
        ensemble = build_seed_ensemble(
            seed_validation_predictions, validation_rank_ic=seed_rank_ic
        )
        prototype = _m2_model_factory(best_config, factor_order)
        model_store.write(
            model_id,
            model_card=torch_model_card(
                card_family=m2_card_family(best_config["family"]),
                dataset_id=dataset_id,
                model_code_version=m2_model_code_version(best_config["family"]),
            ),
            config=torch_bundle_config(
                best_config,
                card_family=m2_card_family(best_config["family"]),
                factor_count=len(factor_order),
            ),
            signature=asdict(prototype.artifact_signature()),
            feature_order=factor_order,
            preprocessing=_preprocessing_artifact(preprocessor),
            seed_checkpoints=seed_checkpoints,
            validation_predictions=ensemble.predictions.rename(
                columns={"ensemble_score": "percentile_rank"}
            ),
        )

        model_evidence_id = _existing_evidence_id(
            workspace / "selections", selection_store, "model", fold_id, dataset_id
        )
        if model_evidence_id is None:
            model_evidence = selection_store.write_model_selection(
                fold_id=fold_id,
                model_family="M2",
                model_artifact_id=model_id,
                model_bundle_digest=model_store.bundle_digest(model_id),
                training_dataset_id=train_dataset_id,
                validation_dataset_id=validation_dataset_id,
                test_dataset_id=dataset_id,
                baseline_selection_id=baseline_evidence_id,
                validation_rank_ic=float(best["ensemble_validation_rank_ic"]),
            )
            model_evidence_id = model_evidence.evidence_id

        loaded = load_bundle_for_evaluation(
            model_store,
            model_id,
            expected_dataset_id=dataset_id,
            expected_feature_order=factor_order,
            expected_preprocessor_hash=preprocessor.state_hash,
            device=torch.device("cpu"),
        )
        candidate_scores = _concat_date_scores(loaded.models, test_batches)
        candidate_artifact = prediction_store.write(
            selection_id=model_evidence_id,
            kind="candidate",
            fold_id=fold_id,
            dataset_id=dataset_id,
            model_id=model_id,
            scores=candidate_scores,
            selections=selection_store,
        )
        baseline_scores = baseline_predictions[fold_id].loc[
            :, ["signal_date", "ts_code", "score"]
        ]
        baseline_artifact = prediction_store.write(
            selection_id=baseline_evidence_id,
            kind="best_simple",
            fold_id=fold_id,
            dataset_id=dataset_id,
            model_id=str(entry["selected_baseline_id"]),
            scores=baseline_scores,
            selections=selection_store,
        )
        oos_folds[fold_id] = ModelOosFold(
            model_selection_id=model_evidence_id,
            candidate_prediction_id=candidate_artifact.prediction_id,
            best_simple_prediction_id=baseline_artifact.prediction_id,
        )
        fold_reports.append(
            {
                "fold_id": fold_id,
                "dataset_id": dataset_id,
                "model_id": model_id,
                "best_config": best_config,
                "search_results": search_results,
                "baseline_selection_id": baseline_evidence_id,
                "model_selection_id": model_evidence_id,
            }
        )
        print(f"  {fold_id} complete: model={model_id}", flush=True)

    walk_forward = evaluate_model_walk_forward(
        oos_folds,
        dataset_store=dataset_store,
        model_store=model_store,
        selection_root=workspace / "selections",
        prediction_root=workspace / "predictions",
        minimum_cross_section=30,
    )
    evaluation = walk_forward.evaluation
    return {
        "family": "M2",
        "mode": "full",
        "folds": fold_reports,
        "unscored_scoreable_rows_by_fold": dict(walk_forward.unscored_scoreable_rows_by_fold),
        "oos": {
            "valid_day_count": evaluation.summary.valid_day_count,
            "mean_rank_ic": evaluation.summary.mean_rank_ic,
            "icir": evaluation.summary.icir,
            "hac_t_statistic": evaluation.hac.t_statistic,
            "baseline_mean_rank_ic": evaluation.baseline_mean_rank_ic,
            "delta_rank_ic": evaluation.delta_rank_ic,
            "yearly_mean_rank_ic": {
                str(year): value for year, value in evaluation.yearly_mean_rank_ic.items()
            },
            "gates": {
                gate: {
                    "status": gate_result.status.value,
                    "observed_value": gate_result.observed_value,
                    "threshold": gate_result.threshold,
                }
                for gate, gate_result in sorted(evaluation.gate_report.results.items())
            },
            "all_passed": evaluation.gate_report.all_passed,
        },
    }


def _maybe_emit_promotion_evidence(args: argparse.Namespace, report: dict[str, Any]) -> None:
    """Turn a gate-passing run into something the lifecycle registry can actually consume.

    ``load_walk_forward_evidence`` is the only route from WALK_FORWARD_PASSED onward and it
    does not read this driver's report shape, so a run that passed all six gates still could
    not be promoted. Emission re-verifies the signed artifacts from disk rather than trusting
    the in-memory numbers, and cross-checks them against the report just written.

    Only a passing run emits: a failing one has nothing to promote, and writing a document the
    registry would reject anyway would just invite someone to hand-edit it.
    """
    oos = report.get("oos")
    if not isinstance(oos, dict) or oos.get("all_passed") is not True:
        print("promotion evidence not emitted: the run did not pass all six gates")
        return
    if args.workspace is None:
        print("promotion evidence not emitted: no --workspace to re-verify against")
        return
    out_path = Path(args.workspace) / PROMOTION_EVIDENCE_FILENAME
    try:
        summary = emit_promotion_evidence(Path(args.workspace), args.report, out_path)
    except (PromotionEvidenceError, ValueError) as error:
        # The report is already on disk, so nothing is lost -- but a run that passed the gates
        # and still cannot produce registry evidence is a real failure, not a warning.
        print(f"promotion evidence emission FAILED: {error}", file=sys.stderr)
        raise SystemExit(3) from error
    rendered = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"promotion evidence: {rendered}")


def main() -> None:
    args = parse_args()
    execution = _parallel_execution(args)
    _check_worker_memory(args, execution)
    if execution.torch_threads is not None:
        configure_torch_threads(execution.torch_threads)
    if args.smoke:
        # Smoke mode verifies orchestration only; the E-plan execution order
        # guard applies to real training runs, not to test harnesses.
        workspace = args.report.parent / f"smoke_workspace_{args.family.lower()}"
        workspace.mkdir(parents=True, exist_ok=True)
        if args.family == "M0":
            report = train_m0_smoke(workspace)
        elif args.family == "M0R":
            report = train_m0r_smoke(workspace)
        elif args.family == "M1":
            report = train_m1_smoke(workspace)
        elif args.family == "M2":
            report = train_m2_smoke(workspace)
        else:
            report = train_m3_smoke(workspace)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report["oos"], ensure_ascii=False, indent=2))
        print(f"smoke complete; report={args.report}")
        return
    passed_priors = prior_families_from_reports(args.prior_report)
    acknowledged_failed_priors = prior_families_from_failed_reports(args.prior_report_failed)
    validate_family_order(
        args.family,
        prior_families_passed=passed_priors,
        acknowledged_failed_priors=acknowledged_failed_priors,
        deviation_reason=args.family_order_deviation,
        deviation_owner=args.family_order_deviation_owner,
    )
    deviation_record: dict[str, Any] | None = None
    if missing_prerequisites(args.family, passed_priors):
        deviation_record = family_order_deviation_record(
            args.family,
            passed_priors,
            acknowledged_failed_priors,
            cast(str, args.family_order_deviation),
            cast(str, args.family_order_deviation_owner),
        )
        print(
            "WARNING: running "
            f"{args.family} under an owner-authorized E-plan order deviation: "
            f"{deviation_record['reason']} (authorized by {deviation_record['authorized_by']})",
            flush=True,
        )
    if args.baseline_artifacts is None:
        raise SystemExit("--baseline-artifacts is required for full runs")
    if args.baseline_report is None:
        raise SystemExit("--baseline-report is required for full runs")
    if args.workspace is None:
        raise SystemExit("--workspace is required for full runs")
    load_baseline_predictions(Path(args.baseline_artifacts))
    if args.family == "M0":
        report = run_full_m0(args)
    elif args.family == "M0R":
        report = run_full_m0r(args)
    elif args.family == "M1":
        report = run_full_m1(args)
    elif args.family == "M2":
        report = run_full_m2(args)
    elif args.family == "M3":
        report = run_full_m3(args)
    else:
        raise SystemExit(f"unsupported family: {args.family}")
    if deviation_record is not None:
        report["family_order_deviation"] = deviation_record
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["oos"], ensure_ascii=False, indent=2))
    print(f"full run complete; report={args.report}")
    _maybe_emit_promotion_evidence(args, report)


if __name__ == '__main__':
    main()
