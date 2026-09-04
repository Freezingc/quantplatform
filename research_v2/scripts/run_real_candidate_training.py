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
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from research_v2.datasets.preprocessing import FoldPreprocessor
from research_v2.evaluation.model_evaluator import evaluate_frozen_oos
from research_v2.evaluation.rank_ic import compute_daily_rank_ic, summarize_rank_ic
from research_v2.inference.artifact_loader import load_inference_artifact
from research_v2.models.residual_mlp import ResidualMLP, ResidualMLPConfig
from research_v2.models.temporal import TemporalGRU, TemporalGRUConfig
from research_v2.training.artifacts import CheckpointStore
from research_v2.training.batches import TemporalTrainingBatch, TrainingBatch
from research_v2.training.ensemble import REQUIRED_SEEDS, build_seed_ensemble
from research_v2.training.model_bundle import ModelArtifactBundleStore
from research_v2.training.runner import run_validation_governed_training
from research_v2.training.static_batches import build_static_training_batches
from research_v2.training.temporal_batches import build_temporal_training_batches
from research_v2.training.trainer import DatasetRole, DatasetView, TrainerConfig

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


def _percentile_rank_series(scores: torch.Tensor) -> torch.Tensor:
    ranks = scores.argsort().argsort().to(dtype=torch.float)
    return ranks / (len(scores) - 1)


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


def _predict_validation_percentiles(
    model: torch.nn.Module, validation_batches: Mapping[date, TrainingBatch]
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, object]] = []
    for signal_date in sorted(validation_batches):
        batch = validation_batches[signal_date]
        with torch.no_grad():
            scores = model(batch.inputs)[0]
        ranks = _percentile_rank_series(scores)
        for stock_index, code in enumerate(batch.inputs.ts_codes[0]):
            rows.append(
                {
                    "signal_date": signal_date,
                    "ts_code": code,
                    "percentile_rank": float(ranks[stock_index]),
                }
            )
    return pd.DataFrame(rows)


def _prepare_frame(
    frame: pd.DataFrame, preprocessor: FoldPreprocessor, factor_order: tuple[str, ...]
) -> pd.DataFrame:
    """Apply the train-fitted preprocessor and re-join business keys and labels."""
    transformed = preprocessor.transform(frame.loc[:, list(factor_order)])
    keys = ["signal_date", "ts_code"]
    extras = [c for c in ("raw_return_5d", "label_valid") if c in frame.columns]
    base = frame.loc[:, keys + extras].reset_index(drop=True)
    return pd.concat([base, transformed.values.reset_index(drop=True)], axis=1)


def train_m1_smoke(workspace: Path) -> dict[str, Any]:
    """End-to-end M1 orchestration on synthetic frames (not model evidence)."""
    factor_count = SMOKE_FACTOR_COUNT
    frames = build_smoke_frames(factor_count)
    factor_order = tuple(f"f{index:02d}" for index in range(factor_count))
    preprocessor = FoldPreprocessor(factor_order=factor_order)
    preprocessor.fit(
        frames["train"].loc[:, list(factor_order)], role="TRAIN"
    )
    train_batches = build_static_training_batches(
        _prepare_frame(frames["train"], preprocessor, factor_order), factor_order=factor_order
    )
    validation_batches = build_static_training_batches(
        _prepare_frame(frames["validation"], preprocessor, factor_order),
        factor_order=factor_order,
    )
    config = ResidualMLPConfig(
        raw_factor_count=factor_count, hidden_dim=16, num_blocks=1, dropout=0.0
    )
    config_payload = {
        "raw_factor_count": factor_count,
        "hidden_dim": 16,
        "num_blocks": 1,
        "dropout": 0.0,
    }
    checkpoint_store = CheckpointStore(workspace / "checkpoints")
    seed_checkpoints: dict[int, Path] = {}
    seed_validation_predictions: dict[int, pd.DataFrame] = {}
    seed_rank_ic: dict[int, float] = {}
    prototype = ResidualMLP(config)
    for seed in REQUIRED_SEEDS:
        model = ResidualMLP(config)
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
            checkpoint_prefix=f"smoke-m1-{seed}",
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
    model_id = "smoke-m1"
    bundle_store.write(
        model_id,
        model_card={
            "family": "RESIDUAL_MLP",
            "dataset_id": "smoke-train",
            "model_status": "WALK_FORWARD_PASSED",
            "model_code_version": "residual-mlp-1",
        },
        config=config_payload,
        signature=asdict(prototype.artifact_signature()),
        feature_order=factor_order,
        preprocessing={"state_hash": preprocessor.state_hash},
        seed_checkpoints=seed_checkpoints,
        validation_predictions=ensemble.predictions.rename(
            columns={"ensemble_score": "percentile_rank"}
        ),
    )
    loaded = load_inference_artifact(
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
    rows: list[dict[str, object]] = []
    for signal_date in sorted(test_batches):
        batch = test_batches[signal_date]
        accumulated: torch.Tensor | None = None
        for seed_model in loaded.models:
            with torch.no_grad():
                scores = seed_model(batch.inputs)[0]
            ranks = _percentile_rank_series(scores)
            accumulated = ranks if accumulated is None else accumulated + ranks
        assert accumulated is not None
        mean_ranks = accumulated / len(loaded.models)
        for stock_index, code in enumerate(batch.inputs.ts_codes[0]):
            rows.append(
                {
                    "signal_date": signal_date,
                    "ts_code": code,
                    "score": float(mean_ranks[stock_index]),
                }
            )
    candidate_predictions = pd.DataFrame(rows)
    evaluation = evaluate_frozen_oos(
        candidate_predictions,
        frames["labels"],
        baseline_predictions=frames["baseline_predictions"],
        role="TEST",
        minimum_cross_section=30,
    )
    return {
        "family": "M1",
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
    model.eval()
    rows: list[dict[str, object]] = []
    for signal_date in sorted(validation_batches):
        batch = validation_batches[signal_date]
        with torch.no_grad():
            scores = model(batch.inputs)[0]
        ranks = _percentile_rank_series(scores)
        for stock_index, code in enumerate(batch.inputs.ts_codes[0]):
            rows.append(
                {
                    "signal_date": signal_date,
                    "ts_code": code,
                    "percentile_rank": float(ranks[stock_index]),
                }
            )
    return pd.DataFrame(rows)


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
        model_card={
            "family": "TEMPORAL_GRU",
            "dataset_id": "smoke-train",
            "model_status": "WALK_FORWARD_PASSED",
            "model_code_version": "temporal-gru-1",
        },
        config=config_payload,
        signature=asdict(prototype.artifact_signature()),
        feature_order=factor_order,
        preprocessing={"state_hash": preprocessor.state_hash},
        seed_checkpoints=seed_checkpoints,
        validation_predictions=ensemble.predictions.rename(
            columns={"ensemble_score": "percentile_rank"}
        ),
    )
    loaded = load_inference_artifact(
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
    rows: list[dict[str, object]] = []
    for signal_date in sorted(test_batches):
        batch = test_batches[signal_date]
        accumulated: torch.Tensor | None = None
        for seed_model in loaded.models:
            with torch.no_grad():
                scores = seed_model(batch.inputs)[0]
            ranks = _percentile_rank_series(scores)
            accumulated = ranks if accumulated is None else accumulated + ranks
        assert accumulated is not None
        mean_ranks = accumulated / len(loaded.models)
        for stock_index, code in enumerate(batch.inputs.ts_codes[0]):
            rows.append(
                {
                    "signal_date": signal_date,
                    "ts_code": code,
                    "score": float(mean_ranks[stock_index]),
                }
            )
    candidate_predictions = pd.DataFrame(rows)
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


def main() -> None:
    args = parse_args()
    if args.smoke:
        # Smoke mode verifies orchestration only; the E-plan execution order
        # guard applies to real training runs, not to test harnesses.
        workspace = args.report.parent / f"smoke_workspace_{args.family.lower()}"
        workspace.mkdir(parents=True, exist_ok=True)
        if args.family == "M1":
            report = train_m1_smoke(workspace)
        elif args.family == "M2":
            report = train_m2_smoke(workspace)
        else:
            raise SystemExit("M3 smoke wiring lands later in Task 4 of the candidate plan")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report["oos"], ensure_ascii=False, indent=2))
        print(f"smoke complete; report={args.report}")
        return
    validate_family_order(args.family, prior_families_passed=())
    if args.baseline_artifacts is None:
        raise SystemExit("--baseline-artifacts is required for full runs")
    load_baseline_predictions(args.baseline_artifacts)
    raise SystemExit(
        "full-run training wiring is gated on Task 1.6 baseline artifacts and "
        "plan approval (Task 3.4); refusing to run"
    )


if __name__ == '__main__':
    main()
