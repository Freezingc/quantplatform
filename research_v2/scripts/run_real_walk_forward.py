"""Run the real-data B0-B3 walk-forward for the approved F01-F06 folds.

Assembles TRAIN/VALIDATION/TEST datasets from the V2 PIT tables
(factor_observation_clean / label_daily / universe_daily, current version per
partition), fits the frozen B0-B3 candidates per fold, selects by validation
RankIC, evaluates each winner on TEST, and produces the concatenated OOS
evaluation with G1-G6 gates.

Usage:
  run_real_walk_forward.py --folds F01,F02,F03,F04,F05,F06 --report <path>

The factor value source is switchable (reconstruction plan Task 2.3):
  run_real_walk_forward.py --factor-source frozen --report <path>
``legacy`` (the default) is bit-identical to the criterion-303 reproduction
contract; ``frozen`` uses ``factor_daily`` with FROZEN_FACTOR_SET_V1.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from walk_forward_common import (
    current_versions as _current_versions,
)
from walk_forward_common import (
    factor_members_for,
    new_holder,
)
from walk_forward_common import (
    load_calendar as _load_calendar,
)
from walk_forward_factor_daily import (
    FACTOR_SOURCES,
    extract_fn_for,
    factor_source_plan,
)

from research_v2.baselines.base import BaselineDataset
from research_v2.baselines.orchestration import BaselineFold
from research_v2.datasets.coverage import (
    DEFAULT_MINIMUM_CROSS_SECTION,
    summarize_frame_coverage,
)
from research_v2.datasets.splits import SplitResolver
from research_v2.evaluation.walk_forward import (
    BaselineWalkForwardResult,
    run_walk_forward_baselines,
)


def save_baseline_oos_artifacts(
    result: BaselineWalkForwardResult, root: Path
) -> dict[str, str]:
    """Persist each ordinary fold's winner-baseline TEST predictions as parquet.

    These fold-matched baseline predictions are the comparison artifacts required
    for candidate evaluation (E8.3 delta computation).
    """
    expected = ("F01", "F02", "F03", "F04", "F05", "F06")
    missing = [fold for fold in expected if fold not in result.fold_results]
    if missing:
        raise ValueError(f"missing fold results for baseline artifacts: {missing}")
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for fold_id in expected:
        fold_result = result.fold_results[fold_id]
        if fold_result.test_evaluation is None:
            raise ValueError(f"missing TEST evaluation for baseline artifacts: {fold_id}")
        frame = fold_result.test_evaluation.predictions.frame
        target = root / f"{fold_id}.parquet"
        frame.to_parquet(target, index=False)
        paths[fold_id] = str(target)
    return paths


def build_walk_forward_report(
    fold_ids: Sequence[str],
    result: BaselineWalkForwardResult,
    versions: Mapping[str, Mapping[str, str]],
    *,
    environment: Mapping[str, object] | None = None,
    assembly_coverage: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Serialize the walk-forward result for archival.

    Note: GateReport.all_passed is a property; calling it raises TypeError.

    ``environment`` is emitted verbatim and only when supplied. The legacy source is
    under a bit-identical reproduction contract (Task 2.4 re-runs the driver without
    the factor-source flag and compares against the archived criterion-303 report), so
    the key must not appear there; the frozen trust root is the one that has to
    self-describe. See :func:`describe_run_environment`.
    """
    report: dict[str, object] = {
        "folds": list(fold_ids),
        "data_versions": {
            table: dict(sorted(partitions.items())) for table, partitions in versions.items()
        },
        "selections": [
            {
                "fold_id": record.fold_id,
                "selected_baseline_id": record.selected_baseline_id,
                "validation_scores": [
                    {
                        "baseline_id": score.baseline_id,
                        "mean_rank_ic": score.mean_rank_ic,
                        "valid_day_count": score.valid_day_count,
                        "invalid_day_count": score.invalid_day_count,
                    }
                    for score in record.validation_scores
                ],
                "unselected_reasons": [list(reason) for reason in record.unselected_reasons],
            }
            for record in result.selection_records
        ],
        "oos": {
            "valid_day_count": result.evaluation.summary.valid_day_count,
            "mean_rank_ic": result.evaluation.summary.mean_rank_ic,
            "icir": result.evaluation.summary.icir,
            "hac_t_statistic": result.evaluation.hac.t_statistic,
            "baseline_mean_rank_ic": result.evaluation.baseline_mean_rank_ic,
            "delta_rank_ic": result.evaluation.delta_rank_ic,
            "yearly_mean_rank_ic": {
                str(year): value for year, value in result.evaluation.yearly_mean_rank_ic.items()
            },
            "gates": {
                gate: {
                    "status": gate_result.status.value,
                    "observed_value": gate_result.observed_value,
                    "threshold": gate_result.threshold,
                }
                for gate, gate_result in sorted(result.evaluation.gate_report.results.items())
            },
            "all_passed": result.evaluation.gate_report.all_passed,
        },
    }
    if environment is not None:
        report["environment"] = dict(environment)
    if assembly_coverage is not None:
        report["assembly_coverage"] = dict(assembly_coverage)
    return report


def describe_run_environment(
    factor_source: str, factor_ids: Sequence[str]
) -> dict[str, object]:
    """Self-describe a trust-root run: what it consumed, and on what runtime.

    The archived criterion-303 report records ``data_versions`` but nothing about the
    factor set or the runtime, so it cannot answer "which factor set produced these
    numbers?" or "were they fitted at the same thread count?". Both matter once a
    second factor source exists: ``legacy`` and ``frozen`` write reports with the same
    shape and nothing in the file distinguishes them, and thread count is
    numerics-affecting (floating-point reductions are not associative) while defaulting
    to half the host core count, i.e. to a host property nobody declared.

    Recording is observation only. Pinning device or threads is a separate, adjudicated
    change because it would alter results, so this reads the live values and reports
    ``OMP_NUM_THREADS`` as ``None`` when it was never set rather than inventing one.
    """
    import importlib.metadata as metadata
    import os
    import platform
    import sys

    from research_v2.training.seeds import describe_runtime_environment

    libraries: dict[str, str | None] = {}
    for distribution in ("numpy", "pandas", "pyarrow", "scikit-learn", "lightgbm", "torch"):
        try:
            libraries[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:  # pragma: no cover - all six are dependencies
            libraries[distribution] = None
    return {
        "factor_source": factor_source,
        "factor_ids": list(factor_ids),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "host_cpu_count": os.cpu_count(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "library_versions": libraries,
        "torch": asdict(describe_runtime_environment()),
    }


def _constant_loader(dataset: BaselineDataset) -> Callable[[], BaselineDataset]:
    def loader() -> BaselineDataset:
        return dataset

    return loader


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", default="F01,F02,F03,F04,F05,F06")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--factor-source",
        choices=FACTOR_SOURCES,
        default="legacy",
        help="legacy: factor_observation_clean + the 14 walk-forward ids "
        "(bit-identical to criterion-303); frozen: factor_daily + FROZEN_FACTOR_SET_V1",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    fold_ids = tuple(value.strip().upper() for value in args.folds.split(",") if value.strip())

    holder: dict[str, Any] = new_holder()
    calendar = _load_calendar(holder)
    resolver = SplitResolver(calendar)
    factor_ids, version_tables = factor_source_plan(args.factor_source)
    extract = extract_fn_for(args.factor_source)
    factor_members = factor_members_for(factor_ids)
    versions = {table: _current_versions(holder, table) for table in version_tables}
    print(
        f"factor source: {args.factor_source} "
        f"({len(factor_ids)} factors, tables={','.join(version_tables)})",
        flush=True,
    )

    feature_order = tuple(member.feature_name for member in factor_members)
    folds: dict[str, BaselineFold] = {}
    # The legacy source is under a bit-identical reproduction contract, so neither the
    # environment nor the coverage block may appear in its report; see
    # build_walk_forward_report. Recording both is observation only and costs one
    # groupby per segment against a multi-hour run.
    self_describe = args.factor_source != "legacy"
    coverage_by_fold: dict[str, dict[str, Any]] = {}
    for fold_id in fold_ids:
        print(f"assembling datasets for {fold_id}", flush=True)
        train = extract(holder, resolver, fold_id, "train", factor_members, versions)
        validation = extract(holder, resolver, fold_id, "validation", factor_members, versions)
        test = extract(holder, resolver, fold_id, "test", factor_members, versions)
        dataset_id = f"real-{fold_id.lower()}-v1"
        if self_describe:
            # Measured BEFORE the dropna below, on the frame as assembled, so the
            # report records what the baselines were actually fitted on: how many rows
            # survived, how many days fell under the evaluation's cross-section floor,
            # and how wide the median day was. This is the quantity adjudication 4 is
            # about, and G1-G6 cannot see it because they read TEST evidence only.
            coverage_by_fold[fold_id] = {
                role: summarize_frame_coverage(frame, feature_order).to_json()
                for role, frame in (("train", train), ("validation", validation), ("test", test))
            }
        # Baselines require finite features: keep only complete-coverage rows.
        train = train.dropna(subset=list(feature_order)).copy()
        validation = validation.dropna(subset=list(feature_order)).copy()
        test = test.dropna(subset=list(feature_order)).copy()
        train_ds = BaselineDataset(
            role="TRAIN", fold_id=fold_id, dataset_id=f"{dataset_id}-train",
            features=train.loc[:, ["signal_date", "ts_code", *feature_order]],
            labels=train.loc[:, ["signal_date", "ts_code", "raw_return_5d", "label_valid"]],
        )
        validation_ds = BaselineDataset(
            role="VALIDATION", fold_id=fold_id, dataset_id=f"{dataset_id}-validation",
            features=validation.loc[:, ["signal_date", "ts_code", *feature_order]],
            labels=validation.loc[:, ["signal_date", "ts_code", "raw_return_5d", "label_valid"]],
        )
        test_ds = BaselineDataset(
            role="TEST", fold_id=fold_id, dataset_id=f"{dataset_id}-test",
            features=test.loc[:, ["signal_date", "ts_code", *feature_order]],
            labels=test.loc[:, ["signal_date", "ts_code", "raw_return_5d", "label_valid"]],
        )
        fold = BaselineFold(
            fold_id=fold_id,
            train=train_ds,
            validation=validation_ds,
            factor_order=feature_order,
            frozen_directions={name: 1 for name in feature_order},
            test_loader=_constant_loader(test_ds),
        )
        folds[fold_id] = fold
        print(
            f"{fold_id}: train={len(train)} validation={len(validation)} test={len(test)}",
            flush=True,
        )

    print("running walk-forward baselines", flush=True)
    result = run_walk_forward_baselines(folds, minimum_cross_section=30)

    environment = (
        None
        if args.factor_source == "legacy"
        else describe_run_environment(args.factor_source, factor_ids)
    )
    assembly_coverage = (
        None
        if not self_describe
        else {
            "minimum_cross_section": DEFAULT_MINIMUM_CROSS_SECTION,
            "measured_on": (
                "assembled frames, after the universe and label joins and before the "
                "dropna the baselines train through"
            ),
            "folds": coverage_by_fold,
        }
    )
    report = build_walk_forward_report(
        fold_ids,
        result,
        versions,
        environment=environment,
        assembly_coverage=assembly_coverage,
    )
    artifact_paths = save_baseline_oos_artifacts(result, args.report.parent / "baseline_oos")
    report["baseline_oos_artifacts"] = artifact_paths
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["oos"], ensure_ascii=False, indent=2), flush=True)
    print(f"walk-forward complete; report={args.report}", flush=True)


if __name__ == "__main__":
    main()
