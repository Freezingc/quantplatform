"""Run the real-data B0-B3 walk-forward for the approved F01-F06 folds.

Assembles TRAIN/VALIDATION/TEST datasets from the V2 PIT tables
(factor_observation_clean / label_daily / universe_daily, current version per
partition), fits the frozen B0-B3 candidates per fold, selects by validation
RankIC, evaluates each winner on TEST, and produces the concatenated OOS
evaluation with G1-G6 gates.

Usage:
  run_real_walk_forward.py --folds F01,F02 --report <path> [--build-only]
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path

import dolphindb as ddb
import pandas as pd

from research_v2.baselines.base import BaselineDataset
from research_v2.baselines.orchestration import BaselineFold
from research_v2.datasets.builder import (
    DatasetBuildRequest,
    FactorInput,
    build_point_in_time_dataset,
)
from research_v2.datasets.splits import SplitResolver, annual_folds
from research_v2.evaluation.walk_forward import (
    run_walk_forward_baselines,
)
from research_v2.pit.calendar import TradingCalendar

TARGET_DATABASE = "dfs://QuantResearchV2Test"
WALK_FORWARD_FACTOR_IDS: tuple[str, ...] = (
    "yeslab_376",
    "yeslab_377",
    "yeslab_380",
    "yeslab_381",
    "yeslab_387",
    "yeslab_388",
    "yeslab_391",
    "yeslab_396",
    "yeslab_400",
    "yeslab_405",
    "yeslab_425",
    "yeslab_432",
    "yeslab_433",
    "yeslab_442",
)


def _connect() -> object:
    session = ddb.session()
    session.connect(
        os.getenv("QRV2_DDB_HOST", "10.176.55.211"),
        int(os.getenv("QRV2_DDB_PORT", "8848")),
        os.getenv("QRV2_DDB_USERNAME", "xiahongjie"),
        os.getenv("QRV2_DDB_PASSWORD", "123456"),
    )
    return session


def _factor_members() -> tuple[FactorInput, ...]:
    return tuple(
        FactorInput(factor_id=factor_id, factor_version=1, feature_name=f"f{index:02d}")
        for index, factor_id in enumerate(WALK_FORWARD_FACTOR_IDS)
    )


def _load_calendar(session: object) -> TradingCalendar:
    frame = session.run(
        "select cal_date from loadTable('dfs://YesLab', 'trade_calendar') "
        "where exchange='SSE' and is_open=1 and cal_date between 2013.01.01 and 2026.12.31 "
        "order by cal_date"
    )
    return TradingCalendar.from_dates([pd.Timestamp(v).date() for v in frame["cal_date"]])


def _current_versions(session: object, object_name: str) -> dict[str, str]:
    frame = session.run(
        f"select logical_partition, data_version, published_at "
        f"from loadTable('{TARGET_DATABASE}','published_version') "
        f"where object_name='{object_name}'"
    )
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return {}
    frame = frame.sort_values("published_at", kind="mergesort")
    return {
        str(row.logical_partition): str(row.data_version)
        for row in frame.itertuples(index=False)
    }


def _filter_current(
    frame: pd.DataFrame, date_column: str, versions: dict[str, str]
) -> pd.DataFrame:
    partitions = pd.to_datetime(frame[date_column]).dt.strftime("%Y-%m")
    expected = partitions.map(versions)
    return frame.loc[expected.notna() & (frame["data_version"] == expected)].copy()


def _extract_dataset(
    session: object,
    resolver: SplitResolver,
    fold_id: str,
    role: str,
    factor_members: tuple[FactorInput, ...],
    versions: dict[str, dict[str, str]],
) -> pd.DataFrame:
    segment = resolver.resolve(fold_id, role)
    start = min(segment.signal_dates)
    end = max(segment.signal_dates)
    fold_definition = next(fold for fold in annual_folds() if fold.fold_id == fold_id)
    role_years = {
        "train": fold_definition.train_years,
        "validation": fold_definition.validation_years,
        "test": fold_definition.test_years,
    }[role]
    segment_end = date(max(role_years), 12, 31)
    years = list(range(start.year, end.year + 1))
    universe_parts = []
    for year in years:
        part = session.run(
            f"select signal_date, ts_code, data_version "
            f"from loadTable('{TARGET_DATABASE}','universe_daily') "
            f"where signal_date between date('{year}.01.01') and date('{year}.12.31') "
            f"and is_member=1"
        )
        if isinstance(part, pd.DataFrame) and not part.empty:
            universe_parts.append(part)
    universe_rows = pd.concat(universe_parts, ignore_index=True)
    universe_rows = _filter_current(universe_rows, "signal_date", versions["universe_daily"])
    factor_parts = []
    for member in factor_members:
        for year in years:
            part = session.run(
                f"select signal_date, ts_code, factor_id, factor_version, value, "
                f"available_date, data_version "
                f"from loadTable('{TARGET_DATABASE}','factor_observation_clean') "
                f"where signal_date between date('{year}.01.01') and date('{year}.12.31') "
                f"and factor_id='{member.factor_id}'"
            )
            if isinstance(part, pd.DataFrame) and not part.empty:
                factor_parts.append(part)
        print(f"  extracted factor {member.factor_id} for {fold_id}/{role}", flush=True)
    factor_rows = pd.concat(factor_parts, ignore_index=True)
    factor_rows = _filter_current(factor_rows, "signal_date", versions["factor_observation_clean"])
    label_parts = []
    for year in years:
        part = session.run(
            f"select signal_date, ts_code, raw_return, label_valid, exit_date, data_version "
            f"from loadTable('{TARGET_DATABASE}','label_daily') "
            f"where signal_date between date('{year}.01.01') and date('{year}.12.31')"
        )
        if isinstance(part, pd.DataFrame) and not part.empty:
            label_parts.append(part)
    label_rows = pd.concat(label_parts, ignore_index=True)
    label_rows = _filter_current(label_rows, "signal_date", versions["label_daily"])
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    universe_rows = universe_rows.loc[
        (universe_rows["signal_date"] >= start_ts) & (universe_rows["signal_date"] <= end_ts)
    ]
    factor_rows = factor_rows.loc[
        (factor_rows["signal_date"] >= start_ts) & (factor_rows["signal_date"] <= end_ts)
    ]
    label_rows = label_rows.loc[
        (label_rows["signal_date"] >= start_ts) & (label_rows["signal_date"] <= end_ts)
    ]
    dataset = build_point_in_time_dataset(
        universe_rows.loc[:, ["signal_date", "ts_code"]],
        factor_rows.loc[
            :, ["signal_date", "ts_code", "factor_id", "factor_version", "value", "available_date"]
        ],
        label_rows.loc[:, ["signal_date", "ts_code", "raw_return", "label_valid", "exit_date"]],
        factor_members=factor_members,
        segment_end=segment_end,
        request=DatasetBuildRequest(fold_id=fold_id, split_role=role.upper()),
    )
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", default="F01,F02,F03,F04,F05,F06")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    fold_ids = tuple(value.strip().upper() for value in args.folds.split(",") if value.strip())

    session = _connect()
    calendar = _load_calendar(session)
    resolver = SplitResolver(calendar)
    factor_members = _factor_members()
    versions = {
        table: _current_versions(session, table)
        for table in ("universe_daily", "factor_observation_clean", "label_daily")
    }

    feature_order = tuple(member.feature_name for member in factor_members)
    folds: dict[str, BaselineFold] = {}
    for fold_id in fold_ids:
        print(f"assembling datasets for {fold_id}", flush=True)
        train = _extract_dataset(session, resolver, fold_id, "train", factor_members, versions)
        validation = _extract_dataset(
            session, resolver, fold_id, "validation", factor_members, versions
        )
        test = _extract_dataset(session, resolver, fold_id, "test", factor_members, versions)
        dataset_id = f"real-{fold_id.lower()}-v1"
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
            test_loader=lambda ds=test_ds: ds,
        )
        folds[fold_id] = fold
        print(
            f"{fold_id}: train={len(train)} validation={len(validation)} test={len(test)}",
            flush=True,
        )

    print("running walk-forward baselines", flush=True)
    result = run_walk_forward_baselines(folds, minimum_cross_section=30)

    report = {
        "folds": list(fold_ids),
        "selections": [
            {
                "fold_id": record.fold_id,
                "selected_baseline_id": record.selected_baseline_id,
                "validation_scores": [
                    {
                        "baseline_id": score.baseline_id,
                        "mean_rank_ic": score.mean_rank_ic,
                        "valid_day_count": score.valid_day_count,
                    }
                    for score in record.validation_scores
                ],
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
            "all_passed": result.evaluation.gate_report.all_passed(),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["oos"], ensure_ascii=False, indent=2), flush=True)
    print(f"walk-forward complete; report={args.report}", flush=True)


if __name__ == "__main__":
    main()
