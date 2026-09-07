"""JSON and HTML reports for validation-selected walk-forward experiments."""

from __future__ import annotations

import html
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research_v2.baselines.orchestration import (
    SIMPLE_BASELINE_IDS,
    BaselineFoldResult,
    BaselineSelectionRecord,
)
from research_v2.evaluation.walk_forward import (
    ORDINARY_TEST_FOLDS,
    BaselineWalkForwardResult,
)


@dataclass(frozen=True)
class ExperimentReport:
    payload: Mapping[str, object]

    def to_json(self) -> str:
        return (
            json.dumps(_json_value(self.payload), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        )

    def to_html(self) -> str:
        escaped = html.escape(self.to_json())
        return (
            "<!doctype html>\n"
            "<html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<title>Quant Research V2 experiment</title></head>"
            "<body><main><h1>Quant Research V2 experiment</h1>"
            f"<pre>{escaped}</pre></main></body></html>\n"
        )


def build_experiment_report(
    result: BaselineWalkForwardResult,
    *,
    experiment_id: str,
    fold_metadata: Mapping[str, Mapping[str, object]] | None = None,
    factor_set: Mapping[str, object] | None = None,
    preprocessing: Mapping[str, object] | None = None,
    environment: Mapping[str, object] | None = None,
    artifact_links: Mapping[str, object] | None = None,
) -> ExperimentReport:
    """Build one report without recomputing or averaging the underlying OOS metrics."""
    validate_experiment_id(experiment_id)
    if set(result.fold_results) != set(ORDINARY_TEST_FOLDS):
        raise ValueError("experiment report requires exactly F1-F4 fold results")
    if len(result.selection_records) != len(ORDINARY_TEST_FOLDS):
        raise ValueError("experiment report requires one selection record per ordinary fold")
    metadata = fold_metadata or {}
    folds: dict[str, object] = {}
    for fold_id, fold_result in sorted(result.fold_results.items()):
        folds[fold_id] = _fold_payload(
            fold_result,
            metadata=metadata.get(fold_id, {}),
        )
    evaluation = result.evaluation
    daily_columns = [
        "signal_date",
        "rank_ic",
        "sample_count",
        "eligible_count",
        "valid_label_count",
        "coverage",
        "prediction_unique_count",
        "prediction_std",
        "rank_ic_valid",
    ]
    daily = evaluation.daily_rank_ic.loc[
        :,
        [column for column in daily_columns if column in evaluation.daily_rank_ic.columns],
    ]
    overall = {
        "summary": asdict(evaluation.summary),
        "yearly_summary": _records(evaluation.yearly_summary),
        "hac": asdict(evaluation.hac),
        "gates": {
            name: {
                "status": gate.status.value,
                "observed_value": gate.observed_value,
                "threshold": gate.threshold,
            }
            for name, gate in sorted(evaluation.gate_report.results.items())
        },
        "coverage": _records(
            evaluation.daily_rank_ic.loc[
                :,
                [
                    column
                    for column in ("signal_date", "eligible_count", "valid_label_count", "coverage")
                    if column in evaluation.daily_rank_ic.columns
                ],
            ]
        ),
        "score_health": {
            "daily": _records(daily),
            "valid_day_count": evaluation.summary.valid_day_count,
        },
    }
    payload = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "factor_set": dict(factor_set or {}),
        "preprocessing": dict(preprocessing or {}),
        "environment": dict(environment or {}),
        "artifact_links": dict(artifact_links or {}),
        "folds": folds,
        "selection_records": [
            _selection_payload(record) for record in result.selection_records
        ],
        "overall_oos": overall,
    }
    return ExperimentReport(payload=payload)


def write_experiment_report(
    report: ExperimentReport,
    output: Path,
    *,
    format: str,
) -> None:
    """Write JSON or HTML report content, refusing unsupported formats."""
    if format not in {"json", "html"}:
        raise ValueError("report format must be json or html")
    content = report.to_json() if format == "json" else report.to_html()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")


def _fold_payload(
    result: BaselineFoldResult,
    *,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    selection = result.selection_record
    score_by_id = {item.baseline_id: item for item in selection.validation_scores}
    reason_by_id = dict(selection.unselected_reasons)
    test_result = result.test_evaluation
    baselines: dict[str, object] = {}
    for baseline_id in SIMPLE_BASELINE_IDS:
        validation = score_by_id.get(baseline_id)
        if validation is None:
            raise ValueError(f"selection record is missing {baseline_id}")
        is_winner = baseline_id == selection.selected_baseline_id
        test_payload: dict[str, object] = {"evaluated": False}
        if is_winner:
            if test_result is None or test_result.baseline_id != baseline_id:
                raise ValueError("selected baseline has no matching test evaluation")
            test_payload = {
                "evaluated": True,
                "summary": asdict(test_result.summary),
                "daily_rank_ic": _records(test_result.daily_rank_ic),
                "coverage": _records(
                    test_result.daily_rank_ic.loc[
                        :,
                        [
                            column
                            for column in (
                                "signal_date",
                                "eligible_count",
                                "valid_label_count",
                                "coverage",
                            )
                            if column in test_result.daily_rank_ic.columns
                        ],
                    ]
                ),
            }
        baselines[baseline_id] = {
            "validation": asdict(validation),
            "test": test_payload,
            "selected": is_winner,
            "unselected_reason": reason_by_id.get(baseline_id),
        }
    return {
        "metadata": dict(metadata),
        "selected_baseline_id": selection.selected_baseline_id,
        "baselines": baselines,
    }


def _selection_payload(record: BaselineSelectionRecord) -> dict[str, object]:
    return {
        "fold_id": record.fold_id,
        "selected_baseline_id": record.selected_baseline_id,
        "validation_scores": [asdict(score) for score in record.validation_scores],
        "unselected_reasons": dict(record.unselected_reasons),
    }


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    if frame.empty:
        return []
    return [_json_value(dict(row)) for row in frame.to_dict(orient="records")]


def validate_experiment_id(experiment_id: str) -> None:
    if (
        not experiment_id.startswith("sha256:")
        or len(experiment_id) != len("sha256:") + 64
        or any(character not in "0123456789abcdef" for character in experiment_id[7:])
    ):
        raise ValueError("experiment_id must be sha256:<64 lowercase hex characters>")


def _json_value(value: object) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, pd.Timedelta):
        return value.isoformat()
    return value


__all__ = [
    "ExperimentReport",
    "build_experiment_report",
    "validate_experiment_id",
    "write_experiment_report",
]
