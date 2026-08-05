"""Read-only 2025 validation evidence for the structural factor representatives.

The training direction is loaded from the already audited TRAIN gate report and
is never re-estimated on the validation sample.  This command deliberately
does not publish a V2 snapshot: the result is evidence for the admission-policy
review, not a factor freeze or a model-training trigger.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_gp_candidates_strict import (
    _connect,
    _coverage,
    _metadata,
    _month_inputs,
)
from record_validation_quality_snapshot import compute_validation_metrics

from research_v2.db.session import V2WriteSession, YesLabReadOnlySession
from research_v2.factors.source_backfill import build_factor_definitions
from research_v2.factors.source_evaluation import evaluate_source_factor_range

DEFAULT_FACTOR_IDS: tuple[str, ...] = (
    "yeslab_115",
    "yeslab_118",
    "yeslab_163",
    "yeslab_357",
    "yeslab_363",
    "yeslab_365",
    "yeslab_376",
    "yeslab_380",
    "yeslab_383",
    "yeslab_385",
    "yeslab_396",
    "yeslab_95",
    "yeslab_96",
    "yeslab_98",
)
TRAIN_SCOPE_ID = "SCREEN_TRAIN2020_2024_DAILY_HAC_FDR_V2"
DEFAULT_TRAIN_GATE_REPORT = Path(
    "research_v2/artifacts/diagnostics/factor_quality_train26_gate_report_v5.json"
)
DEFAULT_OUTPUT = Path(
    "research_v2/artifacts/diagnostics/train26_representatives_validation_2025_v1"
)


def normalize_validation_daily(
    daily: pd.DataFrame,
    *,
    factor_ids: tuple[str, ...],
    start: date,
    end: date,
) -> pd.DataFrame:
    """Normalize a daily RankIC panel and enforce the validation boundary."""

    if start > end:
        raise ValueError("validation start must not be after end")
    expected = tuple(sorted(set(factor_ids)))
    if len(expected) != len(factor_ids) or not expected:
        raise ValueError("factor_ids must be non-empty and unique")
    frame = daily.copy(deep=True)
    if "factor_id" not in frame.columns and "factor_name" in frame.columns:
        frame = frame.rename(columns={"factor_name": "factor_id"})
    required = {"factor_id", "signal_date", "rank_ic"}
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"validation RankIC is missing columns: {missing}")
    frame = frame.loc[:, ["factor_id", "signal_date", "rank_ic"]].copy()
    frame["factor_id"] = frame["factor_id"].astype(str)
    unexpected = sorted(set(frame["factor_id"]).difference(expected))
    if unexpected:
        raise ValueError(f"validation RankIC contains unexpected factor IDs: {unexpected}")
    frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="raise").dt.date
    outside = (frame["signal_date"] < start) | (frame["signal_date"] > end)
    if outside.any():
        dates = sorted(set(frame.loc[outside, "signal_date"]))
        raise ValueError(f"validation RankIC contains dates outside validation window: {dates[:3]}")
    frame["rank_ic"] = pd.to_numeric(frame["rank_ic"], errors="coerce")
    frame = frame.loc[np.isfinite(frame["rank_ic"].to_numpy(dtype=float))].copy()
    if frame.duplicated(["factor_id", "signal_date"]).any():
        raise ValueError("validation RankIC contains duplicate factor/date keys")
    return frame.sort_values(["factor_id", "signal_date"], kind="mergesort").reset_index(drop=True)


def load_train_directions(
    report_path: Path,
    *,
    factor_ids: tuple[str, ...],
    scope_id: str = TRAIN_SCOPE_ID,
) -> dict[str, int]:
    """Load fixed TRAIN directions from an audited gate report."""

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    try:
        rows = payload["scopes"][scope_id]["rows"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"gate report is missing TRAIN scope {scope_id}") from error
    directions: dict[str, int] = {}
    for row in rows:
        factor_id = str(row.get("factor_id", ""))
        if factor_id not in factor_ids:
            continue
        metrics = row.get("metrics", {})
        direction = metrics.get("direction") if isinstance(metrics, dict) else None
        if direction not in (-1, 1):
            raise ValueError(f"invalid TRAIN direction for {factor_id}: {direction}")
        directions[factor_id] = int(direction)
    missing = sorted(set(factor_ids).difference(directions))
    if missing:
        raise ValueError(f"gate report is missing TRAIN directions: {missing}")
    return directions


def build_fixed_direction_validation_report(
    daily: pd.DataFrame,
    *,
    factor_ids: tuple[str, ...],
    train_directions: dict[str, int],
    start: date = date(2025, 1, 1),
    end: date = date(2025, 12, 31),
) -> pd.DataFrame:
    """Score observed factors and explicitly retain factors with no RankIC rows."""

    normalized = normalize_validation_daily(
        daily,
        factor_ids=factor_ids,
        start=start,
        end=end,
    )
    missing_directions = sorted(set(factor_ids).difference(train_directions))
    if missing_directions:
        raise ValueError(f"TRAIN directions are missing: {missing_directions}")
    observed = (
        compute_validation_metrics(
            normalized,
            train_directions={factor_id: train_directions[factor_id] for factor_id in factor_ids},
        )
        if not normalized.empty
        else pd.DataFrame()
    )
    observed_ids = set(observed["factor_id"].astype(str)) if not observed.empty else set()
    missing_rows: list[dict[str, object]] = []
    for factor_id in sorted(set(factor_ids).difference(observed_ids)):
        missing_rows.append(
            {
                "factor_id": factor_id,
                "valid_day_count": 0,
                "mean_rank_ic": np.nan,
                "rank_ic_std": np.nan,
                "icir": np.nan,
                "direction": int(train_directions[factor_id]),
                "signed_mean_rank_ic": np.nan,
                "directional_day_share": np.nan,
                "hac_t_statistic": np.nan,
                "hac_p_value": np.nan,
                "fdr_q_value": np.nan,
                "decision_status": "VALIDATION_2025_NO_RANKIC",
                "reason_codes": ["NO_VALIDATION_RANKIC"],
            }
        )
    result = pd.concat([observed, pd.DataFrame(missing_rows)], ignore_index=True)
    if result.empty:
        raise ValueError("validation report has no factor rows")
    return result.sort_values("factor_id", kind="mergesort").reset_index(drop=True)


def _parse_factor_ids(value: str) -> tuple[str, ...]:
    factor_ids = tuple(item.strip() for item in value.split(",") if item.strip())
    if not factor_ids or len(set(factor_ids)) != len(factor_ids):
        raise ValueError("factor-ids must be non-empty and unique")
    return factor_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--purge-last-dates", type=int, default=6)
    parser.add_argument("--chunk-months", type=int, default=3)
    parser.add_argument("--factor-ids", default=",".join(DEFAULT_FACTOR_IDS))
    parser.add_argument("--train-gate-report", type=Path, default=DEFAULT_TRAIN_GATE_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    start_date = pd.Timestamp(args.start).date()
    end_date = pd.Timestamp(args.end).date()
    if end_date > date(2025, 12, 31):
        raise ValueError("validation evaluator cannot read dates after 2025-12-31")
    factor_ids = _parse_factor_ids(args.factor_ids)
    source_client, v2_client = _connect()
    source_session = YesLabReadOnlySession(source_client)
    v2_session = V2WriteSession(v2_client, target_database="dfs://QuantResearchV2Test")
    metadata = _metadata(source_session, factor_ids)
    definitions = build_factor_definitions(metadata, factor_ids)
    month_inputs, purged_dates = _month_inputs(
        v2_session,
        start_date=start_date,
        end_date=end_date,
        purge_last_dates=args.purge_last_dates,
    )
    evaluations = evaluate_source_factor_range(
        source_session,
        metadata,
        factor_ids=factor_ids,
        start_date=start_date,
        end_date=end_date,
        month_inputs=month_inputs,
        minimum_cross_section=30,
        chunk_months=args.chunk_months,
    )
    daily = pd.concat([evaluation.daily_rank_ic for evaluation in evaluations], ignore_index=True)
    report = build_fixed_direction_validation_report(
        daily,
        factor_ids=factor_ids,
        train_directions=load_train_directions(args.train_gate_report, factor_ids=factor_ids),
        start=start_date,
        end=end_date,
    )
    coverage = _coverage(evaluations)
    args.output.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(args.output / "daily_rank_ic.parquet", index=False)
    report.to_csv(args.output / "quality_summary.csv", index=False)
    coverage.to_csv(args.output / "coverage.csv", index=False)
    metadata_payload = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "purged_signal_dates": [value.isoformat() for value in purged_dates],
        "factor_ids": list(factor_ids),
        "factor_names": [definition.display_name for definition in definitions],
        "target_database": "dfs://QuantResearchV2Test",
        "locked_test_read": False,
        "database_writes": False,
        "train_direction_source": str(args.train_gate_report),
        "train_direction_scope": TRAIN_SCOPE_ID,
        "minimum_cross_section": 30,
        "source_materialization": "YesLab strict PIT normalization/deduplication",
        "output_rows_daily_rank_ic": len(daily),
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(report["decision_status"].value_counts().sort_index().to_string())
    print(
        f"validation evidence complete: factors={len(factor_ids)} "
        f"daily_rows={len(daily)} output={args.output}"
    )


if __name__ == "__main__":
    main()
