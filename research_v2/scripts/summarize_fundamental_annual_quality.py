"""Summarize 2020-2024 PIT financial quality into a strict annual gate report."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

YEARS = (2020, 2021, 2022, 2023, 2024)
CANDIDATE_METRICS = (
    "cfoa",
    "roa",
    "road",
    "roed",
    "roe_yoy",
    "roicd",
    "ta_yoy",
    "np_qoq",
)
FACTOR_IDS = {
    "cfoa": "yeslab_219",
    "roa": "yeslab_220",
    "road": "yeslab_221",
    "roed": "yeslab_223",
    "roe_yoy": "yeslab_226",
    "ta_yoy": "yeslab_227",
    "roicd": "yeslab_229",
    "np_qoq": "yeslab_231",
}
FACTOR_NAMES = {
    "cfoa": "CFOA",
    "roa": "ROA",
    "road": "ROAD",
    "roed": "ROED",
    "roe_yoy": "ROE_YOY",
    "ta_yoy": "TA_YOY",
    "roicd": "ROICD",
    "np_qoq": "NP_QOQ",
}


def build_annual_report(
    annual_by_year: Mapping[int, pd.DataFrame],
    daily_by_year: Mapping[int, pd.DataFrame],
    *,
    minimum_finite_ratio: float = 0.70,
    minimum_annual_pass_years: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return annual rows and five-year summary rows using frozen direction."""
    if not 0 <= minimum_finite_ratio <= 1:
        raise ValueError("minimum_finite_ratio must be in [0, 1]")
    if minimum_annual_pass_years < 1 or minimum_annual_pass_years > len(YEARS):
        raise ValueError("minimum_annual_pass_years must be within the year count")
    annual_rows: list[pd.DataFrame] = []
    daily_rows: list[pd.DataFrame] = []
    for year in YEARS:
        if year not in annual_by_year or year not in daily_by_year:
            raise ValueError(f"missing annual or daily evidence for {year}")
        annual = _normalize_annual(annual_by_year[year], year)
        daily = _normalize_daily(daily_by_year[year], year)
        annual_rows.append(annual)
        daily_rows.append(daily)
    annual_frame = pd.concat(annual_rows, ignore_index=True)
    daily_frame = pd.concat(daily_rows, ignore_index=True)
    if annual_frame.duplicated(["year", "metric_id"]).any():
        raise ValueError("annual evidence has duplicate year/metric rows")
    if daily_frame.duplicated(["year", "metric_id", "signal_date"]).any():
        raise ValueError("daily evidence has duplicate year/metric/date rows")

    aggregate_rows: list[dict[str, object]] = []
    for metric_id, group in daily_frame.groupby("metric_id", sort=True):
        values = pd.to_numeric(group["rank_ic"], errors="coerce").dropna()
        if values.empty:
            aggregate_mean = float("nan")
            aggregate_icir = float("nan")
        else:
            aggregate_mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            aggregate_icir = aggregate_mean / std if np.isfinite(std) and std > 0 else float("nan")
        direction = 1 if np.isfinite(aggregate_mean) and aggregate_mean >= 0 else -1
        annual_metric = annual_frame.loc[annual_frame["metric_id"].eq(metric_id)].copy()
        annual_metric["signed_mean_rank_ic"] = annual_metric["mean_rank_ic"] * direction
        annual_metric["signed_directional_day_share"] = np.where(
            direction > 0,
            annual_metric["positive_day_share"],
            1.0 - annual_metric["positive_day_share"],
        )
        annual_metric["coverage_pass"] = annual_metric["finite_ratio"] >= minimum_finite_ratio
        annual_metric["annual_gate_pass"] = (
            (annual_metric["valid_day_count"] >= 60)
            & (annual_metric["signed_mean_rank_ic"] > 0.02)
            & (annual_metric["icir"].abs() >= 0.50)
            & (annual_metric["signed_directional_day_share"] >= 0.55)
            & annual_metric["coverage_pass"]
        )
        annual_frame.loc[annual_metric.index, "signed_mean_rank_ic"] = annual_metric[
            "signed_mean_rank_ic"
        ]
        annual_frame.loc[annual_metric.index, "signed_directional_day_share"] = annual_metric[
            "signed_directional_day_share"
        ]
        annual_frame.loc[annual_metric.index, "coverage_pass"] = annual_metric["coverage_pass"]
        annual_frame.loc[annual_metric.index, "annual_gate_pass"] = annual_metric["annual_gate_pass"]
        aggregate_rows.append(
            {
                "metric_id": metric_id,
                "factor_id": FACTOR_IDS[metric_id],
                "factor_name": FACTOR_NAMES[metric_id],
                "aggregate_valid_day_count": int(len(values)),
                "aggregate_mean_rank_ic": aggregate_mean,
                "aggregate_icir": aggregate_icir,
                "frozen_direction": direction,
                "strict_annual_pass_years": int(annual_metric["annual_gate_pass"].sum()),
                "coverage_pass_years": int(annual_metric["coverage_pass"].sum()),
                "minimum_annual_finite_ratio": float(annual_metric["finite_ratio"].min()),
                "stable_4_of_5": bool(
                    int(annual_metric["annual_gate_pass"].sum()) >= minimum_annual_pass_years
                ),
                "decision": "ANNUAL_STABLE_PASS"
                if int(annual_metric["annual_gate_pass"].sum()) >= minimum_annual_pass_years
                else "ANNUAL_STABILITY_FAIL",
            }
        )
    summary = pd.DataFrame(aggregate_rows).sort_values("factor_id", ignore_index=True)
    annual_frame = annual_frame.sort_values(["metric_id", "year"], ignore_index=True)
    return annual_frame, summary


def _normalize_annual(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    required = {
        "metric_id",
        "valid_day_count",
        "mean_rank_ic",
        "icir",
        "positive_day_share",
        "finite_ratio",
    }
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"annual {year} summary is missing columns: {missing}")
    result = frame.loc[frame["metric_id"].isin(CANDIDATE_METRICS), list(required)].copy()
    if set(result["metric_id"]) != set(CANDIDATE_METRICS):
        raise ValueError(f"annual {year} summary does not contain all candidates")
    result["year"] = year
    for column in required.difference({"metric_id"}):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _normalize_daily(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    required = {"factor_name", "signal_date", "rank_ic"}
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"daily {year} evidence is missing columns: {missing}")
    result = frame.rename(columns={"factor_name": "metric_id"}).loc[
        :, ["metric_id", "signal_date", "rank_ic"]
    ].copy()
    result = result.loc[result["metric_id"].isin(CANDIDATE_METRICS)]
    result["signal_date"] = pd.to_datetime(result["signal_date"], errors="raise").dt.date
    if not result.empty and set(pd.to_datetime(result["signal_date"]).dt.year) != {year}:
        raise ValueError(f"daily evidence has dates outside {year}")
    result["rank_ic"] = pd.to_numeric(result["rank_ic"], errors="coerce")
    result["year"] = year
    return result


def _load_year(root: Path, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if year == 2020:
        summary_paths = (
            root / "fundamental_pit_snapshot_quality_2020_v1" / "summary.csv",
            root / "fundamental_pit_derived_quality_2020_v1" / "summary.csv",
            root / "fundamental_pit_ta_np_quality_2020_v1" / "summary.csv",
        )
        daily_paths = (
            root / "fundamental_pit_snapshot_quality_2020_v1" / "daily_rank_ic.csv",
            root / "fundamental_pit_derived_quality_2020_v1" / "daily_rank_ic.csv",
            root / "fundamental_pit_ta_np_quality_2020_v1" / "daily_rank_ic.csv",
        )
    else:
        summary_paths = (root / f"fundamental_pit_quality_{year}_v1" / "summary.csv",)
        daily_paths = (root / f"fundamental_pit_quality_{year}_v1" / "daily_rank_ic.csv",)
    summary = pd.concat([pd.read_csv(path) for path in summary_paths], ignore_index=True)
    daily = pd.concat([pd.read_csv(path) for path in daily_paths], ignore_index=True)
    return summary, daily


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("research_v2/artifacts/diagnostics"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    annual_inputs: dict[int, pd.DataFrame] = {}
    daily_inputs: dict[int, pd.DataFrame] = {}
    for year in YEARS:
        annual_inputs[year], daily_inputs[year] = _load_year(args.root, year)
    annual, summary = build_annual_report(annual_inputs, daily_inputs)
    args.output.mkdir(parents=True, exist_ok=True)
    annual.to_csv(args.output / "annual.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    metadata = {
        "scope": "FINANCIAL_PIT_ANNUAL_QUALITY_2020_2024_V1",
        "market_scope": "SH/SZ",
        "excluded_exchanges": ["BJ"],
        "label": "raw 5-trading-day forward adjusted return",
        "years": list(YEARS),
        "candidate_count": len(CANDIDATE_METRICS),
        "annual_gate": "signed mean RankIC > 0.02, abs ICIR >= 0.50, directional share >= 0.55, finite ratio >= 0.70",
        "strict_stability_gate": "at least 4 of 5 annual gates pass",
        "database_writes": False,
        "locked_test_read": False,
        "stable_candidate_count": int(summary["stable_4_of_5"].sum()),
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    return metadata


if __name__ == "__main__":
    main()
