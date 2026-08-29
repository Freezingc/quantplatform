from __future__ import annotations

import pandas as pd
import pytest
import summarize_fundamental_annual_quality as annual


def _inputs() -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
    annual_by_year = {}
    daily_by_year = {}
    for year in annual.YEARS:
        annual_by_year[year] = pd.DataFrame(
            {
                "metric_id": list(annual.CANDIDATE_METRICS),
                "valid_day_count": [243] * len(annual.CANDIDATE_METRICS),
                "mean_rank_ic": [0.03] * len(annual.CANDIDATE_METRICS),
                "icir": [0.60] * len(annual.CANDIDATE_METRICS),
                "positive_day_share": [0.60] * len(annual.CANDIDATE_METRICS),
                "finite_ratio": [0.90] * len(annual.CANDIDATE_METRICS),
            }
        )
        daily_by_year[year] = pd.DataFrame(
            {
                "factor_name": ["cfoa", "cfoa"],
                "signal_date": [f"{year}-01-03", f"{year}-01-04"],
                "rank_ic": [0.02, 0.04],
            }
        )
    return annual_by_year, daily_by_year


def test_annual_report_freezes_direction_and_counts_four_of_five() -> None:
    annual_by_year, daily_by_year = _inputs()
    annual_by_year[2024].loc[annual_by_year[2024]["metric_id"].eq("cfoa"), "icir"] = 0.1

    annual_frame, summary = annual.build_annual_report(annual_by_year, daily_by_year)

    cfoa = summary.loc[summary["metric_id"].eq("cfoa")].iloc[0]
    assert cfoa["frozen_direction"] == 1
    assert cfoa["strict_annual_pass_years"] == 4
    assert cfoa["stable_4_of_5"]
    assert annual_frame.loc[annual_frame["metric_id"].eq("cfoa"), "annual_gate_pass"].sum() == 4


def test_annual_report_rejects_missing_candidate_year() -> None:
    annual_by_year, daily_by_year = _inputs()
    del annual_by_year[2024]
    with pytest.raises(ValueError, match="2024"):
        annual.build_annual_report(annual_by_year, daily_by_year)
