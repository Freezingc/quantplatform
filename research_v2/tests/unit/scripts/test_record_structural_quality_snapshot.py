from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from record_raw_conflict_snapshot import classify_conflict_metrics
from record_raw_pit_key_snapshot import summarize_finite_key_coverage
from record_raw_source_quality_snapshot import classify_raw_metrics, summarize_raw_aggregates
from record_repaired_factor_rankic import merge_daily_rankic
from record_source_quality_combined_snapshot import combine_source_statuses
from record_structural_quality_snapshot import compute_structural_metrics
from record_train_rankic_similarity import rankic_similarity_pairs
from record_train_similarity_full_day import (
    apply_candidate_overlay,
    daily_similarity_values,
    query_clean_factor_month,
    vectorized_clean_technical_sample,
)
from record_train_similarity_sample import _pairwise_sample_similarity, classify_similarity_pairs
from record_validation_quality_snapshot import compute_validation_metrics


def test_structural_metrics_uses_reference_cross_section_and_monthly_coverage() -> None:
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    daily = pd.DataFrame(
        {
            "factor_id": ["yeslab_1"] * 4 + ["yeslab_2"] * 2,
            "signal_date": list(dates) + list(dates[:2]),
            "sample_count": [80, 80, 80, 80, 30, 30],
        }
    )
    reference = pd.DataFrame(
        {"signal_date": dates, "sample_count": [100, 100, 100, 100]}
    )

    result = compute_structural_metrics(
        daily,
        reference,
        expected_factor_ids=("yeslab_1", "yeslab_2", "yeslab_3"),
        minimum_day_ratio=0.75,
        minimum_valid_day_count=1,
        minimum_median_coverage_ratio=0.70,
        minimum_p05_coverage_ratio=0.70,
        minimum_monthly_coverage_ratio=0.70,
    ).set_index("factor_id")

    assert bool(result.loc["yeslab_1", "coverage_gate_pass"])
    assert result.loc["yeslab_1", "median_coverage_ratio"] == 0.8
    assert not bool(result.loc["yeslab_2", "coverage_gate_pass"])
    assert result.loc["yeslab_3", "decision_status"] == "STRUCTURAL_DQ_PENDING_INPUT"


def test_raw_sample_audit_requires_finite_values_but_allows_expected_revisions() -> None:
    passed = classify_raw_metrics(
        {"rows": 100, "finite": 90, "duplicate_rows": 50}, minimum_finite_ratio=0.70
    )
    assert passed["decision_status"] == "RAW_SAMPLE_AUDIT_PASS_PENDING_FULL_RANGE_PIT"
    failed = classify_raw_metrics(
        {"rows": 100, "finite": 50, "duplicate_rows": 0}, minimum_finite_ratio=0.70
    )
    assert failed["decision_status"] == "RAW_SAMPLE_AUDIT_FAIL"


def test_full_range_raw_aggregate_counts_duplicate_rows_and_missing_days() -> None:
    rows = pd.DataFrame(
        {
            "factor_name": ["F", "F", "F"],
            "trade_date": pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-03"]),
            "rows": [10, 8, 8],
            "finite": [9, 7, 8],
        }
    )
    securities = pd.DataFrame(
        {
            "factor_name": ["F", "F", "F"],
            "trade_date": pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-03"]),
            "securities": [5, 5, 8],
        }
    )
    result = summarize_raw_aggregates(
        rows,
        securities,
        expected_dates=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
        expected_factor_ids=("F",),
        minimum_day_ratio=0.50,
    ).set_index("factor_name")

    assert result.loc["F", "raw_rows"] == 26
    assert result.loc["F", "duplicate_rows"] == 13
    assert result.loc["F", "observed_day_count"] == 2
    assert result.loc["F", "missing_month_count"] == 0


def test_conflicting_source_values_are_blocked_even_when_duplicates_are_expected() -> None:
    assert (
        classify_conflict_metrics(0)["decision_status"]
        == "RAW_CONFLICT_CHECK_PASS_PENDING_MATERIALIZATION"
    )
    failed = classify_conflict_metrics(3)
    assert failed["decision_status"] == "RAW_CONFLICT_CHECK_FAIL"
    assert "CONFLICT_VALUE_KEYS" in failed["reason_codes"]


def test_finite_key_coverage_compares_deduplicated_source_to_reference() -> None:
    finite_keys = pd.DataFrame(
        {
            "factor_name": ["F", "F"],
            "trade_date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
            "finite_keys": [90, 90],
        }
    )
    reference = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
            "sample_count": [100, 100],
        }
    )
    result = summarize_finite_key_coverage(
        finite_keys,
        reference,
        expected_factor_names=("F",),
        minimum_day_ratio=0.5,
        minimum_median_coverage_ratio=0.8,
        minimum_p05_coverage_ratio=0.7,
    ).set_index("factor_name")
    assert bool(result.loc["F", "pit_coverage_gate_pass"])
    assert result.loc["F", "median_pit_coverage_ratio"] == 0.9


def test_combined_source_gate_requires_all_three_source_checks() -> None:
    passed = combine_source_statuses(
        raw_status="RAW_FULL_RANGE_PASS_PENDING_PIT_CONFLICT",
        conflict_status="RAW_CONFLICT_CHECK_PASS_PENDING_MATERIALIZATION",
        pit_status="PIT_KEY_DQ_PASS_PENDING_COMPACT_UNIVERSE",
    )
    assert passed["decision_status"] == "SOURCE_DQ_PASS_PENDING_SIMILARITY"
    failed = combine_source_statuses(
        raw_status="RAW_FULL_RANGE_FAIL",
        conflict_status="RAW_CONFLICT_CHECK_PASS_PENDING_MATERIALIZATION",
        pit_status="PIT_KEY_DQ_PASS_PENDING_COMPACT_UNIVERSE",
    )
    assert failed["decision_status"] == "SOURCE_DQ_FAIL"


def test_train_similarity_sample_marks_only_highest_pair_as_redundant() -> None:
    pairs = pd.DataFrame(
        {
            "factor_id_left": ["yeslab_1", "yeslab_1", "yeslab_2"],
            "factor_id_right": ["yeslab_2", "yeslab_3", "yeslab_3"],
            "median_daily_abs_spearman": [0.95, 0.50, 0.92],
            "valid_days": [20, 20, 20],
        }
    )
    result = classify_similarity_pairs(
        pairs, factor_ids=("yeslab_1", "yeslab_2", "yeslab_3"), threshold=0.90
    ).set_index("factor_id")
    assert result.loc["yeslab_1", "nearest_similarity"] == 0.95
    assert result.loc["yeslab_2", "decision_status"] == "SIMILARITY_SAMPLE_REDUNDANT_PENDING_FULL"


def test_pairwise_similarity_counts_common_members_not_boolean_presence() -> None:
    observations = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2020-01-02"] * 40 * 2).date,
            "ts_code": [f"S{i:03d}" for i in range(40)] * 2,
            "factor_id": ["yeslab_1"] * 40 + ["yeslab_2"] * 40,
            "factor_version": [1] * 80,
            "value": list(range(40)) + list(range(40)),
        }
    )
    result = _pairwise_sample_similarity(observations, ("yeslab_1", "yeslab_2"))
    assert result.loc[0, "valid_days"] == 1
    assert result.loc[0, "median_daily_abs_spearman"] == 1.0


def test_rankic_series_similarity_is_train_scope_bounded() -> None:
    daily = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(
                ["2020-01-02", "2020-01-03", "2020-01-06"] * 2
            ),
            "factor_id": ["yeslab_1"] * 3 + ["yeslab_2"] * 3,
            "rank_ic": [0.1, 0.2, 0.3, -0.1, -0.2, -0.3],
        }
    )
    result = rankic_similarity_pairs(
        daily, factor_ids=("yeslab_1", "yeslab_2"), minimum_valid_days=2
    )
    assert result.loc[0, "valid_days"] == 3
    assert result.loc[0, "abs_rankic_correlation"] == 1.0


def test_full_day_similarity_returns_one_value_per_valid_date() -> None:
    observations = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2020-01-02"] * 40 * 2).date,
            "ts_code": [f"S{i:03d}" for i in range(40)] * 2,
            "factor_id": ["yeslab_1"] * 40 + ["yeslab_2"] * 40,
            "factor_version": [1] * 80,
            "value": list(range(40)) + list(range(40)),
        }
    )
    values = daily_similarity_values(observations, ("yeslab_1", "yeslab_2"))
    assert len(values[("yeslab_1", "yeslab_2")]) == 1
    assert values[("yeslab_1", "yeslab_2")][0] == 1.0


def test_vectorized_clean_sample_filters_invalid_values_and_collapses_duplicates() -> None:
    class Definition:
        def __init__(self, factor_id: str, display_name: str) -> None:
            self.factor_id = factor_id
            self.display_name = display_name

    raw = pd.DataFrame(
        {
            "ts_code": ["S1", "S1", "S2"],
            "trade_date": pd.to_datetime(["2020-01-02"] * 3),
            "factor_name": ["F", "F", "F"],
            "value": [1.0, 1.0, float("nan")],
        }
    )
    result = vectorized_clean_technical_sample(raw, (Definition("yeslab_1", "F"),))
    assert len(result) == 1
    assert result.iloc[0]["factor_id"] == "yeslab_1"


def test_candidate_overlay_only_adds_missing_factor_keys() -> None:
    observations = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2020-09-18"]),
            "ts_code": ["S1"],
            "factor_id": ["yeslab_377"],
            "factor_version": [1],
            "value": [10.0],
        }
    )
    candidates = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2020-09-18"] * 2),
            "ts_code": ["S1", "S2"],
            "factor_id": ["yeslab_377"] * 2,
            "factor_version": [1, 1],
            "value": [99.0, 20.0],
        }
    )

    result = apply_candidate_overlay(observations, candidates)

    assert result.sort_values("ts_code")["value"].tolist() == [10.0, 20.0]


def test_candidate_overlay_rejects_duplicate_source_keys() -> None:
    observations = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2020-09-18"] * 2),
            "ts_code": ["S1", "S1"],
            "factor_id": ["yeslab_377"] * 2,
            "factor_version": [1, 1],
            "value": [10.0, 11.0],
        }
    )
    candidates = observations.iloc[:0].copy()

    with pytest.raises(ValueError, match="observations contain duplicate"):
        apply_candidate_overlay(observations, candidates)


def test_query_clean_factor_month_uses_v2_factor_panel() -> None:
    class Session:
        def query(self, script: str) -> object:
            if "published_version" in script:
                return pd.DataFrame({"data_version": ["v1"]})
            assert "factor_observation_clean" in script
            assert "2020.01.01" in script
            assert "data_version='v1'" in script
            return pd.DataFrame(
                {
                    "signal_date": pd.to_datetime(["2020-01-02"]),
                    "ts_code": ["S1"],
                    "factor_id": ["yeslab_377"],
                    "factor_version": [1],
                    "value": [1.0],
                }
            )

    result = query_clean_factor_month(
        Session(),
        factor_ids=("yeslab_377",),
        start=date(2020, 1, 1),
        end=date(2020, 1, 31),
    )

    assert result[["factor_id", "ts_code"]].to_dict("records") == [
        {"factor_id": "yeslab_377", "ts_code": "S1"}
    ]


def test_merge_daily_rankic_replaces_only_repaired_factor_dates() -> None:
    existing = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2020-09-18", "2020-09-19"]),
            "factor_id": ["yeslab_377", "yeslab_377"],
            "factor_version": [1, 1],
            "rank_ic": [0.1, 0.2],
            "sample_count": [100, 100],
        }
    )
    repaired = existing.iloc[[0]].assign(rank_ic=0.3)

    result = merge_daily_rankic(existing, repaired)

    assert result.sort_values("signal_date")["rank_ic"].tolist() == [0.3, 0.2]


def test_validation_metrics_use_fixed_train_direction_and_fdr_gate() -> None:
    daily = pd.DataFrame(
        {
            "factor_id": ["yeslab_1"] * 80 + ["yeslab_2"] * 80,
            "signal_date": list(pd.date_range("2025-01-01", periods=80)) * 2,
            "rank_ic": [0.05] * 80 + [0.001] * 80,
        }
    )
    result = compute_validation_metrics(
        daily,
        train_directions={"yeslab_1": -1, "yeslab_2": 1},
        minimum_valid_days=60,
        minimum_abs_mean_rank_ic=0.02,
        minimum_abs_icir=0.50,
        minimum_directional_share=0.55,
        maximum_fdr_q=0.10,
    ).set_index("factor_id")
    assert result.loc["yeslab_1", "decision_status"] == "VALIDATION_2025_FAIL"
    assert result.loc["yeslab_2", "decision_status"] == "VALIDATION_2025_FAIL"
