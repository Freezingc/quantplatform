from __future__ import annotations

import evaluate_tushare_scope_validation as evaluation
import pandas as pd
import pytest


def test_restrict_scope_inputs_keeps_only_sh_sz_members_with_valid_labels() -> None:
    universe = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2025-01-02"] * 4),
            "ts_code": ["000001.SZ", "600000.SH", "830001.BJ", "000002.SZ"],
            "is_member": [True, True, True, False],
        }
    )
    labels = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2025-01-02"] * 3),
            "ts_code": ["000001.SZ", "600000.SH", "830001.BJ"],
            "raw_return": [0.1, 0.2, 0.3],
            "label_valid": [True, True, True],
        }
    )

    result = evaluation.restrict_scope_inputs(universe, labels)

    assert result.ts_code.tolist() == ["000001.SZ", "600000.SH"]
    assert result["raw_return"].tolist() == [0.1, 0.2]


def test_compute_daily_rank_ic_uses_rank_correlation() -> None:
    replay = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2025-01-02"] * 3),
            "ts_code": ["000001.SZ", "000002.SZ", "600000.SH"],
            "replay_value": [3.0, 1.0, 2.0],
        }
    )
    labels = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2025-01-02"] * 3),
            "ts_code": ["000001.SZ", "000002.SZ", "600000.SH"],
            "raw_return": [0.3, 0.1, 0.2],
        }
    )

    result = evaluation.compute_daily_rank_ic(replay, labels, minimum_cross_section=3)

    assert len(result) == 1
    assert result.loc[0, "rank_ic"] == 1.0
    assert result.loc[0, "sample_count"] == 3


def test_summarize_rank_ic_applies_fixed_direction() -> None:
    daily = pd.DataFrame(
        {
            "factor_id": ["yeslab_94", "yeslab_94", "yeslab_94"],
            "signal_date": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
            "rank_ic": [-0.2, -0.1, 0.1],
        }
    )

    result = evaluation.summarize_rank_ic(daily, direction=-1)

    assert result["valid_day_count"] == 3
    assert result["direction"] == -1
    assert result["signed_mean_rank_ic"] > 0
    assert result["directional_day_share"] == 2 / 3


def test_compute_pair_daily_spearman_reports_median_absolute_similarity() -> None:
    left = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2025-01-02"] * 3),
            "ts_code": ["000001.SZ", "000002.SZ", "600000.SH"],
            "replay_value": [1.0, 2.0, 3.0],
        }
    )
    right = left.assign(replay_value=[3.0, 2.0, 1.0])

    result = evaluation.compute_pair_daily_spearman(
        {"yeslab_left": left, "yeslab_right": right}, minimum_cross_section=3
    )

    assert len(result) == 1
    assert result.loc[0, "factor_id_left"] == "yeslab_left"
    assert result.loc[0, "factor_id_right"] == "yeslab_right"
    assert result.loc[0, "median_abs_spearman"] == 1.0
    assert result.loc[0, "valid_day_count"] == 1


def test_parser_accepts_stored_factor_comparison() -> None:
    args = evaluation.parse_args(
        [
            "--factor-id",
            "yeslab_155",
            "--stored-factor-id",
            "yeslab_380",
            "--start",
            "2025-01-01",
            "--end",
            "2025-12-31",
            "--output",
            "/tmp/scope-similarity",
        ]
    )

    assert args.factor_id == ("yeslab_155",)
    assert args.stored_factor_id == ("yeslab_380",)


def test_load_directions_accepts_frozen_direction_artifact(tmp_path) -> None:
    path = tmp_path / "directions.csv"
    pd.DataFrame(
        {
            "factor_id": ["yeslab_376"],
            "frozen_direction": [-1],
        }
    ).to_csv(path, index=False)

    assert evaluation._load_directions(path, ("yeslab_376",)) == {"yeslab_376": -1}


def test_parser_can_skip_expensive_pair_similarity() -> None:
    args = evaluation.parse_args(
        [
            "--factor-id",
            "yeslab_376",
            "--start",
            "2020-01-01",
            "--end",
            "2024-12-31",
            "--skip-similarity",
            "--output",
            "/tmp/train-quality-only",
        ]
    )

    assert args.skip_similarity is True


def test_parser_accepts_pit_industry_cache() -> None:
    args = evaluation.parse_args(
        [
            "--factor-id",
            "yeslab_357",
            "--pit-industry-cache",
            "/tmp/pit-industry-cache",
            "--pit-industry-cutover",
            "2021-12-10",
            "--start",
            "2025-01-01",
            "--end",
            "2025-12-31",
            "--output",
            "/tmp/pit-quality",
        ]
    )

    assert str(args.pit_industry_cache) == "/tmp/pit-industry-cache"
    assert str(args.pit_industry_cutover) == "2021-12-10"


def test_daily_basic_contract_includes_total_mv() -> None:
    assert evaluation.requires_daily_basic(("pct_chg", "total_mv")) is True


def test_adjusted_field_contract_is_required_for_price_based_replay() -> None:
    required_fields = ("close_hfq", "high_hfq", "low_hfq")
    include_adjusted = bool(
        {"open_hfq", "high_hfq", "low_hfq", "close_hfq", "pre_close_hfq"}
        & set(required_fields)
    )

    assert include_adjusted is True


def test_load_validation_inputs_reads_current_ranges_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object, object]] = []

    def fake_read_current_universe_range(
        session: object, *, start: object, end: object
    ) -> pd.DataFrame:
        del session
        calls.append(("universe", start, end))
        return pd.DataFrame(
            {
                "batch_id": [1, 1],
                "signal_date": pd.to_datetime(["2020-02-03", "2020-02-03"]),
                "ts_code": ["000001.SZ", "600000.SH"],
                "is_member": [True, True],
            }
        )

    def fake_read_current_label_range(
        session: object, *, start: object, end: object
    ) -> pd.DataFrame:
        del session
        calls.append(("label", start, end))
        return pd.DataFrame(
            {
                "batch_id": [1, 1],
                "signal_date": pd.to_datetime(["2020-02-03", "2020-02-03"]),
                "ts_code": ["000001.SZ", "600000.SH"],
                "raw_return": [0.1, 0.2],
                "label_valid": [True, True],
            }
        )

    monkeypatch.setattr(
        evaluation,
        "_read_current_universe_range",
        fake_read_current_universe_range,
        raising=False,
    )
    monkeypatch.setattr(
        evaluation,
        "_read_current_label_range",
        fake_read_current_label_range,
        raising=False,
    )
    result = evaluation._load_validation_inputs(
        object(),
        pd.Timestamp("2020-02-01").date(),
        pd.Timestamp("2020-02-29").date(),
    )

    assert calls == [
        ("universe", pd.Timestamp("2020-02-01").date(), pd.Timestamp("2020-02-29").date()),
        ("label", pd.Timestamp("2020-02-01").date(), pd.Timestamp("2020-02-29").date()),
    ]
    assert result["raw_return"].tolist() == [0.1, 0.2]


def test_read_current_label_range_casts_boolean_to_int_in_dolphindb_query() -> None:
    class RecordingSession:
        target_database = "dfs://QuantResearchV2Test"

        def __init__(self) -> None:
            self.queries: list[str] = []

        def query(self, script: str) -> pd.DataFrame:
            self.queries.append(script)
            return pd.DataFrame(
                {
                    "batch_id": [7],
                    "signal_date": pd.to_datetime(["2020-02-03"]),
                    "ts_code": ["000001.SZ"],
                    "raw_return": [0.01],
                    "label_valid": [1],
                }
            )

    session = RecordingSession()

    result = evaluation._read_current_label_range(
        session,
        start=pd.Timestamp("2020-01-02").date(),
        end=pd.Timestamp("2020-12-31").date(),
    )

    assert "latest_manifest" in session.queries[0]
    assert "label_daily_latest" in session.queries[0]
    assert "batch_id,status from" in session.queries[0]
    assert "context by logical_partition csort published_at desc" in session.queries[0]
    assert "where status=`PUBLISHED" in session.queries[0]
    assert "logical_partition between '2020-01':'2020-12'" in session.queries[0]
    assert "int(label_valid) as label_valid" in session.queries[0]
    assert result["label_valid"].tolist() == [True]
