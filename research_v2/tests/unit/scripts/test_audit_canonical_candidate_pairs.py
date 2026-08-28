"""Tests for the canonical candidate-pair comparison contract."""

from __future__ import annotations

from datetime import date

import audit_canonical_candidate_pairs as audit
import pandas as pd
import pytest


def test_registry_has_exactly_six_versioned_candidates() -> None:
    assert [item.candidate_id for item in audit.CANDIDATES] == [
        "yeslab_114_tushare_shsz_v2",
        "yeslab_381_tushare_shsz_v1",
        "yeslab_405_tushare_shsz_v1",
        "yeslab_425_tushare_shsz_v2",
        "yeslab_484_tushare_shsz_v1",
        "yeslab_206_median_abs_v2",
    ]


def test_parser_defaults_to_the_fixed_comparison_window() -> None:
    args = audit.parse_args(["--output", "/tmp/x"])

    assert args.start == date(2020, 1, 2)
    assert args.end == date(2025, 12, 31)


def test_parser_requires_an_output_location() -> None:
    with pytest.raises(SystemExit):
        audit.parse_args([])


def test_parser_rejects_locked_test_range() -> None:
    with pytest.raises(SystemExit):
        audit.parse_args(
            ["--start", "2025-01-01", "--end", "2026-01-01", "--output", "/tmp/x"]
        )


def test_classify_pair_requires_complete_phase_days() -> None:
    assert audit.classify_pair(0.20, 1212, 0.20, 242) == "INCOMPLETE"


def test_classify_pair_marks_a_pair_below_monitor_threshold_as_unique() -> None:
    assert audit.classify_pair(0.79, 1212, 0.20, 243) == "PAIR_UNIQUE"


def test_classify_pair_marks_a_pair_at_monitor_threshold() -> None:
    assert audit.classify_pair(0.80, 1212, 0.20, 243) == "ELEVATED_PAIR_REDUNDANCY_MONITOR"


def test_classify_pair_marks_a_pair_at_redundancy_threshold() -> None:
    assert audit.classify_pair(0.20, 1212, 0.90, 243) == "REDUNDANT"


def test_alpha206_median_repair_uses_rolling_median() -> None:
    market = pd.DataFrame(
        {
            "trade_date": pd.date_range("2025-01-01", periods=60).repeat(2),
            "ts_code": ["000001.SZ", "600000.SH"] * 60,
            "vol": [1.0, 2.0] * 59 + [100.0, 2.0],
        }
    )

    result = audit.materialize_alpha206_median_repair(
        market, pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-01")
    )

    assert list(result.columns) == ["trade_date", "ts_code", "replay_value"]
    assert result.loc[result.ts_code == "000001.SZ", "replay_value"].iloc[0] == pytest.approx(
        99.0
    )


def test_summarize_phase_pairs_rejects_missing_edge() -> None:
    with pytest.raises(ValueError, match="incomplete candidate pair matrix"):
        audit.summarize_phase_pairs({"yeslab_114_tushare_shsz_v2": pd.DataFrame()}, 1)


def test_summarize_phase_pairs_returns_each_unordered_candidate_pair_once() -> None:
    keys = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2025-01-02")] * 30,
            "ts_code": [f"{index:06d}.SZ" for index in range(30)],
        }
    )
    replay_frames = {
        candidate.candidate_id: keys.assign(replay_value=range(30))
        for candidate in audit.CANDIDATES
    }

    result = audit.summarize_phase_pairs(replay_frames, expected_days=1)

    assert len(result) == 15
    assert not result.duplicated(["factor_id_left", "factor_id_right"]).any()
    assert set(result["valid_day_count"]) == {1}
    assert set(result["status"]) == {"COMPLETE"}


def test_load_comparison_keys_normalizes_the_v2_member_label_panel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        audit,
        "_load_validation_inputs",
        lambda _client, _start, _end: pd.DataFrame(
            {
                "signal_date": [pd.Timestamp("2025-01-03")],
                "ts_code": ["000001.SZ"],
                "raw_return": [0.02],
            }
        ),
    )

    result = audit.load_comparison_keys(
        object(), date(2025, 1, 2), date(2025, 1, 31)
    )

    assert result.to_dict(orient="records") == [
        {"trade_date": pd.Timestamp("2025-01-03"), "ts_code": "000001.SZ"}
    ]
