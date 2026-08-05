from __future__ import annotations

import pandas as pd

import evaluate_tushare_scope_validation as evaluation


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
