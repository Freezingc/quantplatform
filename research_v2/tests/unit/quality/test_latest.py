from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from research_v2.quality import latest


def _formal_row(factor_id: str, *, monitoring_required: bool = False) -> dict[str, object]:
    return {
        "factor_id": factor_id,
        "status": "PASS_WITH_MONITORING" if monitoring_required else "PASS",
        "monitoring_required": monitoring_required,
        "reason_code": "FORMAL_NINE_SCOREBOARD",
        "frozen_direction": -1,
        "train_valid_days": 1212,
        "train_signed_rank_ic": 0.04,
        "train_icir": 0.60,
        "train_directional_share": 0.65,
        "annual_gate_years": 5,
        "train_max_similarity": 0.70,
        "validation_valid_days": 243,
        "validation_signed_rank_ic": 0.03,
        "validation_icir": 0.55,
        "validation_directional_share": 0.60,
        "validation_max_similarity": 0.71,
        "evidence_path": "formal.json",
    }


def _alpha95_payload() -> dict[str, object]:
    candidates: dict[str, object] = {}
    for window in (10, 20, 40):
        factor_id = f"yeslab_163_relative_amount_volatility_{window}_v1"
        candidates[factor_id] = {
            "train_2020_2024": {
                "direction": -1,
                "valid_day_count": 1212,
                "signed_mean_rank_ic": 0.03,
                "signed_icir": 0.40,
                "directional_day_share": 0.62,
                "annual_double_gate_pass_years": 2,
                "max_similarity": 0.60,
            },
            "validation_2025": {},
        }
    return {"scope": "ALPHA95", "candidates": candidates}


def _alpha484_payload(*, status: str = "READY_FOR_EXPLICIT_ADMISSION_REVIEW") -> dict[str, object]:
    return {
        "candidate_id": "yeslab_484_tushare_shsz_v1",
        "source_factor_id": "yeslab_484",
        "status": status,
        "training": {
            "direction": -1,
            "valid_day_count": 1212,
            "signed_mean_rank_ic": 0.04,
            "signed_icir": 0.60,
            "directional_day_share": 0.65,
            "annual_gate_years": 5,
        },
        "validation": {
            "valid_day_count": 243,
            "signed_mean_rank_ic": 0.03,
            "signed_icir": 0.55,
            "directional_day_share": 0.60,
        },
        "train_similarity": {"maximum_median_abs_spearman": 0.70},
        "validation_similarity": {"maximum_median_abs_spearman": 0.71},
    }


def test_build_quality_latest_frame_has_catalog_and_isolated_candidates() -> None:
    catalog = pd.DataFrame({"factor_id": [1, 2]})

    frame = latest.build_quality_latest_frame(
        catalog=catalog,
        scoreboard={"rows": [_formal_row("yeslab_1")]},
        alpha95=_alpha95_payload(),
        alpha484=_alpha484_payload(),
        effective_date=date(2026, 8, 14),
        evidence_commit="a" * 40,
    )

    assert set(frame["factor_id"]) == {
        "yeslab_1",
        "yeslab_2",
        "yeslab_163_relative_amount_volatility_10_v1",
        "yeslab_163_relative_amount_volatility_20_v1",
        "yeslab_163_relative_amount_volatility_40_v1",
        "yeslab_484_tushare_shsz_v1",
    }
    assert frame["factor_id"].is_unique
    assert frame.loc[frame["factor_id"] == "yeslab_1", "model_eligible"].item()
    assert not frame.loc[
        frame["factor_id"].str.contains("relative_amount"), "model_eligible"
    ].any()
    assert not frame.loc[
        frame["factor_id"] == "yeslab_484_tushare_shsz_v1", "model_eligible"
    ].item()


def test_validate_quality_latest_frame_rejects_lineage_fields() -> None:
    with pytest.raises(ValueError, match="forbidden columns"):
        latest.validate_quality_latest_frame(
            pd.DataFrame([latest.empty_quality_row("yeslab_1", date(2026, 8, 14))]).assign(
                run_id="forbidden"
            ),
            expected_catalog_count=1,
        )


def test_select_model_eligible_returns_only_sorted_current_shsz_rows() -> None:
    formal_ids = tuple(f"yeslab_{value}" for value in range(1, 10))
    frame = latest.build_quality_latest_frame(
        catalog=pd.DataFrame({"factor_id": list(range(1, 10))}),
        scoreboard={"rows": [_formal_row(factor_id) for factor_id in formal_ids]},
        alpha95=_alpha95_payload(),
        alpha484=_alpha484_payload(),
        effective_date=date(2026, 8, 14),
        evidence_commit="b" * 40,
    )

    eligible = latest.select_model_eligible(frame)

    assert eligible["factor_id"].tolist() == sorted(formal_ids)
    assert eligible["market_scope"].eq("SHSZ").all()
    assert eligible["label_id"].eq("forward_open_5d:v1").all()
    assert len(eligible) == 9
