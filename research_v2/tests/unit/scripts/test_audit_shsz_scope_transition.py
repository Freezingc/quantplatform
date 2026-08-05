from __future__ import annotations

import audit_shsz_scope_transition as audit


def test_scope_code_summary_counts_exchange_suffixes() -> None:
    result = audit.summarize_code_scope(
        ["000001.SZ", "600000.SH", "920001.BJ", "OTHER"]
    )

    assert result == {
        "unique_code_count": 4,
        "bj_code_count": 1,
        "sh_code_count": 1,
        "sz_code_count": 1,
        "other_code_count": 1,
        "scope_status": "FAIL",
    }


def test_scope_code_summary_passes_only_pure_sh_sz() -> None:
    result = audit.summarize_code_scope(["000001.SZ", "600000.SH", "000001.SZ"])

    assert result["unique_code_count"] == 2
    assert result["bj_code_count"] == 0
    assert result["scope_status"] == "PASS"


def test_factor_semantics_require_pct_chg_and_no_forward_window() -> None:
    semantics = audit.inspect_factor_semantics(
        ("yeslab_376", "yeslab_396", "yeslab_442")
    )

    assert [row["factor_id"] for row in semantics] == [
        "yeslab_376",
        "yeslab_396",
        "yeslab_442",
    ]
    for row in semantics:
        assert row["required_market_fields"] == ["pct_chg"]
        assert row["required_non_market_fields"] == []
        assert row["uses_cross_sectional_standardization"] is True
        assert row["uses_forward_lookahead"] is False
        assert row["scope_contract"] == "SH_SZ_ONLY"


def test_build_scope_review_stays_pending_until_manual_approval() -> None:
    report = audit.build_scope_review(
        factor_semantics=[{"factor_id": "yeslab_376"}],
        scope_evidence={
            "v2_universe": {"scope_status": "PASS"},
            "stored_source": {"scope_status": "FAIL"},
        },
    )

    assert report["formula_scope_gate"] == "PENDING_SCOPE_TRANSITION_REVIEW"
    assert report["review_status"] == "READY_FOR_MANUAL_SCOPE_APPROVAL"
    assert report["database_writes"] is False
    assert report["locked_test_read"] is False
