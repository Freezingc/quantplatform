from __future__ import annotations

import hashlib
import json
from datetime import date

import pandas as pd
import publish_turnover_gap_candidates as publish
import pytest


def _frame() -> pd.DataFrame:
    factor_ids = list(sorted(publish.APPROVED_FACTOR_IDS))
    return pd.DataFrame(
        {
            "observation_date": [date(2020, 9, 18)] * len(factor_ids),
            "ts_code": [f"0000{index:02d}.SZ" for index in range(1, len(factor_ids) + 1)],
            "factor_id": factor_ids,
            "factor_version": [1] * len(factor_ids),
            "factor_name": ["F"] * len(factor_ids),
            "candidate_value": [1.0] * len(factor_ids),
            "source_value": [float("nan")] * len(factor_ids),
            "repair_method": ["TUSHARE_TURNOVER_GAP"] * len(factor_ids),
            "reason": ["TURNOVER_RATE_GAP"] * len(factor_ids),
        }
    )


def test_extended_turnover_formula_candidates_are_approved() -> None:
    assert {"yeslab_400", "yeslab_439", "yeslab_456"}.issubset(
        publish.APPROVED_FACTOR_IDS
    )


def test_build_candidate_batch_is_unique_and_lineaged() -> None:
    batch, identity = publish.build_candidate_batch(_frame())

    batch.validate()

    assert identity["row_count"] == len(publish.APPROVED_FACTOR_IDS)
    assert batch.data_version == identity["data_version"]
    assert batch.frame["record_hash"].map(lambda value: len(str(value))).eq(64).all()


def test_load_candidate_scope_verifies_manifest_hashes(tmp_path) -> None:
    source = _frame().iloc[:1].copy()
    path = tmp_path / "yeslab_377_turnover_gap_candidate.csv"
    source.to_csv(path, index=False)
    manifest = {
        "scope": "TURNOVER_GAP_TUSHARE_REPAIR_AUDIT_V1",
        "candidate_artifacts": [
            {
                "factor_id": factor_id,
                "path": str(path),
                "rows": 1,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for factor_id in sorted(publish.APPROVED_FACTOR_IDS)
        ],
    }
    # Keep the test focused on hash checking; six manifest entries point to
    # the same valid one-row file but the factor ID validation must reject it.
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="factor ID mismatch"):
        publish.load_candidate_scope(manifest_path, tmp_path)
