from __future__ import annotations

import hashlib
import json
from datetime import date

import pandas as pd
import publish_turnover_gap_candidates as publish
import pytest


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "observation_date": [date(2020, 9, 18)] * 6,
            "ts_code": [
                "000001.SZ",
                "000002.SZ",
                "000003.SZ",
                "000004.SZ",
                "000005.SZ",
                "000006.SZ",
            ],
            "factor_id": list(publish.EXPECTED_FACTOR_IDS),
            "factor_version": [1] * 6,
            "factor_name": ["F"] * 6,
            "candidate_value": [1.0] * 6,
            "source_value": [float("nan")] * 6,
            "repair_method": ["TUSHARE_TURNOVER_GAP"] * 6,
            "reason": ["TURNOVER_RATE_GAP"] * 6,
        }
    )


def test_build_candidate_batch_is_unique_and_lineaged() -> None:
    batch, identity = publish.build_candidate_batch(_frame())

    batch.validate()

    assert identity["row_count"] == 6
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
            for factor_id in publish.EXPECTED_FACTOR_IDS
        ],
    }
    # Keep the test focused on hash checking; six manifest entries point to
    # the same valid one-row file but the factor ID validation must reject it.
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="factor ID mismatch"):
        publish.load_candidate_scope(manifest_path, tmp_path)
