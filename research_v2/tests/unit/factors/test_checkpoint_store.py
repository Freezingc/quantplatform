from __future__ import annotations

import pandas as pd

from research_v2.factors.checkpoint import FactorCheckpointStore


def test_factor_checkpoint_store_writes_and_verifies_an_immutable_group(tmp_path: object) -> None:
    store = FactorCheckpointStore(tmp_path / "factors")  # type: ignore[operator]
    clean = pd.DataFrame(
        {"signal_date": ["2025-01-02"], "ts_code": ["000001.SZ"], "factor_id": ["f1"]}
    )
    quarantine = pd.DataFrame(
        {
            "observation_date": ["2025-01-02"],
            "ts_code": ["000002.SZ"],
            "factor_id": ["f1"],
            "reason": ["X"],
        }
    )

    checkpoint = store.write(
        identity={
            "source_table": "factor_data1",
            "factor_ids": ["f1"],
            "start": "2025-01-02",
            "end": "2025-01-10",
        },
        clean=clean,
        quarantine=quarantine,
    )

    assert checkpoint.path.is_dir()
    assert store.verify(checkpoint.checkpoint_id)
