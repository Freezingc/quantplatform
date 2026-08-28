from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import pandas as pd

from research_v2.export.compatibility import export_compatibility_predictions

_MODEL_ID = "sha256:" + "a" * 64


def test_compatibility_csv_matches_the_approved_golden_file(tmp_path: Path) -> None:
    predictions = pd.DataFrame(
        {
            "signal_date": [date(2026, 1, 2), date(2026, 1, 2)],
            "ts_code": ["600000.SH", "000001.SZ"],
            "score": [0.9, 0.1],
        }
    )
    result = export_compatibility_predictions(
        predictions,
        model_id=_MODEL_ID,
        model_version=1,
        signal_date=date(2026, 1, 2),
        source_prediction_data_version="prediction-v1",
        artifact_root=tmp_path / "artifacts",
        legacy_output_roots=(tmp_path / "legacy-v1-output",),
    )
    expected = Path(__file__).parent / "data" / "expected_export.csv"

    assert result.csv_path.read_bytes() == expected.read_bytes()
    expected_hash = (Path(__file__).parent / "data" / "expected_export.pkl.sha256").read_text(
        encoding="utf-8"
    ).strip()
    assert hashlib.sha256(result.pkl_path.read_bytes()).hexdigest() == expected_hash
    pkl = pd.read_pickle(result.pkl_path)
    assert pkl.dtypes.astype(str).to_dict() == {
        "instrument": "object",
        "datetime": "object",
        "score": "float64",
    }
    assert pkl.to_dict(orient="records") == [
        {"instrument": "000001.SZ", "datetime": "2026-01-02", "score": 0.1},
        {"instrument": "600000.SH", "datetime": "2026-01-02", "score": 0.9},
    ]
