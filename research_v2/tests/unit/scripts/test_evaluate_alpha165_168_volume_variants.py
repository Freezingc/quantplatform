from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import evaluate_alpha165_168_volume_variants as evaluation


def test_compute_variants_preserves_window_and_variant_names() -> None:
    dates = pd.date_range("2025-01-01", periods=4)
    market = pd.DataFrame(
        {
            "trade_date": dates.repeat(2),
            "ts_code": ["000001.SZ", "600000.SH"] * 4,
            "vol": [1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0],
        }
    )
    result = evaluation.compute_variants(market, window=3)
    assert tuple(result) == evaluation.VARIANTS
    assert all(frame.shape == (4, 2) for frame in result.values())
    np.testing.assert_allclose(result["raw_volume_std"].iloc[2].to_numpy(), [1.0, 2.0])
    assert result["volume_cv"].iloc[2].notna().all()


def test_parser_rejects_locked_test() -> None:
    with pytest.raises(SystemExit):
        evaluation.parse_args(["--start", "2026-01-01", "--end", "2026-01-02", "--output", "/tmp/x"])
