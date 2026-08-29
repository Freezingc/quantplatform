from __future__ import annotations

import audit_preclose_variant as audit
import pandas as pd
import pytest


def test_preclose_variant_uses_tushare_pre_close() -> None:
    market = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2025-01-02")] * 2,
            "ts_code": ["000001.SZ", "000002.SZ"],
            "open": [11.0, 20.0],
            "pre_close": [10.0, 20.0],
        }
    )

    result = audit.compute_preclose_skip_variant(market)

    assert result.set_index("ts_code").loc["000001.SZ", "replay_value"] == pytest.approx(0.05)
    assert result.set_index("ts_code").loc["000002.SZ", "replay_value"] == pytest.approx(0.05)


def test_preclose_variant_rejects_duplicate_keys() -> None:
    market = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2025-01-02")] * 2,
            "ts_code": ["000001.SZ", "000001.SZ"],
            "open": [11.0, 11.0],
            "pre_close": [10.0, 10.0],
        }
    )

    with pytest.raises(ValueError, match="duplicate"):
        audit.compute_preclose_skip_variant(market)


def test_parser_rejects_locked_test() -> None:
    with pytest.raises(ValueError, match="2026"):
        audit.parse_args(
            ["--start", "2026-01-01", "--end", "2026-01-02", "--output", "/tmp/x.json"]
        )
