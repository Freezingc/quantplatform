"""Golden boundary samples for deterministic market cleaning (B8)."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from research_v2.pit.market import clean_market_daily

_DATA = Path(__file__).parent / "data"


def _canonical_value(value: object) -> object:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and value != value:
        return None
    return value


def _canonical_rows_sha256(rows: list[dict[str, object]]) -> str:
    normalized = [
        {key: _canonical_value(value) for key, value in record.items()} for record in rows
    ]
    normalized.sort(key=lambda record: (record["trade_date"], record["ts_code"]))
    payload = json.dumps(normalized, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_golden_market_clean_matches_independent_reference() -> None:
    raw = pd.read_csv(_DATA / "pit_market_raw.csv")
    expected = json.loads((_DATA / "expected_pit_market_clean.json").read_text(encoding="utf-8"))

    clean = clean_market_daily(raw)

    assert clean.shape[0] == len(expected["rows"])
    for index, expected_row in enumerate(expected["rows"]):
        actual = clean.iloc[index]
        assert str(actual["trade_date"]) == expected_row["trade_date"]
        assert str(actual["ts_code"]) == expected_row["ts_code"]
        for column in ("open", "high", "low", "close", "vol", "amount", "adj_factor"):
            assert float(actual[column]) == pytest.approx(expected_row[column], abs=1e-12)
        assert str(actual["dedup_resolution"]) == expected_row["dedup_resolution"]
        assert float(actual["adjusted_open"]) == pytest.approx(
            expected_row["adjusted_open"], abs=1e-12
        )


def test_golden_market_clean_canonical_rows_sha256_is_stable() -> None:
    raw = pd.read_csv(_DATA / "pit_market_raw.csv")
    expected = json.loads((_DATA / "expected_pit_market_clean.json").read_text(encoding="utf-8"))

    clean = clean_market_daily(raw)

    assert _canonical_rows_sha256(clean.to_dict(orient="records")) == expected[
        "canonical_rows_sha256"
    ]
