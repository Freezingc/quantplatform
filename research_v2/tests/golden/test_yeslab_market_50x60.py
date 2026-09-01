"""Golden equality evidence for the real 50x60 boundary sample (B8.2/B8.3).

Compares the production cleaning/history/label pipelines against expectations
recomputed by an independent hand-written implementation
(``scripts/compute_pit_golden_expected.py``) using canonical row SHA-256 hashes.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd

from research_v2.labels.forward_return import build_forward_open_labels
from research_v2.pit.calendar import TradingCalendar
from research_v2.pit.market import add_market_history_features, clean_market_daily

_DATA = Path(__file__).parent / "data"


def _expected() -> dict:
    return json.loads(
        (_DATA / "expected_yeslab_market_50x60.json").read_text(encoding="utf-8")
    )


def _canonical_sha256(rows: list[dict]) -> str:
    payload = json.dumps(rows, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_production_clean_matches_independent_expectations() -> None:
    extract = pd.read_parquet(_DATA / "yeslab_market_50x60.parquet")
    expected = _expected()

    clean = clean_market_daily(extract)
    enriched = add_market_history_features(clean)

    ordered = enriched.sort_values(["trade_date", "ts_code"], kind="mergesort")
    rows = [
        {
            "ts_code": record.ts_code,
            "trade_date": pd.Timestamp(record.trade_date).date().isoformat(),
            "adjusted_open": round(float(record.adjusted_open), 9),
            "valid_trade_day": bool(record.valid_trade_day),
            "valid_days_20": int(record.valid_days_20),
            "adv20": None if pd.isna(record.adv20) else round(float(record.adv20), 3),
        }
        for record in ordered.itertuples(index=False)
    ]

    assert len(rows) == expected["clean_row_count"]
    assert _canonical_sha256(rows) == expected["clean_rows_sha256"]


def test_production_labels_match_independent_expectations() -> None:
    extract = pd.read_parquet(_DATA / "yeslab_market_50x60.parquet")
    extension = pd.read_parquet(_DATA / "yeslab_label_extension_50x60.parquet")
    expected = _expected()

    calendar = TradingCalendar.from_dates(
        [date.fromisoformat(day) for day in expected["calendar"]]
    )
    market = pd.DataFrame(
        {
            "trade_date": extension["trade_date"],
            "ts_code": extension["ts_code"],
            "adjusted_open": extension["open"] * extension["adj_factor"],
        }
    )
    members = extract.loc[:, ["trade_date", "ts_code"]].rename(
        columns={"trade_date": "signal_date"}
    )

    labels = build_forward_open_labels(members, market, calendar)
    valid = labels.loc[labels["label_valid"].astype(bool)]
    rows = [
        {
            "signal_date": record.signal_date.isoformat(),
            "ts_code": record.ts_code,
            "entry_date": record.entry_date.isoformat(),
            "exit_date": record.exit_date.isoformat(),
            "raw_return_5d": float(record.raw_return_5d),
        }
        for record in valid.sort_values(
            ["signal_date", "ts_code"], kind="mergesort"
        ).itertuples(index=False)
    ]

    assert len(rows) == expected["label_row_count"]
    assert _canonical_sha256(rows) == expected["label_rows_sha256"]
