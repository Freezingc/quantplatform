from __future__ import annotations

import audit_extension_candidate_pairs as ext
import pandas as pd
import pytest


def test_required_pairs_cover_extensions_and_base() -> None:
    pairs = ext._required_pairs()
    expected_count = len(ext.EXTENSION_CANDIDATES) * len(ext.CANDIDATES)
    expected_count += len(ext.EXTENSION_CANDIDATES) * (len(ext.EXTENSION_CANDIDATES) - 1) // 2
    assert len(pairs) == expected_count == 30
    assert len(set(pairs)) == len(pairs)
    base_ids = {candidate.candidate_id for candidate in ext.CANDIDATES}
    for left, right in pairs:
        assert left in ext.EXTENSION_CANDIDATES
        assert right in base_ids or right in ext.EXTENSION_CANDIDATES


def test_summarize_extension_pairs_requires_expected_dates() -> None:
    date = pd.Timestamp("2025-01-02")
    frames = {}
    candidate_ids = [
        *ext.EXTENSION_CANDIDATES,
        *[candidate.candidate_id for candidate in ext.CANDIDATES],
    ]
    for candidate_id in candidate_ids:
        frames[candidate_id] = pd.DataFrame(
            {"trade_date": [date], "ts_code": ["000001.SZ"], "replay_value": [0.1]}
        )
    with pytest.raises(ValueError, match="incomplete extension pair dates"):
        ext.summarize_extension_pairs(frames, [date, pd.Timestamp("2025-01-03")])
