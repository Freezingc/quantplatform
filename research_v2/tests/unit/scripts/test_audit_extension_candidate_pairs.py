from __future__ import annotations

import pandas as pd
import pytest
import audit_extension_candidate_pairs as ext


def test_required_pairs_cover_extensions_times_base() -> None:
    pairs = ext._required_pairs()
    assert len(pairs) == len(ext.EXTENSION_CANDIDATES) * len(ext.CANDIDATES)
    assert len(set(pairs)) == len(pairs)
    for extension, base in pairs:
        assert extension in ext.EXTENSION_CANDIDATES
        assert base in {candidate.candidate_id for candidate in ext.CANDIDATES}


def test_summarize_extension_pairs_requires_expected_dates() -> None:
    date = pd.Timestamp("2025-01-02")
    frames = {}
    for candidate_id in [*ext.EXTENSION_CANDIDATES, *[candidate.candidate_id for candidate in ext.CANDIDATES]]:
        frames[candidate_id] = pd.DataFrame(
            {"trade_date": [date], "ts_code": ["000001.SZ"], "replay_value": [0.1]}
        )
    with pytest.raises(ValueError, match="incomplete extension pair dates"):
        ext.summarize_extension_pairs(frames, [date, pd.Timestamp("2025-01-03")])
