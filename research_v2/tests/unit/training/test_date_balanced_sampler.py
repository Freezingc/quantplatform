from __future__ import annotations

from datetime import date

import pytest

from research_v2.training.sampler import DateBalancedSampler


def test_sampler_visits_each_complete_date_once_per_epoch_reproducibly() -> None:
    dates = (date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 6))
    sampler = DateBalancedSampler(dates, seed=20260801)

    first = sampler.for_epoch(0)
    repeated = sampler.for_epoch(0)
    next_epoch = sampler.for_epoch(1)

    assert first == repeated
    assert set(first) == set(dates)
    assert len(first) == len(dates)
    assert set(next_epoch) == set(dates)
    assert first != next_epoch


def test_sampler_rejects_duplicate_dates_and_invalid_epoch() -> None:
    with pytest.raises(ValueError, match="unique"):
        DateBalancedSampler((date(2025, 1, 2), date(2025, 1, 2)), seed=1)
    sampler = DateBalancedSampler((date(2025, 1, 2),), seed=1)
    with pytest.raises(ValueError, match="epoch"):
        sampler.for_epoch(-1)
