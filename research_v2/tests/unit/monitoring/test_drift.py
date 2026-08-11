from __future__ import annotations

import pytest

from research_v2.monitoring.drift import DriftSeverity, population_stability_index


def test_psi_is_zero_for_identical_distributions_and_escalates_for_shift() -> None:
    unchanged = population_stability_index((0.5, 0.5), (0.5, 0.5))
    shifted = population_stability_index((0.99, 0.01), (0.01, 0.99))

    assert unchanged.value == pytest.approx(0.0)
    assert unchanged.severity is DriftSeverity.INFO
    assert shifted.value > 0.25
    assert shifted.severity is DriftSeverity.ERROR
