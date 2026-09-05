"""Tests for the admission-consistency verifier.

Fixtures use the values actually recorded in factor_quality_snapshot for the frozen set,
so the tests encode the incident: six admitted factors carried COVERAGE_GATE_FAIL while
the contract declared no conflicts.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "verify_admission_consistency.py"


def _load():
    spec = importlib.util.spec_from_file_location("verify_admission_consistency", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_admission_consistency"] = module
    spec.loader.exec_module(module)
    return module


M = _load()


def _record(factor_id, scope, status, codes=(), median=None, minimum=None):
    return M.ScreeningRecord(
        factor_id=factor_id, scope_id=scope, eligibility_status=status,
        reason_codes=tuple(codes), median_coverage=median, minimum_coverage=minimum,
    )


# Values as recorded in SCREEN_TRAIN2020_2024_STRUCTURAL_COVERAGE_V1/_V2.
FAIL_372 = _record(
    "yeslab_372", "SCREEN_TRAIN2020_2024_STRUCTURAL_COVERAGE_V1", "ELIGIBILITY_FAIL",
    ("COVERAGE_GATE_FAIL", "LOW_MEDIAN_SAMPLE_COVERAGE"), 0.46357704421254553,
    0.028759894459102903,
)
FAIL_384 = _record(
    "yeslab_384", "SCREEN_TRAIN2020_2024_STRUCTURAL_COVERAGE_V2", "ELIGIBILITY_FAIL",
    ("COVERAGE_GATE_FAIL",), 0.49218312512090573, 0.1608840914931894,
)
PASS_484 = _record(
    "yeslab_484", "SCREEN_TRAIN2020_2024_STRUCTURAL_COVERAGE_V1", "PASS",
    ("COVERAGE_GATE_PASS",), 1.0018288222384784, 1.0,
)
SIX = ("yeslab_372", "yeslab_375", "yeslab_384", "yeslab_389", "yeslab_390", "yeslab_406")


def test_blocking_coverage_failure_on_an_admitted_factor_is_caught() -> None:
    audit = M.audit_admission(("yeslab_372", "yeslab_484"), (FAIL_372, PASS_484))
    assert not audit.passed
    assert any("yeslab_372" in f and "COVERAGE_GATE_FAIL" in f for f in audit.failures)
    assert any("median=0.4636" in f and "min=0.0288" in f for f in audit.failures)
    assert not any("yeslab_484" in f for f in audit.failures)


def test_clean_set_passes() -> None:
    audit = M.audit_admission(("yeslab_484",), (PASS_484,))
    assert audit.passed
    assert audit.failures == ()
    assert audit.blocking_failures == ()


def test_undocumented_conflict_is_called_out_separately() -> None:
    """The incident: blocking failures existed and the contract's conflicts was empty."""
    audit = M.audit_admission(("yeslab_372",), (FAIL_372,), declared_conflicts=())
    assert not audit.passed
    assert any("declares no conflicts" in f for f in audit.failures)


def test_declaring_a_conflict_is_not_the_same_as_an_override() -> None:
    """Recording the conflict removes the 'undocumented' failure but not the block.

    Admission despite a COVERAGE_GATE_FAIL must be an explicit waiver naming the factor,
    not merely an acknowledgement that some conflict exists somewhere.
    """
    audit = M.audit_admission(
        ("yeslab_372",), (FAIL_372,), declared_conflicts=("yeslab_372 coverage",)
    )
    assert not audit.passed
    assert not any("declares no conflicts" in f for f in audit.failures)
    assert any("yeslab_372" in f for f in audit.failures)


def test_explicit_override_downgrades_to_a_note_and_passes() -> None:
    audit = M.audit_admission(("yeslab_372",), (FAIL_372,), overrides=("yeslab_372",))
    assert audit.passed
    assert audit.overridden == ("yeslab_372",)
    assert any(n.startswith("OVERRIDDEN") and "yeslab_372" in n for n in audit.notes)


def test_unused_override_is_reported_so_waivers_do_not_rot() -> None:
    audit = M.audit_admission(("yeslab_484",), (PASS_484,), overrides=("yeslab_999",))
    assert audit.passed
    assert any("match no blocking failure" in n and "yeslab_999" in n for n in audit.notes)


def test_non_blocking_scope_failure_does_not_block_admission() -> None:
    """ELIGIBILITY_FAIL in a predictive screen is a selection outcome, not a DQ block."""
    other = _record(
        "yeslab_372", "SCREEN_TRAIN2020_2024_DAILY_HAC_FDR_V1", "ELIGIBILITY_FAIL",
        ("LOW_ICIR",),
    )
    audit = M.audit_admission(("yeslab_372",), (other,))
    assert audit.passed
    assert audit.blocking_failures == ()


def test_all_six_incident_factors_are_caught_together() -> None:
    records = tuple(
        _record(fid, "SCREEN_TRAIN2020_2024_STRUCTURAL_COVERAGE_V1",
                "ELIGIBILITY_FAIL", ("COVERAGE_GATE_FAIL",))
        for fid in SIX
    )
    audit = M.audit_admission(SIX + ("yeslab_484",), records + (PASS_484,))
    assert not audit.passed
    assert {r.factor_id for r in audit.blocking_failures} == set(SIX)


def test_384_v2_record_is_also_blocking() -> None:
    audit = M.audit_admission(("yeslab_384",), (FAIL_384,))
    assert not audit.passed
    assert any("median=0.4922" in f for f in audit.failures)


def test_empty_admitted_set_is_rejected() -> None:
    with pytest.raises(ValueError):
        M.audit_admission((), ())


def test_parse_metrics_reads_the_real_blob_shape() -> None:
    blob = json.dumps({
        "decision_status": "STRUCTURAL_DQ_FAIL",
        "reason_codes": ["COVERAGE_GATE_FAIL", "LOW_MEDIAN_SAMPLE_COVERAGE"],
        "structural": {
            "coverage_gate_pass": False,
            "median_coverage_ratio": 0.46357704421254553,
            "minimum_coverage_ratio": 0.028759894459102903,
            "p05_coverage_ratio": 0.127335,
        },
    })
    codes, median, minimum = M.parse_metrics(blob)
    assert codes == ("COVERAGE_GATE_FAIL", "LOW_MEDIAN_SAMPLE_COVERAGE")
    assert median == pytest.approx(0.46357704421254553)
    assert minimum == pytest.approx(0.028759894459102903)


def test_parse_metrics_tolerates_doubly_encoded_and_missing_fields() -> None:
    doubled = json.dumps({"reason_codes": json.dumps(["A", "B"]),
                          "structural": json.dumps({"median_coverage_ratio": 0.5})})
    codes, median, minimum = M.parse_metrics(doubled)
    assert codes == ("A", "B")
    assert median == pytest.approx(0.5)
    assert minimum is None
    assert M.parse_metrics("not json") == ((), None, None)
    assert M.parse_metrics({"reason_codes": "SOLO"}) == (("SOLO",), None, None)
    assert M.parse_metrics(None) == ((), None, None)


def test_coverage_at_or_above_the_gate_threshold_is_noted_not_silenced() -> None:
    """A blocking status with healthy coverage means another criterion fired; surface it."""
    odd = _record("yeslab_999", "SCREEN_X_STRUCTURAL_COVERAGE_V1", "ELIGIBILITY_FAIL",
                  ("MISSING_MONTH",), 0.9978, 0.99)
    audit = M.audit_admission(("yeslab_999",), (odd,))
    assert not audit.passed
    assert any("despite median coverage" in n for n in audit.notes)
