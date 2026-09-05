"""Cross-check an admitted factor set against its own screening evidence.

Why this exists: the frozen set of 17 was admitted while the project's own
``factor_quality_snapshot`` recorded ``ELIGIBILITY_FAIL`` / ``COVERAGE_GATE_FAIL`` for
six of its members (``yeslab_372`` median coverage 0.4636 and minimum 0.0288,
``yeslab_384`` 0.4922 / 0.1609, plus 389/390/406), against a dense control
(``yeslab_484``) at 1.0018 / 1.0000 and a C6.2 gate threshold of 0.70. The gate fired
and was recorded, but nothing propagated it: ``factor_quality_grade`` scores only
train_predictive / oos_validation / regime_robustness / uniqueness, the admission
decision sheet has no coverage column, and the frozen contract's ``conflicts`` array is
empty. This script closes that link so a blocking screening failure cannot be admitted
silently again.

It is deliberately a verifier rather than a schema change: it reads the snapshots that
already exist, so it can gate future admissions without migrating ``factor_quality_grade``
or re-scoring anything already decided.

Usage:
  verify_admission_consistency.py                      # the frozen v1 contract
  verify_admission_consistency.py --contract <path> --allow-override-reason <text>
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "artifacts" / "diagnostics" / "factor_set_frozen_v1.json"
)
# Scopes whose ELIGIBILITY_FAIL blocks admission. Structural coverage is the one that
# was recorded and ignored; the pattern list keeps the rule declarative so a new
# blocking screen can be added without touching the audit logic.
BLOCKING_SCOPE_SUBSTRINGS: tuple[str, ...] = ("STRUCTURAL_COVERAGE",)
COVERAGE_GATE_THRESHOLD = 0.70  # C plan C6.2: train scope coverage < 70% fails


@dataclass(frozen=True)
class ScreeningRecord:
    factor_id: str
    scope_id: str
    eligibility_status: str
    reason_codes: tuple[str, ...]
    median_coverage: float | None
    minimum_coverage: float | None


@dataclass(frozen=True)
class AdmissionAudit:
    admitted: tuple[str, ...]
    blocking_failures: tuple[ScreeningRecord, ...]
    declared_conflicts: tuple[str, ...]
    overridden: tuple[str, ...]
    passed: bool
    failures: tuple[str, ...]
    notes: tuple[str, ...]


def _is_blocking(scope_id: str) -> bool:
    return any(token in scope_id for token in BLOCKING_SCOPE_SUBSTRINGS)


def audit_admission(
    admitted: tuple[str, ...],
    records: tuple[ScreeningRecord, ...],
    *,
    declared_conflicts: tuple[str, ...] = (),
    overrides: tuple[str, ...] = (),
) -> AdmissionAudit:
    """Pure verdict: did anything blocking get admitted without a recorded override?

    ``overrides`` names factor ids that a human explicitly waved through; an override
    downgrades a blocking failure to a note but never hides it, and an override that
    matches nothing is itself reported so stale waivers do not accumulate.
    """
    if not admitted:
        raise ValueError("admitted factor set is empty")

    admitted_set = set(admitted)
    blocking = tuple(
        record
        for record in records
        if record.factor_id in admitted_set
        and _is_blocking(record.scope_id)
        and record.eligibility_status != "PASS"
    )

    overridden_set = set(overrides)
    blocked_ids = sorted({record.factor_id for record in blocking})
    unresolved = [fid for fid in blocked_ids if fid not in overridden_set]
    waived = [fid for fid in blocked_ids if fid in overridden_set]
    unused_overrides = sorted(overridden_set - set(blocked_ids))

    failures: list[str] = []
    notes: list[str] = []
    for record in sorted(blocking, key=lambda r: (r.factor_id, r.scope_id)):
        coverage = (
            f" median={record.median_coverage:.4f} min={record.minimum_coverage:.4f}"
            if record.median_coverage is not None and record.minimum_coverage is not None
            else ""
        )
        line = (
            f"{record.factor_id}: {record.eligibility_status} in {record.scope_id}"
            f"{coverage} reasons={list(record.reason_codes)}"
        )
        if record.factor_id in overridden_set:
            notes.append(f"OVERRIDDEN {line}")
        else:
            failures.append(line)

    if unresolved and not declared_conflicts:
        failures.append(
            f"{len(unresolved)} admitted factor(s) carry a blocking screening failure "
            f"({', '.join(unresolved)}) while the contract declares no conflicts; the "
            "contract's conflicts array must record this or the factors must be removed"
        )
    if unused_overrides:
        notes.append(f"override(s) match no blocking failure: {unused_overrides}")
    for record in blocking:
        if (
            record.median_coverage is not None
            and record.median_coverage >= COVERAGE_GATE_THRESHOLD
        ):
            notes.append(
                f"{record.factor_id}: blocking status despite median coverage "
                f"{record.median_coverage:.4f} >= {COVERAGE_GATE_THRESHOLD}; "
                "inspect the scope's other criteria"
            )

    return AdmissionAudit(
        admitted=tuple(admitted),
        blocking_failures=blocking,
        declared_conflicts=tuple(declared_conflicts),
        overridden=tuple(waived),
        passed=not failures,
        failures=tuple(failures),
        notes=tuple(notes),
    )


def parse_metrics(payload: Any) -> tuple[tuple[str, ...], float | None, float | None]:
    """Pull (reason_codes, median_coverage, minimum_coverage) out of a metrics_json blob."""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return (), None, None
    if not isinstance(payload, dict):
        return (), None, None
    codes = payload.get("reason_codes") or []
    if isinstance(codes, str):
        try:
            codes = json.loads(codes)
        except json.JSONDecodeError:
            codes = [codes]
    structural = payload.get("structural")
    if isinstance(structural, str):
        try:
            structural = json.loads(structural)
        except json.JSONDecodeError:
            structural = None
    median = minimum = None
    if isinstance(structural, dict):
        raw_median = structural.get("median_coverage_ratio")
        raw_min = structural.get("minimum_coverage_ratio")
        median = float(raw_median) if isinstance(raw_median, (int, float)) else None
        minimum = float(raw_min) if isinstance(raw_min, (int, float)) else None
    return tuple(str(code) for code in codes), median, minimum


def collect_records(
    holder: dict[str, Any], factor_ids: tuple[str, ...]
) -> tuple[ScreeningRecord, ...]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pandas as pd
    from walk_forward_common import TARGET_DATABASE, query

    ids = ",".join(f"`{fid}" for fid in factor_ids)
    table = f"loadTable('{TARGET_DATABASE}','factor_quality_snapshot')"
    ok, frame = query(
        holder,
        f"select factor_id, scope_id, eligibility_status, metrics_json from {table} "
        f"where factor_id in [{ids}]",
    )
    if not ok or not isinstance(frame, pd.DataFrame) or frame.empty:
        return ()
    records: list[ScreeningRecord] = []
    for factor_id, scope_id, status, metrics in zip(
        frame["factor_id"], frame["scope_id"], frame["eligibility_status"],
        frame["metrics_json"], strict=True,
    ):
        codes, median, minimum = parse_metrics(metrics)
        records.append(
            ScreeningRecord(
                factor_id=str(factor_id),
                scope_id=str(scope_id),
                eligibility_status=str(status),
                reason_codes=codes,
                median_coverage=median,
                minimum_coverage=minimum,
            )
        )
    return tuple(records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--override", action="append", default=[],
        help="factor_id with an explicit human waiver for a blocking failure (repeatable)",
    )
    args = parser.parse_args(argv)

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    admitted = tuple(str(fid) for fid in contract["factors"])
    declared = tuple(str(item) for item in contract.get("conflicts") or [])
    print(f"contract: {args.contract}")
    print(f"admitted factors: {len(admitted)}; declared conflicts: {len(declared)}")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from walk_forward_common import new_holder

    records = collect_records(new_holder(), admitted)
    print(f"screening records found: {len(records)}")
    audit = audit_admission(
        admitted, records, declared_conflicts=declared, overrides=tuple(args.override)
    )
    for note in audit.notes:
        print(f"  NOTE {note}")
    for failure in audit.failures:
        print(f"  FAIL {failure}")
    print(
        f"blocking failures: {len(audit.blocking_failures)} across "
        f"{len({r.factor_id for r in audit.blocking_failures})} admitted factor(s); "
        f"overridden: {list(audit.overridden) or 'none'}"
    )
    print("ADMISSION CONSISTENCY:", "PASS" if audit.passed else "FAIL")
    return 0 if audit.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
