"""Audit formula semantics and code-scope evidence for SH/SZ factor rematerialization.

The audit is read-only.  It separates two questions that are easy to conflate:
whether a calculator is technically safe to run on an explicitly scoped
cross-section, and whether the historical stored values came from that same
cross-section.  The latter is expected to fail for the three scope-transition
candidates because their old source panels contain BJ securities.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
import os
import sys
from collections.abc import Iterable, Sequence
from datetime import date
from pathlib import Path

import pandas as pd

import audit_formula_replay_consistency as replay

LOCKED_TEST_START = date(2026, 1, 1)
DEFAULT_FACTOR_IDS = ("yeslab_376", "yeslab_396", "yeslab_442")
REPO_ROOT = Path(__file__).resolve().parents[2]
CALCULATOR_ROOT = REPO_ROOT / "数据库"


def summarize_code_scope(codes: Iterable[object]) -> dict[str, object]:
    """Summarize unique exchange suffixes and require SH/SZ-only membership."""

    unique = sorted({str(code) for code in codes if pd.notna(code)})
    frame = pd.Series(unique, dtype="string")
    bj = frame.str.endswith(".BJ")
    sh = frame.str.endswith(".SH")
    sz = frame.str.endswith(".SZ")
    other = ~(bj | sh | sz)
    return {
        "unique_code_count": len(unique),
        "bj_code_count": int(bj.sum()),
        "sh_code_count": int(sh.sum()),
        "sz_code_count": int(sz.sum()),
        "other_code_count": int(other.sum()),
        "scope_status": "PASS" if len(unique) > 0 and not bool((bj | other).any()) else "FAIL",
    }


class _FormulaVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.rolling_windows: list[int] = []
        self.shift_periods: list[int] = []

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "rolling" and node.args:
                value = node.args[0]
                if isinstance(value, ast.Constant) and isinstance(value.value, int):
                    self.rolling_windows.append(int(value.value))
            if node.func.attr == "shift" and node.args:
                value = node.args[0]
                if isinstance(value, ast.Constant) and isinstance(value.value, int):
                    self.shift_periods.append(int(value.value))
        self.generic_visit(node)


def _calculator_class(definition: replay.ReplayDefinition) -> type:
    if str(CALCULATOR_ROOT) not in sys.path:
        sys.path.insert(0, str(CALCULATOR_ROOT))
    module = importlib.import_module(f"factor_calculators.{definition.calculator_module}")
    return getattr(module, definition.calculator_class)


def inspect_factor_semantics(factor_ids: Sequence[str]) -> list[dict[str, object]]:
    """Extract calculator inputs and look-ahead properties without running a DB query."""

    rows: list[dict[str, object]] = []
    for factor_id in factor_ids:
        definition = replay.REPLAY_DEFINITIONS[str(factor_id)]
        calculator_class = _calculator_class(definition)
        calculator = calculator_class(None, definition.lookback_days)
        source = inspect.getsource(calculator_class)
        tree = ast.parse(source)
        visitor = _FormulaVisitor()
        visitor.visit(tree)
        # Cross-sectional operators appear as either explicit z-scores or
        # percentile ranks in the GP calculators.  Both require the full
        # SH/SZ universe to be supplied in one batch.
        source_has_cross_section = (
            (".mean(axis=1)" in source and ".std(axis=1)" in source)
            or ".rank(axis=1" in source
        )
        required_market = sorted(str(value) for value in calculator.required_factors)
        required_non_market = sorted(
            str(value)
            for name in (
                "required_other_factors",
                "required_financial_factors",
                "required_balance_sheet_factors",
                "required_cashflow_factors",
                "required_income_factors",
            )
            for value in getattr(calculator, name, ())
        )
        rows.append(
            {
                "factor_id": str(factor_id),
                "factor_name": definition.factor_name,
                "store_factor": definition.store_factor,
                "calculator_module": definition.calculator_module,
                "calculator_class": definition.calculator_class,
                "required_market_fields": required_market,
                "required_non_market_fields": required_non_market,
                "lookback_days": int(definition.lookback_days),
                "replay_lookback_days": int(definition.effective_replay_lookback_days),
                "rolling_windows": sorted(set(visitor.rolling_windows)),
                "shift_periods": sorted(set(visitor.shift_periods)),
                "uses_cross_sectional_standardization": bool(source_has_cross_section),
                "uses_forward_lookahead": any(value < 0 for value in visitor.shift_periods),
                "scope_contract": "SH_SZ_ONLY",
            }
        )
    return rows


def build_scope_review(
    *,
    factor_semantics: list[dict[str, object]],
    scope_evidence: dict[str, object],
) -> dict[str, object]:
    """Build an approval packet while keeping the manual gate closed."""

    v2_scope = scope_evidence.get("v2_universe", {})
    stored = scope_evidence.get("stored_sources", scope_evidence.get("stored_source", {}))
    stored_rows = stored.values() if isinstance(stored, dict) else (stored,)
    formula_static_pass = all(
        row.get("required_market_fields")
        and row.get("required_non_market_fields") == []
        and row.get("uses_cross_sectional_standardization") is True
        and row.get("uses_forward_lookahead") is False
        for row in factor_semantics
    )
    source_scope_mismatch = any(
        isinstance(row, dict) and row.get("scope_status") == "FAIL" for row in stored_rows
    )
    return {
        "scope": "TUSHARE_SHSZ_FORMULA_SCOPE_TRANSITION_AUDIT_V1",
        "market_scope": ["SH", "SZ"],
        "excluded_exchanges": ["BJ"],
        "formula_static_audit": "PASS" if formula_static_pass else "FAIL",
        "v2_scope_audit": v2_scope.get("scope_status", "MISSING")
        if isinstance(v2_scope, dict)
        else "MISSING",
        "stored_source_scope_mismatch": bool(source_scope_mismatch),
        "formula_scope_gate": "PENDING_SCOPE_TRANSITION_REVIEW",
        "review_status": "READY_FOR_MANUAL_SCOPE_APPROVAL",
        "reason_codes": [
            "CROSS_SECTIONAL_FORMULA_REQUIRES_EXPLICIT_SH_SZ_SCOPE",
            "STORED_SOURCE_SCOPE_CONTAINS_BJ",
            "NEW_VERSION_MUST_RETAIN_SH_SZ_SCOPE_AND_PIT_CONTRACT",
        ],
        "database_writes": False,
        "locked_test_read": False,
    }


def _connect() -> object:
    import dolphindb as ddb  # type: ignore[import-untyped]

    client = ddb.session()
    client.connect(
        os.getenv("QRV2_DDB_HOST", "10.176.55.211"),
        int(os.getenv("QRV2_DDB_PORT", "8848")),
        os.getenv("QRV2_DDB_USERNAME", "xiahongjie"),
        os.getenv("QRV2_DDB_PASSWORD", "123456"),
    )
    return client


def _query_codes(client: object, table: str, factor_names: Sequence[str], start: date, end: date) -> pd.Series:
    names = ", ".join(f"'{str(name).replace(chr(39), chr(39) + chr(39))}'" for name in factor_names)
    if table == "universe_daily_latest":
        query = (
            "select distinct ts_code from loadTable('dfs://QuantResearchV2Test','universe_daily_latest') "
            "where is_member == true "
            f"and signal_date between date('{start:%Y.%m.%d}') and date('{end:%Y.%m.%d}')"
        )
    else:
        query = (
            f"select distinct ts_code from loadTable('dfs://YesLab','{table}') "
            f"where factor_name in [{names}] "
            f"and trade_date between date('{start:%Y.%m.%d}') and date('{end:%Y.%m.%d}')"
        )
    result = client.run(query)
    if not isinstance(result, pd.DataFrame) or "ts_code" not in result.columns:
        raise RuntimeError(f"scope code query did not return ts_code: {table}")
    return result["ts_code"]


def run_audit(args: argparse.Namespace) -> dict[str, object]:
    definitions = [replay.REPLAY_DEFINITIONS[factor_id] for factor_id in args.factor_id]
    factor_semantics = inspect_factor_semantics(args.factor_id)
    client = _connect()
    v2_codes = _query_codes(client, "universe_daily_latest", (), args.start, args.end)
    source_groups: dict[str, list[str]] = {}
    for definition in definitions:
        source_groups.setdefault(definition.store_factor, []).append(definition.factor_name)
    stored_evidence: dict[str, dict[str, object]] = {}
    for definition in definitions:
        codes = _query_codes(
            client, definition.store_factor, [definition.factor_name], args.start, args.end
        )
        stored_evidence[definition.factor_id] = summarize_code_scope(codes)
    report = build_scope_review(
        factor_semantics=factor_semantics,
        scope_evidence={
            "v2_universe": summarize_code_scope(v2_codes),
            "stored_sources": stored_evidence,
        },
    )
    report.update(
        {
            "factor_ids": list(args.factor_id),
            "start": args.start.isoformat(),
            "end": args.end.isoformat(),
            "factor_semantics": factor_semantics,
            "scope_evidence": {
                "v2_universe": summarize_code_scope(v2_codes),
                "stored_sources": stored_evidence,
            },
            "related_artifacts": [
                "tushare_shsz_rematerialized_quality_16_v1.json",
                "tushare_shsz_candidate_extension_similarity_v1.json",
            ],
        }
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor-id", action="append")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2025, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.factor_id = tuple(dict.fromkeys(args.factor_id or DEFAULT_FACTOR_IDS))
    unknown = sorted(set(args.factor_id).difference(replay.REPLAY_DEFINITIONS))
    if unknown:
        parser.error(f"unsupported factor ids: {unknown}")
    if args.start > args.end or args.end >= LOCKED_TEST_START:
        parser.error("audit range must be before 2026 and start <= end")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "review_status": result["review_status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
