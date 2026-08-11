from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DAILY_TASK_DIR = ROOT / "数据库" / "daily_task"
DOC_PATH = ROOT / "agent_gateway" / "FIELD_REFERENCE.md"
SKIP_FILES = {"update_alpha191.py", "update_factor.py"}
ID_COLUMNS = {
    "ts_code",
    "trade_date",
    "ann_date",
    "end_date",
    "cal_date",
    "index_code",
    "con_code",
}


@dataclass
class TableFields:
    script: str
    table: str
    fields: list[str]


def generate_field_reference(task_dir: Path = DAILY_TASK_DIR, output_path: Path = DOC_PATH) -> Path:
    records = parse_daily_task_fields(task_dir)
    output_path.write_text(render_markdown(records), encoding="utf-8")
    return output_path


def parse_daily_task_fields(task_dir: Path = DAILY_TASK_DIR) -> list[TableFields]:
    records: list[TableFields] = []
    for script in sorted(task_dir.glob("update_*.py")):
        if script.name in SKIP_FILES:
            continue
        table = _extract_table_name(script)
        fields = sorted(_extract_fields(script))
        if not table:
            continue
        records.append(TableFields(script=script.name, table=table, fields=fields))
    return records


def render_markdown(records: list[TableFields]) -> str:
    lines = [
        "# Database Field Reference",
        "",
        "This document is generated from `数据库/daily_task/update_*.py` for agent-side factor generation.",
        "`update_alpha191.py` and `update_factor.py` are skipped because they update factor workflows rather than base database tables.",
        "",
        "Use these fields as the source of truth before generating DSL expressions or production `FactorCalculator` drafts.",
        "",
        "## Summary",
        "",
        "| Table | Update Script | Field Count |",
        "| --- | --- | ---: |",
    ]
    for record in records:
        lines.append(f"| `{record.table}` | `{record.script}` | {len(record.fields)} |")

    lines.extend(["", "## Tables", ""])
    for record in records:
        lines.extend([
            f"### `{record.table}`",
            "",
            f"- Update script: `数据库/daily_task/{record.script}`",
            f"- Field count: {len(record.fields)}",
            "",
            "| Field | Role | Agent Usage |",
            "| --- | --- | --- |",
        ])
        for field in record.fields:
            role = "identifier/date" if field in ID_COLUMNS else _field_role(field)
            usage = _agent_usage(field, role)
            lines.append(f"| `{field}` | {role} | {usage} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _extract_table_name(path: Path) -> str | None:
    tree = _parse(path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node.func).endswith("TableAppender"):
            for kw in node.keywords:
                if kw.arg == "tableName" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    return kw.value.value
    return None


def _extract_fields(path: Path) -> set[str]:
    tree = _parse(path)
    fields: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg == "fields":
                fields.update(_string_fields(kw.value))
    return fields


def _string_fields(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {part.strip() for part in node.value.split(",") if part.strip()}
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return {
            item.value
            for item in node.elts
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        }
    return set()


def _parse(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8", errors="ignore"))


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _field_role(field: str) -> str:
    if field.endswith("_name") or field in {"name", "fullname"}:
        return "metadata"
    if field in {"open", "high", "low", "close", "pre_close", "change", "pct_chg"}:
        return "price"
    if field in {"vol", "amount", "turnover_rate", "turnover_rate_f", "volume_ratio"}:
        return "volume/liquidity"
    if field in {"pe", "pe_ttm", "pb", "ps", "ps_ttm", "dv_ratio", "dv_ttm", "total_mv", "circ_mv"}:
        return "valuation/market-cap"
    if any(token in field for token in ("income", "profit", "revenue", "cashflow", "assets", "liab", "equity")):
        return "financial"
    if any(field.startswith(prefix) for prefix in ("ma_", "ema_", "macd_", "kdj_", "rsi_", "boll_", "atr_", "cci_", "dmi_")):
        return "technical"
    return "raw"


def _agent_usage(field: str, role: str) -> str:
    if role == "identifier/date":
        return "Join key; do not use as a factor signal."
    if role == "metadata":
        return "Use for filtering or labeling, not numeric DSL expressions."
    return "Allowed for candidate expressions if coverage is sufficient."


if __name__ == "__main__":
    path = generate_field_reference()
    print(path)
