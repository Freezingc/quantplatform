from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import re
from pathlib import Path
from typing import Iterable

from .models import FieldSpec


ROOT = Path(__file__).resolve().parents[1]
CALCULATOR_DIR = ROOT / "数据库" / "factor_calculators"
DAILY_TASK_DIRS = (
    ROOT / "数据库" / "daily_task",
    ROOT / "数据库" / "daily_task_new" / "update_mission",
)


DEFAULT_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("open", "stock_factor_daily", "market", "daily", "开盘价", ("open_hfq", "开盘价")),
    FieldSpec("close", "stock_factor_daily", "market", "daily", "收盘价", ("close_hfq", "收盘价")),
    FieldSpec("high", "stock_factor_daily", "market", "daily", "最高价", ("high_hfq", "最高价")),
    FieldSpec("low", "stock_factor_daily", "market", "daily", "最低价", ("low_hfq", "最低价")),
    FieldSpec("vol", "stock_factor_daily", "market", "daily", "成交量", ("volume", "成交量")),
    FieldSpec("amount", "stock_factor_daily", "market", "daily", "成交额", ("成交额",)),
    FieldSpec("pct_chg", "stock_factor_daily", "market", "daily", "涨跌幅", ("returns", "return", "收益率")),
    FieldSpec("turnover_rate", "stock_factor_daily", "market", "daily", "换手率", ("turnover", "换手率")),
    FieldSpec("total_mv", "stock_factor_daily", "market_cap", "daily", "总市值", ("market_cap", "市值")),
    FieldSpec("circ_mv", "stock_factor_daily", "market_cap", "daily", "流通市值", ("float_market_cap", "流通市值")),
    FieldSpec("pe", "stock_factor_daily", "valuation", "daily", "市盈率", ("PE", "市盈率")),
    FieldSpec("pb", "stock_factor_daily", "valuation", "daily", "市净率", ("PB", "市净率")),
    FieldSpec("ps", "stock_factor_daily", "valuation", "daily", "市销率", ("PS", "市销率")),
    FieldSpec("dv_ratio", "stock_factor_daily", "valuation", "daily", "股息率", ("dividend_yield", "股息率")),
    FieldSpec("roe", "financial_metrics_table", "financial", "quarterly", "ROE", ("净资产收益率",)),
    FieldSpec("roa", "financial_metrics_table", "financial", "quarterly", "ROA", ("总资产收益率",)),
    FieldSpec("grossprofit_margin", "financial_metrics_table", "financial", "quarterly", "毛利率", ("gp_margin", "毛利率")),
    FieldSpec("netprofit_margin", "financial_metrics_table", "financial", "quarterly", "净利率", ("np_margin", "净利率")),
    FieldSpec("debt_to_assets", "balance_sheet", "financial", "quarterly", "资产负债率", ("debt_ratio", "资产负债率")),
    FieldSpec("industry", "code_index", "industry", "daily", "行业分类", ("行业", "申万行业"), False),
)


class FieldRegistry:
    """Registry of fields that factor generation is allowed to use.

    The registry starts with a curated production-safe field list, can enrich it
    from existing calculator files, and can optionally refresh from DolphinDB
    when a live session is available.
    """

    def __init__(self, fields: Iterable[FieldSpec] | None = None):
        self._fields: dict[str, FieldSpec] = {}
        self._aliases: dict[str, str] = {}
        for spec in fields or DEFAULT_FIELDS:
            self.add(spec)

    def add(self, spec: FieldSpec) -> None:
        self._fields[spec.name] = spec
        self._aliases[self._normalize(spec.name)] = spec.name
        for alias in spec.aliases:
            self._aliases[self._normalize(alias)] = spec.name

    def get(self, name: str) -> FieldSpec | None:
        canonical = self.resolve_name(name)
        return self._fields.get(canonical) if canonical else None

    def resolve_name(self, name: str) -> str | None:
        if name in self._fields:
            return name
        return self._aliases.get(self._normalize(name))

    def list_fields(self, expression_enabled: bool | None = None) -> list[FieldSpec]:
        specs = sorted(self._fields.values(), key=lambda x: (x.category, x.name))
        if expression_enabled is None:
            return specs
        return [spec for spec in specs if spec.expression_enabled == expression_enabled]

    def as_dicts(self) -> list[dict]:
        return [spec.to_dict() for spec in self.list_fields()]

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.as_dicts(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "FieldRegistry":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(FieldSpec(**item) for item in data)

    @classmethod
    def from_project(cls, calculator_dir: str | Path = CALCULATOR_DIR) -> "FieldRegistry":
        registry = cls()
        registry.enrich_from_daily_tasks()
        registry.enrich_from_calculators(calculator_dir)
        return registry

    def enrich_from_daily_tasks(self, task_dirs: Iterable[str | Path] = DAILY_TASK_DIRS) -> None:
        """Enrich the registry from daily update scripts.

        The daily task scripts are the closest local source of truth for fields
        that are actually written to DolphinDB. We statically parse each script
        for the target `TableAppender(... tableName=...)` and Tushare
        `fields=[...]` lists, then add those fields under the detected table.
        """
        for task_dir in task_dirs:
            path = Path(task_dir)
            if not path.exists():
                continue
            for script in path.glob("update_*.py"):
                table = _extract_table_name(script)
                fields = _extract_tushare_fields(script)
                if not table or not fields:
                    continue
                for field in fields:
                    if field in {"ts_code", "trade_date", "ann_date", "end_date"}:
                        continue
                    if field not in self._fields:
                        self.add(FieldSpec(
                            name=field,
                            table=table,
                            category=_category_for_table(table),
                            frequency=_infer_frequency(table),
                            description="",
                            aliases=(),
                            expression_enabled=_is_expression_field(table, field),
                        ))

    def enrich_from_calculators(self, calculator_dir: str | Path = CALCULATOR_DIR) -> None:
        calculator_dir = Path(calculator_dir)
        if not calculator_dir.exists():
            return
        for path in calculator_dir.glob("*.py"):
            if path.name in {"__init__.py", "base.py"}:
                continue
            for field in _extract_required_fields(path):
                if field not in self._fields:
                    self.add(FieldSpec(field, "inferred_from_calculators", "existing_factor_input", "daily"))

    def enrich_from_dolphindb(self, session) -> None:
        """Best-effort schema enrichment from DolphinDB.

        This method is deliberately optional; tests and local development do not
        require DolphinDB connectivity.
        """
        table_map = {
            "stock_factor_daily": "market",
            "financial_metrics_table": "financial",
            "balance_sheet_table": "financial",
            "cashflow_table": "financial",
            "company_income_table": "financial",
            "factor_metadata": "metadata",
        }
        for table, category in table_map.items():
            try:
                schema = session.run(f"schema(loadTable('dfs://YesLab', '{table}'))")
                names = schema["colDefs"].name.tolist()
            except Exception:
                continue
            for name in names:
                if name in {"ts_code", "trade_date", "ann_date", "end_date"}:
                    continue
                if name not in self._fields:
                    self.add(FieldSpec(name, table, category, _infer_frequency(table)))

        try:
            factors = session.run("select factor_name from loadTable('dfs://YesLab', 'factor_metadata')")
            for name in factors.get("factor_name", []):
                if name not in self._fields:
                    self.add(FieldSpec(str(name), "factor_metadata", "existing_factor", "daily"))
        except Exception:
            pass

    def map_terms(self, terms: Iterable[str]) -> tuple[dict[str, str], list[str]]:
        mapped: dict[str, str] = {}
        missing: list[str] = []
        for term in terms:
            resolved = self.resolve_name(term)
            if resolved:
                mapped[term] = resolved
            else:
                missing.append(term)
        return mapped, sorted(set(missing))

    @staticmethod
    def _normalize(value: str) -> str:
        return re.sub(r"[\s_\-()（）]+", "", value.strip().lower())


def _infer_frequency(table: str) -> str:
    return "quarterly" if "financial" in table or "balance" in table or "cashflow" in table or "income" in table else "daily"


def _category_for_table(table: str) -> str:
    if table in {"stock_daily_data", "stock_factor_daily", "index_daily_data"}:
        return "market"
    if table in {"balance_sheet", "cashflow_table", "company_income_table", "financial_metrics_table"}:
        return "financial"
    if table in {"stock_info_table", "index_info_table", "index_component_table", "sw_ind_code", "code_index"}:
        return "metadata"
    return "database"


def _is_expression_field(table: str, field: str) -> bool:
    if field.endswith("_name") or field in {"name", "fullname", "exchange", "list_status", "is_hs"}:
        return False
    return table not in {"stock_info_table", "index_info_table", "trade_calendar", "stock_used_name"}


def _extract_table_name(path: Path) -> str | None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = _call_name(node.func)
            if func_name.endswith("TableAppender"):
                for kw in node.keywords:
                    if kw.arg == "tableName" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                        return kw.value.value
    return None


def _extract_tushare_fields(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return set()
    fields: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "fields":
                    fields.update(_literal_string_list(kw.value))
    return fields


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _extract_required_fields(path: Path) -> set[str]:
    fields: set[str] = set()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return fields
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr.startswith("required_"):
                    fields.update(_literal_string_list(node.value))
    return fields


def _literal_string_list(node: ast.AST) -> set[str]:
    if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return set()
    values: set[str] = set()
    for item in node.elts:
        if isinstance(item, ast.Constant) and isinstance(item.value, str):
            values.add(item.value)
    return values
