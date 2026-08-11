from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from .field_registry import FieldRegistry


ROOT = Path(__file__).resolve().parents[1]
DB_ROOT = ROOT / "数据库"


class DolphinDBDataAdapter:
    """Fetch registry-approved fields from the existing DolphinDB data layer."""

    def __init__(self, session, registry: FieldRegistry | None = None):
        self.session = session
        self.registry = registry or FieldRegistry.from_project()
        if str(DB_ROOT) not in sys.path:
            sys.path.insert(0, str(DB_ROOT))
        from get_data import DataFetcher

        self.fetcher = DataFetcher(s=session)

    def fetch_wide_fields(self, fields: list[str], start_time: str, end_time: str) -> dict[str, pd.DataFrame]:
        groups: dict[str, list[str]] = {
            "stock_factor_daily": [],
            "financial_metrics_table": [],
            "balance_sheet_table": [],
            "cashflow_table": [],
            "company_income_table": [],
            "factor_metadata": [],
            "inferred_from_calculators": [],
        }
        for field in fields:
            spec = self.registry.get(field)
            if not spec:
                raise ValueError(f"Unknown field: {field}")
            table = spec.table
            if table == "existing_factor":
                table = "factor_metadata"
            groups.setdefault(table, []).append(spec.name)

        result: dict[str, pd.DataFrame] = {}
        self._merge_result(result, groups["stock_factor_daily"], self.fetcher.get_stock_daily_matrix, start_time, end_time)
        self._merge_result(result, groups["inferred_from_calculators"], self.fetcher.get_stock_daily_matrix, start_time, end_time)
        self._merge_result(result, groups["financial_metrics_table"], self.fetcher.get_financial_metrics_matrix, start_time, end_time)
        self._merge_result(result, groups["balance_sheet_table"], self.fetcher.get_balance_sheet_matrix, start_time, end_time)
        self._merge_result(result, groups["cashflow_table"], self.fetcher.get_cashflow_matrix, start_time, end_time)
        self._merge_result(result, groups["company_income_table"], self.fetcher.get_income_matrix, start_time, end_time)
        self._merge_result(result, groups["factor_metadata"], self.fetcher.get_factor_matrix_new, start_time, end_time)
        return result

    @staticmethod
    def _merge_result(target, fields, fetcher, start_time, end_time):
        if not fields:
            return
        data = fetcher(factor_names=fields, start_time=start_time, end_time=end_time)
        data = data.T
        for field in fields:
            target[field] = data.xs(field, level=0)
