"""Explicit read-only adapter for the existing YesLab source database."""

from __future__ import annotations

import hashlib
import re
from datetime import date
from typing import Protocol

import pandas as pd

from research_v2.contracts.source import SourceBatch


class SourceSchemaDrift(ValueError):
    """Raised when an approved YesLab source contract changes."""


class ReadOnlyQuerySession(Protocol):
    """The subset of YesLabReadOnlySession required by this adapter."""

    def query(self, script: str) -> object: ...


_MARKET_COLUMNS = (
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "vol",
    "amount",
    "adj_factor",
)
_CALENDAR_COLUMNS = ("exchange", "cal_date", "is_open", "pretrade_date")
_SECURITY_COLUMNS = (
    "ts_code",
    "exchange",
    "market",
    "list_date",
    "delist_date",
    "name",
    "list_status",
)
_NAME_HISTORY_COLUMNS = ("ts_code", "name", "start_date", "end_date", "ann_date", "change_reason")
_SUSPENSION_COLUMNS = ("ts_code", "trade_date", "suspend_timing", "suspend_type")
_EXCHANGE_PATTERN = re.compile(r"^[A-Z]{2,8}$")
_TS_CODE_PATTERN = re.compile(r"^[0-9]{6}\.(?:SH|SZ|BJ)$")


def _date_literal(value: date) -> str:
    return value.strftime("%Y.%m.%d")


def _schema_fingerprint(schema: pd.DataFrame) -> str:
    required = [column for column in ("name", "typeString") if column in schema.columns]
    canonical = schema.loc[:, required].sort_values("name").to_csv(index=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class YesLabAdapter:
    """Load approved source fields through the read-only session boundary."""

    def __init__(self, session: ReadOnlyQuerySession) -> None:
        self._session = session
        self.last_query = ""

    def fetch_market(self, start_date: date, end_date: date) -> SourceBatch:
        if start_date > end_date:
            raise ValueError("start_date cannot be after end_date")
        schema = self._query_dataframe(
            "schema(loadTable('dfs://YesLab', 'stock_factor_daily')).colDefs"
        )
        source_columns = set(schema["name"])
        missing_columns = sorted(set(_MARKET_COLUMNS).difference(source_columns))
        if missing_columns:
            raise SourceSchemaDrift(f"stock_factor_daily is missing columns: {missing_columns}")

        selected_columns = ", ".join(_MARKET_COLUMNS)
        self.last_query = (
            f"select {selected_columns} from loadTable('dfs://YesLab', 'stock_factor_daily') "
            f"where trade_date between date('{_date_literal(start_date)}') "
            f"and date('{_date_literal(end_date)}') order by trade_date, ts_code"
        )
        frame = self._query_dataframe(self.last_query)
        if frame.empty:
            max_observation_date = end_date
        else:
            max_observation_date = pd.Timestamp(frame["trade_date"].max()).date()
        fingerprint = _schema_fingerprint(schema)
        watermark_payload = (
            f"YesLab|stock_factor_daily|{max_observation_date.isoformat()}|"
            f"{len(frame)}|{fingerprint}"
        )
        return SourceBatch(
            frame=frame,
            source_name="YesLab",
            table_name="stock_factor_daily",
            watermark_id=hashlib.sha256(watermark_payload.encode("utf-8")).hexdigest(),
            max_observation_date=max_observation_date,
            source_row_count=len(frame),
            schema_fingerprint=fingerprint,
            extracted_at=pd.Timestamp.now(tz="UTC").to_pydatetime(),
        )

    def fetch_market_observations(
        self,
        ts_codes: set[str],
        start_date: date,
        end_date: date,
    ) -> SourceBatch:
        """Fetch a narrow market-history extract for audited profile reconstruction."""
        if start_date > end_date:
            raise ValueError("start_date cannot be after end_date")
        if not ts_codes or not all(_TS_CODE_PATTERN.fullmatch(code) for code in ts_codes):
            raise ValueError("ts_codes must contain valid mainland security codes")
        symbols = ", ".join(f"`{code}" for code in sorted(ts_codes))
        return self._fetch_explicit_table(
            table_name="stock_factor_daily",
            columns=("ts_code", "trade_date"),
            query=(
                "select ts_code, trade_date from "
                "loadTable('dfs://YesLab', 'stock_factor_daily') "
                f"where ts_code in [{symbols}] and trade_date between "
                f"date('{_date_literal(start_date)}') and date('{_date_literal(end_date)}') "
                "order by ts_code, trade_date"
            ),
            max_observation_date=end_date,
        )

    def fetch_trading_calendar(
        self,
        exchange: str,
        start_date: date,
        end_date: date,
    ) -> SourceBatch:
        """Fetch only source-confirmed open days for one exchange and date range."""
        if not _EXCHANGE_PATTERN.fullmatch(exchange):
            raise ValueError("exchange must be an uppercase source symbol")
        if start_date > end_date:
            raise ValueError("start_date cannot be after end_date")
        schema = self._query_dataframe(
            "schema(loadTable('dfs://YesLab', 'trade_calendar')).colDefs"
        )
        source_columns = set(schema["name"])
        missing_columns = sorted(set(_CALENDAR_COLUMNS).difference(source_columns))
        if missing_columns:
            raise SourceSchemaDrift(f"trade_calendar is missing columns: {missing_columns}")
        selected_columns = ", ".join(_CALENDAR_COLUMNS)
        self.last_query = (
            f"select {selected_columns} from loadTable('dfs://YesLab', 'trade_calendar') "
            f"where exchange = `{exchange}, is_open = 1 and cal_date between "
            f"date('{_date_literal(start_date)}') and date('{_date_literal(end_date)}') "
            "order by cal_date"
        )
        frame = self._query_dataframe(self.last_query)
        max_observation_date = (
            pd.Timestamp(frame["cal_date"].max()).date() if not frame.empty else end_date
        )
        fingerprint = _schema_fingerprint(schema)
        watermark_payload = (
            f"YesLab|trade_calendar|{exchange}|{max_observation_date.isoformat()}|"
            f"{len(frame)}|{fingerprint}"
        )
        return SourceBatch(
            frame=frame,
            source_name="YesLab",
            table_name="trade_calendar",
            watermark_id=hashlib.sha256(watermark_payload.encode("utf-8")).hexdigest(),
            max_observation_date=max_observation_date,
            source_row_count=len(frame),
            schema_fingerprint=fingerprint,
            extracted_at=pd.Timestamp.now(tz="UTC").to_pydatetime(),
        )

    def fetch_security_snapshot(self, as_of_date: date) -> SourceBatch:
        """Fetch the current security snapshot while binding extraction to an as-of date."""
        return self._fetch_explicit_table(
            table_name="stock_info_table",
            columns=_SECURITY_COLUMNS,
            query=(
                f"select {', '.join(_SECURITY_COLUMNS)} from "
                "loadTable('dfs://YesLab', 'stock_info_table') order by ts_code"
            ),
            max_observation_date=as_of_date,
        )

    def fetch_name_history(self, start_date: date, end_date: date) -> SourceBatch:
        """Fetch name/ST intervals intersecting the requested historical range."""
        if start_date > end_date:
            raise ValueError("start_date cannot be after end_date")
        return self._fetch_explicit_table(
            table_name="stock_used_name",
            columns=_NAME_HISTORY_COLUMNS,
            query=(
                f"select {', '.join(_NAME_HISTORY_COLUMNS)} from "
                "loadTable('dfs://YesLab', 'stock_used_name') "
                f"where start_date <= date('{_date_literal(end_date)}') and "
                f"(isNull(end_date) or end_date >= date('{_date_literal(start_date)}')) "
                "order by ts_code, start_date"
            ),
            max_observation_date=end_date,
        )

    def fetch_suspensions(self, start_date: date, end_date: date) -> SourceBatch:
        """Fetch source-recorded suspension events for the closed date range."""
        if start_date > end_date:
            raise ValueError("start_date cannot be after end_date")
        return self._fetch_explicit_table(
            table_name="stock_trading_suspension_info",
            columns=_SUSPENSION_COLUMNS,
            query=(
                f"select {', '.join(_SUSPENSION_COLUMNS)} from "
                "loadTable('dfs://YesLab', 'stock_trading_suspension_info') "
                f"where trade_date between date('{_date_literal(start_date)}') "
                f"and date('{_date_literal(end_date)}') order by trade_date, ts_code"
            ),
            max_observation_date=end_date,
        )

    def _fetch_explicit_table(
        self,
        *,
        table_name: str,
        columns: tuple[str, ...],
        query: str,
        max_observation_date: date,
    ) -> SourceBatch:
        schema = self._query_dataframe(
            f"schema(loadTable('dfs://YesLab', '{table_name}')).colDefs"
        )
        source_columns = set(schema["name"])
        missing_columns = sorted(set(columns).difference(source_columns))
        if missing_columns:
            raise SourceSchemaDrift(f"{table_name} is missing columns: {missing_columns}")
        self.last_query = query
        frame = self._query_dataframe(query)
        fingerprint = _schema_fingerprint(schema)
        watermark_payload = (
            f"YesLab|{table_name}|{max_observation_date.isoformat()}|{len(frame)}|{fingerprint}"
        )
        return SourceBatch(
            frame=frame,
            source_name="YesLab",
            table_name=table_name,
            watermark_id=hashlib.sha256(watermark_payload.encode("utf-8")).hexdigest(),
            max_observation_date=max_observation_date,
            source_row_count=len(frame),
            schema_fingerprint=fingerprint,
            extracted_at=pd.Timestamp.now(tz="UTC").to_pydatetime(),
        )

    def _query_dataframe(self, script: str) -> pd.DataFrame:
        result = self._session.query(script)
        if not isinstance(result, pd.DataFrame):
            raise TypeError("YesLab query did not return a DataFrame")
        return result.copy(deep=True)
