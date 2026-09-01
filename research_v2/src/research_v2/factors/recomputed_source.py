"""Recomputed factor source for periods absent from stored YesLab factor tables.

Serves GP factor values recomputed from raw market inputs (the registered
calculators under ``数据库/factor_calculators``) through the same
``YesLabFactorSession`` query contract consumed by
:mod:`research_v2.factors.source_backfill`, so the existing audit and publish
path can materialize recomputed months (e.g. the 2014–2019 TRAIN years) with
identical quarantine/watermark/lineage discipline.

Safety basis: the admitted-factor replay verification
(``docs/reports/2026-09-01-admitted16-formula-replay-verification.md``) proved
which calculators reproduce stored observations at machine precision; only
verified factors may be routed through this session.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol

import pandas as pd

_SOURCE_COLUMNS = ("ts_code", "trade_date", "factor_name", "value")
_SAFE_TABLE = re.compile(r"factor_data(?:[1-9]|10|11)")
_QUERY_SHAPE = re.compile(
    r"select ts_code, trade_date, factor_name, value from "
    r"loadTable\('dfs://YesLab', '(?P<table>[^']+)'\) "
    r"where factor_name in \[(?P<names>(?:'[^']+',? ?)+)\] "
    r"and trade_date between date\('(?P<start>\d{4}\.\d{2}\.\d{2})'\) "
    r"and date\('(?P<end>\d{4}\.\d{2}\.\d{2})'\)"
)


class CalculatorLike(Protocol):
    def calculate(self, start_time: str, end_time: str) -> pd.DataFrame: ...


@dataclass(frozen=True)
class RecomputeRoute:
    """One factor's recomputation recipe: source-table identity + calculator."""

    source_table: str
    lookback_days: int
    factory: Callable[[Any, int], CalculatorLike]


@dataclass(frozen=True)
class ParsedRecomputationQuery:
    source_table: str
    factor_names: tuple[str, ...]
    start_date: date
    end_date: date


def parse_recomputation_query(script: str) -> ParsedRecomputationQuery:
    match = _QUERY_SHAPE.fullmatch(script.strip())
    if match is None:
        raise ValueError("recomputation query does not match the source contract")
    table = match.group("table")
    if not _SAFE_TABLE.fullmatch(table):
        raise ValueError(f"unsafe recomputation source table: {table}")
    names = tuple(
        part.strip().strip("'")
        for part in match.group("names").split(",")
        if part.strip()
    )
    if not names:
        raise ValueError("recomputation query requested no factor names")
    start = pd.Timestamp(match.group("start")).date()
    end = pd.Timestamp(match.group("end")).date()
    if start > end:
        raise ValueError("recomputation window is inverted")
    return ParsedRecomputationQuery(table, names, start, end)


class RecomputedFactorSourceSession:
    """Query-compatible factor source backed by calculator recomputation."""

    def __init__(
        self,
        market_client: object,
        factor_routes: Mapping[str, RecomputeRoute],
    ) -> None:
        if not factor_routes:
            raise ValueError("recomputed source requires at least one factor route")
        self._market_client = market_client
        self._routes = dict(factor_routes)

    def query(self, script: str) -> object:
        parsed = parse_recomputation_query(script)
        frames: list[pd.DataFrame] = []
        for factor_name in parsed.factor_names:
            route = self._routes.get(factor_name)
            if route is None:
                raise ValueError(f"no recomputation route for factor: {factor_name}")
            if route.source_table != parsed.source_table:
                raise ValueError(
                    f"factor {factor_name} is routed to {route.source_table}, "
                    f"not {parsed.source_table}"
                )
            calculator = route.factory(self._market_client, route.lookback_days)
            wide = calculator.calculate(
                parsed.start_date.strftime("%Y.%m.%d"),
                parsed.end_date.strftime("%Y.%m.%d"),
            )
            long = (
                wide.rename_axis("trade_date")
                .reset_index()
                .melt(id_vars=["trade_date"], var_name="ts_code", value_name="value")
            )
            long["trade_date"] = pd.to_datetime(long["trade_date"]).dt.date
            long = long.loc[
                (long["trade_date"] >= parsed.start_date)
                & (long["trade_date"] <= parsed.end_date)
            ]
            long["factor_name"] = factor_name
            frames.append(long)
        if not frames:
            return pd.DataFrame(columns=list(_SOURCE_COLUMNS))
        combined = pd.concat(frames, ignore_index=True)
        return combined.loc[:, list(_SOURCE_COLUMNS)].copy(deep=True)
