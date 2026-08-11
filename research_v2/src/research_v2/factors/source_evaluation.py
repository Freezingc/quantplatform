"""Fast, reproducible factor coverage and RankIC evaluation before publication."""

from __future__ import annotations

import gc
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from research_v2.contracts.factor import FactorDefinition
from research_v2.evaluation.factor_rank_ic import (
    compute_factor_daily_rank_ic,
    summarize_factor_rank_ic,
)
from research_v2.factors.materializer import build_technical_factor_batches
from research_v2.factors.source_backfill import (
    FactorBackfillError,
    FactorMonth,
    YesLabFactorSession,
    build_factor_definitions,
    build_factor_month_context,
    iter_factor_materialization_batches,
)


@dataclass(frozen=True)
class FactorMonthEvaluation:
    """Coverage and predictive evidence for one source-materialized month."""

    logical_partition: str
    data_version: str
    clean_row_count: int
    coverage: pd.DataFrame
    daily_rank_ic: pd.DataFrame
    factor_summary: pd.DataFrame


def evaluate_materialized_factor_frames(
    clean_frames: Iterable[pd.DataFrame],
    *,
    universe: pd.DataFrame,
    labels: pd.DataFrame,
    logical_partition: str,
    minimum_cross_section: int = 30,
    expected_factor_ids: tuple[str, ...] = (),
) -> FactorMonthEvaluation:
    """Evaluate already-clean factor frames against mature PIT labels.

    The denominator is the intersection of universe members and valid labels,
    matching the production dataset/admission boundary. Missing factor values
    therefore reduce finite coverage instead of being silently excluded from
    the denominator.
    """
    collected = [frame.copy(deep=True) for frame in clean_frames]
    if collected:
        factors = _combine_clean_frames(collected)
    elif expected_factor_ids:
        factors = pd.DataFrame(columns=["signal_date", "ts_code", "factor_id", "value"])
    else:
        raise ValueError("at least one clean factor frame is required")
    expected = tuple(sorted(set(expected_factor_ids)))
    if len(expected) != len(expected_factor_ids):
        raise ValueError("expected_factor_ids must be unique")
    universe_frame = _normalize_universe(universe)
    labels_frame = _normalize_labels(labels)
    mature = universe_frame.merge(
        labels_frame.loc[labels_frame["label_valid"]],
        on=["signal_date", "ts_code"],
        how="inner",
        validate="one_to_one",
    )
    coverage_rows: list[dict[str, object]] = []
    observed_factor_ids = set(factors["factor_id"].astype(str))
    factor_ids = tuple(sorted(observed_factor_ids.union(expected)))
    for factor_id in factor_ids:
        group = factors.loc[factors["factor_id"].astype(str) == factor_id]
        merged = mature.loc[:, ["signal_date", "ts_code"]].merge(
            group.loc[:, ["signal_date", "ts_code", "value"]],
            on=["signal_date", "ts_code"],
            how="left",
            validate="one_to_one",
        )
        values = pd.to_numeric(merged["value"], errors="coerce").to_numpy(dtype=float)
        finite_count = int(np.isfinite(values).sum())
        mature_count = len(merged)
        coverage_rows.append(
            {
                "factor_id": str(factor_id),
                "finite_count": finite_count,
                "mature_row_count": mature_count,
                "finite_ratio": finite_count / mature_count if mature_count else 0.0,
            }
        )
    coverage = pd.DataFrame(
        coverage_rows,
        columns=["factor_id", "finite_count", "mature_row_count", "finite_ratio"],
    )
    factors_for_ic = factors.loc[:, ["signal_date", "ts_code", "factor_id", "value"]].copy()
    factors_for_ic = factors_for_ic.rename(columns={"factor_id": "factor_name"})
    daily_rank_ic = compute_factor_daily_rank_ic(
        factors_for_ic,
        labels_frame,
        minimum_cross_section=minimum_cross_section,
    )
    factor_summary = summarize_factor_rank_ic(daily_rank_ic)
    return FactorMonthEvaluation(
        logical_partition=logical_partition,
        data_version="",
        clean_row_count=len(factors),
        coverage=coverage,
        daily_rank_ic=daily_rank_ic,
        factor_summary=factor_summary,
    )


def evaluate_source_factor_month(
    source_session: YesLabFactorSession,
    metadata: pd.DataFrame,
    *,
    factor_ids: tuple[str, ...],
    logical_partition: str,
    universe: pd.DataFrame,
    labels: pd.DataFrame,
    attempt_nonce: str | None = None,
    minimum_cross_section: int = 30,
) -> FactorMonthEvaluation:
    """Materialize one YesLab month without lineage hashes and evaluate it."""
    month = FactorMonth.from_partition(logical_partition)
    definitions = build_factor_definitions(metadata, factor_ids)
    context = build_factor_month_context(month, definitions, attempt_nonce=attempt_nonce)
    materialized = (
        item.clean_batch.frame
        for item in iter_factor_materialization_batches(
            source_session,
            definitions,
            month,
            context,
            include_lineage=False,
        )
    )
    evaluation = evaluate_materialized_factor_frames(
        materialized,
        universe=universe,
        labels=labels,
        logical_partition=logical_partition,
        minimum_cross_section=minimum_cross_section,
    )
    return FactorMonthEvaluation(
        logical_partition=evaluation.logical_partition,
        data_version=context.data_version,
        clean_row_count=evaluation.clean_row_count,
        coverage=evaluation.coverage,
        daily_rank_ic=evaluation.daily_rank_ic,
        factor_summary=evaluation.factor_summary,
    )


def evaluate_source_factor_range(
    source_session: YesLabFactorSession,
    metadata: pd.DataFrame,
    *,
    factor_ids: tuple[str, ...],
    start_date: date,
    end_date: date,
    month_inputs: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]],
    attempt_nonce: str | None = None,
    minimum_cross_section: int = 30,
    chunk_months: int = 3,
) -> tuple[FactorMonthEvaluation, ...]:
    """Evaluate a date range with one source-table read per factor table.

    The range path avoids re-reading and re-normalizing the same source table
    once per month. It still materializes each factor deterministically and
    evaluates every calendar month separately, so no cross-month filling or
    label leakage is introduced.
    """
    if start_date > end_date:
        raise ValueError("start_date cannot be after end_date")
    if chunk_months < 1:
        raise ValueError("chunk_months must be positive")
    periods = tuple(str(period) for period in pd.period_range(start_date, end_date, freq="M"))
    missing_inputs = sorted(set(periods).difference(month_inputs))
    if missing_inputs:
        raise ValueError(f"month_inputs is missing months: {missing_inputs}")
    definitions = build_factor_definitions(metadata, factor_ids)
    expected_factor_ids = tuple(definition.factor_id for definition in definitions)
    first_month = FactorMonth.from_partition(periods[0])
    context = build_factor_month_context(
        first_month,
        definitions,
        attempt_nonce=attempt_nonce,
    )
    grouped: dict[str, list[FactorDefinition]] = defaultdict(list)
    for definition in definitions:
        grouped[definition.source_table].append(definition)
    evaluations: list[FactorMonthEvaluation] = []
    for chunk_start in range(0, len(periods), chunk_months):
        chunk_periods = periods[chunk_start : chunk_start + chunk_months]
        chunk_start_date = pd.Period(chunk_periods[0], freq="M").start_time.date()
        chunk_end_date = pd.Period(chunk_periods[-1], freq="M").end_time.date()
        monthly_frames: dict[str, list[pd.DataFrame]] = defaultdict(list)
        for source_table, raw_definitions in sorted(grouped.items()):
            table_definitions = tuple(
                sorted(raw_definitions, key=lambda definition: definition.factor_id)
            )
            try:
                source = _query_source_range(
                    source_session,
                    source_table,
                    table_definitions,
                    chunk_start_date,
                    chunk_end_date,
                )
            except FactorBackfillError as error:
                if "have no source rows" not in str(error):
                    raise
                source = pd.DataFrame()
            if source.empty:
                continue
            watermark = _range_watermark(
                source_table,
                chunk_start_date,
                chunk_end_date,
                source,
            )
            for definition in table_definitions:
                try:
                    materialized = build_technical_factor_batches(
                        source,
                        definition,
                        source_factor_name=definition.display_name,
                        source_watermark_id=watermark,
                        context=context,
                        record_hash_strategy="fast",
                        include_lineage=False,
                    )
                except ValueError as error:
                    if str(error) != "technical factor has no clean observations to publish":
                        raise
                    continue
                clean = materialized.clean_batch.frame
                clean["signal_date"] = pd.to_datetime(
                    clean["signal_date"], errors="raise"
                ).dt.date
                for logical_partition, month_frame in clean.groupby(
                    clean["signal_date"].map(lambda value: value.strftime("%Y-%m")),
                    sort=True,
                ):
                    monthly_frames[str(logical_partition)].append(month_frame.copy(deep=True))
            del source
            gc.collect()
        for logical_partition in chunk_periods:
            universe, labels = month_inputs[logical_partition]
            evaluation = evaluate_materialized_factor_frames(
                monthly_frames[logical_partition],
                universe=universe,
                labels=labels,
                logical_partition=logical_partition,
                minimum_cross_section=minimum_cross_section,
                expected_factor_ids=expected_factor_ids,
            )
            evaluations.append(evaluation)
        del monthly_frames
        gc.collect()
    return tuple(evaluations)


def _query_source_range(
    session: YesLabFactorSession,
    source_table: str,
    definitions: tuple[FactorDefinition, ...],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    if not re.fullmatch(r"factor_data(?:[1-9]|10|11)", source_table):
        raise FactorBackfillError(f"unsafe YesLab factor table: {source_table}")
    names = ", ".join(
        "'" + definition.display_name.replace("'", "''") + "'"
        for definition in definitions
    )
    query = (
        "select ts_code, trade_date, factor_name, value from "
        f"loadTable('dfs://YesLab', '{source_table}') "
        f"where factor_name in [{names}] "
        f"and trade_date between date('{start_date.strftime('%Y.%m.%d')}') "
        f"and date('{end_date.strftime('%Y.%m.%d')}')"
    )
    result = session.query(query)
    if not isinstance(result, pd.DataFrame):
        raise FactorBackfillError("YesLab factor range query did not return a DataFrame")
    required = {"ts_code", "trade_date", "factor_name", "value"}
    if missing := sorted(required.difference(result.columns)):
        raise FactorBackfillError(f"YesLab factor range query is missing columns: {missing}")
    return result.loc[:, sorted(required)].copy(deep=True)


def _range_watermark(
    source_table: str,
    start_date: date,
    end_date: date,
    source: pd.DataFrame,
) -> str:
    payload = {
        "source_table": source_table,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "min_date": str(source["trade_date"].min()),
        "max_date": str(source["trade_date"].max()),
        "row_count": len(source),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _combine_clean_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    collected = [frame.copy(deep=True) for frame in frames]
    if not collected:
        raise ValueError("at least one clean factor frame is required")
    factors = pd.concat(collected, ignore_index=True)
    required = {"signal_date", "ts_code", "factor_id", "value"}
    if missing := sorted(required.difference(factors.columns)):
        raise ValueError(f"clean factor frames are missing columns: {missing}")
    factors["signal_date"] = pd.to_datetime(factors["signal_date"], errors="raise").dt.date
    if factors.duplicated(["signal_date", "ts_code", "factor_id"]).any():
        raise ValueError("clean factor frames contain duplicate business keys")
    return factors.loc[:, ["signal_date", "ts_code", "factor_id", "value"]].copy()


def _normalize_universe(universe: pd.DataFrame) -> pd.DataFrame:
    required = {"signal_date", "ts_code", "is_member"}
    if missing := sorted(required.difference(universe.columns)):
        raise ValueError(f"universe is missing columns: {missing}")
    frame = universe.loc[:, ["signal_date", "ts_code", "is_member"]].copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="raise").dt.date
    if frame.duplicated(["signal_date", "ts_code"]).any():
        raise ValueError("universe contains duplicate business keys")
    return frame.loc[frame["is_member"].astype(bool)].drop(columns=["is_member"])


def _normalize_labels(labels: pd.DataFrame) -> pd.DataFrame:
    label_column = "raw_return" if "raw_return" in labels.columns else "raw_return_5d"
    required = {"signal_date", "ts_code", label_column, "label_valid"}
    if missing := sorted(required.difference(labels.columns)):
        raise ValueError(f"labels are missing columns: {missing}")
    frame = labels.loc[:, ["signal_date", "ts_code", label_column, "label_valid"]].copy()
    frame = frame.rename(columns={label_column: "raw_return"})
    frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="raise").dt.date
    if frame.duplicated(["signal_date", "ts_code"]).any():
        raise ValueError("labels contain duplicate business keys")
    return frame
