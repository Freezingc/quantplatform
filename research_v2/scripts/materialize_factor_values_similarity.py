"""Materialize a non-test factor panel for scope-bounded similarity analysis.

This is a read-only diagnostic.  It applies the same YesLab normalization,
deduplication, and PIT alignment used by publication, but intentionally writes
only a local parquet artifact.  The 2025 panel is sufficient for redundancy
screening because it is outside the training selection window and is never
used to change the locked 2026 test.
"""

from __future__ import annotations

import argparse
import gc
import os
from datetime import date
from pathlib import Path

import dolphindb as ddb
import pandas as pd

from research_v2.db.session import YesLabReadOnlySession
from research_v2.factors.materializer import build_technical_factor_batches
from research_v2.factors.source_backfill import (
    FactorMonth,
    build_factor_definitions,
    build_factor_month_context,
)
from research_v2.factors.source_evaluation import _query_source_range

DEFAULT_FACTOR_IDS: tuple[str, ...] = (
    "yeslab_376",
    "yeslab_377",
    "yeslab_380",
    "yeslab_381",
    "yeslab_383",
    "yeslab_385",
    "yeslab_387",
    "yeslab_388",
    "yeslab_391",
    "yeslab_393",
    "yeslab_395",
    "yeslab_396",
    "yeslab_398",
    "yeslab_400",
    "yeslab_404",
    "yeslab_405",
    "yeslab_425",
    "yeslab_431",
    "yeslab_432",
    "yeslab_433",
    "yeslab_442",
    "yeslab_443",
    "yeslab_458",
    "yeslab_464",
)


def _connect() -> ddb.session:
    client = ddb.session()
    client.connect(
        os.getenv("QRV2_DDB_HOST", "10.176.55.211"),
        int(os.getenv("QRV2_DDB_PORT", "8848")),
        os.getenv("QRV2_DDB_USERNAME", "xiahongjie"),
        os.getenv("QRV2_DDB_PASSWORD", "123456"),
    )
    return client


def _metadata(session: YesLabReadOnlySession, factor_ids: tuple[str, ...]) -> pd.DataFrame:
    ids = ", ".join(str(int(value.removeprefix("yeslab_"))) for value in factor_ids)
    result = session.query(
        "select factor_id,factor_name,creator,update_frequency,store_factor "
        "from loadTable('dfs://YesLab','factor_metadata') "
        f"where factor_id in [{ids}]"
    )
    if not isinstance(result, pd.DataFrame):
        raise RuntimeError("factor metadata query did not return a DataFrame")
    return result.copy(deep=True)


def _parse_date(value: str) -> date:
    return pd.Timestamp(value).date()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--chunk-months", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factor-ids", default=",".join(DEFAULT_FACTOR_IDS))
    args = parser.parse_args()
    start_date = _parse_date(args.start)
    end_date = _parse_date(args.end)
    factor_ids = tuple(value.strip() for value in args.factor_ids.split(",") if value.strip())
    if start_date > end_date or args.chunk_months < 1:
        raise ValueError("invalid date range or chunk size")
    if not factor_ids or len(set(factor_ids)) != len(factor_ids):
        raise ValueError("factor_ids must be non-empty and unique")

    client = _connect()
    session = YesLabReadOnlySession(client)
    metadata = _metadata(session, factor_ids)
    definitions = build_factor_definitions(metadata, factor_ids)
    periods = tuple(str(period) for period in pd.period_range(start_date, end_date, freq="M"))
    context = build_factor_month_context(FactorMonth.from_partition(periods[0]), definitions)
    grouped: dict[str, list] = {}
    for definition in definitions:
        grouped.setdefault(definition.source_table, []).append(definition)
    collected: list[pd.DataFrame] = []
    for chunk_start in range(0, len(periods), args.chunk_months):
        chunk_periods = periods[chunk_start : chunk_start + args.chunk_months]
        chunk_start_date = pd.Period(chunk_periods[0], freq="M").start_time.date()
        chunk_end_date = pd.Period(chunk_periods[-1], freq="M").end_time.date()
        for source_table, raw_definitions in sorted(grouped.items()):
            table_definitions = tuple(
                sorted(raw_definitions, key=lambda definition: definition.factor_id)
            )
            source = _query_source_range(
                session,
                source_table,
                table_definitions,
                chunk_start_date,
                chunk_end_date,
            )
            if source.empty:
                continue
            watermark = f"similarity|{source_table}|{chunk_start_date}|{chunk_end_date}"
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
                    if str(error) == "technical factor has no clean observations to publish":
                        continue
                    raise
                frame = materialized.clean_batch.frame.loc[
                    :, ["signal_date", "ts_code", "factor_id", "factor_version", "value"]
                ].copy()
                collected.append(frame)
            del source
            gc.collect()

    if not collected:
        raise RuntimeError("no clean factor observations were materialized")
    output = pd.concat(collected, ignore_index=True)
    output["signal_date"] = pd.to_datetime(output["signal_date"], errors="raise").dt.date
    output = output.sort_values(
        ["signal_date", "ts_code", "factor_id", "factor_version"],
        kind="mergesort",
        ignore_index=True,
    )
    if output.duplicated(["signal_date", "ts_code", "factor_id", "factor_version"]).any():
        raise RuntimeError("materialized similarity panel contains duplicate business keys")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    print(
        f"materialized similarity panel: factors={len(factor_ids)} rows={len(output)} "
        f"dates={output['signal_date'].nunique()} output={args.output}"
    )


if __name__ == "__main__":
    main()
