"""Read-only compact-universe coverage audit for selected factors."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Protocol

import pandas as pd

from research_v2.db.session import V2WriteSession, YesLabReadOnlySession
from research_v2.factors.compact_coverage import summarize_compact_coverage

LOCKED_TEST_START = date(2026, 1, 1)
_FACTOR_ID_RE = re.compile(r"^yeslab_[0-9]+$")
_TABLE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_SYMBOL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


class QuerySession(Protocol):
    def query(self, script: str) -> object: ...


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor-id", action="append", required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--universe-id", default="cn_liquid_a")
    parser.add_argument("--universe-version", type=int, default=1)
    parser.add_argument("--label-id", default="forward_open_5d")
    parser.add_argument("--label-version", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.start > args.end:
        raise ValueError("coverage start date must not be after end date")
    if args.end >= LOCKED_TEST_START:
        raise ValueError("compact coverage audit cannot read 2026 locked-test data")
    if args.start < date(2020, 1, 1) or args.end > date(2025, 12, 31):
        raise ValueError("compact coverage audit must remain in TRAIN/VALIDATION bounds")
    if not all(_FACTOR_ID_RE.fullmatch(str(value)) for value in args.factor_id):
        raise ValueError("factor_id must use yeslab_<integer> form")
    if not _SYMBOL_RE.fullmatch(str(args.universe_id)):
        raise ValueError("universe_id contains unsafe characters")
    if not _SYMBOL_RE.fullmatch(str(args.label_id)):
        raise ValueError("label_id contains unsafe characters")
    return args


def run_audit(
    args: argparse.Namespace,
    *,
    source_session: QuerySession,
    target_session: QuerySession,
) -> list[dict[str, object]]:
    ids = tuple(sorted(set(str(value) for value in args.factor_id)))
    numeric_ids = ",".join(str(value.removeprefix("yeslab_")) for value in ids)
    metadata = source_session.query(
        "select factor_id,factor_name,store_factor from "
        "loadTable('dfs://YesLab','factor_metadata') "
        f"where factor_id in [{numeric_ids}]"
    )
    if not isinstance(metadata, pd.DataFrame):
        raise RuntimeError("factor metadata query did not return a DataFrame")
    metadata = metadata.assign(
        factor_id=metadata["factor_id"].map(lambda value: f"yeslab_{int(value)}")
    ).set_index("factor_id")
    missing = sorted(set(ids).difference(metadata.index))
    if missing:
        raise ValueError(f"factor metadata is missing requested IDs: {missing}")
    universe = _query_universe(target_session, args)
    labels = _query_labels(target_session, args)
    output: list[dict[str, object]] = []
    for factor_id in ids:
        meta = metadata.loc[factor_id]
        table = str(meta["store_factor"])
        factor_name = str(meta["factor_name"])
        if _TABLE_RE.fullmatch(table) is None:
            raise ValueError(f"unsafe source table: {table!r}")
        source = _query_factor(source_session, table, factor_name, args.start, args.end)
        result = summarize_compact_coverage(
            universe,
            labels,
            source,
            factor_id=factor_id,
            factor_name=factor_name,
            logical_partition=f"{args.start.isoformat()}..{args.end.isoformat()}",
        )
        output.append(dict(result))
    return output


def _query_universe(target: QuerySession, args: argparse.Namespace) -> pd.DataFrame:
    return _frame_query(
        target,
        "select signal_date,ts_code,is_member from "
        "loadTable('dfs://QuantResearchV2Test','universe_daily') "
        f"where universe_id=`{args.universe_id}` and universe_version={int(args.universe_version)} "
        f"and signal_date between date('{args.start:%Y.%m.%d}') and date('{args.end:%Y.%m.%d}')",
        "universe",
    )


def _query_labels(target: QuerySession, args: argparse.Namespace) -> pd.DataFrame:
    return _frame_query(
        target,
        "select signal_date,ts_code,label_valid from "
        "loadTable('dfs://QuantResearchV2Test','label_daily') "
        f"where label_id=`{args.label_id}` and label_version={int(args.label_version)} "
        f"and signal_date between date('{args.start:%Y.%m.%d}') and date('{args.end:%Y.%m.%d}')",
        "labels",
    )


def _query_factor(
    source: QuerySession,
    table: str,
    factor_name: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    escaped = factor_name.replace("'", "''")
    return _frame_query(
        source,
        "select trade_date,ts_code,value from "
        f"loadTable('dfs://YesLab','{table}') "
        f"where factor_name='{escaped}' and trade_date between date('{start:%Y.%m.%d}') "
        f"and date('{end:%Y.%m.%d}')",
        f"factor {factor_name}",
    )


def _frame_query(session: QuerySession, script: str, label: str) -> pd.DataFrame:
    result = session.query(script)
    if not isinstance(result, pd.DataFrame):
        raise RuntimeError(f"{label} query did not return a DataFrame")
    return result


def _connect(target_database: str) -> tuple[YesLabReadOnlySession, V2WriteSession]:
    import dolphindb as ddb  # type: ignore[import-untyped]

    client = ddb.session()
    client.connect(
        os.getenv("QRV2_DDB_HOST", "10.176.55.211"),
        int(os.getenv("QRV2_DDB_PORT", "8848")),
        os.getenv("QRV2_DDB_USERNAME", "xiahongjie"),
        os.getenv("QRV2_DDB_PASSWORD", "123456"),
    )
    return YesLabReadOnlySession(client), V2WriteSession(client, target_database=target_database)


def main(argv: Sequence[str] | None = None) -> list[dict[str, object]]:
    args = parse_args(argv)
    source, target = _connect("dfs://QuantResearchV2Test")
    rows = run_audit(args, source_session=source, target_session=target)
    payload = {
        "scope": "COMPACT_UNIVERSE_COVERAGE_READ_ONLY_V1",
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "universe_id": args.universe_id,
        "universe_version": args.universe_version,
        "label_id": args.label_id,
        "label_version": args.label_version,
        "database_writes": False,
        "rows": rows,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return rows


if __name__ == "__main__":
    main()
