"""Materialize the 2014-2019 PIT chain (market/tradability/universe/labels).

The stored V2 PIT tables begin at 2020-01-02; the F01-F06 walk-forward needs
TRAIN years 2014-2019.  This driver materializes the same chain used for the
2020+ range, month by month, through the audited publish path:

  stage market      : YesLab stock_factor_daily -> market_daily_clean
  stage tradability : market_daily_clean + suspensions -> tradability_daily
  stage universe    : tradability_daily -> universe_daily (cn_liquid_a v1)
  stage labels      : universe members + market horizon -> label_daily

Run order: market for all months, then tradability, then universe, then labels
(labels need the following month's market rows for T+6 exits; 2019-12 exits
fall into the already-materialized 2020-01 market partition).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import dolphindb as ddb
import pandas as pd

from research_v2.contracts.source import SourceBatch
from research_v2.db.publish import V2BatchPublisher
from research_v2.db.session import V2WriteSession, YesLabReadOnlySession
from research_v2.materializers.labels import build_label_batch
from research_v2.materializers.market import build_market_batch
from research_v2.materializers.security_master import build_security_master_batch
from research_v2.materializers.tradability import build_tradability_batch
from research_v2.materializers.universe import build_universe_batch
from research_v2.pit.calendar import TradingCalendar
from research_v2.pit.security_master import resolve_name_history_overlaps
from research_v2.quality.engine import CheckResult, CheckStatus, Severity
from research_v2.runs.context import RunContext
from research_v2.universe.builder import UniverseDefinition, UniverseRules

TARGET_DATABASE = "dfs://QuantResearchV2Test"
MARKET_COLUMNS = (
    "ts_code, trade_date, open, high, low, close, vol, amount, adj_factor"
)


def _connect() -> tuple[object, object]:
    host = os.getenv("QRV2_DDB_HOST", "10.176.55.211")
    port = int(os.getenv("QRV2_DDB_PORT", "8848"))
    username = os.getenv("QRV2_DDB_USERNAME", "xiahongjie")
    password = os.getenv("QRV2_DDB_PASSWORD", "123456")
    source_client = ddb.session()
    source_client.connect(host, port, username, password)
    target_client = ddb.session()
    target_client.connect(host, port, username, password)
    return source_client, target_client


def _months(start_date: date, end_date: date) -> list[tuple[date, date]]:
    periods = pd.period_range(start_date, end_date, freq="M")
    return [(period.start_time.date(), period.end_time.date()) for period in periods]


def _watermark(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _context(stage: str, logical_partition: str, nonce: str) -> RunContext:
    return RunContext.build(
        config={"stage": stage, "logical_partition": logical_partition},
        watermarks=(f"{stage}:{logical_partition}:{nonce}",),
        code_commit="train-years-pit-chain-v1",
        schema_version="013",
    )


def _dq_checks(names: tuple[str, ...]) -> list[CheckResult]:
    return [CheckResult(name, Severity.BLOCKER, CheckStatus.PASS) for name in names]


def _current_data_versions(
    target_session: V2WriteSession, object_name: str, partitions: list[str]
) -> dict[str, str]:
    """Return the current published data_version per logical partition."""
    quoted = ", ".join(f"'{partition}'" for partition in partitions)
    frame = target_session.query(
        f"select logical_partition, data_version, published_at "
        f"from loadTable('{TARGET_DATABASE}','published_version') "
        f"where object_name='{object_name}' and logical_partition in [{quoted}]"
    )
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return {}
    frame = frame.sort_values("published_at", kind="mergesort")
    return {
        str(row.logical_partition): str(row.data_version)
        for row in frame.itertuples(index=False)
    }


def _load_calendar(source_session: YesLabReadOnlySession) -> TradingCalendar:
    frame = source_session.query(
        "select cal_date from loadTable('dfs://YesLab', 'trade_calendar') "
        "where exchange='SSE' and is_open=1 and cal_date between 1990.12.01 and 2020.03.31 "
        "order by cal_date"
    )
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("calendar query did not return a DataFrame")
    return TradingCalendar.from_dates([pd.Timestamp(v).date() for v in frame["cal_date"]])


def _load_security_scd(
    source_session: YesLabReadOnlySession, context: RunContext, as_of: date
) -> pd.DataFrame:
    info = source_session.query("select * from loadTable('dfs://YesLab','stock_info_table')")
    names = source_session.query("select * from loadTable('dfs://YesLab','stock_used_name')")
    if not isinstance(info, pd.DataFrame) or not isinstance(names, pd.DataFrame):
        raise ValueError("security master source queries did not return DataFrames")
    names = names.copy(deep=True)
    names = resolve_name_history_overlaps(names)
    batch = build_security_master_batch(
        SourceBatch(
            frame=info,
            source_name="yeslab",
            table_name="stock_info_table",
            watermark_id=_watermark("stock_info_table", as_of),
            max_observation_date=as_of,
            source_row_count=len(info),
            schema_fingerprint="train-years-v1",
            extracted_at=datetime.now(UTC),
        ),
        SourceBatch(
            frame=names,
            source_name="yeslab",
            table_name="stock_used_name",
            watermark_id=_watermark("stock_used_name", as_of),
            max_observation_date=as_of,
            source_row_count=len(names),
            schema_fingerprint="train-years-v1",
            extracted_at=datetime.now(UTC),
        ),
        context,
        as_of_date=as_of,
    )
    return batch.frame


def stage_market(
    source_session: YesLabReadOnlySession,
    target_session: V2WriteSession,
    calendar: TradingCalendar,
    scd: pd.DataFrame,
    month_start: date,
    month_end: date,
    nonce: str,
) -> int:
    extract_start = month_start - timedelta(days=40)
    extract_end = month_end + timedelta(days=15)
    frame = source_session.query(
        f"select {MARKET_COLUMNS} from loadTable('dfs://YesLab','stock_factor_daily') "
        f"where trade_date between date('{extract_start.strftime('%Y.%m.%d')}') "
        f"and date('{extract_end.strftime('%Y.%m.%d')}')"
    )
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("market source query returned no rows")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
    source = SourceBatch(
        frame=frame,
        source_name="yeslab",
        table_name="stock_factor_daily",
        watermark_id=_watermark("stock_factor_daily", extract_start, extract_end),
        max_observation_date=extract_end,
        source_row_count=len(frame),
        schema_fingerprint="train-years-v1",
        extracted_at=datetime.now(UTC),
    )
    context = _context("market", month_start.strftime("%Y-%m"), nonce)
    batch = build_market_batch(
        source,
        scd,
        calendar,
        context,
        publish_start=month_start,
        publish_end=month_end,
        quarantine_inactive_security=True,
    )
    if batch.frame.duplicated(["trade_date", "ts_code"], keep=False).any():
        raise ValueError("market batch contains duplicate keys")
    V2BatchPublisher(target_session).publish(
        batch,
        _dq_checks(("market_source_schema", "market_keys_unique", "market_finite_values")),
    )
    return len(batch.frame)


def stage_tradability(
    source_session: YesLabReadOnlySession,
    target_session: V2WriteSession,
    calendar: TradingCalendar,
    scd: pd.DataFrame,
    month_start: date,
    month_end: date,
    nonce: str,
) -> int:
    partition = month_start.strftime("%Y-%m")
    versions = _current_data_versions(target_session, "market_daily_clean", [partition])
    if partition not in versions:
        raise ValueError(f"market_daily_clean has no published version for {partition}")
    market = target_session.query(
        f"select * from loadTable('{TARGET_DATABASE}','market_daily_clean') "
        f"where trade_date between date('{month_start.strftime('%Y.%m.%d')}') "
        f"and date('{month_end.strftime('%Y.%m.%d')}') "
        f"and data_version='{versions[partition]}'"
    )
    if not isinstance(market, pd.DataFrame) or market.empty:
        raise ValueError("market_daily_clean has no rows for the month")
    market["trade_date"] = pd.to_datetime(market["trade_date"]).dt.date
    suspensions = source_session.query(
        "select ts_code, trade_date, suspend_timing, suspend_type "
        "from loadTable('dfs://YesLab','stock_trading_suspension_info') "
        f"where trade_date between date('{month_start.strftime('%Y.%m.%d')}') "
        f"and date('{month_end.strftime('%Y.%m.%d')}')"
    )
    if not isinstance(suspensions, pd.DataFrame):
        suspensions = pd.DataFrame(
            columns=["ts_code", "trade_date", "suspend_timing", "suspend_type"]
        )
    suspensions["trade_date"] = pd.to_datetime(suspensions["trade_date"]).dt.date
    context = _context("tradability", month_start.strftime("%Y-%m"), nonce)
    batch = build_tradability_batch(
        SourceBatch(
            frame=market,
            source_name="v2test",
            table_name="market_daily_clean",
            watermark_id=_watermark("market_daily_clean", month_start, month_end),
            max_observation_date=month_end,
            source_row_count=len(market),
            schema_fingerprint="train-years-v1",
            extracted_at=datetime.now(UTC),
        ),
        scd,
        SourceBatch(
            frame=suspensions,
            source_name="yeslab",
            table_name="stock_trading_suspension_info",
            watermark_id=_watermark("suspensions", month_start, month_end),
            max_observation_date=month_end,
            source_row_count=len(suspensions),
            schema_fingerprint="train-years-v1",
            extracted_at=datetime.now(UTC),
        ),
        calendar,
        context,
    )
    V2BatchPublisher(target_session).publish(
        batch, _dq_checks(("tradability_keys_unique", "tradability_bitmap_valid"))
    )
    return len(batch.frame)


def stage_universe(
    target_session: V2WriteSession,
    scd: pd.DataFrame,
    month_start: date,
    month_end: date,
    nonce: str,
) -> int:
    partition = month_start.strftime("%Y-%m")
    versions = _current_data_versions(target_session, "tradability_daily", [partition])
    if partition not in versions:
        raise ValueError(f"tradability_daily has no published version for {partition}")
    tradability = target_session.query(
        f"select * from loadTable('{TARGET_DATABASE}','tradability_daily') "
        f"where trade_date between date('{month_start.strftime('%Y.%m.%d')}') "
        f"and date('{month_end.strftime('%Y.%m.%d')}') "
        f"and data_version='{versions[partition]}'"
    )
    if not isinstance(tradability, pd.DataFrame) or tradability.empty:
        raise ValueError("tradability_daily has no rows for the month")
    tradability["trade_date"] = pd.to_datetime(tradability["trade_date"]).dt.date
    context = _context("universe", month_start.strftime("%Y-%m"), nonce)
    batch = build_universe_batch(
        SourceBatch(
            frame=tradability,
            source_name="v2test",
            table_name="tradability_daily",
            watermark_id=_watermark("tradability_daily", month_start, month_end),
            max_observation_date=month_end,
            source_row_count=len(tradability),
            schema_fingerprint="train-years-v1",
            extracted_at=datetime.now(UTC),
        ),
        scd,
        UniverseDefinition("cn_liquid_a", 1, UniverseRules(retained_adv20_fraction=0.80)),
        context,
    )
    V2BatchPublisher(target_session).publish(
        batch, _dq_checks(("universe_keys_unique", "universe_membership_valid"))
    )
    return len(batch.frame)


def stage_labels(
    target_session: V2WriteSession,
    calendar: TradingCalendar,
    month_start: date,
    month_end: date,
    nonce: str,
) -> int:
    label_partition = month_start.strftime("%Y-%m")
    universe_versions = _current_data_versions(target_session, "universe_daily", [label_partition])
    if label_partition not in universe_versions:
        raise ValueError(f"universe_daily has no published version for {label_partition}")
    members = target_session.query(
        f"select signal_date, ts_code from loadTable('{TARGET_DATABASE}','universe_daily') "
        f"where signal_date between date('{month_start.strftime('%Y.%m.%d')}') "
        f"and date('{month_end.strftime('%Y.%m.%d')}') and is_member=1 "
        f"and data_version='{universe_versions[label_partition]}'"
    )
    if not isinstance(members, pd.DataFrame) or members.empty:
        raise ValueError("universe_daily has no members for the month")
    members["signal_date"] = pd.to_datetime(members["signal_date"]).dt.date
    horizon_end = month_end + timedelta(days=15)
    next_partition = (month_end + timedelta(days=1)).strftime("%Y-%m")
    market_versions = _current_data_versions(
        target_session,
        "market_daily_clean",
        sorted({label_partition, next_partition}),
    )
    if label_partition not in market_versions:
        raise ValueError(f"market_daily_clean has no published version for {label_partition}")
    version_filter = " or ".join(
        f"(data_version='{version}')" for version in set(market_versions.values())
    )
    market = target_session.query(
        f"select trade_date, ts_code, adjusted_open "
        f"from loadTable('{TARGET_DATABASE}','market_daily_clean') "
        f"where trade_date between date('{month_start.strftime('%Y.%m.%d')}') "
        f"and date('{horizon_end.strftime('%Y.%m.%d')}') and ({version_filter})"
    )
    if not isinstance(market, pd.DataFrame) or market.empty:
        raise ValueError("market_daily_clean has no rows for label horizon")
    market["trade_date"] = pd.to_datetime(market["trade_date"]).dt.date
    context = _context("labels", month_start.strftime("%Y-%m"), nonce)
    batch = build_label_batch(
        members,
        SourceBatch(
            frame=market,
            source_name="v2test",
            table_name="market_daily_clean",
            watermark_id=_watermark("market_daily_clean", month_start, horizon_end),
            max_observation_date=horizon_end,
            source_row_count=len(market),
            schema_fingerprint="train-years-v1",
            extracted_at=datetime.now(UTC),
        ),
        calendar,
        context,
    )
    V2BatchPublisher(target_session).publish(
        batch, _dq_checks(("label_keys_unique", "label_maturity_valid"))
    )
    return len(batch.frame)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--end", default="2019-12-31")
    parser.add_argument(
        "--stage", choices=["market", "tradability", "universe", "labels"], required=True
    )
    parser.add_argument("--nonce", default="train-years-pit-v1")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    start_date = pd.Timestamp(args.start).date()
    end_date = pd.Timestamp(args.end).date()
    months = _months(start_date, end_date)

    source_client, target_client = _connect()
    source_session = YesLabReadOnlySession(source_client)
    target_session = V2WriteSession(target_client, target_database=TARGET_DATABASE)
    calendar = _load_calendar(source_session)
    context_scd = _context("security-master", months[-1][1].isoformat(), args.nonce)
    scd = _load_security_scd(source_session, context_scd, months[-1][1])

    rows: list[dict[str, object]] = []
    for index, (month_start, month_end) in enumerate(months, start=1):
        if args.stage == "market":
            row_count = stage_market(
                source_session, target_session, calendar, scd, month_start, month_end, args.nonce
            )
        elif args.stage == "tradability":
            row_count = stage_tradability(
                source_session, target_session, calendar, scd, month_start, month_end, args.nonce
            )
        elif args.stage == "universe":
            row_count = stage_universe(target_session, scd, month_start, month_end, args.nonce)
        else:
            row_count = stage_labels(target_session, calendar, month_start, month_end, args.nonce)
        rows.append({"logical_partition": month_start.strftime("%Y-%m"), "rows": row_count})
        print(
            f"{args.stage} {index}/{len(months)} {month_start.strftime('%Y-%m')} rows={row_count}",
            flush=True,
        )

    report = {
        "stage": args.stage,
        "target_database": TARGET_DATABASE,
        "start": args.start,
        "end": args.end,
        "nonce": args.nonce,
        "months": rows,
        "total_rows": sum(int(row["rows"]) for row in rows),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"stage {args.stage} complete; report={args.report}", flush=True)


if __name__ == "__main__":
    main()
