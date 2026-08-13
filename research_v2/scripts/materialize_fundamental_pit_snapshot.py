"""Materialize selected announcement-safe financial metrics into a new V2 table.

This command never mutates YesLab.  It reads financial disclosure events and
the V2 SH/SZ universe, expands events from the next trading day after their
announcement, and writes only to ``fundamental_pit_snapshot`` when explicitly
asked to publish.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from bisect import bisect_right
from collections.abc import Sequence
from datetime import date

import dolphindb as ddb
import pandas as pd

from research_v2.compact.publisher import read_latest_partition
from research_v2.db.session import V2WriteSession, YesLabReadOnlySession
from research_v2.factors.fundamental_snapshot import build_fundamental_snapshot
from research_v2.pit.calendar import TradingCalendar

TARGET_DATABASE = "dfs://QuantResearchV2Test"
OBJECT_NAME = "fundamental_pit_snapshot"
LOCKED_TEST_START = date(2026, 1, 1)
DEFAULT_PUBLISH_CHUNK_ROWS = 25_000
DIRECT_METRICS = ("roa", "roe", "roic")
DERIVED_METRICS = {
    "road": ("roa", "difference"),
    "roed": ("roe", "difference"),
    "roe_yoy": ("roe", "relative_change"),
    "roicd": ("roic", "difference"),
    "ta_yoy": ("total_assets", "relative_change"),
    "np_qoq": ("net_income_cumulative", "quarter_relative_change"),
}
SUPPORTED_METRICS = (*DIRECT_METRICS, "cfoa", *DERIVED_METRICS)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2020, 1, 2))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2020, 12, 31))
    parser.add_argument("--metric", action="append", default=[])
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--publish-chunk-rows", type=int, default=DEFAULT_PUBLISH_CHUNK_ROWS)
    args = parser.parse_args(argv)
    args.metric = tuple(dict.fromkeys(args.metric or SUPPORTED_METRICS))
    unsupported = sorted(set(args.metric).difference(SUPPORTED_METRICS))
    if unsupported:
        parser.error(f"unsupported metrics: {unsupported}")
    if args.start > args.end or args.end >= LOCKED_TEST_START:
        parser.error("snapshot range must be before 2026 and start <= end")
    if args.publish_chunk_rows < 1:
        parser.error("publish-chunk-rows must be positive")
    return args


def build_snapshot_frame(
    events: pd.DataFrame,
    universe: pd.DataFrame,
    calendar: TradingCalendar,
    *,
    data_version: str,
    source_watermark_id: str,
    run_id: str,
) -> pd.DataFrame:
    snapshot = build_fundamental_snapshot(events, universe, calendar)
    if snapshot.empty:
        raise ValueError("fundamental PIT snapshot has no materialized rows")
    result = snapshot.copy()
    result["run_id"] = run_id
    result["data_version"] = data_version
    result["created_at"] = pd.Timestamp.now(tz="UTC").tz_localize(None)
    result["source_watermark_id"] = source_watermark_id
    result["record_hash"] = result.apply(_record_hash, axis=1)
    return result


def _record_hash(row: pd.Series) -> str:
    payload = {
        key: (value.isoformat() if isinstance(value, (date, pd.Timestamp)) else value)
        for key, value in row.items()
        if key != "record_hash"
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True).encode()
    ).hexdigest()


def _connect() -> object:
    client = ddb.session()
    client.connect(
        os.getenv("QRV2_DDB_HOST", "10.176.55.211"),
        int(os.getenv("QRV2_DDB_PORT", "8848")),
        os.getenv("QRV2_DDB_USERNAME", "xiahongjie"),
        os.getenv("QRV2_DDB_PASSWORD", ""),
    )
    return client


def _query_events(
    source: YesLabReadOnlySession,
    *,
    start: date,
    end: date,
    metrics: tuple[str, ...],
) -> pd.DataFrame:
    history_start = date(start.year - 1, 1, 1)
    rows: list[pd.DataFrame] = []
    requested_direct = set(metrics).intersection(DIRECT_METRICS)
    requested_source = {
        source_metric
        for output_metric, (source_metric, _) in DERIVED_METRICS.items()
        if output_metric in metrics
    }
    source_metrics = tuple(sorted(requested_direct | requested_source))
    source_rows: list[pd.DataFrame] = []
    for metric in source_metrics:
        if metric == "total_assets":
            source_rows.append(
                _query_total_assets_events(source, history_start=history_start, end=end)
            )
        elif metric == "net_income_cumulative":
            source_rows.append(
                _query_net_income_events(source, history_start=history_start, end=end)
            )
        else:
            source_rows.append(
                _query_indicator_events(
                    source,
                    history_start=history_start,
                    end=end,
                    metric=metric,
                )
            )
    if "cfoa" in metrics:
        rows.append(_query_cfoa_events(source, history_start=history_start, end=end))
    if source_rows:
        source_events = pd.concat(source_rows, ignore_index=True)
        rows.extend(
            [
                source_events.loc[source_events["metric_id"].eq(metric)].copy()
                for metric in sorted(requested_direct)
            ]
        )
        for output_metric, (source_metric, operation) in DERIVED_METRICS.items():
            if output_metric in metrics:
                if operation == "quarter_relative_change":
                    rows.append(
                        derive_quarter_over_quarter_events(
                            source_events,
                            source_metric=source_metric,
                            output_metric=output_metric,
                        )
                    )
                else:
                    rows.append(
                        derive_year_over_year_events(
                            source_events,
                            source_metric=source_metric,
                            output_metric=output_metric,
                            operation=operation,
                        )
                    )
    if not rows:
        return pd.DataFrame(
            columns=[
                "metric_id",
                "ts_code",
                "report_period",
                "ann_date",
                "source_revision_id",
                "raw_value",
            ]
        )
    return pd.concat(rows, ignore_index=True).drop_duplicates()


def _query_indicator_events(
    source: YesLabReadOnlySession,
    *,
    history_start: date,
    end: date,
    metric: str,
) -> pd.DataFrame:
    result = source.query(
        "select ts_code,ann_date,end_date,update_flag,"
        f"{metric} as raw_value from loadTable('dfs://YesLab','financial_metrics_table') "
        f"where ann_date between date('{history_start:%Y.%m.%d}') and "
        f"date('{end:%Y.%m.%d}') and isValid({metric})"
    )
    if not isinstance(result, pd.DataFrame):
        raise RuntimeError(f"financial event query did not return a DataFrame: {metric}")
    frame = result.rename(columns={"end_date": "report_period"}).copy()
    frame.insert(0, "metric_id", metric)
    frame["ann_date"] = pd.to_datetime(frame["ann_date"], errors="raise").dt.date
    frame["report_period"] = pd.to_datetime(
        frame["report_period"], errors="raise"
    ).dt.date
    frame["source_revision_id"] = frame["update_flag"].astype(str)
    return frame.loc[
        :,
        [
            "metric_id",
            "ts_code",
            "report_period",
            "ann_date",
            "source_revision_id",
            "raw_value",
        ],
    ].drop_duplicates()


def _query_total_assets_events(
    source: YesLabReadOnlySession, *, history_start: date, end: date
) -> pd.DataFrame:
    result = source.query(
        "select ts_code,ann_date,end_date,update_flag,total_cur_assets,total_nca "
        "from loadTable('dfs://YesLab','balance_sheet') "
        f"where ann_date between date('{history_start:%Y.%m.%d}') and "
        f"date('{end:%Y.%m.%d}') and isValid(total_cur_assets) and isValid(total_nca)"
    )
    if not isinstance(result, pd.DataFrame):
        raise RuntimeError("total-assets source query did not return a DataFrame")
    frame = result.rename(columns={"end_date": "report_period"}).copy()
    frame.insert(0, "metric_id", "total_assets")
    frame["raw_value"] = pd.to_numeric(frame["total_cur_assets"], errors="coerce") + pd.to_numeric(
        frame["total_nca"], errors="coerce"
    )
    frame["ann_date"] = pd.to_datetime(frame["ann_date"], errors="raise").dt.date
    frame["report_period"] = pd.to_datetime(
        frame["report_period"], errors="raise"
    ).dt.date
    frame["source_revision_id"] = frame["update_flag"].astype(str)
    return frame.loc[
        :,
        [
            "metric_id",
            "ts_code",
            "report_period",
            "ann_date",
            "source_revision_id",
            "raw_value",
        ],
    ].dropna(subset=["raw_value"]).drop_duplicates()


def _query_net_income_events(
    source: YesLabReadOnlySession, *, history_start: date, end: date
) -> pd.DataFrame:
    result = source.query(
        "select ts_code,ann_date,end_date,update_flag,n_income_attr_p "
        "from loadTable('dfs://YesLab','company_income_table') "
        f"where ann_date between date('{history_start:%Y.%m.%d}') and "
        f"date('{end:%Y.%m.%d}') and isValid(n_income_attr_p)"
    )
    if not isinstance(result, pd.DataFrame):
        raise RuntimeError("net-income source query did not return a DataFrame")
    frame = result.rename(columns={"end_date": "report_period"}).copy()
    frame.insert(0, "metric_id", "net_income_cumulative")
    frame = frame.rename(columns={"n_income_attr_p": "raw_value"})
    frame["ann_date"] = pd.to_datetime(frame["ann_date"], errors="raise").dt.date
    frame["report_period"] = pd.to_datetime(
        frame["report_period"], errors="raise"
    ).dt.date
    frame["source_revision_id"] = frame["update_flag"].astype(str)
    return frame.loc[
        :,
        [
            "metric_id",
            "ts_code",
            "report_period",
            "ann_date",
            "source_revision_id",
            "raw_value",
        ],
    ].dropna(subset=["raw_value"]).drop_duplicates()


def _query_cfoa_events(
    source: YesLabReadOnlySession,
    *,
    history_start: date,
    end: date,
) -> pd.DataFrame:
    """Build CFOA events from independently announced statements.

    A CFOA observation is effective only after both the balance-sheet and
    cash-flow inputs are available.  For each security/report period we walk
    the union of announcement dates and carry the latest disclosed value from
    each statement, so a later correction in either source creates a new
    deterministic event without changing the raw YesLab tables.
    """
    bounds = (
        f"date('{history_start:%Y.%m.%d}') and date('{end:%Y.%m.%d}')"
    )
    balance = source.query(
        "select ts_code,ann_date,end_date,update_flag,total_cur_assets,total_nca "
        "from loadTable('dfs://YesLab','balance_sheet') "
        f"where ann_date between {bounds} and isValid(total_cur_assets) "
        "and isValid(total_nca)"
    )
    cashflow = source.query(
        "select ts_code,ann_date,end_date,update_flag,n_cashflow_act "
        "from loadTable('dfs://YesLab','cashflow_table') "
        f"where ann_date between {bounds} and isValid(n_cashflow_act)"
    )
    if not isinstance(balance, pd.DataFrame) or not isinstance(cashflow, pd.DataFrame):
        raise RuntimeError("CFOA source queries did not return DataFrames")
    balance = balance.rename(columns={"end_date": "report_period"}).copy()
    cashflow = cashflow.rename(columns={"end_date": "report_period"}).copy()
    for frame in (balance, cashflow):
        frame["ts_code"] = frame["ts_code"].astype(str)
        frame["ann_date"] = pd.to_datetime(frame["ann_date"], errors="raise").dt.date
        frame["report_period"] = pd.to_datetime(
            frame["report_period"], errors="raise"
        ).dt.date
        frame["source_revision_id"] = frame["update_flag"].astype(str)
    balance = balance.sort_values(
        ["ts_code", "report_period", "ann_date", "source_revision_id"],
        kind="mergesort",
    ).drop_duplicates(["ts_code", "report_period", "ann_date"], keep="last")
    cashflow = cashflow.sort_values(
        ["ts_code", "report_period", "ann_date", "source_revision_id"],
        kind="mergesort",
    ).drop_duplicates(["ts_code", "report_period", "ann_date"], keep="last")

    event_rows: list[dict[str, object]] = []
    balance_groups = balance.groupby(["ts_code", "report_period"], sort=False)
    cash_groups = cashflow.groupby(["ts_code", "report_period"], sort=False)
    for key, balance_group in balance_groups:
        if key not in cash_groups.groups:
            continue
        cash_group = cash_groups.get_group(key)
        balance_group = balance_group.sort_values("ann_date", kind="mergesort")
        cash_group = cash_group.sort_values("ann_date", kind="mergesort")
        dates = sorted(set(balance_group["ann_date"]) | set(cash_group["ann_date"]))
        for announcement_date in dates:
            available_balance = balance_group.loc[
                balance_group["ann_date"] <= announcement_date
            ]
            available_cash = cash_group.loc[cash_group["ann_date"] <= announcement_date]
            if available_balance.empty or available_cash.empty:
                continue
            balance_row = available_balance.iloc[-1]
            cash_row = available_cash.iloc[-1]
            denominator = float(balance_row["total_cur_assets"]) + float(
                balance_row["total_nca"]
            )
            numerator = float(cash_row["n_cashflow_act"])
            if not pd.notna(denominator) or denominator == 0 or not pd.notna(numerator):
                continue
            event_rows.append(
                {
                    "metric_id": "cfoa",
                    "ts_code": key[0],
                    "report_period": key[1],
                    "ann_date": announcement_date,
                    "source_revision_id": (
                        f"balance:{balance_row['source_revision_id']}|"
                        f"cashflow:{cash_row['source_revision_id']}"
                    ),
                    "raw_value": numerator / denominator,
                }
            )
    return pd.DataFrame(
        event_rows,
        columns=[
            "metric_id",
            "ts_code",
            "report_period",
            "ann_date",
            "source_revision_id",
            "raw_value",
        ],
    ).drop_duplicates()


def derive_year_over_year_events(
    events: pd.DataFrame,
    *,
    source_metric: str,
    output_metric: str,
    operation: str,
) -> pd.DataFrame:
    """Derive announcement-safe exact-prior-report-period events.

    The legacy calculators approximate four reporting quarters with a fixed
    trading-day lag.  This version pairs an event with the same report period
    one calendar year earlier, selecting the latest prior disclosure known at
    the current announcement date.  A restatement creates a fresh derived
    event and never rewrites an earlier event's information set.
    """
    if operation not in {"difference", "relative_change"}:
        raise ValueError("operation must be difference or relative_change")
    required = {
        "metric_id",
        "ts_code",
        "report_period",
        "ann_date",
        "source_revision_id",
        "raw_value",
    }
    if missing := sorted(required.difference(events.columns)):
        raise ValueError(f"financial events are missing columns: {missing}")
    source = events.loc[events["metric_id"].astype(str).eq(source_metric), sorted(required)].copy()
    if source.empty:
        return pd.DataFrame(columns=sorted(required))
    source["ts_code"] = source["ts_code"].astype(str)
    source["ann_date"] = pd.to_datetime(source["ann_date"], errors="raise").dt.date
    source["report_period"] = pd.to_datetime(
        source["report_period"], errors="raise"
    ).dt.date
    source["raw_value"] = pd.to_numeric(source["raw_value"], errors="coerce")
    source = source.loc[source["raw_value"].notna()].sort_values(
        ["ts_code", "report_period", "ann_date", "source_revision_id"], kind="mergesort"
    )
    rows: list[dict[str, object]] = []
    records = source.to_dict("records")
    prior_lookup: dict[tuple[str, date], list[dict[str, object]]] = {}
    for record in records:
        prior_lookup.setdefault(
            (str(record["ts_code"]), record["report_period"]), []
        ).append(record)
    for candidates in prior_lookup.values():
        candidates.sort(key=lambda item: (item["ann_date"], str(item["source_revision_id"])))
    for current in records:
        prior_period = _shift_report_period_one_year(current["report_period"])
        candidates = prior_lookup.get((str(current["ts_code"]), prior_period), [])
        if not candidates:
            continue
        announcement_dates = [item["ann_date"] for item in candidates]
        prior_index = bisect_right(announcement_dates, current["ann_date"]) - 1
        if prior_index < 0:
            continue
        prior = candidates[prior_index]
        prior_value = float(prior["raw_value"])
        current_value = float(current["raw_value"])
        if operation == "difference":
            raw_value = current_value - prior_value
        else:
            if prior_value == 0:
                continue
            raw_value = (current_value - prior_value) / prior_value
        rows.append(
            {
                "metric_id": output_metric,
                "ts_code": current["ts_code"],
                "report_period": current["report_period"],
                "ann_date": current["ann_date"],
                "source_revision_id": (
                    f"current:{current['source_revision_id']}|prior:{prior['source_revision_id']}"
                ),
                "raw_value": raw_value,
            }
        )
    return pd.DataFrame(rows, columns=sorted(required)).sort_values(
        ["ts_code", "report_period", "ann_date", "source_revision_id"],
        ignore_index=True,
    )


def derive_quarter_over_quarter_events(
    events: pd.DataFrame,
    *,
    source_metric: str,
    output_metric: str,
) -> pd.DataFrame:
    """Recover single-quarter values and compare with the prior report period."""
    required = {
        "metric_id",
        "ts_code",
        "report_period",
        "ann_date",
        "source_revision_id",
        "raw_value",
    }
    source = events.loc[events["metric_id"].astype(str).eq(source_metric), sorted(required)].copy()
    if source.empty:
        return pd.DataFrame(columns=sorted(required))
    source["ts_code"] = source["ts_code"].astype(str)
    source["ann_date"] = pd.to_datetime(source["ann_date"], errors="raise").dt.date
    source["report_period"] = pd.to_datetime(
        source["report_period"], errors="raise"
    ).dt.date
    source["raw_value"] = pd.to_numeric(source["raw_value"], errors="coerce")
    source = source.loc[source["raw_value"].notna()].sort_values(
        ["ts_code", "report_period", "ann_date", "source_revision_id"], kind="mergesort"
    )
    rows: list[dict[str, object]] = []
    grouped = source.groupby("ts_code", sort=False)
    for ts_code, group in grouped:
        records = group.to_dict("records")
        records.sort(
            key=lambda item: (
                item["report_period"],
                item["ann_date"],
                str(item["source_revision_id"]),
            )
        )
        by_period: dict[date, list[dict[str, object]]] = {}
        for record in records:
            by_period.setdefault(record["report_period"], []).append(record)
        periods = sorted(by_period)
        for period in periods:
            current_candidates = by_period[period]
            for current in current_candidates:
                asof = current["ann_date"]
                previous_period = _previous_report_period(period)
                previous = _latest_event_before(by_period.get(previous_period, []), asof)
                if previous is None:
                    continue
                previous_previous_period = _previous_report_period(previous_period)
                previous_previous = _latest_event_before(
                    by_period.get(previous_previous_period, []), asof
                )
                current_cumulative = float(current["raw_value"])
                previous_cumulative = float(previous["raw_value"])
                if period.month == 3:
                    current_value = current_cumulative
                else:
                    current_value = current_cumulative - previous_cumulative
                if previous_period.month == 3:
                    previous_value = previous_cumulative
                else:
                    if previous_previous is None:
                        continue
                    previous_value = previous_cumulative - float(previous_previous["raw_value"])
                if previous_value == 0:
                    continue
                rows.append(
                    {
                        "metric_id": output_metric,
                        "ts_code": ts_code,
                        "report_period": current["report_period"],
                        "ann_date": current["ann_date"],
                        "source_revision_id": (
                            f"current:{current['source_revision_id']}|prior:{previous['source_revision_id']}"
                        ),
                        "raw_value": (current_value - previous_value) / previous_value,
                    }
                )
    return pd.DataFrame(rows, columns=sorted(required)).sort_values(
        ["ts_code", "report_period", "ann_date", "source_revision_id"], ignore_index=True
    )


def _latest_event_before(
    candidates: list[dict[str, object]], asof: date
) -> dict[str, object] | None:
    eligible = [item for item in candidates if item["ann_date"] <= asof]
    if not eligible:
        return None
    return max(eligible, key=lambda item: (item["ann_date"], str(item["source_revision_id"])))


def _shift_report_period_one_year(report_period: date) -> date:
    try:
        return report_period.replace(year=report_period.year - 1)
    except ValueError:
        # Financial report periods are normally quarter ends, but retain a
        # deterministic fallback for a leap-day source record.
        return report_period.replace(year=report_period.year - 1, day=28)


def _previous_report_period(report_period: date) -> date:
    month = report_period.month - 3
    year = report_period.year
    while month <= 0:
        month += 12
        year -= 1
    if month in (3, 6, 9, 12):
        day = {3: 31, 6: 30, 9: 30, 12: 31}[month]
    else:
        day = 28
    return report_period.replace(year=year, month=month, day=day)


def _query_universe(
    session: V2WriteSession,
    *,
    start: date,
    end: date,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for period in pd.period_range(start, end, freq="M"):
        frame = read_latest_partition(
            session,
            object_name="universe_daily_latest",
            logical_partition=str(period),
            columns=("batch_id", "signal_date", "ts_code", "is_member"),
        )
        parts.append(frame)
    universe = pd.concat(parts, ignore_index=True)
    universe["signal_date"] = pd.to_datetime(universe["signal_date"], errors="raise").dt.date
    universe["ts_code"] = universe["ts_code"].astype(str)
    return universe.loc[
        universe["is_member"].fillna(False).astype(bool)
        & universe["ts_code"].str.endswith((".SH", ".SZ")),
        ["signal_date", "ts_code"],
    ].drop_duplicates()


def _query_calendar(
    source: YesLabReadOnlySession,
    *,
    start: date,
    end: date,
) -> TradingCalendar:
    history_start = date(start.year - 1, 1, 1)
    result = source.query(
        "select cal_date from loadTable('dfs://YesLab','trade_calendar') "
        "where exchange='SSE' and is_open=1 and cal_date between "
        f"date('{history_start:%Y.%m.%d}') and date('{end.replace(year=end.year + 1, month=1, day=31):%Y.%m.%d}') "
        "order by cal_date"
    )
    if not isinstance(result, pd.DataFrame):
        raise RuntimeError("trading-calendar query did not return a DataFrame")
    return TradingCalendar.from_dates(
        pd.to_datetime(result["cal_date"], errors="raise").dt.date.tolist()
    )


def _quality_summary(frame: pd.DataFrame, universe: pd.DataFrame) -> dict[str, object]:
    counts = frame.groupby("metric_id", sort=True).size().to_dict()
    expected = len(universe)
    coverage = {metric: count / expected for metric, count in counts.items()}
    if frame.duplicated(["signal_date", "ts_code", "metric_id"], keep=False).any():
        raise ValueError("fundamental PIT snapshot has duplicate business keys")
    if (frame["available_date"] > frame["signal_date"]).any():
        raise ValueError("fundamental PIT snapshot contains PIT violations")
    if (frame["stale_days"] > 180).any():
        raise ValueError("fundamental PIT snapshot contains stale values")
    return {
        "row_count": len(frame),
        "expected_universe_rows": expected,
        "coverage_by_metric": coverage,
        "minimum_coverage": min(coverage.values(), default=0.0),
    }


def _publish(
    session: V2WriteSession,
    frame: pd.DataFrame,
    *,
    chunk_rows: int = DEFAULT_PUBLISH_CHUNK_ROWS,
) -> None:
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    schema = session.query(
        f"schema(loadTable('{session.target_database}','{OBJECT_NAME}')).colDefs"
    )
    if not isinstance(schema, pd.DataFrame):
        raise RuntimeError("fundamental PIT target schema query did not return a DataFrame")
    types = dict(zip(schema["name"].astype(str), schema["typeString"].astype(str), strict=True))
    casts = {"DATE": "date", "INT": "int", "DOUBLE": "double", "UUID": "uuid", "TIMESTAMP": "timestamp"}
    for chunk in _iter_publish_chunks(frame, chunk_rows=chunk_rows):
        if _chunk_already_published(session, chunk):
            continue
        stage = f"fundamental_pit_{uuid.uuid4().hex[:16]}"
        session.upload_frame(stage, chunk)
        selected = ", ".join(
            f"{casts[types[column]]}({column}) as {column}"
            if types.get(column) in casts
            else column
            for column in chunk.columns
        )
        session.execute(
            f"{stage}_typed = select {selected} from {stage}\n"
            f"tableInsert{{loadTable('{session.target_database}','{OBJECT_NAME}')}}({stage}_typed)\n"
            f"undef(`{stage}, VAR)\nundef(`{stage}_typed, VAR)"
        )


def _iter_publish_chunks(frame: pd.DataFrame, *, chunk_rows: int) -> list[pd.DataFrame]:
    """Split by metric/month first, then cap each DolphinDB insert size."""
    working = frame.copy(deep=False)
    working["_logical_month"] = pd.to_datetime(
        working["signal_date"], errors="raise"
    ).dt.strftime("%Y-%m")
    chunks: list[pd.DataFrame] = []
    for _, group in working.groupby(["metric_id", "_logical_month"], sort=True):
        group = group.drop(columns="_logical_month")
        for start in range(0, len(group), chunk_rows):
            chunks.append(group.iloc[start : start + chunk_rows].copy())
    return chunks


def _chunk_already_published(session: V2WriteSession, chunk: pd.DataFrame) -> bool:
    """Skip only an exact business-key/data-version match; reject conflicts."""
    start = pd.Timestamp(chunk["signal_date"].min()).strftime("%Y.%m.%d")
    end = pd.Timestamp(chunk["signal_date"].max()).strftime("%Y.%m.%d")
    metric = str(chunk["metric_id"].iloc[0])
    existing = session.query(
        "select signal_date,ts_code,metric_id,data_version from "
        f"loadTable('{session.target_database}','{OBJECT_NAME}') "
        f"where metric_id=`{metric} and signal_date between date('{start}') and date('{end}')"
    )
    if not isinstance(existing, pd.DataFrame):
        raise RuntimeError("fundamental PIT chunk preflight did not return a DataFrame")
    if existing.empty:
        return False
    keys = ["signal_date", "ts_code", "metric_id"]
    left = chunk.loc[:, keys + ["data_version"]].copy()
    right = existing.loc[:, keys + ["data_version"]].copy()
    for frame in (left, right):
        frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="raise").dt.date
        frame["ts_code"] = frame["ts_code"].astype(str)
        frame["metric_id"] = frame["metric_id"].astype(str)
        frame["data_version"] = frame["data_version"].astype(str)
    matched = left.merge(right, on=keys, how="left", suffixes=("_incoming", "_existing"), validate="one_to_one")
    if matched["data_version_existing"].notna().sum() == 0:
        return False
    if matched["data_version_existing"].isna().any():
        raise RuntimeError(
            f"fundamental PIT chunk is partially present: metric={metric}, range={start}/{end}"
        )
    if (matched["data_version_incoming"] == matched["data_version_existing"]).all():
        return True
    raise RuntimeError(
        f"fundamental PIT chunk conflicts with existing rows: metric={metric}, range={start}/{end}"
    )

def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    client = _connect()
    source = YesLabReadOnlySession(client)
    target = V2WriteSession(client, target_database=TARGET_DATABASE)
    events = _query_events(source, start=args.start, end=args.end, metrics=args.metric)
    universe = _query_universe(target, start=args.start, end=args.end)
    calendar = _query_calendar(source, start=args.start, end=args.end)
    watermark = hashlib.sha256(
        json.dumps(
            {
                "metrics": args.metric,
                "start": args.start.isoformat(),
                "end": args.end.isoformat(),
                "event_rows": len(events),
                "universe_rows": len(universe),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    data_version = hashlib.sha256(("FUNDAMENTAL_PIT_V1|" + watermark).encode()).hexdigest()
    frame = build_snapshot_frame(
        events,
        universe,
        calendar,
        data_version=data_version,
        source_watermark_id=watermark,
        run_id=str(uuid.uuid4()),
    )
    summary = {
        "scope": "FUNDAMENTAL_PIT_SNAPSHOT_V1",
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "metrics": list(args.metric),
        "database_writes": bool(args.publish),
        "locked_test_read": False,
        "data_version": data_version,
        **_quality_summary(frame, universe),
    }
    if args.publish:
        _publish(target, frame, chunk_rows=args.publish_chunk_rows)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
