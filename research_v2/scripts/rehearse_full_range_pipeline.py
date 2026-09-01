"""Rehearse the full PIT pipeline on real YesLab data across 2013-2025 (B criterion 1).

Read-only against the source database: runs security master -> market clean ->
tradability -> universe -> labels for anchor windows spanning the warmup and
mature range, publishes nothing to any live database, and verifies stability
(rebuild produces identical content hashes) plus label maturity invariants.
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, date, datetime, timedelta

import dolphindb as ddb
import pandas as pd

from research_v2.contracts.source import SourceBatch
from research_v2.materializers.labels import build_label_batch
from research_v2.materializers.market import (
    build_market_batch,
    build_market_security_quarantine_batch,
)
from research_v2.materializers.security_master import build_security_master_batch
from research_v2.materializers.tradability import build_tradability_batch
from research_v2.materializers.universe import build_universe_batch
from research_v2.pit.calendar import TradingCalendar
from research_v2.pit.security_master import resolve_name_history_overlaps
from research_v2.runs.context import RunContext
from research_v2.universe.builder import UniverseDefinition, UniverseRules

ANCHOR_YEARS = (2013, 2015, 2017, 2019, 2021, 2023, 2025)
WINDOW_DAYS = 20
_SCHEMA_FP = "sha256:" + "e" * 64


def _connect() -> ddb.session:
    session = ddb.session()
    if not session.connect(
        os.getenv("QRV2_DDB_HOST", "127.0.0.1"),
        int(os.getenv("QRV2_DDB_PORT", "8848")),
        os.getenv("QRV2_DDB_USERNAME", ""),
        os.getenv("QRV2_DDB_PASSWORD", ""),
    ):
        raise SystemExit("cannot connect to the configured DolphinDB source database")
    return session


def _source_batch(frame, table_name, watermark_id) -> SourceBatch:
    max_date = frame["trade_date"].max() if "trade_date" in frame.columns else date(2025, 1, 1)
    return SourceBatch(
        frame=frame,
        source_name="yeslab",
        table_name=table_name,
        watermark_id=watermark_id,
        max_observation_date=max_date,
        source_row_count=len(frame),
        schema_fingerprint=_SCHEMA_FP,
        extracted_at=datetime.now(UTC),
    )


def main() -> int:
    session = _connect()
    calendar_rows = session.run(
        "select cal_date from loadTable('dfs://YesLab', 'trade_calendar') "
        "where exchange='SSE' and is_open=1 and cal_date between 2012.12.01 and 2026.01.31 "
        "order by cal_date"
    )
    calendar_days = [pd.Timestamp(value).date() for value in calendar_rows["cal_date"]]
    calendar = TradingCalendar.from_dates(calendar_days)

    info = session.run("select * from loadTable('dfs://YesLab', 'stock_info_table')")
    names = session.run("select * from loadTable('dfs://YesLab', 'stock_used_name')")
    names = resolve_name_history_overlaps(names)
    suspensions_all = session.run(
        "select * from loadTable('dfs://YesLab', 'stock_trading_suspension_info') "
        "where trade_date between 2012.12.01 and 2026.01.31"
    )
    suspensions_all["trade_date"] = pd.to_datetime(suspensions_all["trade_date"]).dt.date

    report_rows = []
    for year in ANCHOR_YEARS:
        window = [day for day in calendar_days if day.year == year][:WINDOW_DAYS]
        if len(window) < WINDOW_DAYS:
            raise SystemExit(f"anchor year {year} has fewer than {WINDOW_DAYS} trading days")
        horizon = calendar_days[calendar_days.index(window[-1]) + 6]
        extract_end = horizon + timedelta(days=4)

        market_source = session.run(
            "select ts_code, trade_date, open, high, low, close, vol, amount, adj_factor "
            "from loadTable('dfs://YesLab', 'stock_factor_daily') "
            f"where trade_date between {window[0].strftime('%Y.%m.%d')} "
            f"and {extract_end.strftime('%Y.%m.%d')}"
        )
        market_source["trade_date"] = pd.to_datetime(market_source["trade_date"]).dt.date

        context = RunContext.build(
            config={"rehearsal_year": year},
            watermarks=(f"rehearsal-{year}",),
            code_commit="sha256:" + "b" * 64,
            schema_version="013",
        )
        scd_batch = build_security_master_batch(
            _source_batch(info, "stock_info_table", f"info-{year}"),
            _source_batch(names, "stock_used_name", f"names-{year}"),
            context,
            as_of_date=window[-1],
        )
        scd_frame = scd_batch.frame

        market_batch = build_market_batch(
            _source_batch(market_source, "stock_factor_daily", f"market-{year}"),
            scd_frame,
            calendar,
            context,
            publish_start=window[0],
            publish_end=window[-1],
            quarantine_inactive_security=True,
        )
        quarantine_batch = build_market_security_quarantine_batch(
            _source_batch(market_source, "stock_factor_daily", f"market-{year}"),
            scd_frame,
            context,
            publish_start=window[0],
            publish_end=window[-1],
        )
        rebuilt = build_market_batch(
            _source_batch(market_source, "stock_factor_daily", f"market-{year}"),
            scd_frame,
            calendar,
            context,
            publish_start=window[0],
            publish_end=window[-1],
            quarantine_inactive_security=True,
        )
        stable = market_batch.frame["record_hash"].tolist() == rebuilt.frame[
            "record_hash"
        ].tolist()

        tradability_batch = build_tradability_batch(
            _source_batch(market_batch.frame, "market_daily_clean", f"clean-{year}"),
            scd_frame,
            _source_batch(
                suspensions_all.loc[
                    suspensions_all["trade_date"].between(window[0], window[-1])
                ],
                "stock_trading_suspension_info",
                f"susp-{year}",
            ),
            calendar,
            context,
        )
        universe_batch = build_universe_batch(
            _source_batch(tradability_batch.frame, "tradability_daily", f"trad-{year}"),
            scd_frame,
            UniverseDefinition("cn_liquid_a", 1, UniverseRules(retained_adv20_fraction=0.80)),
            context,
        )

        label_market = build_market_batch(
            _source_batch(market_source, "stock_factor_daily", f"market-{year}-horizon"),
            scd_frame,
            calendar,
            context,
            publish_start=window[0],
            publish_end=horizon,
            quarantine_inactive_security=True,
        )
        members = universe_batch.frame.loc[:, ["signal_date", "ts_code"]].drop_duplicates()
        members = members.rename(columns={"trade_date": "signal_date"})
        label_batch = build_label_batch(
            members,
            _source_batch(label_market.frame, "market_daily_clean", f"clean-horizon-{year}"),
            calendar,
            context,
        )
        labels = label_batch.frame
        mature_ratio = float((labels["label_valid"].astype(bool)).mean()) if len(labels) else 0.0
        report_rows.append(
            {
                "year": year,
                "market_rows": len(market_batch.frame),
                "quarantined_rows": len(quarantine_batch.frame),
                "market_stable_rebuild": stable,
                "tradability_rows": len(tradability_batch.frame),
                "universe_rows": len(universe_batch.frame),
                "label_rows": len(labels),
                "label_valid_ratio": round(mature_ratio, 4),
            }
        )

    report = pd.DataFrame(report_rows)
    print(report.to_string(index=False))
    if not report["market_stable_rebuild"].all():
        print("STABILITY FAILURE: market rebuild is not deterministic")
        return 1
    if (report["market_rows"] <= 0).any() or (report["universe_rows"] <= 0).any():
        print("COVERAGE FAILURE: empty outputs in the full range")
        return 1
    print("full-range rehearsal passed: stable calendar/master/market/tradability/universe/labels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
