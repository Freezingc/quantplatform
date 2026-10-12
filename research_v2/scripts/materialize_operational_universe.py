"""Rebuild cn_liquid_a membership for dates the published pipeline has not reached.

WHY THIS EXISTS

``universe_daily`` stops at 2026-07-31, and so do ``tradability_daily`` and ``market_daily_clean``,
while the Tushare market source runs on to 2026-10-08. Every date after that has no published
universe and therefore cannot be scored: the models' output is a *within-date* percentile rank, and
a rank computed over a cross-section that was never published is not a decision. The owner's point
is that the raw data for those dates does exist, and it does -- it is on Tushare.

This script rebuilds membership for such a window through the same code the platform used for the
published table: ``add_market_history_features`` for ``valid_days_20``/``adv20``,
``build_tradability_daily`` for the tradability contract, ``_active_boards`` for board membership,
and ``build_universe_snapshot`` with the same ``cn_liquid_a`` v1 rules (allowed boards, 252 trading
days of listing age, 15 valid days in 20, top 80% by ADV20). No rule is re-derived here; if a rule
changes in ``UniverseRules`` the rebuild changes with it.

WHAT MAKES IT TRUSTWORTHY

A rebuild is only worth using if it can be checked, so the script refuses to write until it has
reproduced the *published* universe on an overlapping window inside the published range. The
agreement is printed and recorded in the sidecar, and ``--minimum-member-agreement`` defaults to
1.0: a cross-section that differs from the published one by even one member changes every rank
downstream, so the gate has to be exact until someone decides otherwise on the evidence.

WHAT IT DOES NOT DO

It writes a parquet, never ``universe_daily``. It does not run the ingestion/publish machinery, does
not mint a ``data_version``, and does not read any label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "research_v2" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "research_v2" / "src"))

from materialize_operational_factors import (  # noqa: E402
    as_plain_date,
    query_open_dates,
)

from research_v2.materializers.universe import _active_boards  # noqa: E402
from research_v2.pit.calendar import TradingCalendar  # noqa: E402
from research_v2.pit.market import add_market_history_features  # noqa: E402
from research_v2.universe.builder import (  # noqa: E402
    UniverseDefinition,
    build_universe_snapshot,
)
from research_v2.universe.rules import UniverseRules  # noqa: E402
from research_v2.universe.tradability import build_tradability_daily  # noqa: E402

SUSPENSION_COLUMNS = ["trade_date", "ts_code", "suspend_type"]
SECURITY_SCD_COLUMNS = [
    "ts_code",
    "board",
    "list_date",
    "delist_date",
    "is_st",
    "valid_from",
    "valid_to",
]
#: Same identifier and the same rules object the published ``cn_liquid_a`` v1 was built with.
DEFINITION = UniverseDefinition(
    "cn_liquid_a", 1, UniverseRules(retained_adv20_fraction=0.80)
)
#: Calendar days of market history fetched before the first built date, so the 20-trading-day
#: liquidity features are defined from the window's first day rather than its twenty-first.
WARMUP_CALENDAR_DAYS = 60
CALENDAR_START = date(1990, 1, 1)


def derive_suspensions(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize Tushare ``suspend_d`` rows into the tradability contract's columns.

    Only suspension rows are kept. ``suspend_d`` also returns resumption rows ("R"), and a
    resumption row would mark a security suspended on the very day it started trading again.
    """
    required = {"ts_code", "trade_date", "suspend_type"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"suspension frame is missing columns: {missing}")
    frame = raw.loc[:, ["ts_code", "trade_date", "suspend_type"]].copy()
    frame["ts_code"] = frame["ts_code"].astype(str)
    frame["suspend_type"] = frame["suspend_type"].astype(str)
    frame = frame.loc[frame["suspend_type"].str.upper().str.contains("S")]
    values = frame["trade_date"]
    if pd.api.types.is_datetime64_any_dtype(values):
        parsed = pd.to_datetime(values, errors="coerce")
    else:
        rendered = values.astype(str).str.replace("-", "", regex=False)
        parsed = pd.to_datetime(rendered, format="%Y%m%d", errors="coerce")
    frame["trade_date"] = parsed.dt.date
    frame = frame.loc[frame["trade_date"].notna()]
    frame = frame.drop_duplicates(["trade_date", "ts_code"], keep="last")
    return frame.loc[:, SUSPENSION_COLUMNS].sort_values(
        ["trade_date", "ts_code"], ignore_index=True
    )


def validate_against_official(
    derived: pd.DataFrame, official: pd.DataFrame
) -> dict[str, float | int]:
    """Per-date membership agreement between a rebuild and the published universe.

    Agreement is the Jaccard index of the two member sets, so it is 1.0 only when the date's
    cross-section is reproduced exactly. Dates present in one side and absent from the other count
    as full disagreement rather than being skipped, because a missing date is the failure this
    rebuild is most likely to produce.
    """
    keys = ["signal_date", "ts_code"]

    def member_sets(frame: pd.DataFrame) -> dict[date, set[str]]:
        columns = [*keys, "is_member"] if "is_member" in frame.columns else list(keys)
        normalized = frame.loc[:, columns].copy()
        if "is_member" in normalized.columns:
            normalized = normalized.loc[normalized["is_member"].astype(bool)]
        normalized["signal_date"] = [
            as_plain_date(value) for value in normalized["signal_date"]
        ]
        normalized["ts_code"] = normalized["ts_code"].astype(str)
        return {
            day: set(group["ts_code"]) for day, group in normalized.groupby("signal_date")
        }

    left, right = member_sets(derived), member_sets(official)
    dates = sorted(set(left) | set(right))
    agreements: list[float] = []
    count_diffs: list[int] = []
    exact = 0
    for day in dates:
        derived_members = left.get(day, set())
        official_members = right.get(day, set())
        union = derived_members | official_members
        agreements.append(
            len(derived_members & official_members) / len(union) if union else 1.0
        )
        count_diffs.append(abs(len(derived_members) - len(official_members)))
        if derived_members == official_members:
            exact += 1
    return {
        "dates": len(dates),
        "exact_dates": exact,
        "min_member_agreement": min(agreements) if agreements else 0.0,
        "mean_member_agreement": (sum(agreements) / len(agreements)) if agreements else 0.0,
        "member_count_max_abs_diff": max(count_diffs) if count_diffs else 0,
    }


def require_member_agreement(
    summary: dict[str, float | int], *, minimum: float
) -> None:
    """Refuse to publish a rebuild that does not reproduce the published membership."""
    measured = float(summary.get("min_member_agreement", 0.0))
    if measured < minimum:
        raise ValueError(
            f"universe rebuild reproduces the published members at only {measured:.6f} "
            f"agreement (minimum {minimum:.6f}) over {summary.get('dates')} dates "
            f"({summary.get('exact_dates')} exact); refusing to write"
        )


def load_security_scd(client: object) -> pd.DataFrame:
    """The published security master, collapsed to one state per (ts_code, board).

    The stored table keeps every publication, so several identical rows describe the same security
    and ``_active_security_states`` would reject the duplication as a conflict. Collapsing to one
    row per distinct interval start, newest publication winning, keeps the state table intact: see
    ``collapse_security_master`` for why keeping only the newest row per board is wrong.
    """
    frame = client.run(
        "select ts_code, board, list_date, delist_date, is_st, valid_from, valid_to "
        ", data_version from loadTable('dfs://QuantResearchV2Test','security_master_scd')"
    )
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise RuntimeError("security master query returned no rows")
    publications = client.run(
        "select data_version, published_at from "
        "loadTable('dfs://QuantResearchV2Test','published_version') "
        "where object_name='security_master_scd'"
    )
    if not isinstance(publications, pd.DataFrame) or publications.empty:
        raise RuntimeError("security master publication log returned no rows")
    merged = frame.merge(publications, on="data_version", how="left")
    return collapse_security_master(merged)


def collapse_security_master(scd: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct interval start, with the newest publication winning ties.

    Two failure modes live in this table. The same state is republished, so exact duplicates must
    collapse or ``_active_security_states`` raises a conflict. And a state can be *corrected*
    inside one interval start, so the newest publication has to win rather than an arbitrary row.

    What must NOT happen is collapsing to one row per board: a security whose board record gained a
    later ``valid_from`` would then lose the interval that was still in force before it, no board
    would resolve for the earlier dates, and the security would be excluded as OUTSIDE_BOARD while
    the published universe still carried it. That is exactly what 600768.SH and 688309.SH showed on
    2026-07-20.
    """
    required = {"ts_code", "board", "valid_from", "valid_to", "published_at"}
    missing = sorted(required.difference(scd.columns))
    if missing:
        raise ValueError(f"security master is missing columns: {missing}")
    frame = scd.copy(deep=True)
    frame["ts_code"] = frame["ts_code"].astype(str)
    frame["board"] = frame["board"].astype(str)
    frame["_published"] = pd.to_datetime(frame["published_at"], errors="coerce")
    frame = frame.sort_values("_published").drop_duplicates(
        ["ts_code", "board", "valid_from"], keep="last"
    )
    frame = frame.drop(columns=["_published"])
    available = [column for column in SECURITY_SCD_COLUMNS if column in frame.columns]
    return frame.loc[:, available].reset_index(drop=True)


def load_official_universe(client: object, start: date, end: date) -> pd.DataFrame:
    frame = client.run(
        "select signal_date, ts_code, is_member from "
        "loadTable('dfs://QuantResearchV2Test','universe_daily') "
        f"where signal_date between {start:%Y.%m.%d} and {end:%Y.%m.%d} "
        "and universe_id='cn_liquid_a' and universe_version=1"
    )
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise RuntimeError("official universe query returned no rows")
    return frame


def load_calendar(client: object, end: date) -> TradingCalendar:
    frame = client.run(
        "select cal_date from loadTable('dfs://YesLab','trade_calendar') "
        "where exchange='SSE' and is_open=1 "
        f"and cal_date between {CALENDAR_START:%Y.%m.%d} and {end:%Y.%m.%d} order by cal_date"
    )
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise RuntimeError("trade calendar query returned no rows")
    return TradingCalendar.from_dates(
        [as_plain_date(value) for value in pd.to_datetime(frame["cal_date"]).dropna()]
    )


def fetch_suspensions(start: date, end: date, token: str) -> pd.DataFrame:
    """Tushare suspension rows for the window; the platform stores no raw suspension table."""
    if not token.strip():
        raise ValueError("QRV2_TUSHARE_TOKEN is required to fetch suspensions")
    import tushare as ts  # type: ignore[import-untyped]

    pro = ts.pro_api(token)
    raw = pro.suspend_d(
        suspend_type="S", start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d")
    )
    if not isinstance(raw, pd.DataFrame):
        raise RuntimeError("Tushare suspend_d did not return a frame")
    return derive_suspensions(raw)


def build_membership(
    market: pd.DataFrame,
    security_scd: pd.DataFrame,
    suspensions: pd.DataFrame,
    calendar: TradingCalendar,
    *,
    windows: tuple[tuple[date, date], ...],
) -> pd.DataFrame:
    """Tradability then universe snapshots, for every date inside ``windows``."""
    enriched = add_market_history_features(market)
    market_features = enriched.loc[
        :, ["trade_date", "ts_code", "valid_days_20", "adv20"]
    ].copy()
    market_features["trade_date"] = [
        as_plain_date(value) for value in market_features["trade_date"]
    ]
    tradability = build_tradability_daily(
        market_features, security_scd, suspensions, calendar
    )
    wanted = {
        day for start, end in windows for day in calendar.window(start, end)
    }
    snapshots: list[pd.DataFrame] = []
    for signal_date, records in tradability.groupby("trade_date", sort=True):
        day = as_plain_date(signal_date)
        if day not in wanted:
            continue
        candidates = records.copy(deep=True)
        candidates["trade_date"] = day
        candidates["board"] = candidates["ts_code"].map(_active_boards(security_scd, day))
        candidates["board"] = candidates["board"].fillna("OTHER")
        snapshots.append(build_universe_snapshot(candidates, DEFINITION))
    if not snapshots:
        raise RuntimeError("no universe snapshots were built for the requested windows")
    return pd.concat(snapshots, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-start", type=date.fromisoformat, default=date(2026, 7, 1))
    parser.add_argument("--validate-end", type=date.fromisoformat, default=date(2026, 7, 31))
    parser.add_argument("--minimum-member-agreement", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.start > args.end:
        raise SystemExit("--start must not be after --end")
    if args.validate_start > args.validate_end:
        raise SystemExit("--validate-start must not be after --validate-end")

    import audit_formula_replay_tushare as tushare_replay
    import materialize_frozen_factor_daily as frozen_mod

    client = tushare_replay._connect()
    try:
        security_scd = load_security_scd(client)
        calendar = load_calendar(client, max(args.end, args.validate_end))
        official = load_official_universe(client, args.validate_start, args.validate_end)
        first = min(args.start, args.validate_start) - timedelta(days=WARMUP_CALENDAR_DAYS)
        open_dates = query_open_dates(client, first, max(args.end, args.validate_end))
        market = tushare_replay.fetch_full_tushare_shsz_market(
            open_dates,
            token=os.getenv("QRV2_TUSHARE_TOKEN", ""),
            cache_dir=frozen_mod.CACHE,
            include_daily_basic=False,
        )
        market["trade_date"] = pd.to_datetime(market["trade_date"])
        market["ts_code"] = market["ts_code"].astype(str)
        print(
            f"market rows: {len(market)} over {market['trade_date'].nunique()} dates "
            f"from {first}",
            flush=True,
        )
    finally:
        client.close()

    suspensions = fetch_suspensions(
        first, max(args.end, args.validate_end), os.getenv("QRV2_TUSHARE_TOKEN", "")
    )
    print(f"suspension rows: {len(suspensions)}", flush=True)
    membership = build_membership(
        market,
        security_scd,
        suspensions,
        calendar,
        windows=((args.validate_start, args.validate_end), (args.start, args.end)),
    )
    derived_validation = membership.loc[
        membership["signal_date"].between(args.validate_start, args.validate_end)
    ]
    summary = validate_against_official(derived_validation, official)
    print(f"validation against published universe: {summary}", flush=True)
    require_member_agreement(summary, minimum=args.minimum_member_agreement)

    requested = membership.loc[
        membership["signal_date"].between(args.start, args.end)
    ].reset_index(drop=True)
    if requested.empty:
        raise SystemExit(f"no universe rows built inside {args.start} .. {args.end}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    requested.to_parquet(args.output, index=False)
    sidecar = {
        "scope": "OPERATIONAL_UNIVERSE_REBUILD_V1",
        "note": "rebuilt cn_liquid_a membership; not the published universe_daily table",
        "universe_id": DEFINITION.universe_id,
        "universe_version": DEFINITION.version,
        "rules": {
            "allowed_boards": sorted(DEFINITION.rules.allowed_boards),
            "minimum_listing_trading_days": DEFINITION.rules.minimum_listing_trading_days,
            "minimum_valid_days_20": DEFINITION.rules.minimum_valid_days_20,
            "retained_adv20_fraction": DEFINITION.rules.retained_adv20_fraction,
        },
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "signal_dates": int(requested["signal_date"].nunique()),
        "members": int(requested["is_member"].astype(bool).sum()),
        "validation_window": [args.validate_start.isoformat(), args.validate_end.isoformat()],
        "validation": summary,
        "minimum_member_agreement": args.minimum_member_agreement,
        "output": str(args.output),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "touched_locked_test": args.end >= date(2026, 1, 1),
    }
    sidecar_path = args.output.with_suffix(".meta.json")
    sidecar_path.write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {len(requested)} universe rows "
        f"({sidecar['members']} members over {sidecar['signal_dates']} dates) to {args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
