"""Materialize the frozen factor set's daily values into factor_daily.

Replays each frozen factor over the requested window from Tushare market data
(isolated canonical versions; the 206 median-repair and 443 reg30_max20 repair
variant use their approved formulas), restricts each signal date to the
official cn_liquid_a universe (latest data_version per date), and appends the
values to factor_daily (created by migration 016). Idempotent per
(factor_id, signal_date range): re-running the same window skips factors whose
rows already exist. Read-only with respect to all existing tables; writes only
factor_daily in the V2 target database.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "research_v2" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "research_v2" / "src"))

import audit_formula_replay_consistency as replay  # noqa: E402
from audit_canonical_candidate_pairs import materialize_alpha206_median_repair  # noqa: E402
from audit_formula_replay_tushare import (  # noqa: E402
    TushareMarketSession,
    _calculator_market_start,
    _connect,
    _query_open_dates,
    fetch_full_tushare_shsz_market,
    required_market_fields,
    requires_daily_basic,
)
from audit_tushare_pit_industry import VersionedPITIndustryProvider  # noqa: E402
from repair_alpha443_window_scan import compute_gp074_variant  # noqa: E402

CACHE = REPO_ROOT / "research_v2" / "artifacts" / "tushare_cache"
DIAG = REPO_ROOT / "research_v2" / "artifacts" / "diagnostics"
TARGET_DATABASE = "dfs://QuantResearchV2Test"
REPAIR_VARIANT_443 = "yeslab_443_reg30_max20_v1"
MEDIAN_206 = "yeslab_206"
PIT_FACTORS = {"yeslab_357", "yeslab_363", "yeslab_471"}
UNIVERSE_ID = "cn_liquid_a"
PIT_CUTOVER = date(2021, 12, 10)


def load_frozen_factors() -> list[str]:
    frozen = json.loads((DIAG / "factor_set_frozen_v1.json").read_text(encoding="utf-8"))
    return frozen["factors"]


def replay_factor_values(fid: str, session, market: pd.DataFrame, provider, start: date, end: date) -> pd.DataFrame:
    """Return a long frame (trade_date, ts_code, replay_value) for one factor."""
    if fid == MEDIAN_206:
        return materialize_alpha206_median_repair(market, start, end)
    if fid == REPAIR_VARIANT_443:
        wide = {
            "amount": market.pivot_table(index="trade_date", columns="ts_code", values="amount", aggfunc="last").sort_index(),
            "pct_chg": market.pivot_table(index="trade_date", columns="ts_code", values="pct_chg", aggfunc="last").sort_index(),
        }
        factor_wide = compute_gp074_variant(wide, 30, 20)
        long = (
            factor_wide.rename_axis("trade_date").reset_index()
            .melt(id_vars=["trade_date"], var_name="ts_code", value_name="replay_value")
        )
        long["trade_date"] = pd.to_datetime(long["trade_date"])
        return long.loc[long["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end))].reset_index(drop=True)
    definition = replay.REPLAY_DEFINITIONS[fid]
    prov = provider if fid in PIT_FACTORS else None
    return replay._replay_factor(session, definition, start, end, industry_provider=prov)


def load_official_universe(client, start: date, end: date) -> dict:
    uni = client.run(
        "select signal_date, ts_code, data_version "
        f"from loadTable('{TARGET_DATABASE}','universe_daily') "
        f"where signal_date between {start:%Y.%m.%d} and {end:%Y.%m.%d} "
        f"and universe_id='{UNIVERSE_ID}' and is_member=1"
    )
    if uni is None or len(uni) == 0:
        raise RuntimeError("no cn_liquid_a universe members in window")
    uni["signal_date"] = pd.to_datetime(uni["signal_date"]).dt.normalize()
    uni["ts_code"] = uni["ts_code"].astype(str)
    latest = uni.groupby("signal_date")["data_version"].max().rename("latest_version")
    uni = uni.merge(latest, on="signal_date", how="left")
    uni = uni[uni["data_version"] == uni["latest_version"]]
    return {dt: set(grp["ts_code"]) for dt, grp in uni.groupby("signal_date")}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2020, 1, 2))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2024, 12, 31))
    parser.add_argument("--factors", type=str, default=None, help="comma-separated subset (default all frozen)")
    parser.add_argument("--apply", action="store_true", help="write to factor_daily (default dry-run)")
    args = parser.parse_args(argv)

    frozen = load_frozen_factors()
    factors = args.factors.split(",") if args.factors else frozen
    factors = [f for f in factors if f in frozen]
    print(f"frozen factors: {len(frozen)}; materializing: {len(factors)}")

    client = _connect()
    import dolphindb as ddb
    from research_v2.db.migrations import MigrationRunner
    from research_v2.db.session import V2WriteSession
    session_write = V2WriteSession(client, target_database=TARGET_DATABASE)
    MigrationRunner(session_write, migration_directory=REPO_ROOT / "research_v2" / "migrations").apply()

    definitions = [replay.REPLAY_DEFINITIONS[f] for f in factors if f not in (MEDIAN_206, REPAIR_VARIANT_443)]
    market_start = args.start - timedelta(days=120)
    if definitions:
        market_start = min(_calculator_market_start(client, d, args.start, args.end) for d in definitions) - timedelta(days=90)
    open_dates = _query_open_dates(client, market_start, args.end)
    fields = required_market_fields(definitions) if definitions else ("amount", "pct_chg")
    fields = tuple(sorted(set(fields) | {"amount", "pct_chg"}))
    market = fetch_full_tushare_shsz_market(open_dates, token=os.getenv("QRV2_TUSHARE_TOKEN", ""), cache_dir=CACHE, include_daily_basic=requires_daily_basic(fields))
    market["trade_date"] = pd.to_datetime(market["trade_date"])
    market["ts_code"] = market["ts_code"].astype(str)
    print(f"market rows: {len(market)}", flush=True)

    session = TushareMarketSession(client, market, tuple(sorted(market["ts_code"].unique())))
    provider = VersionedPITIndustryProvider.from_cache(CACHE, cutover=PIT_CUTOVER)
    universe = load_official_universe(client, args.start, args.end)
    print(f"universe dates: {len(universe)}", flush=True)

    watermark = hashlib.sha256(json.dumps(
        {"scope": "FROZEN_FACTOR_DAILY_V1", "start": args.start.isoformat(), "end": args.end.isoformat()},
        sort_keys=True).encode()).hexdigest()
    data_version = hashlib.sha256((watermark + str(args.start) + str(args.end)).encode()).hexdigest()

    for fid in factors:
        # idempotency: skip if rows already exist for this factor+window
        existing = session_write.query(
            f"select count(*) as n from loadTable('{TARGET_DATABASE}','factor_daily') "
            f"where factor_id='{fid}' and signal_date between {args.start:%Y.%m.%d} and {args.end:%Y.%m.%d} "
            f"and data_version='{data_version}'")
        n_existing = int(existing["n"][0]) if isinstance(existing, pd.DataFrame) and len(existing) else 0
        if n_existing:
            print(f"{fid}: skipped ({n_existing} rows already present)")
            continue

        values = replay_factor_values(fid, session, market, provider, args.start, args.end)
        values["trade_date"] = pd.to_datetime(values["trade_date"]).dt.normalize()
        values["ts_code"] = values["ts_code"].astype(str)
        # restrict to official universe per date
        keep_mask = [
            (ts in universe.get(dt, set())) and np.isfinite(v)
            for dt, ts, v in zip(values["trade_date"], values["ts_code"], values["replay_value"])
        ]
        values = values.loc[keep_mask]
        if values.empty:
            print(f"{fid}: no universe-aligned finite values - skipped")
            continue

        run_id = str(uuid.uuid4())
        rec_rows = pd.DataFrame({
            "signal_date": values["trade_date"].dt.date,
            "ts_code": values["ts_code"].astype(str),
            "factor_id": fid,
            "factor_version": 1,
            "universe_id": UNIVERSE_ID,
            "value": values["replay_value"].astype(float),
            "run_id": run_id,
            "data_version": data_version,
            "created_at": pd.Timestamp.utcnow().tz_localize(None),
            "source_watermark_id": watermark,
            "record_hash": "",
        })
        _dates = rec_rows["signal_date"].astype(str).to_numpy()
        _codes = rec_rows["ts_code"].to_numpy()
        _md5 = hashlib.md5
        rec_rows["record_hash"] = [
            _md5(f"{fid}|{d}|{c}|{data_version}".encode()).hexdigest()
            for d, c in zip(_dates, _codes)
        ]
        print(f"{fid}: {len(rec_rows)} universe-aligned rows", flush=True)
        if not args.apply:
            continue
        stage = f"factorDailyStage_{uuid.uuid4().hex[:12]}"
        session_write.upload_frame(stage, rec_rows)
        session_write.execute(
            f"typed = select date(signal_date) as signal_date, symbol(ts_code) as ts_code, "
            f"symbol(factor_id) as factor_id, int(factor_version) as factor_version, "
            f"symbol(universe_id) as universe_id, double(value) as value, uuid(run_id) as run_id, "
            f"string(data_version) as data_version, timestamp(created_at) as created_at, "
            f"string(source_watermark_id) as source_watermark_id, string(record_hash) as record_hash from {stage}\n"
            f"tableInsert{{loadTable('{TARGET_DATABASE}','factor_daily')}}(typed)\n"
            f"undef(`{stage}, VAR)\nundef(`typed, VAR)")
        print(f"{fid}: written")
    if not args.apply:
        print("\nDry-run only. Re-run with --apply to write.")
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
