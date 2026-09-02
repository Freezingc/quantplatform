"""Replicate cn_liquid_a v1 universe membership for pre-2020 research windows.

Uses the official build_universe_snapshot code path with tradability_daily
(backfilled to 2014) and security_master_scd. Validates against the official
universe_daily for 2020-01 first (membership + rank agreement); only if the
validation passes does it emit research membership for 2019.

Read-only research artifact; no database writes, no locked-test reads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

import dolphindb as ddb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DIAG = REPO_ROOT / "research_v2" / "artifacts" / "diagnostics"
sys.path.insert(0, str(REPO_ROOT / "research_v2" / "src"))

from research_v2.materializers.universe import _active_boards  # noqa: E402
from research_v2.universe.builder import UniverseDefinition, UniverseRules, build_universe_snapshot  # noqa: E402

TRADABILITY_COLUMNS = [
    "trade_date", "ts_code", "is_listed", "is_st", "is_suspended",
    "listing_age_trading_days", "valid_days_20", "adv20", "reason_bitmap",
]
DEFINITION = UniverseDefinition("cn_liquid_a", 1, UniverseRules(retained_adv20_fraction=0.80))


def _connect():
    session = ddb.session()
    session.connect(
        os.getenv("QRV2_DDB_HOST", "10.176.55.211"),
        int(os.getenv("QRV2_DDB_PORT", "8848")),
        os.getenv("QRV2_DDB_USERNAME", "xiahongjie"),
        os.getenv("QRV2_DDB_PASSWORD", "123456"),
    )
    return session


def load_tradability(client, start: date, end: date) -> pd.DataFrame:
    cols = ",".join(TRADABILITY_COLUMNS)
    frame = client.run(
        f'''
        t = loadTable(database("dfs://QuantResearchV2Test"), "tradability_daily")
        select {cols} from t
        where trade_date >= {start:%Y.%m.%d} and trade_date <= {end:%Y.%m.%d}
        '''
    )
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["ts_code"] = frame["ts_code"].astype(str)
    return frame


def load_security_scd(client) -> pd.DataFrame:
    frame = client.run(
        '''
        t = loadTable(database("dfs://QuantResearchV2Test"), "security_master_scd")
        select ts_code, board, valid_from, valid_to from t
        '''
    )
    frame["ts_code"] = frame["ts_code"].astype(str)
    # Read-only dedup: the stored SCD contains duplicated publications.
    frame = frame.drop_duplicates(["ts_code", "board", "valid_from", "valid_to"]).copy()
    frame["_vf"] = pd.to_datetime(frame["valid_from"], errors="coerce")
    frame = frame.sort_values("_vf").drop_duplicates(["ts_code", "board"], keep="last").drop(columns=["_vf"])
    return frame


def build_members(tradability: pd.DataFrame, security_scd: pd.DataFrame) -> pd.DataFrame:
    snapshots = []
    for signal_date, records in tradability.groupby("trade_date", sort=True):
        resolved_date = signal_date.date() if hasattr(signal_date, "date") else signal_date
        candidates = records.copy(deep=True)
        candidates["board"] = candidates["ts_code"].map(_active_boards(security_scd, resolved_date))
        candidates["board"] = candidates["board"].fillna("OTHER")
        snapshots.append(build_universe_snapshot(candidates, DEFINITION))
    return pd.concat(snapshots, ignore_index=True)


def load_official_universe(client, start: date, end: date) -> pd.DataFrame:
    frame = client.run(
        f'''
        t = loadTable(database("dfs://QuantResearchV2Test"), "universe_daily")
        select signal_date, ts_code, is_member, membership_rank from t
        where signal_date >= {start:%Y.%m.%d} and signal_date <= {end:%Y.%m.%d}
          and universe_id = "cn_liquid_a" and universe_version = 1
        '''
    )
    frame["signal_date"] = pd.to_datetime(frame["signal_date"])
    frame["ts_code"] = frame["ts_code"].astype(str)
    return frame


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-start", type=date.fromisoformat, default=date(2020, 1, 2))
    parser.add_argument("--validate-end", type=date.fromisoformat, default=date(2020, 1, 31))
    parser.add_argument("--research-start", type=date.fromisoformat, default=date(2019, 1, 2))
    parser.add_argument("--research-end", type=date.fromisoformat, default=date(2019, 12, 31))
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    client = _connect()
    security_scd = load_security_scd(client)

    # validation vs official 2020 universe
    trad_val = load_tradability(client, args.validate_start, args.validate_end)
    replicated = build_members(trad_val, security_scd)
    official = load_official_universe(client, args.validate_start, args.validate_end)
    merged = replicated.merge(official, on=["signal_date", "ts_code"], how="outer",
                              suffixes=("_rep", "_off"), indicator=True)
    member_rep = merged["is_member_rep"].fillna(False).astype(bool)
    member_off = merged["is_member_off"].fillna(False).astype(bool)
    agreement = float((member_rep == member_off).mean())
    both_members = member_rep & member_off
    rank_rep = merged.loc[both_members, "membership_rank_rep"]
    rank_off = merged.loc[both_members, "membership_rank_off"]
    rank_exact = float((rank_rep.to_numpy() == rank_off.to_numpy()).mean()) if both_members.any() else float("nan")
    validation = {
        "window": f"{args.validate_start}/{args.validate_end}",
        "merged_rows": int(len(merged)),
        "membership_agreement": agreement,
        "member_rows": int(both_members.sum()),
        "membership_rank_exact_agreement": rank_exact,
        "passed": bool(agreement >= 0.999 and (not both_members.any() or rank_exact >= 0.999)),
    }
    print("validation:", json.dumps(validation))
    if not validation["passed"]:
        (args.output / "universe_validation_result.json").write_text(json.dumps(validation, indent=2) + "\n")
        print("VALIDATION FAILED - not emitting research membership")
        return 1

    # research membership for 2019
    trad_res = load_tradability(client, args.research_start, args.research_end)
    research_members = build_members(trad_res, security_scd)
    members_only = research_members.loc[research_members["is_member"].astype(bool),
                                        ["signal_date", "ts_code"]].drop_duplicates()
    members_only.to_parquet(args.output / "research_universe_members_2019.parquet", index=False)
    summary = {
        "validation": validation,
        "research_window": f"{args.research_start}/{args.research_end}",
        "research_member_rows": int(len(members_only)),
        "research_signal_dates": int(members_only["signal_date"].nunique()),
        "note": "Research-only cn_liquid_a membership replicated via official builder for historical-window evidence; not published.",
        "database_writes": False,
        "locked_test_read": False,
    }
    (args.output / "universe_validation_result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("research membership:", json.dumps({k: summary[k] for k in ("research_member_rows", "research_signal_dates")}))
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
