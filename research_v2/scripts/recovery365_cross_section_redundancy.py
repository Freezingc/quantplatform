"""Cross-sectional redundancy check for recovery candidate yeslab_365.

Replays 365 plus its industry-family relatives (357/363) and the formal nine
over a representative window with the PIT industry provider, then computes
pairwise median-absolute daily Spearman (the project's redundancy metric).
Read-only; no database writes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "research_v2" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "research_v2" / "src"))

import audit_formula_replay_consistency as replay  # noqa: E402
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
from evaluate_tushare_scope_validation import compute_pair_daily_spearman  # noqa: E402

CACHE = REPO_ROOT / "research_v2" / "artifacts" / "tushare_cache"
FACTORS = ["yeslab_365", "yeslab_357", "yeslab_363", "yeslab_380", "yeslab_431",
           "yeslab_437", "yeslab_155", "yeslab_98", "yeslab_94", "yeslab_376", "yeslab_396", "yeslab_442"]
PIT = {"yeslab_365", "yeslab_357", "yeslab_363"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2022, 1, 4))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2022, 12, 30))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    client = _connect()
    definitions = [replay.REPLAY_DEFINITIONS[f] for f in FACTORS]
    market_start = min(_calculator_market_start(client, d, args.start, args.end) for d in definitions) - timedelta(days=90)
    open_dates = _query_open_dates(client, market_start, args.end)
    fields = required_market_fields(definitions)
    market = fetch_full_tushare_shsz_market(open_dates, token=os.getenv("QRV2_TUSHARE_TOKEN", ""), cache_dir=CACHE, include_daily_basic=requires_daily_basic(fields))
    market["trade_date"] = pd.to_datetime(market["trade_date"])
    market["ts_code"] = market["ts_code"].astype(str)
    session = TushareMarketSession(client, market, tuple(sorted(market["ts_code"].unique())))
    provider = VersionedPITIndustryProvider.from_cache(CACHE, cutover=date(2021, 12, 10))

    frames = {}
    for fid in FACTORS:
        prov = provider if fid in PIT else None
        frames[fid] = replay._replay_factor(session, replay.REPLAY_DEFINITIONS[fid], args.start, args.end, industry_provider=prov)
        print(f"replayed {fid}", flush=True)

    pairs = compute_pair_daily_spearman(frames)
    if isinstance(pairs, pd.DataFrame):
        pair_records = pairs.to_dict("records")
    else:
        pair_records = list(pairs)
    edges = []
    for p in pair_records:
        if "yeslab_365" in (p["factor_id_left"], p["factor_id_right"]):
            other = p["factor_id_right"] if p["factor_id_left"] == "yeslab_365" else p["factor_id_left"]
            edges.append({"other": other, "median_abs_spearman": round(float(p["median_abs_spearman"]), 4), "valid_day_count": int(p["valid_day_count"])})
    edges.sort(key=lambda e: -e["median_abs_spearman"])
    payload = {
        "scope": "RECOVERY365_CROSS_SECTION_REDUNDANCY_V1",
        "window": f"{args.start}/{args.end}",
        "metric": "median abs daily Spearman, PIT industry provider (SW2014->SW2021 cutover 2021-12-10)",
        "yeslab_365_edges": edges,
        "max_edge": edges[0] if edges else None,
        "database_writes": False,
        "locked_test_read": False,
    }
    (args.output / "recovery365_redundancy.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for e in edges:
        print(f"  365 x {e['other']}: {e['median_abs_spearman']} ({e['valid_day_count']}d)")
    print(f"wrote {args.output}/recovery365_redundancy.json")
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
