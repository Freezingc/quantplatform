"""Pre-training regime redundancy audit for the READY candidate pool.

The admission pack's redundancy evidence (nine-edge / pool pairs) covers only
2020 onward. Since regimes demonstrably change factor behavior (alpha443's
2017 collapse), pairs that are safe in-training could become elevated or
redundant in pre-training regimes. This audit materializes all 23 READY
candidates plus the alpha443 repair variant over one window and computes the
pairwise median-absolute-daily-Spearman matrix (same metric as the 464
admission audit), flagging pairs >= 0.80 (ELEVATED) and >= 0.90 (REDUNDANT).

Daily Spearman is computed cross-sectionally on wide rank matrices
(equivalent to the long-format merge used by compute_pair_daily_spearman,
but vectorized per day). Read-only; no database writes; no locked-test reads.
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
PIT_CUTOVER = date(2021, 12, 10)

CANDIDATES = {
    "yeslab_114": -1, "yeslab_381": -1, "yeslab_405": -1, "yeslab_425": -1,
    "yeslab_484": -1, "yeslab_206_median_abs_v2": -1, "yeslab_363": -1,
    "yeslab_357": -1, "yeslab_96": -1, "yeslab_443": -1, "yeslab_464": -1,
    "yeslab_370": -1, "yeslab_375": -1, "yeslab_389": -1, "yeslab_390": -1,
    "yeslab_471": 1, "yeslab_398": -1, "yeslab_395": -1, "yeslab_371": -1,
    "yeslab_372": -1, "yeslab_379": -1, "yeslab_384": -1, "yeslab_406": -1,
}
REPAIR_ID = "yeslab_443_reg30_max20_v1"
FORMAL_NINE = {
    "yeslab_380": -1, "yeslab_431": -1, "yeslab_437": -1, "yeslab_155": -1,
    "yeslab_98": -1, "yeslab_94": -1, "yeslab_376": -1, "yeslab_396": -1,
    "yeslab_442": 1,
}
ALPHA206_MEDIAN_CANDIDATE = "yeslab_206_median_abs_v2"
ALPHA206_SOURCE = "yeslab_206"
PIT_FACTORS = {"yeslab_357", "yeslab_363", "yeslab_471"}
THRESHOLD_MONITOR = 0.80
THRESHOLD_REDUNDANT = 0.90
MIN_CROSS_SECTION = 30


def to_wide(values: pd.DataFrame, index: pd.Index, columns: pd.Index) -> pd.DataFrame:
    values = values.copy()
    values["trade_date"] = pd.to_datetime(values["trade_date"]).dt.normalize()
    values["ts_code"] = values["ts_code"].astype(str)
    wide = values.pivot_table(index="trade_date", columns="ts_code", values="replay_value", aggfunc="last")
    return wide.reindex(index=index, columns=columns)


def pair_median_abs_spearman(wides: dict[str, pd.DataFrame]) -> list[dict]:
    ids = sorted(wides)
    all_idx = wides[ids[0]].index
    ranks = {fid: wides[fid].rank(axis=1) for fid in ids}
    n = len(ids)
    daily = np.full((len(all_idx), n, n), np.nan)
    for di, dt in enumerate(all_idx):
        cols_all = None
        day_ranks = []
        for fid in ids:
            day_ranks.append(ranks[fid].loc[dt])
        mat = pd.concat(day_ranks, axis=1)
        mat.columns = ids
        valid = mat.notna().all(axis=1)
        if int(valid.sum()) < MIN_CROSS_SECTION:
            continue
        m = mat[valid].to_numpy(float)
        c = np.corrcoef(m, rowvar=False)
        daily[di] = c
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            series = daily[:, i, j]
            series = series[np.isfinite(series)]
            if len(series) == 0:
                continue
            rows.append({
                "factor_id_left": ids[i],
                "factor_id_right": ids[j],
                "median_abs_spearman": round(float(np.median(np.abs(series))), 4),
                "valid_day_count": int(len(series)),
            })
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-formal-nine", action="store_true", help="also materialize the formal nine factors (full pool + formal set matrix)")
    parser.add_argument("--formal-only", action="store_true", help="materialize only the formal nine factors")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    client = _connect()
    if args.formal_only:
        series_ids = dict(FORMAL_NINE)
    else:
        series_ids = dict(CANDIDATES)
        if args.include_formal_nine:
            series_ids.update(FORMAL_NINE)
    definitions = [
        replay.REPLAY_DEFINITIONS[ALPHA206_SOURCE if fid == ALPHA206_MEDIAN_CANDIDATE else fid]
        for fid in series_ids
    ]
    market_start = min(
        _calculator_market_start(client, d, args.start, args.end) for d in definitions
    ) - timedelta(days=90)
    open_dates = _query_open_dates(client, market_start, args.end)
    fields = tuple(sorted(set(required_market_fields(definitions)) | {"amount", "pct_chg"}))
    market = fetch_full_tushare_shsz_market(
        open_dates, token=os.getenv("QRV2_TUSHARE_TOKEN", ""), cache_dir=CACHE,
        include_daily_basic=requires_daily_basic(fields),
    )
    print(f"market rows: {len(market)}", flush=True)
    market["trade_date"] = pd.to_datetime(market["trade_date"])
    market["ts_code"] = market["ts_code"].astype(str)

    session = TushareMarketSession(client, market, tuple(sorted(market["ts_code"].unique())))
    provider = VersionedPITIndustryProvider.from_cache(CACHE, cutover=PIT_CUTOVER)

    # wide price index for alignment
    prices = market.pivot_table(index="trade_date", columns="ts_code", values="close", aggfunc="last").sort_index()
    window_idx = prices.index[(prices.index >= pd.Timestamp(args.start)) & (prices.index <= pd.Timestamp(args.end))]
    codes = prices.columns

    wides: dict[str, pd.DataFrame] = {}
    amount = market.pivot_table(index="trade_date", columns="ts_code", values="amount", aggfunc="last").sort_index()
    pct_chg = market.pivot_table(index="trade_date", columns="ts_code", values="pct_chg", aggfunc="last").sort_index()
    if not args.formal_only:
        repair_wide = compute_gp074_variant({"amount": amount, "pct_chg": pct_chg}, 30, 20)
        wides[REPAIR_ID] = repair_wide.reindex(index=window_idx, columns=codes)

    for fid in series_ids:
        if fid == ALPHA206_MEDIAN_CANDIDATE:
            values = materialize_alpha206_median_repair(market, args.start, args.end)
        else:
            prov = provider if fid in PIT_FACTORS else None
            values = replay._replay_factor(session, replay.REPLAY_DEFINITIONS[fid], args.start, args.end, industry_provider=prov)
        wides[fid] = to_wide(values, window_idx, codes)
        print(f"materialized {fid}", flush=True)

    pairs = pair_median_abs_spearman(wides)
    pairs.sort(key=lambda r: -r["median_abs_spearman"])
    redundant = [p for p in pairs if p["median_abs_spearman"] >= THRESHOLD_REDUNDANT]
    elevated = [p for p in pairs if THRESHOLD_MONITOR <= p["median_abs_spearman"] < THRESHOLD_REDUNDANT]
    payload = {
        "scope": "PRETRAINING_REGIME_REDUNDANCY_V2_WITH_FORMAL_NINE" if args.include_formal_nine else "PRETRAINING_REGIME_REDUNDANCY_V1",
        "window": f"{args.start}/{args.end}",
        "market_scope": "SH/SZ full market cross-section",
        "metric": "median |daily cross-sectional Spearman| over window days (min cross-section 30)",
        "series": sorted(wides.keys()),
        "thresholds": {"monitor": THRESHOLD_MONITOR, "redundant": THRESHOLD_REDUNDANT},
        "redundant_pairs": redundant,
        "elevated_pairs": elevated,
        "all_pairs": pairs,
        "database_writes": False,
        "locked_test_read": False,
    }
    out_path = args.output / f"pretraining_redundancy_{args.start.year}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"redundant: {len(redundant)}, elevated: {len(elevated)}")
    for p in pairs[:8]:
        print(f"  {p['factor_id_left']} x {p['factor_id_right']}: {p['median_abs_spearman']}")
    print(f"wrote {out_path}")
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
