"""Historical-window (2019) robustness evaluation for READY candidates.

Computes 2019 daily RankIC for each candidate using Tushare-sourced factors
and labels. The label construction (adjusted open T+1 -> T+6) was validated
against the official V2 label_daily for 2020 at machine precision
(historical_label_replica validation + Tushare-label equivalence check), so
2019 numbers are comparable with the 2020+ evidence. Universe variants:
full_market (every SH/SZ code present on the signal date) and adv20_liquid
(bottom-20% adv20 excluded, approximating cn_liquid_a until the official
pre-2020 universe backfill lands).

Read-only research artifact; no database writes, no locked-test reads.
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
DIAG = REPO_ROOT / "research_v2" / "artifacts" / "diagnostics"
CACHE = REPO_ROOT / "research_v2" / "artifacts" / "tushare_cache"
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
from audit_canonical_candidate_pairs import materialize_alpha206_median_repair  # noqa: E402
from audit_tushare_pit_industry import VersionedPITIndustryProvider  # noqa: E402

CANDIDATES = {
    "yeslab_114": -1, "yeslab_381": -1, "yeslab_405": -1, "yeslab_425": -1,
    "yeslab_484": -1, "yeslab_206_median_abs_v2": -1, "yeslab_363": -1,
    "yeslab_357": -1, "yeslab_96": -1, "yeslab_443": -1, "yeslab_464": -1,
    "yeslab_370": -1, "yeslab_375": -1, "yeslab_389": -1, "yeslab_390": -1,
    "yeslab_471": 1, "yeslab_398": -1, "yeslab_395": -1,
}
ALPHA206_MEDIAN_CANDIDATE = "yeslab_206_median_abs_v2"
ALPHA206_SOURCE_FACTOR = "yeslab_206"
PIT_FACTORS = {"yeslab_357", "yeslab_363", "yeslab_471"}
PIT_CUTOVER = date(2021, 12, 10)


def spearman_rows(factor: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    common_cols = factor.columns.intersection(labels.columns)
    f = factor.loc[:, common_cols]
    lab = labels.loc[:, common_cols]
    common_idx = f.index.intersection(lab.index)
    f = f.loc[common_idx]
    lab = lab.loc[common_idx]
    rows = []
    fr = f.rank(axis=1)
    lr = lab.rank(axis=1)
    for dt in common_idx:
        fv = fr.loc[dt]
        lv = lr.loc[dt]
        both = fv.notna() & lv.notna()
        if int(both.sum()) < 100:
            continue
        a = fv[both].to_numpy(dtype=float)
        b = lv[both].to_numpy(dtype=float)
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        rows.append({"signal_date": dt, "ic": float(np.corrcoef(a, b)[0, 1]), "n": int(both.sum())})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2019, 1, 2))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2019, 12, 31))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--candidates",
        type=str,
        default=None,
        help="optional override: comma-separated fid:direction pairs (e.g. yeslab_371:-1,yeslab_372:-1)",
    )
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    candidates = dict(CANDIDATES)
    if args.candidates:
        candidates = {}
        for token in args.candidates.split(","):
            fid, direction = token.strip().split(":")
            candidates[fid.strip()] = int(direction)

    client = _connect()
    # 206_median_abs_v2 is a repair variant: it has no standalone replay
    # definition; its market requirements come from the source yeslab_206
    # definition and its values are materialized below via the canonical
    # median-repair path (same handling as audit_candidate464_pairs.py).
    definitions = [
        replay.REPLAY_DEFINITIONS[
            ALPHA206_SOURCE_FACTOR if fid == ALPHA206_MEDIAN_CANDIDATE else fid
        ]
        for fid in candidates
    ]
    market_start = min(
        _calculator_market_start(client, definition, args.start, args.end) for definition in definitions
    )
    # Safety buffer: some calculators (e.g. AttentionSpillover/yeslab_471)
    # need more warmup than their declared lookback_days (pattern validated
    # in audit_candidate464_pairs.py).
    market_start = market_start - timedelta(days=90)
    market_end = args.end + timedelta(days=15)
    open_dates = _query_open_dates(client, market_start, market_end)
    fields = required_market_fields(definitions)
    market = fetch_full_tushare_shsz_market(
        open_dates,
        token=os.getenv("QRV2_TUSHARE_TOKEN", ""),
        cache_dir=CACHE,
        include_daily_basic=requires_daily_basic(fields),
    )
    print(f"market rows: {len(market)}", flush=True)
    market["trade_date"] = pd.to_datetime(market["trade_date"])
    market["ts_code"] = market["ts_code"].astype(str)

    # ---- labels from adjusted open T+1 -> T+6 ----
    m2 = market.dropna(subset=["open_hfq"] if "open_hfq" in market.columns else ["open"]).copy()
    if "open_hfq" not in m2.columns:
        m2["open_hfq"] = m2["open"] * m2.get("adj_factor", 1.0)
    prices = m2.pivot_table(index="trade_date", columns="ts_code", values="open_hfq", aggfunc="last").sort_index()
    prices = prices.loc[(prices.index >= pd.Timestamp(args.start) - pd.Timedelta(days=5)) & (prices.index <= pd.Timestamp(market_end))]
    trading_dates = prices.index
    pos = {ts: i for i, ts in enumerate(trading_dates)}
    sig_dates = [d for d in trading_dates if pd.Timestamp(args.start) <= d <= pd.Timestamp(args.end)]
    mat = prices.to_numpy()
    code_list = list(prices.columns)
    code_pos = {c: i for i, c in enumerate(code_list)}
    lab_rows = []
    for sd in sig_dates:
        i = pos[sd]
        if i + 6 >= len(trading_dates):
            continue
        for j, c in enumerate(code_list):
            ep = mat[i + 1, j]
            xp = mat[i + 6, j]
            if np.isfinite(ep) and np.isfinite(xp) and ep > 0 and xp > 0:
                lab_rows.append((sd, c, xp / ep - 1))
    labels = pd.DataFrame(lab_rows, columns=["signal_date", "ts_code", "raw_return"])
    labels_wide = labels.pivot_table(index="signal_date", columns="ts_code", values="raw_return", aggfunc="last")
    print(f"label rows: {len(labels)} dates: {labels_wide.shape}", flush=True)

    # ---- adv20 liquidity universe ----
    amount = m2.pivot_table(index="trade_date", columns="ts_code", values="amount", aggfunc="last").sort_index()
    adv20 = amount.rolling(20, min_periods=15).mean()

    session = TushareMarketSession(client, market, tuple(sorted(market["ts_code"].astype(str).unique())))
    provider = VersionedPITIndustryProvider.from_cache(CACHE, cutover=PIT_CUTOVER)

    results = {}
    for fid, direction in candidates.items():
        if fid == ALPHA206_MEDIAN_CANDIDATE:
            values = materialize_alpha206_median_repair(market, args.start, args.end)
        else:
            definition = replay.REPLAY_DEFINITIONS[fid]
            prov = provider if fid in PIT_FACTORS else None
            values = replay._replay_factor(session, definition, args.start, args.end, industry_provider=prov)
        values["trade_date"] = pd.to_datetime(values["trade_date"]).dt.normalize()
        values["ts_code"] = values["ts_code"].astype(str)
        factor_wide = values.pivot_table(index="trade_date", columns="ts_code", values="replay_value", aggfunc="last")
        factor_wide = factor_wide.reindex(labels_wide.index)
        # full market IC
        daily_full = spearman_rows(direction * factor_wide, labels_wide)
        # adv20 liquid variant
        liq = adv20.reindex(index=labels_wide.index, columns=factor_wide.columns)
        keep = liq.apply(lambda row: row >= row.quantile(0.20) if row.notna().sum() > 100 else row.notna(), axis=1)
        factor_liq = factor_wide.where(keep.reindex(index=factor_wide.index).astype(bool), np.nan)
        labels_liq = labels_wide.where(keep.astype(bool), np.nan)
        daily_liq = spearman_rows(direction * factor_liq, labels_liq)
        def summarize(daily):
            if daily.empty:
                return {"days": 0}
            ic = daily["ic"]
            return {
                "days": int(len(ic)),
                "mean_rank_ic": round(float(ic.mean()), 6),
                "icir": round(float(ic.mean() / ic.std()), 4) if ic.std() > 0 else None,
                "directional_day_share": round(float((ic > 0).mean()), 4),
            }
        results[fid] = {
            "direction": direction,
            "full_market": summarize(daily_full),
            "adv20_liquid": summarize(daily_liq),
        }
        print(f"{fid}: full={results[fid]['full_market']} liquid={results[fid]['adv20_liquid']}", flush=True)

    payload = {
        "scope": f"HISTORICAL_WINDOW_{args.start.year}_ROBUSTNESS_V1",
        "window": f"{args.start}/{args.end}",
        "market_scope": "SH/SZ",
        "label": "forward_open_5d:v1 equivalent (Tushare adjusted open T+1 -> T+6; validated machine-precision-equal to V2 labels on 2020)",
        "universe_variants": {
            "full_market": "every SH/SZ code with a valid adjusted open on the signal date",
            "adv20_liquid": "full market minus bottom-20% adv20 (approximation of cn_liquid_a until pre-2020 universe backfill lands)",
        },
        "candidates": results,
        "database_writes": False,
        "locked_test_read": False,
        "formal_factor_set_changed": False,
        "model_training_started": False,
    }
    out_name = f"historical_window_{args.start.year}_robustness.json"
    (args.output / out_name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output / out_name}")
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
