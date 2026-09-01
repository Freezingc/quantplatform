"""Rank-turnover (tradability) analysis for the eighteen READY candidates.

Materializes every candidate over the training and validation windows,
restricts values to the V2 universe keys, and measures how fast factor ranks
turn over day to day:

  - rank_autocorr_mean: mean daily Spearman autocorrelation of factor values
    between consecutive signal dates (1 = perfectly stable ranks).
  - quintile_transition_rate: mean share of stocks whose factor quintile
    changed from one signal date to the next (0 = no turnover).

Read-only research analysis: no database writes, no locked-test reads, no
admission decisions.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import audit_formula_replay_consistency as replay
from audit_canonical_candidate_pairs import (
    load_alpha484_checkpoint_values,
    materialize_alpha206_median_repair,
)
from audit_formula_replay_tushare import (
    TushareMarketSession,
    _calculator_market_start,
    _connect,
    _query_open_dates,
    fetch_full_tushare_shsz_market,
    required_market_fields,
    requires_daily_basic,
)
from audit_tushare_pit_industry import VersionedPITIndustryProvider
from evaluate_tushare_scope_validation import _load_validation_inputs

REPO_ROOT = Path(__file__).resolve().parents[2]
DIAG = REPO_ROOT / "research_v2" / "artifacts" / "diagnostics"
CACHE = REPO_ROOT / "research_v2" / "artifacts" / "tushare_cache"
CHECKPOINT_ROOT = REPO_ROOT / "research_v2" / "artifacts" / "checkpoints" / "alpha484_canonical_v1"
PIT_CUTOVER = date(2021, 12, 10)

WINDOWS = {
    "train": (date(2020, 1, 2), date(2024, 12, 31)),
    "validation": (date(2025, 1, 2), date(2025, 12, 31)),
}

CANDIDATES: dict[str, str] = {
    "yeslab_114_tushare_shsz_v2": "yeslab_114",
    "yeslab_381_tushare_shsz_v1": "yeslab_381",
    "yeslab_405_tushare_shsz_v1": "yeslab_405",
    "yeslab_425_tushare_shsz_v2": "yeslab_425",
    "yeslab_484_tushare_shsz_v1": "yeslab_484",
    "yeslab_206_median_abs_v2": "yeslab_206",
    "yeslab_363_tushare_shsz_v1": "yeslab_363",
    "yeslab_357_tushare_shsz_v1": "yeslab_357",
    "yeslab_96_canonical_replay_v1": "yeslab_96",
    "yeslab_443_tushare_shsz_v1": "yeslab_443",
    "yeslab_464_tushare_shsz_v1": "yeslab_464",
    "yeslab_370_tushare_shsz_v1": "yeslab_370",
    "yeslab_375_tushare_shsz_v1": "yeslab_375",
    "yeslab_389_tushare_shsz_v1": "yeslab_389",
    "yeslab_390_tushare_shsz_v1": "yeslab_390",
    "yeslab_471_tushare_pit_v1": "yeslab_471",
    "yeslab_398_tushare_shsz_v1": "yeslab_398",
    "yeslab_395_tushare_shsz_v1": "yeslab_395",
}
PIT_FACTORS = {"yeslab_357", "yeslab_363", "yeslab_471"}


def materialize_window(start: date, end: date) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    client = _connect()
    inputs = _load_validation_inputs(client, start, end)
    keys = inputs.loc[:, ["signal_date", "ts_code"]].rename(columns={"signal_date": "trade_date"})
    definitions = [replay.REPLAY_DEFINITIONS[source] for source in set(CANDIDATES.values())]
    market_start = min(
        _calculator_market_start(client, definition, start, end) for definition in definitions
    ) - timedelta(days=90)
    open_dates = _query_open_dates(client, market_start, end)
    fields = required_market_fields(definitions)
    market = fetch_full_tushare_shsz_market(
        open_dates,
        token=__import__("os").getenv("QRV2_TUSHARE_TOKEN", ""),
        cache_dir=CACHE,
        include_daily_basic=requires_daily_basic(fields),
    )
    codes = tuple(sorted(market["ts_code"].astype(str).unique()))
    session = TushareMarketSession(client, market, codes)
    industry_provider = VersionedPITIndustryProvider.from_cache(CACHE, cutover=PIT_CUTOVER)
    alpha484_values = load_alpha484_checkpoint_values(CHECKPOINT_ROOT)
    frames: dict[str, pd.DataFrame] = {}
    for candidate_id, source_factor in CANDIDATES.items():
        if candidate_id == "yeslab_206_median_abs_v2":
            values = materialize_alpha206_median_repair(market, start, end)
        elif candidate_id == "yeslab_484_tushare_shsz_v1":
            values = alpha484_values
        else:
            definition = replay.REPLAY_DEFINITIONS[source_factor]
            provider = industry_provider if source_factor in PIT_FACTORS else None
            values = replay._replay_factor(session, definition, start, end, industry_provider=provider)
        left = values.loc[:, ["trade_date", "ts_code", "replay_value"]].copy()
        left["trade_date"] = pd.to_datetime(left["trade_date"], errors="raise").dt.normalize()
        left["ts_code"] = left["ts_code"].astype(str)
        right = keys.loc[:, ["trade_date", "ts_code"]].copy()
        right["trade_date"] = pd.to_datetime(right["trade_date"], errors="raise").dt.normalize()
        right["ts_code"] = right["ts_code"].astype(str)
        merged = left.merge(right, on=["trade_date", "ts_code"], how="inner", validate="one_to_one")
        merged = merged.drop_duplicates(["trade_date", "ts_code"], keep="last")
        frames[candidate_id] = merged.rename(columns={"replay_value": "value"})
        print(f"materialized {candidate_id}: {len(frames[candidate_id])} rows", flush=True)
    return frames, keys


def rank_turnover_metrics(frame: pd.DataFrame) -> dict:
    wide = frame.pivot_table(index="trade_date", columns="ts_code", values="value", aggfunc="last")
    wide = wide.sort_index()
    dates = list(wide.index)
    autocorrs = []
    transitions = []
    prev_ranks = None
    prev_quintiles = None
    for ts in dates:
        row = wide.loc[ts]
        valid = row.dropna()
        if len(valid) < 100:
            prev_ranks = None
            prev_quintiles = None
            continue
        ranks = valid.rank()
        if prev_ranks is not None:
            common = ranks.index.intersection(prev_ranks.index)
            if len(common) >= 100:
                a = ranks.loc[common].to_numpy(dtype=float)
                b = prev_ranks.loc[common].to_numpy(dtype=float)
                if np.std(a) > 0 and np.std(b) > 0:
                    autocorrs.append(float(np.corrcoef(a, b)[0, 1]))
        quintiles = pd.qcut(valid.rank(pct=True), 5, labels=False, duplicates="drop")
        if prev_quintiles is not None:
            common = quintiles.index.intersection(prev_quintiles.index)
            if len(common) >= 100:
                changed = float((quintiles.loc[common] != prev_quintiles.loc[common]).mean())
                transitions.append(changed)
        prev_ranks = ranks
        prev_quintiles = quintiles
    return {
        "days_used": len(autocorrs),
        "rank_autocorr_mean": float(np.mean(autocorrs)) if autocorrs else None,
        "quintile_transition_rate": float(np.mean(transitions)) if transitions else None,
    }


def main() -> None:
    results: dict[str, dict] = {}
    for window, (start, end) in WINDOWS.items():
        print(f"=== window {window}: {start} .. {end} ===", flush=True)
        frames, _ = materialize_window(start, end)
        for candidate_id, frame in frames.items():
            metrics = rank_turnover_metrics(frame)
            results.setdefault(candidate_id, {})[window] = metrics
            print(
                f"{candidate_id} [{window}] autocorr={metrics['rank_autocorr_mean']:.4f} "
                f"quintile_transition={metrics['quintile_transition_rate']:.4f}"
                if metrics["rank_autocorr_mean"] is not None
                else f"{candidate_id} [{window}] NO DATA",
                flush=True,
            )
    payload = {
        "scope": "POOL18_RANK_TURNOVER_V1",
        "market_scope": "SH/SZ",
        "excluded_exchanges": ["BJ"],
        "label": "forward_open_5d:v1",
        "windows": {w: {"start": s.isoformat(), "end": e.isoformat()} for w, (s, e) in WINDOWS.items()},
        "metric_definitions": {
            "rank_autocorr_mean": "mean daily Spearman autocorrelation of factor values between consecutive signal dates (1.0 = perfectly stable ranks)",
            "quintile_transition_rate": "mean share of common stocks whose factor quintile changed between consecutive signal dates (0.0 = no turnover)",
            "note": "values restricted to V2 universe intersection; days with fewer than 100 common stocks are skipped",
        },
        "candidates": results,
        "database_writes": False,
        "locked_test_read": False,
        "formal_factor_set_changed": False,
        "model_training_started": False,
    }
    out = DIAG / "pool18_rank_turnover_v1.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
