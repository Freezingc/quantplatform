"""Market-regime robustness analysis for the eighteen READY candidates.

Read-only analysis over existing daily RankIC artifacts.  Regimes are defined
from CSI 300 (000300.SH): trend = close vs 120-day moving average
(above/below), volatility = terciles of the 20-day realized volatility of
index returns computed within each evaluation window.  No database writes, no
locked-test reads, no admission decisions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DIAG = Path("research_v2/artifacts/diagnostics")
INDEX_CSV = Path("research_v2/artifacts/tushare_cache/index_daily_000300_SH_20190701_20251231.csv")

CANDIDATES: dict[str, dict] = {
    "yeslab_114_tushare_shsz_v2": {"direction": -1, "train": "tushare_alpha114_canonical_train_similarity_v1/daily_rank_ic.csv", "validation": "tushare_alpha114_canonical_validation_v1/daily_rank_ic.csv"},
    "yeslab_381_tushare_shsz_v1": {"direction": -1, "train": "tushare_alpha381_canonical_train_similarity_v1/daily_rank_ic.csv", "validation": "tushare_alpha381_canonical_validation_v1/daily_rank_ic.csv"},
    "yeslab_405_tushare_shsz_v1": {"direction": -1, "train": "tushare_alpha405_canonical_train_similarity_v1/daily_rank_ic.csv", "validation": "tushare_alpha405_canonical_validation_v1/daily_rank_ic.csv"},
    "yeslab_425_tushare_shsz_v2": {"direction": -1, "train": "tushare_alpha425_canonical_train_similarity_v1/daily_rank_ic.csv", "validation": "tushare_alpha425_canonical_validation_v1/daily_rank_ic.csv"},
    "yeslab_484_tushare_shsz_v1": {"direction": -1, "train": "tushare_alpha484_canonical_v1/daily_rank_ic.csv", "validation": "tushare_alpha484_canonical_v1/daily_rank_ic.csv"},
    "yeslab_206_median_abs_v2": {"direction": -1, "train": "alpha206_volume_variants_2020_2025_v1/median_abs/daily_rank_ic.csv", "validation": "alpha206_volume_variants_2020_2025_v1/median_abs/daily_rank_ic.csv"},
    "yeslab_363_tushare_shsz_v1": {"direction": -1, "train": "tushare_shsz_alpha363_formal9_train_2020_2024_v1/daily_rank_ic.csv", "validation": "tushare_shsz_alpha363_formal9_validation_2025_v1/daily_rank_ic.csv"},
    "yeslab_357_tushare_shsz_v1": {"direction": -1, "train": "tushare_shsz_alpha357_formal9_train_2020_2024_v1/daily_rank_ic.csv", "validation": "tushare_shsz_alpha357_formal9_validation_2025_v1/daily_rank_ic.csv"},
    "yeslab_96_canonical_replay_v1": {"direction": -1, "train": "tushare_canonical_formal9_unique7_train_2020_2024_v1/daily_rank_ic.csv", "validation": "tushare_canonical_formal9_unique7_validation_2025_v1/daily_rank_ic.csv"},
    "yeslab_443_tushare_shsz_v1": {"direction": -1, "train": "tushare_shsz_alpha443_formal9_train_2020_2024_v1/daily_rank_ic.csv", "validation": "tushare_shsz_alpha443_formal9_validation_2025_v1/daily_rank_ic.csv"},
    "yeslab_464_tushare_shsz_v1": {"direction": -1, "train": "tushare_shsz_gp_coverage_repair4_formal9_train_2020_2024_v1/daily_rank_ic.csv", "validation": "tushare_shsz_gp_coverage_repair4_formal9_validation_2025_v1/daily_rank_ic.csv"},
    "yeslab_370_tushare_shsz_v1": {"direction": -1, "train": "tushare_shsz_nearmiss6_formal9_train_2020_2024_v1/daily_rank_ic.csv", "validation": "tushare_shsz_nearmiss6_formal9_validation_2025_v1/daily_rank_ic.csv"},
    "yeslab_375_tushare_shsz_v1": {"direction": -1, "train": "tushare_shsz_nearmiss6_formal9_train_2020_2024_v1/daily_rank_ic.csv", "validation": "tushare_shsz_nearmiss6_formal9_validation_2025_v1/daily_rank_ic.csv"},
    "yeslab_389_tushare_shsz_v1": {"direction": -1, "train": "tushare_shsz_nearmiss6_formal9_train_2020_2024_v1/daily_rank_ic.csv", "validation": "tushare_shsz_nearmiss6_formal9_validation_2025_v1/daily_rank_ic.csv"},
    "yeslab_390_tushare_shsz_v1": {"direction": -1, "train": "tushare_shsz_nearmiss6_formal9_train_2020_2024_v1/daily_rank_ic.csv", "validation": "tushare_shsz_nearmiss6_formal9_validation_2025_v1/daily_rank_ic.csv"},
    "yeslab_471_tushare_pit_v1": {"direction": 1, "train": "tushare_pit_alpha471_train_2020_2024_v1/daily_rank_ic.csv", "validation": "tushare_pit_alpha471_formal9_validation_2025_v1/daily_rank_ic.csv"},
    "yeslab_398_tushare_shsz_v1": {"direction": -1, "train": "tushare_shsz_dq3_formal9_train_2020_2024_v1/daily_rank_ic.csv", "validation": "tushare_shsz_dq3_formal9_validation_2025_v1/daily_rank_ic.csv"},
    "yeslab_395_tushare_shsz_v1": {"direction": -1, "train": "tushare_shsz_alpha395_formal9_train_2020_2024_v1/daily_rank_ic.csv", "validation": "tushare_shsz_alpha395_formal9_validation_2025_v1/daily_rank_ic.csv"},
}
SOURCE_FACTOR = {cid: cid.replace("_tushare_shsz_v1", "").replace("_tushare_pit_v1", "").replace("_tushare_shsz_v2", "").replace("_canonical_replay_v1", "") for cid in CANDIDATES}
SOURCE_FACTOR["yeslab_206_median_abs_v2"] = "yeslab_206_median_abs"

WINDOW_RANGES = {
    "train": ("2020-01-02", "2024-12-31"),
    "validation": ("2025-01-02", "2025-12-31"),
}


def build_regimes() -> pd.DataFrame:
    idx = pd.read_csv(INDEX_CSV)
    idx["trade_date"] = pd.to_datetime(idx["trade_date"], format="%Y%m%d")
    idx = idx.sort_values("trade_date").set_index("trade_date")
    close = idx["close"]
    ret = close.pct_change()
    ma120 = close.rolling(120).mean()
    vol20 = ret.rolling(20).std()
    regimes = pd.DataFrame({
        "trend": np.where(close >= ma120, "above_ma120", "below_ma120"),
        "vol20": vol20,
    })
    regimes = regimes.dropna()
    return regimes


def regime_stats(signed_ic: pd.Series) -> dict:
    n = int(len(signed_ic))
    if n == 0:
        return {"days": 0}
    mean = float(signed_ic.mean())
    std = float(signed_ic.std())
    return {
        "days": n,
        "signed_mean_rank_ic": round(mean, 6),
        "signed_icir": round(mean / std, 4) if std > 0 else None,
        "directional_day_share": round(float((signed_ic > 0).mean()), 4),
    }


def main() -> None:
    regimes = build_regimes()
    results: dict[str, dict] = {}
    for cid, spec in CANDIDATES.items():
        source_factor = SOURCE_FACTOR[cid]
        entry: dict[str, dict] = {}
        for window, (start, end) in WINDOW_RANGES.items():
            daily = pd.read_csv(DIAG / spec[window])
            daily["signal_date"] = pd.to_datetime(daily["signal_date"])
            daily = daily[(daily["signal_date"] >= start) & (daily["signal_date"] <= end)]
            if "factor_id" in daily.columns:
                daily = daily[daily["factor_id"].isin({source_factor, cid})]
                if daily.empty:
                    daily = pd.read_csv(DIAG / spec[window])
                    daily["signal_date"] = pd.to_datetime(daily["signal_date"])
                    daily = daily[(daily["signal_date"] >= start) & (daily["signal_date"] <= end)]
                    cand_ids = set(daily["factor_id"].unique())
                    match = [f for f in cand_ids if source_factor.split("_")[-1] in f or f == cid]
                    daily = daily[daily["factor_id"] == sorted(match)[0]] if match else daily.iloc[0:0]
            daily = daily.dropna(subset=["rank_ic"]).set_index("signal_date")
            daily["signed_ic"] = spec["direction"] * daily["rank_ic"]
            merged = daily.join(regimes, how="inner")
            merged = merged.dropna(subset=["trend", "vol20"])
            vol_terciles = merged["vol20"].quantile([1 / 3, 2 / 3]).values
            merged["vol_regime"] = pd.cut(
                merged["vol20"],
                bins=[-np.inf, vol_terciles[0], vol_terciles[1], np.inf],
                labels=["low_vol", "mid_vol", "high_vol"],
            )
            window_result = {"all": regime_stats(merged["signed_ic"])}
            for trend_value in ("above_ma120", "below_ma120"):
                window_result[trend_value] = regime_stats(merged.loc[merged["trend"] == trend_value, "signed_ic"])
            for vol_value in ("low_vol", "mid_vol", "high_vol"):
                window_result[vol_value] = regime_stats(merged.loc[merged["vol_regime"] == vol_value, "signed_ic"])
            entry[window] = window_result
        results[cid] = entry

    payload = {
        "scope": "POOL18_MARKET_REGIME_ROBUSTNESS_V1",
        "market_scope": "SH/SZ",
        "excluded_exchanges": ["BJ"],
        "label": "forward_open_5d:v1",
        "regime_definitions": {
            "trend": "CSI300 close >= 120-day moving average -> above_ma120, else below_ma120",
            "volatility": "within-window terciles of CSI300 20-day realized volatility of daily returns",
            "index": "000300.SH",
            "index_artifact": str(INDEX_CSV),
        },
        "windows": {w: {"start": s, "end": e} for w, (s, e) in WINDOW_RANGES.items()},
        "candidates": results,
        "note": "Read-only decision-support evidence; regimes with signed_mean_rank_ic <= 0 are flagged as monitoring concerns.",
        "database_writes": False,
        "locked_test_read": False,
        "formal_factor_set_changed": False,
        "model_training_started": False,
    }
    out = DIAG / "pool18_market_regime_robustness_v1.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")

    print(f"{'candidate':32s} {'win':4s} {'all':>7s} {'above':>7s} {'below':>7s} {'lowV':>7s} {'midV':>7s} {'highV':>7s}")
    concerns = []
    for cid, entry in results.items():
        for window in ("train", "validation"):
            r = entry[window]
            vals = [r["all"]["signed_mean_rank_ic"], r["above_ma120"]["signed_mean_rank_ic"], r["below_ma120"]["signed_mean_rank_ic"],
                    r["low_vol"]["signed_mean_rank_ic"], r["mid_vol"]["signed_mean_rank_ic"], r["high_vol"]["signed_mean_rank_ic"]]
            print(f"{cid:32s} {window[:4]:4s} " + " ".join(f"{v:+.4f}" for v in vals))
            for regime_name in ("above_ma120", "below_ma120", "low_vol", "mid_vol", "high_vol"):
                if r[regime_name]["days"] >= 30 and r[regime_name]["signed_mean_rank_ic"] <= 0:
                    concerns.append((cid, window, regime_name, r[regime_name]["signed_mean_rank_ic"]))
    print("\nconcerns (signed IC <= 0, >=30 days):")
    for c in concerns:
        print("  ", c)


if __name__ == "__main__":
    main()
