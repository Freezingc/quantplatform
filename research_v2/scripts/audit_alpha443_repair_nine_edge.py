"""Nine-edge redundancy audit for the alpha443 reg30_max20 repair variant.

Materializes the repair variant, canonical 443, and the formal nine over the
requested phase, then computes the standard median-absolute-daily-Spearman
pair edges (same metric/thresholds as the 464 admission audit: monitor 0.80,
redundant 0.90). Required before admission because the redundancy precheck
showed the variant is materially distinct from canonical 443 (0.639 train).
Read-only; no database writes; no locked-test reads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

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
from evaluate_tushare_scope_validation import compute_pair_daily_spearman  # noqa: E402
from repair_alpha443_window_scan import compute_gp074_variant  # noqa: E402

CACHE = REPO_ROOT / "research_v2" / "artifacts" / "tushare_cache"
FORMAL_NINE = (
    "yeslab_380", "yeslab_431", "yeslab_437", "yeslab_155",
    "yeslab_98", "yeslab_94", "yeslab_376", "yeslab_396", "yeslab_442",
)
REPAIR_ID = "yeslab_443_reg30_max20_v1"
CANONICAL_ID = "yeslab_443_canonical_v1"
THRESHOLD_MONITOR = 0.80
THRESHOLD_REDUNDANT = 0.90
PHASES = {
    "train": (date(2020, 1, 2), date(2024, 12, 31)),
    "validation": (date(2025, 1, 2), date(2025, 12, 31)),
}


def melt_wide(wide: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    out = (
        wide.rename_axis("trade_date")
        .reset_index()
        .melt(id_vars=["trade_date"], var_name="ts_code", value_name="replay_value")
    )
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    out = out[(out["trade_date"] >= pd.Timestamp(start)) & (out["trade_date"] <= pd.Timestamp(end))]
    return out.loc[:, ["trade_date", "ts_code", "replay_value"]].reset_index(drop=True)


def run_phase(phase: str, output: Path) -> dict:
    start, end = PHASES[phase]
    client = _connect()
    definitions = [replay.REPLAY_DEFINITIONS[fid] for fid in FORMAL_NINE]
    market_start = min(
        _calculator_market_start(client, d, start, end) for d in definitions
    ) - timedelta(days=90)
    open_dates = _query_open_dates(client, market_start, end)
    fields = required_market_fields(definitions)
    fields = tuple(sorted(set(fields) | {"amount", "pct_chg"}))
    market = fetch_full_tushare_shsz_market(
        open_dates,
        token=os.getenv("QRV2_TUSHARE_TOKEN", ""),
        cache_dir=CACHE,
        include_daily_basic=requires_daily_basic(fields),
    )
    print(f"[{phase}] market rows: {len(market)}", flush=True)
    market["trade_date"] = pd.to_datetime(market["trade_date"])
    market["ts_code"] = market["ts_code"].astype(str)

    wide = {
        f: market.pivot_table(index="trade_date", columns="ts_code", values=f, aggfunc="last").sort_index()
        for f in ("amount", "pct_chg")
    }
    frames: dict[str, pd.DataFrame] = {
        REPAIR_ID: melt_wide(compute_gp074_variant(wide, 30, 20), start, end),
        CANONICAL_ID: melt_wide(compute_gp074_variant(wide, 30, 5), start, end),
    }
    session = TushareMarketSession(client, market, tuple(sorted(market["ts_code"].unique())))
    for fid in FORMAL_NINE:
        frames[fid] = replay._replay_factor(session, replay.REPLAY_DEFINITIONS[fid], start, end)
        print(f"[{phase}] replayed {fid}", flush=True)

    pairs = compute_pair_daily_spearman(frames)
    nine_rows = pairs[
        ((pairs["factor_id_left"] == REPAIR_ID) & (pairs["factor_id_right"].isin(FORMAL_NINE)))
        | ((pairs["factor_id_right"] == REPAIR_ID) & (pairs["factor_id_left"].isin(FORMAL_NINE)))
    ].copy()
    nine_rows["other"] = nine_rows.apply(
        lambda r: r["factor_id_left"] if r["factor_id_right"] == REPAIR_ID else r["factor_id_right"], axis=1
    )
    canonical_row = pairs[
        ((pairs["factor_id_left"] == REPAIR_ID) & (pairs["factor_id_right"] == CANONICAL_ID))
        | ((pairs["factor_id_right"] == REPAIR_ID) & (pairs["factor_id_left"] == CANONICAL_ID))
    ]
    maximum = nine_rows.loc[nine_rows["median_abs_spearman"].idxmax()] if len(nine_rows) else None
    payload = {
        "scope": f"ALPHA443_REPAIR_NINE_EDGE_{phase.upper()}_V1",
        "phase": phase,
        "window": f"{start}/{end}",
        "candidate_id": REPAIR_ID,
        "thresholds": {"monitor": THRESHOLD_MONITOR, "redundant": THRESHOLD_REDUNDANT},
        "nine_edge_maximum": {
            "other": str(maximum["other"]),
            "median_abs_spearman": round(float(maximum["median_abs_spearman"]), 4),
            "valid_day_count": int(maximum["valid_day_count"]),
        } if maximum is not None else None,
        "nine_edges": {
            str(r["other"]): {
                "median_abs_spearman": round(float(r["median_abs_spearman"]), 4),
                "valid_day_count": int(r["valid_day_count"]),
            } for _, r in nine_rows.sort_values("median_abs_spearman", ascending=False).iterrows()
        },
        "repair_vs_canonical_443": {
            "median_abs_spearman": round(float(canonical_row.iloc[0]["median_abs_spearman"]), 4),
            "valid_day_count": int(canonical_row.iloc[0]["valid_day_count"]),
        } if len(canonical_row) else None,
        "database_writes": False,
        "locked_test_read": False,
    }
    verdict = "PASS"
    if payload["nine_edge_maximum"]:
        m = payload["nine_edge_maximum"]["median_abs_spearman"]
        verdict = "REDUNDANT" if m >= THRESHOLD_REDUNDANT else ("ELEVATED_MONITOR" if m >= THRESHOLD_MONITOR else "PASS")
    payload["nine_edge_verdict"] = verdict
    out_path = output / f"alpha443_repair_nine_edge_{phase}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[{phase}] nine-edge max: {payload['nine_edge_maximum']} -> {verdict}", flush=True)
    print(f"[{phase}] wrote {out_path}", flush=True)
    client.close()
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train", "validation", "all"), default="all")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    phases = ("train", "validation") if args.phase == "all" else (args.phase,)
    for phase in phases:
        run_phase(phase, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
