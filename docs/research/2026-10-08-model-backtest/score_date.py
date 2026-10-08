#!/usr/bin/env python3
"""Stock scores for one signal date, read from the signed frozen predictions.

The four ordinary TEST folds already cover every trading day of 2022-2025, and each
day's scores were written to a signed artifact by the run that produced the gates.
So for any date in that window "what does the model score today" is a lookup, not a
re-inference: this reads the artifact, ranks it, and joins the realised five-day
return so a pick can be checked against what actually happened.

Read-only. No database, no weights, no writes except --out.

Dates outside 2022-2025 cannot be served from artifacts and would need live
inference; note that factor_daily ends 2025-12-31 (verified against
dfs://QuantResearchV2Test), so a 2026 date additionally needs factor
materialisation first.

Usage:
    score_date.py --date 2025-06-13 [--top 50] [--models M0R,M1,M2] [--out scores.csv]
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ARTIFACTS = Path("/data/zhangyudong/qrv2_research_artifacts")
MODELS = {"M0R": "m0r_v2", "M1": "m1_full_v1", "M2": "m2_v1", "M0-refined": "m0_v3"}
FOLD_OF_YEAR = {2022: "F1", 2023: "F2", 2024: "F3", 2025: "F4"}


def predictions_by_fold(ws_dir: Path) -> dict[str, tuple[Path, dict]]:
    out: dict[str, tuple[Path, dict]] = {}
    for manifest_path in sorted((ws_dir / "predictions").glob("*/manifest.json")):
        meta = json.loads(manifest_path.read_text())["metadata"]
        if meta["kind"] == "candidate":
            out[meta["fold_id"]] = (manifest_path.parent / "scores.parquet", meta)
    return out


def labels_for(ws_dir: Path, dataset_id: str) -> pd.DataFrame:
    shards = sorted((ws_dir / "datasets" / dataset_id / "split=test").rglob("*.parquet"))
    frame = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    frame["signal_date"] = pd.to_datetime(frame["signal_date"]).dt.date
    return frame.loc[:, ["signal_date", "ts_code", "raw_return_5d", "label_valid"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="signal date, YYYY-MM-DD")
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--models", default="M0R,M1,M2,M0-refined")
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    target = date.fromisoformat(args.date)
    fold = FOLD_OF_YEAR.get(target.year)
    if fold is None:
        raise SystemExit(
            f"{target} is outside the signed TEST folds (2022-2025). Serving it needs live "
            "inference from the bundles; note factor_daily ends 2025-12-31, so a 2026 date "
            "also needs factor materialisation first."
        )

    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in wanted if m not in MODELS]
    if unknown:
        raise SystemExit(f"unknown models {unknown}; choose from {sorted(MODELS)}")

    frames: dict[str, pd.DataFrame] = {}
    labels: pd.DataFrame | None = None
    for model in wanted:
        ws_dir = args.artifacts / MODELS[model]
        preds = predictions_by_fold(ws_dir)
        if fold not in preds:
            print(f"  ! {model}: no candidate artifact for {fold}")
            continue
        path, meta = preds[fold]
        scores = pd.read_parquet(path)
        scores["signal_date"] = pd.to_datetime(scores["signal_date"]).dt.date
        day = scores.loc[scores["signal_date"] == target, ["ts_code", "score"]].copy()
        if day.empty:
            print(f"  ! {model}: {target} is not a scored signal date in {fold}")
            continue
        day = day.rename(columns={"score": f"score_{model}"})
        day[f"rank_{model}"] = day[f"score_{model}"].rank(ascending=False, method="min").astype(int)
        frames[model] = day
        print(
            f"  {model:<11} {fold} model_id={meta['model_id']} "
            f"prediction={meta['selection_id'][:12]} names={len(day)}"
        )
        if labels is None:
            labels = labels_for(ws_dir, meta["dataset_id"]).query("signal_date == @target")

    if not frames:
        raise SystemExit(f"no model has scores for {target}")

    merged = frames[wanted[0]]
    for model in wanted[1:]:
        if model in frames:
            merged = merged.merge(frames[model], on="ts_code", how="outer")
    if labels is not None and not labels.empty:
        merged = merged.merge(labels, on="ts_code", how="left")
        merged["realized_5d_pct"] = (merged["raw_return_5d"] * 100).round(3)
        merged.drop(columns=["signal_date", "raw_return_5d", "label_valid"], inplace=True, errors="ignore")

    first = next(m for m in wanted if m in frames)
    merged = merged.sort_values(f"rank_{first}").head(args.top)
    merged.insert(0, "signal_date", target.isoformat())

    print(f"\n=== top {args.top} by {first} on {target} (fold {fold}) ===")
    show = ["ts_code", f"rank_{first}", f"score_{first}"]
    show += [c for m in wanted[1:] if m in frames for c in (f"rank_{m}", f"score_{m}")]
    if "realized_5d_pct" in merged:
        show.append("realized_5d_pct")
    print(merged.loc[:, show].to_string(index=False))

    if "realized_5d_pct" in merged:
        valid = merged["realized_5d_pct"].dropna()
        if not valid.empty:
            print(
                f"\nequal-weight realised 5-day return of these {len(merged)} names: "
                f"{valid.mean():+.3f}%  (median {valid.median():+.3f}%, "
                f"win rate {(valid > 0).mean():.1%}, n={len(valid)})"
            )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
