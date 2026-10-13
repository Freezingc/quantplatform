"""Separate "the signal decayed" from "the market or the book did" for the operational scores.

A backtest total cannot tell those apart: a falling market makes every long-only book look bad, and
a good market hides a dead signal. This measures the signal on its own terms -- daily
cross-sectional RankIC against realised forward returns, using the platform's own
``compute_daily_rank_ic``/``summarize_rank_ic`` so the numbers are on the same footing as the
published walk-forward gates -- and it measures it for two windows at once:

* a **reference** window (the FINAL fold's validation year) whose labels are already published, so
  the score and the outcome both come from the platform's own tables; and
* the **operational** window the owner asked about, whose labels have to be recomputed from Tushare
  because the platform's ingestion stops earlier.

Each model is scored alone as well as in the ensemble, so a drop can be attributed to one carrier
rather than to "the model".

The operational window falls inside the locked 2026 test segment. As with the backtest script, this
is a diagnostic read and must not be used to select, tune or retrain anything.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from collections.abc import Mapping, Sequence
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

from research_v2.evaluation.rank_ic import (  # noqa: E402
    compute_daily_rank_ic,
    summarize_rank_ic,
)
from research_v2.training.model_bundle import ModelArtifactBundleStore  # noqa: E402

LABEL_COLUMNS = ["signal_date", "ts_code", "raw_return_5d", "label_valid"]


def returns_to_labels(returns: Mapping[date, Mapping[str, float]]) -> pd.DataFrame:
    """Reshape computed forward returns into the evaluator's label schema.

    A name with no finite return is kept with ``label_valid=False`` rather than dropped: the
    evaluator counts those as invalid days, and silently dropping them would turn an unmeasurable
    day into a measured one.
    """
    rows = [
        {
            "signal_date": as_plain_date(day),
            "ts_code": str(code),
            "raw_return_5d": value,
            "label_valid": pd.notna(value),
        }
        for day, realised in returns.items()
        for code, value in realised.items()
    ]
    frame = pd.DataFrame(rows, columns=LABEL_COLUMNS)
    return frame.sort_values(["signal_date", "ts_code"], ignore_index=True)


def ensemble_score(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Mean of the named score columns, aligned on business keys, dropping partial rows."""
    if not columns:
        raise ValueError("at least one score column is required")
    missing = sorted({"signal_date", "ts_code", *columns}.difference(frame.columns))
    if missing:
        raise ValueError(f"score frame is missing columns: {missing}")
    keys = ["signal_date", "ts_code"]
    working = frame.loc[:, [*keys, *columns]].copy()
    working["signal_date"] = [as_plain_date(value) for value in working["signal_date"]]
    working["ts_code"] = working["ts_code"].astype(str)
    for column in columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working.dropna(subset=list(columns))
    working["score"] = working.loc[:, list(columns)].mean(axis=1)
    return working.loc[:, [*keys, "score"]].reset_index(drop=True)


def _ic_summary(scores: pd.DataFrame, labels: pd.DataFrame) -> dict[str, object]:
    daily = compute_daily_rank_ic(scores, labels)
    summary = summarize_rank_ic(daily)
    monthly = (
        pd.DataFrame(
            {
                "signal_date": pd.to_datetime(daily["signal_date"]),
                "rank_ic": pd.to_numeric(daily["rank_ic"], errors="coerce"),
            }
        )
        .assign(month=lambda frame: frame["signal_date"].dt.strftime("%Y-%m"))
        .groupby("month")["rank_ic"]
        .agg(["count", "mean"])
        .reset_index()
    )
    return {
        "valid_days": summary.valid_day_count,
        "invalid_days": summary.invalid_day_count,
        "mean_rank_ic": summary.mean_rank_ic,
        "rank_ic_std": summary.rank_ic_std,
        "icir": summary.icir,
        "positive_day_share": summary.positive_day_share,
        "mean_sample_count": summary.mean_sample_count,
        "monthly_mean_rank_ic": {
            str(row["month"]): float(row["mean"]) for _, row in monthly.iterrows()
        },
        "monthly_day_counts": {
            str(row["month"]): int(row["count"]) for _, row in monthly.iterrows()
        },
    }


def _load_validation_predictions(store: ModelArtifactBundleStore, model_id: str) -> pd.DataFrame:
    bundle = store.load_verified_bundle(model_id)
    frame = pd.read_parquet(io.BytesIO(bundle.files["validation_predictions.parquet"]))
    frame["signal_date"] = [as_plain_date(value) for value in frame["signal_date"]]
    frame["ts_code"] = frame["ts_code"].astype(str)
    return frame.loc[:, ["signal_date", "ts_code", "percentile_rank"]].rename(
        columns={"percentile_rank": "score"}
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--models", type=str, default="m0r-final-v1,m1-final-v1")
    parser.add_argument("--reference-start", type=date.fromisoformat, default=date(2025, 1, 2))
    parser.add_argument("--reference-end", type=date.fromisoformat, default=date(2025, 12, 23))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--priced-through",
        type=date.fromisoformat,
        default=None,
        help="last date whose bar is final (default: yesterday). The calendar extends past the "
        "present, so without this bound a signal's exit date can be a day that has not traded yet.",
    )
    args = parser.parse_args(argv)

    import audit_formula_replay_tushare as tushare_replay
    import backtest_operational_scores as backtest
    import materialize_frozen_factor_daily as frozen_mod
    import walk_forward_common as wfc

    model_ids = tuple(part.strip() for part in args.models.split(",") if part.strip())
    scores = pd.read_parquet(args.scores)
    scores["signal_date"] = [as_plain_date(value) for value in scores["signal_date"]]
    scores["ts_code"] = scores["ts_code"].astype(str)
    scores = scores.loc[
        scores["signal_date"].map(lambda day: args.start <= day <= args.end)
    ].reset_index(drop=True)
    if scores.empty:
        raise SystemExit(f"no scores inside {args.start} .. {args.end}")

    client = tushare_replay._connect()
    try:
        calendar = backtest.load_calendar(client, args.end + timedelta(days=30))
        open_dates = query_open_dates(
            client, args.start - timedelta(days=10), args.end + timedelta(days=30)
        )
    finally:
        client.close()
    priced_through = args.priced_through or (date.today() - timedelta(days=1))
    measurable = backtest.measurable_signal_dates(
        calendar, tuple(sorted(set(scores["signal_date"]))), priced_through=priced_through
    )
    if not measurable:
        raise SystemExit("no signal date in the window has a priced forward return yet")
    # Prices are needed through the last signal's exit date, which is past --end by design.
    last_exit = calendar.shift(measurable[-1], 6)
    price_dates = tuple(day for day in open_dates if day <= last_exit)
    market = tushare_replay.fetch_full_tushare_shsz_market(
        price_dates,
        token=os.getenv("QRV2_TUSHARE_TOKEN", ""),
        cache_dir=frozen_mod.CACHE,
        include_daily_basic=False,
    )
    market = backtest.with_adjusted_open(
        market.assign(
            trade_date=pd.to_datetime(market["trade_date"]),
            ts_code=market["ts_code"].astype(str),
        )
    ).loc[:, ["trade_date", "ts_code", "adjusted_open"]]
    returns = backtest.forward_returns(market, calendar, measurable)
    operational_labels = returns_to_labels(returns)
    print(
        f"operational window: {len(measurable)} measurable signal dates "
        f"({measurable[0]} .. {measurable[-1]}), {len(operational_labels)} label rows",
        flush=True,
    )

    store = ModelArtifactBundleStore(args.workspace)
    reference_frames = {
        model_id: _load_validation_predictions(store, model_id) for model_id in model_ids
    }
    holder = wfc.new_holder()
    # label_daily is versioned: without pinning the published version each key appears once per
    # publication, and the evaluator refuses duplicate signal-date keys.
    label_versions = wfc.current_versions(holder, "label_daily")
    _, published = wfc.query(
        holder,
        "select signal_date, ts_code, raw_return, label_valid, data_version from "
        f"loadTable('{wfc.TARGET_DATABASE}','label_daily') "
        f"where signal_date between {args.reference_start:%Y.%m.%d} "
        f"and {args.reference_end:%Y.%m.%d}",
    )
    if published is None or published.empty:
        raise SystemExit("reference window has no published labels")
    published = wfc.filter_current(published, "signal_date", label_versions)
    if published.empty:
        raise SystemExit("reference window has no labels at the published version")
    reference_labels = published.rename(columns={"raw_return": "raw_return_5d"})
    reference_labels["signal_date"] = [
        as_plain_date(value) for value in reference_labels["signal_date"]
    ]
    reference_labels["ts_code"] = reference_labels["ts_code"].astype(str)
    reference_labels = reference_labels.loc[
        :, ["signal_date", "ts_code", "raw_return_5d", "label_valid"]
    ]

    results: dict[str, object] = {}
    operational_frames = {}
    for model_id in model_ids:
        column = f"{model_id}_percentile"
        if column not in scores.columns:
            raise SystemExit(f"scores are missing the {column} column")
        operational_frames[model_id] = scores.loc[:, ["signal_date", "ts_code", column]].rename(
            columns={column: "score"}
        )

    results: dict[str, object] = {}
    for label, score_frames, label_frame in (
        ("reference", reference_frames, reference_labels),
        ("operational", operational_frames, operational_labels),
    ):
        window = {
            model_id: _ic_summary(frame, label_frame)
            for model_id, frame in score_frames.items()
        }
        # Join each model's scores onto shared business keys before averaging: two models emit
        # their rows in score order, so averaging positionally would blend different securities.
        joined: pd.DataFrame | None = None
        for model_id, frame in score_frames.items():
            current = frame.loc[:, ["signal_date", "ts_code", "score"]].rename(
                columns={"score": model_id}
            )
            joined = (
                current
                if joined is None
                else joined.merge(
                    current, on=["signal_date", "ts_code"], how="inner", validate="one_to_one"
                )
            )
        if joined is None:
            raise SystemExit("no model scores to combine")
        ensemble = ensemble_score(joined, tuple(score_frames))
        window["ensemble"] = _ic_summary(ensemble, label_frame)
        results[label] = window

    payload = {
        "scope": "OPERATIONAL_SCORE_DIAGNOSTIC_V1",
        "warning": (
            "the operational window is inside the locked 2026 test segment; diagnostic read only, "
            "not a basis for selection, tuning or retraining"
        ),
        "operational_window": [args.start.isoformat(), args.end.isoformat()],
        "reference_window": [args.reference_start.isoformat(), args.reference_end.isoformat()],
        "evaluator": "research_v2.evaluation.rank_ic (daily Spearman, minimum cross-section 30)",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for window_name, window in results.items():
        print(f"\n{window_name} window", flush=True)
        for key, summary in window.items():
            print(
                f"  {key:>16}: RankIC {summary['mean_rank_ic']:+.6f} | ICIR {summary['icir']:+.3f} "
                f"| positive {summary['positive_day_share']:.3f} | days {summary['valid_days']}",
                flush=True,
            )
    print(f"\nwrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
