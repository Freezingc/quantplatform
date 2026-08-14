"""Evaluate isolated scale-normalized Alpha95 research candidates."""

import argparse
import json
import os
from collections.abc import Mapping
from datetime import date, timedelta
from pathlib import Path

import audit_formula_replay_consistency as replay
import numpy as np
import pandas as pd
from audit_formula_replay_tushare import (
    TushareMarketSession,
    _calculator_market_start,
    _connect,
    _fetch_tushare_market,
    _query_open_dates,
    required_market_fields,
    requires_daily_basic,
)
from evaluate_tushare_scope_validation import (
    _load_validation_inputs,
    compute_daily_rank_ic,
    compute_pair_daily_spearman,
)

CANDIDATE_WINDOWS = (10, 20, 40)
LOCKED_TEST_START = date(2026, 1, 1)
FORMAL_FACTOR_IDS = (
    "yeslab_380",
    "yeslab_431",
    "yeslab_437",
    "yeslab_155",
    "yeslab_98",
    "yeslab_94",
    "yeslab_376",
    "yeslab_396",
    "yeslab_442",
)
TRAIN_START = date(2020, 1, 2)
TRAIN_END = date(2024, 12, 31)
VALIDATION_START = date(2025, 1, 2)
VALIDATION_END = date(2025, 12, 31)
WARMUP_CALENDAR_DAYS = 120


def compute_relative_amount_volatility(market: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return trailing amount standard deviation relative to trailing mean."""
    required = {"trade_date", "ts_code", "amount"}
    if missing := sorted(required.difference(market.columns)):
        raise ValueError(f"market frame is missing columns: {missing}")
    amount = market.pivot(index="trade_date", columns="ts_code", values="amount").sort_index()
    deviation = amount.rolling(window=window, min_periods=window).std(ddof=1)
    baseline = amount.rolling(window=window, min_periods=window).mean()
    return deviation.div(baseline.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)


def passes_training_gate(row: Mapping[str, float]) -> bool:
    """Return whether one candidate clears every fixed training gate."""
    return (
        row["signed_mean_rank_ic"] > 0.02
        and row["signed_icir"] >= 0.50
        and row["directional_day_share"] >= 0.55
        and row["annual_double_gate_pass_years"] >= 4
        and row["max_similarity"] < 0.90
    )


def select_training_candidate(rows: Mapping[str, Mapping[str, float]]) -> str | None:
    """Select the highest-ICIR eligible candidate using training-only evidence."""
    eligible = [
        (candidate_id, row)
        for candidate_id, row in rows.items()
        if passes_training_gate(row)
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda item: (item[1]["signed_icir"], -item[1]["max_similarity"], item[0]),
    )[0]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse a bounded read-only candidate evaluation command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2020, 1, 2))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("research_v2/artifacts/tushare_cache"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.start > args.end or args.end >= LOCKED_TEST_START:
        parser.error("evaluation range must be before 2026 locked-test and start <= end")
    return args


def passes_validation_gate(row: Mapping[str, float]) -> bool:
    """Return whether a training-selected candidate confirms on validation."""
    return (
        row["signed_mean_rank_ic"] > 0.02
        and row["signed_icir"] >= 0.50
        and row["directional_day_share"] >= 0.55
        and row["max_similarity"] < 0.90
    )


def build_review(
    *,
    selected_id: str | None,
    training: Mapping[str, object],
    validation: Mapping[str, float],
) -> dict[str, object]:
    """Build a non-publishing admission review from frozen candidate evidence."""
    status = (
        "REJECT_TRAINING_GATE"
        if selected_id is None
        else (
            "READY_FOR_EXPLICIT_ADMISSION_REVIEW"
            if passes_validation_gate(validation)
            else "REJECT_VALIDATION_GATE"
        )
    )
    return {
        "selected_id": selected_id,
        "training": dict(training),
        "validation": dict(validation),
        "status": status,
        "database_writes": False,
        "formal_factor_set_changed": False,
        "model_training_started": False,
        "locked_test_read": False,
    }


def summarize_rank_ic(daily: pd.DataFrame, direction: int) -> dict[str, float | int]:
    """Summarize finite daily RankIC values under a frozen direction."""
    required = {"signal_date", "rank_ic"}
    if missing := sorted(required.difference(daily.columns)):
        raise ValueError(f"daily RankIC is missing columns: {missing}")
    if direction not in {-1, 1}:
        raise ValueError("direction must be -1 or 1")
    values = pd.to_numeric(daily["rank_ic"], errors="coerce")
    values = values.loc[np.isfinite(values.to_numpy(float))].to_numpy(float)
    if values.size == 0:
        raise ValueError("daily RankIC has no finite observations")
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if values.size > 1 else float("nan")
    signed_mean = mean * direction
    signed_icir = signed_mean / std if np.isfinite(std) and std > 0 else float("nan")
    return {
        "valid_day_count": int(values.size),
        "mean_rank_ic": mean,
        "icir": mean / std if np.isfinite(std) and std > 0 else float("nan"),
        "signed_mean_rank_ic": signed_mean,
        "signed_icir": signed_icir,
        "directional_day_share": float(((values * direction) > 0).mean()),
        "direction": direction,
    }


def summarize_training_candidate(daily: pd.DataFrame) -> dict[str, object]:
    """Freeze direction from training and count annual double-gate passes."""
    values = pd.to_numeric(daily["rank_ic"], errors="coerce")
    values = values.loc[np.isfinite(values.to_numpy(float))]
    if values.empty:
        raise ValueError("training daily RankIC has no finite observations")
    direction = -1 if float(values.mean()) < 0 else 1
    frame = daily.copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="raise")
    annual: dict[str, dict[str, float | int]] = {}
    annual_passes = 0
    for year, group in frame.groupby(frame["signal_date"].dt.year, sort=True):
        metrics = summarize_rank_ic(group, direction)
        annual[str(int(year))] = metrics
        if (
            metrics["signed_mean_rank_ic"] > 0.02
            and metrics["signed_icir"] >= 0.50
            and metrics["directional_day_share"] >= 0.55
        ):
            annual_passes += 1
    return {
        **summarize_rank_ic(frame, direction),
        "annual": annual,
        "annual_double_gate_pass_years": annual_passes,
    }


def max_similarity_to_references(
    pairs: pd.DataFrame,
    candidate_id: str,
    reference_ids: set[str] | tuple[str, ...],
) -> dict[str, object]:
    """Return the candidate's highest complete similarity to a reference set."""
    required = {
        "factor_id_left",
        "factor_id_right",
        "median_abs_spearman",
        "valid_day_count",
    }
    if missing := sorted(required.difference(pairs.columns)):
        raise ValueError(f"similarity pairs are missing columns: {missing}")
    frame = pairs.copy()
    frame["factor_id_left"] = frame["factor_id_left"].astype(str)
    frame["factor_id_right"] = frame["factor_id_right"].astype(str)
    frame["formal_factor_id"] = np.where(
        frame["factor_id_left"] == candidate_id,
        frame["factor_id_right"],
        frame["factor_id_left"],
    )
    frame = frame.loc[
        (frame["factor_id_left"] == candidate_id)
        | (frame["factor_id_right"] == candidate_id)
    ]
    frame = frame.loc[frame["formal_factor_id"].isin(set(reference_ids))].copy()
    frame["median_abs_spearman"] = pd.to_numeric(
        frame["median_abs_spearman"], errors="coerce"
    )
    frame = frame.loc[np.isfinite(frame["median_abs_spearman"].to_numpy(float))]
    if frame.empty:
        raise ValueError(f"no similarity evidence for {candidate_id}")
    nearest = frame.sort_values(
        ["median_abs_spearman", "valid_day_count", "formal_factor_id"],
        ascending=[False, False, True],
        kind="mergesort",
    ).iloc[0]
    return {
        "factor_id": str(nearest["formal_factor_id"]),
        "value": float(nearest["median_abs_spearman"]),
        "valid_day_count": int(nearest["valid_day_count"]),
    }


def max_formal_similarity(pairs: pd.DataFrame, candidate_id: str) -> dict[str, object]:
    """Return the candidate's highest complete pairwise similarity to the formal nine."""
    return max_similarity_to_references(pairs, candidate_id, FORMAL_FACTOR_IDS)


def _candidate_id(window: int) -> str:
    return f"yeslab_163_relative_amount_volatility_{window}_v1"


def _panel_to_replay_values(panel: pd.DataFrame) -> pd.DataFrame:
    values = panel.stack(dropna=True).rename("replay_value").reset_index()
    values.columns = ["trade_date", "ts_code", "replay_value"]
    values["trade_date"] = pd.to_datetime(values["trade_date"], errors="raise")
    values["ts_code"] = values["ts_code"].astype(str)
    return values


def _evaluate_window(
    client: object,
    *,
    start: date,
    end: date,
    cache_dir: Path,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    inputs = _load_validation_inputs(client, start, end)
    codes = tuple(sorted(inputs["ts_code"].astype(str).unique()))
    definitions = [replay.REPLAY_DEFINITIONS[factor_id] for factor_id in FORMAL_FACTOR_IDS]
    formal_market_start = min(
        _calculator_market_start(client, definition, start, end) for definition in definitions
    )
    market_start = min(formal_market_start, start - timedelta(days=WARMUP_CALENDAR_DAYS))
    open_dates = _query_open_dates(client, market_start, end)
    required_fields = required_market_fields(definitions)
    market = _fetch_tushare_market(
        codes,
        market_start,
        end,
        token=os.getenv("QRV2_TUSHARE_TOKEN", ""),
        open_dates=open_dates,
        cache_dir=cache_dir,
        include_daily_basic=requires_daily_basic(required_fields),
        include_adjusted_fields=bool(
            {"open_hfq", "high_hfq", "low_hfq", "close_hfq", "pre_close_hfq"}
            & set(required_fields)
        ),
    )
    session = TushareMarketSession(client, market, codes)
    replay_frames: dict[str, pd.DataFrame] = {}
    for definition in definitions:
        replay_frames[definition.factor_id] = replay._replay_factor(
            session,
            definition,
            start,
            end,
        )
    candidate_daily: dict[str, pd.DataFrame] = {}
    for window in CANDIDATE_WINDOWS:
        candidate_id = _candidate_id(window)
        values = _panel_to_replay_values(compute_relative_amount_volatility(market, window))
        replay_frames[candidate_id] = values
        candidate_daily[candidate_id] = compute_daily_rank_ic(values, inputs)
    return candidate_daily, compute_pair_daily_spearman(replay_frames)


def _summary_with_similarity(
    daily: pd.DataFrame,
    *,
    direction: int,
    pairs: pd.DataFrame,
    candidate_id: str,
    reference_ids: set[str],
) -> dict[str, object]:
    summary = summarize_rank_ic(daily, direction)
    nearest = max_similarity_to_references(pairs, candidate_id, reference_ids)
    return {
        **summary,
        "max_similarity": nearest["value"],
        "max_similarity_reference": nearest,
    }


def run_evaluation(args: argparse.Namespace, *, client: object | None = None) -> dict[str, object]:
    """Run one training selection and one validation confirmation without writes."""
    own_client = client is None
    client = client or _connect()
    candidate_ids = tuple(_candidate_id(window) for window in CANDIDATE_WINDOWS)
    try:
        train_daily, train_pairs = _evaluate_window(
            client,
            start=TRAIN_START,
            end=TRAIN_END,
            cache_dir=args.cache_dir,
        )
        training: dict[str, dict[str, object]] = {}
        selection_rows: dict[str, dict[str, float]] = {}
        selection_references = set(FORMAL_FACTOR_IDS).union(candidate_ids)
        for candidate_id, daily in train_daily.items():
            summary = summarize_training_candidate(daily)
            nearest = max_similarity_to_references(
                train_pairs,
                candidate_id,
                selection_references.difference({candidate_id}),
            )
            summary["max_similarity"] = nearest["value"]
            summary["max_similarity_reference"] = nearest
            training[candidate_id] = summary
            selection_rows[candidate_id] = {
                "signed_mean_rank_ic": float(summary["signed_mean_rank_ic"]),
                "signed_icir": float(summary["signed_icir"]),
                "directional_day_share": float(summary["directional_day_share"]),
                "annual_double_gate_pass_years": float(summary["annual_double_gate_pass_years"]),
                "max_similarity": float(summary["max_similarity"]),
            }
        selected_id = select_training_candidate(selection_rows)
        validation_daily, validation_pairs = _evaluate_window(
            client,
            start=VALIDATION_START,
            end=VALIDATION_END,
            cache_dir=args.cache_dir,
        )
        validation: dict[str, dict[str, object]] = {}
        for candidate_id, daily in validation_daily.items():
            direction = int(training[candidate_id]["direction"])
            validation[candidate_id] = _summary_with_similarity(
                daily,
                direction=direction,
                pairs=validation_pairs,
                candidate_id=candidate_id,
                reference_ids=set(FORMAL_FACTOR_IDS),
            )
        selected_validation = validation[selected_id] if selected_id is not None else {}
        review = build_review(
            selected_id=selected_id,
            training=training.get(selected_id, {"selection_rows": selection_rows}),
            validation=selected_validation,
        )
        review.update(
            {
                "scope": "ALPHA95_RELATIVE_AMOUNT_VOLATILITY_CANONICAL_V1",
                "candidate_ids": list(candidate_ids),
                "market_scope": "SH/SZ",
                "excluded_exchanges": ["BJ"],
                "label": "forward_open_5d:v1 raw five-trading-day adjusted-open return",
                "training_window": f"{TRAIN_START.isoformat()}/{TRAIN_END.isoformat()}",
                "validation_window": f"{VALIDATION_START.isoformat()}/{VALIDATION_END.isoformat()}",
                "selection_rows": selection_rows,
            }
        )
        output = args.output
        output.mkdir(parents=True, exist_ok=True)
        daily_rows: list[pd.DataFrame] = []
        for phase, frames in (("train", train_daily), ("validation", validation_daily)):
            for candidate_id, daily in frames.items():
                frame = daily.copy()
                frame.insert(0, "factor_id", candidate_id)
                frame.insert(1, "phase", phase)
                daily_rows.append(frame)
        pd.concat(daily_rows, ignore_index=True).to_csv(output / "daily_rank_ic.csv", index=False)
        all_pairs = pd.concat(
            [
                train_pairs.assign(phase="train"),
                validation_pairs.assign(phase="validation"),
            ],
            ignore_index=True,
        )
        all_pairs.to_json(output / "similarity_summary.json", orient="records", indent=2)
        quality = {
            "scope": "ALPHA95_RELATIVE_AMOUNT_VOLATILITY_CANONICAL_V1",
            "candidate_windows": list(CANDIDATE_WINDOWS),
            "candidates": {
                candidate_id: {
                    "train_2020_2024": training[candidate_id],
                    "validation_2025": validation[candidate_id],
                }
                for candidate_id in candidate_ids
            },
            "database_writes": False,
            "formal_factor_set_changed": False,
            "model_training_started": False,
            "locked_test_read": False,
        }
        (output / "quality_summary.json").write_text(
            json.dumps(quality, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        (output / "admission_review.json").write_text(
            json.dumps(review, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        return review
    finally:
        if own_client:
            client.close()


def main(argv: list[str] | None = None) -> int:
    print(json.dumps(run_evaluation(parse_args(argv)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
