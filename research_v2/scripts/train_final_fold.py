#!/usr/bin/env python3
"""Train the FINAL fold (train 2020-2024, validation 2025) for an operational model.

WHY THIS EXISTS SEPARATELY

`run_real_candidate_training.py` can only run the four ORDINARY folds F1-F4. Its fold lists
are module constants, its `--folds` flag is parsed and then never referenced, and it extracts
train/validation/TEST for every fold -- so pointing it at FINAL raises `LockedPeriodError` on
the test segment. FINAL also needs a baseline entry that only exists for F1-F4, and a G1-G6
gate evaluation that is defined only over the ordinary TEST years.

This script trains FINAL for the operational purpose the owner asked for: a model whose train
window ends in 2024 and whose validation year is 2025, usable to score dates that are NOT the
locked segment.

WHAT IT DELIBERATELY DOES NOT DO

It never resolves, reads, writes, or evaluates the 2026 test segment, and it asserts that
every signal date it touches precedes 2026-01-01. It produces NO gate evidence, so the bundle
it writes is marked `OPERATIONAL_UNGATED` and must not be presented as a walk-forward-passed
production artifact. The governed path to a real 2026 evaluation is
`research_v2/runbooks/locked-test-2026.md`, and using 2026 to tune anything is outside what
that runbook authorises.

It reuses the training library and the driver's own helpers rather than reimplementing them,
so the carrier, hyperparameters, seeds and bundle layout stay identical to F1-F4.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "research_v2" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "research_v2" / "src"))

import run_real_candidate_training as drv  # noqa: E402
from walk_forward_common import (  # noqa: E402
    current_versions,
    factor_members_for,
    load_calendar,
    new_holder,
)
from walk_forward_factor_daily import extract_fn_for, factor_source_plan  # noqa: E402

from research_v2.datasets.preprocessing import FoldPreprocessor  # noqa: E402
from research_v2.datasets.splits import SplitResolver  # noqa: E402
from research_v2.inference.torch_scoring import prepare_frame  # noqa: E402
from research_v2.training.artifacts import CheckpointStore  # noqa: E402
from research_v2.training.ensemble import REQUIRED_SEEDS, build_seed_ensemble  # noqa: E402
from research_v2.training.model_bundle import ModelArtifactBundleStore  # noqa: E402
from research_v2.training.static_batches import build_static_training_batches  # noqa: E402

FINAL_FOLD = "FINAL"
#: first day of the locked 2026 test segment; nothing here may touch it
LOCKED_TEST_START = date(2026, 1, 1)
KEYS = ["signal_date", "ts_code"]
#: minimum names in a cross-section before its RankIC is counted, matching the driver
MINIMUM_CROSS_SECTION = 30


def mean_rank_ic(predictions: pd.DataFrame, frame: pd.DataFrame) -> tuple[float, int]:
    """Mean per-date Spearman RankIC between predicted percentile and realised 5-day return."""
    merged = predictions.loc[:, KEYS + ["percentile_rank"]].merge(
        frame.loc[:, KEYS + ["raw_return_5d", "label_valid"]], on=KEYS, validate="one_to_one"
    )
    merged = merged.loc[
        merged["label_valid"].fillna(False).astype(bool) & merged["raw_return_5d"].notna()
    ]
    ics: list[float] = []
    for _, group in merged.groupby("signal_date"):
        if len(group) < MINIMUM_CROSS_SECTION:
            continue
        value = group["percentile_rank"].corr(group["raw_return_5d"], method="spearman")
        if pd.notna(value):
            ics.append(float(value))
    if not ics:
        return float("nan"), 0
    return sum(ics) / len(ics), len(ics)


def _assert_pre_locked(frames: dict[str, pd.DataFrame]) -> dict[str, str]:
    """Refuse to proceed if any segment reaches the locked 2026 test window."""
    spans: dict[str, str] = {}
    for role, frame in frames.items():
        dates = pd.to_datetime(frame["signal_date"], errors="raise").dt.date
        latest = max(dates)
        span = f"{min(dates)} .. {latest}"
        spans[role] = span
        if latest >= LOCKED_TEST_START:
            raise SystemExit(
                f"{role} reaches {latest}, which is in or after the locked 2026 test segment. "
                f"This script must never touch 2026."
            )
    return spans


#: M1's carrier identity, mirroring how the driver names it for the ordinary folds
TORCH_FAMILY = {"M1": ("RESIDUAL_MLP", "residual-mlp-1", "_m1_model_factory")}


def run_torch_final(
    args: argparse.Namespace,
    frames: dict[str, pd.DataFrame],
    factor_order: tuple[str, ...],
    spans: dict[str, str],
) -> dict[str, Any]:
    """Train FINAL for a torch cross-section family (M1), reusing the driver's own fit path.

    The sequence is the driver's, minus everything bound to the ordinary folds: no dataset
    publication, no baseline evidence, no signed selection chain, no TEST extraction and no
    G1-G6 gates. Those all exist to produce governed evidence for F1-F4, which is not what
    this run is for.
    """
    family_name, model_code_version, factory_name = TORCH_FAMILY[args.family]
    model_factory = getattr(drv, factory_name)

    preprocessor = FoldPreprocessor(factor_order=factor_order)
    preprocessor.fit(frames["train"].loc[:, list(factor_order)], role="TRAIN")
    train_batches = build_static_training_batches(
        prepare_frame(frames["train"], preprocessor, factor_order), factor_order=factor_order
    )
    validation_batches = build_static_training_batches(
        prepare_frame(frames["validation"], preprocessor, factor_order), factor_order=factor_order
    )
    print(f"batches: train {len(train_batches)} dates, validation {len(validation_batches)} dates",
          flush=True)

    configs = drv.m1_config_grid(args.max_configs)
    search_seeds = tuple(REQUIRED_SEEDS[: args.search_seeds])
    dataset_ids = {"train": f"final-{args.family.lower()}-train",
                   "validation": f"final-{args.family.lower()}-validation"}
    checkpoint_store = CheckpointStore(args.workspace / "checkpoints")

    print(f"\nscreening {len(configs)} configs with {len(search_seeds)} seed(s), "
          f"max_epochs={args.max_epochs}", flush=True)
    trials: list[dict[str, Any]] = []
    best_config: dict[str, Any] | None = None
    best_ic = float("-inf")
    for index, payload in enumerate(configs, 1):
        ics = []
        for seed in search_seeds:
            model, _ = drv._train_cross_section_seed(
                payload, factor_order, train_batches, validation_batches,
                seed=seed,
                checkpoint_store=checkpoint_store,
                checkpoint_prefix=f"{args.family.lower()}-final-screen-{index}-{seed}",
                train_dataset_id=dataset_ids["train"],
                validation_dataset_id=dataset_ids["validation"],
                max_epochs=args.max_epochs,
                model_factory=model_factory,
                learning_rate=float(payload.get("learning_rate", 0.003)),
            )
            ics.append(drv._validation_rank_ic(
                drv._predict_validation_percentiles(model, validation_batches),
                frames["validation"]))
        ic = sum(ics) / len(ics)
        trials.append({"config": payload, "screened_mean_rank_ic": ic,
                       "screened_seed_rank_ic": ics, "screened_seeds": list(search_seeds)})
        print(f"  [{index:>2}/{len(configs)}] mean RankIC {ic:+.6f}  "
              f"hidden={payload.get('hidden_dim')} blocks={payload.get('num_blocks')} "
              f"dropout={payload.get('dropout')} lr={payload.get('learning_rate')}", flush=True)
        if pd.notna(ic) and ic > best_ic:
            best_ic, best_config = ic, payload
    if best_config is None:
        raise SystemExit("no config produced a finite validation RankIC")

    print(f"\nbest screened config {best_ic:+.6f}; final three-seed fit", flush=True)
    seed_checkpoints: dict[int, Path] = {}
    seed_predictions: dict[int, pd.DataFrame] = {}
    seed_rank_ic: dict[int, float] = {}
    for seed in REQUIRED_SEEDS:
        # the returned model is already the best-validation checkpoint, which is exactly what
        # the driver predicts from; reloading it from disk would only add a way to get it wrong
        model, checkpoint_path = drv._train_cross_section_seed(
            best_config, factor_order, train_batches, validation_batches,
            seed=seed,
            checkpoint_store=checkpoint_store,
            checkpoint_prefix=f"{args.family.lower()}-final-{seed}",
            train_dataset_id=dataset_ids["train"],
            validation_dataset_id=dataset_ids["validation"],
            max_epochs=args.max_epochs,
            model_factory=model_factory,
            learning_rate=float(best_config.get("learning_rate", 0.003)),
        )
        seed_checkpoints[seed] = checkpoint_path
        predictions = drv._predict_validation_percentiles(model, validation_batches)
        seed_predictions[seed] = predictions
        seed_rank_ic[seed] = drv._validation_rank_ic(predictions, frames["validation"])
        print(f"  seed {seed}: validation RankIC {seed_rank_ic[seed]:+.6f}", flush=True)

    ensemble = build_seed_ensemble(seed_predictions, validation_rank_ic=seed_rank_ic)
    print(f"  seed-mean ensemble: RankIC {ensemble.seed_mean_validation_rank_ic:+.6f} "
          f"(std {ensemble.seed_std_validation_rank_ic:.6f}, worst seed {ensemble.worst_seed})",
          flush=True)

    prototype = model_factory(best_config, factor_order)
    bundle = ModelArtifactBundleStore(args.workspace / "models").write(
        args.model_id,
        model_card={
            "family": family_name,
            "dataset_id": f"final-{args.family.lower()}-operational",
            "model_status": "OPERATIONAL_UNGATED",
            "model_code_version": model_code_version,
            "train_span": spans["train"],
            "validation_span": spans["validation"],
            "note": "trained for operational scoring; carries no G1-G6 gate evidence and "
                    "must not be represented as walk-forward-passed",
        },
        config=dict(best_config, raw_factor_count=len(factor_order)),
        signature=asdict(prototype.artifact_signature()),
        feature_order=factor_order,
        preprocessing=drv._preprocessing_artifact(preprocessor),
        seed_checkpoints=seed_checkpoints,
        validation_predictions=ensemble.predictions.rename(
            columns={"ensemble_score": "percentile_rank"}),
    )
    return {
        "configs_screened": len(configs),
        "screened_seeds": list(search_seeds),
        "screening_budget_epochs": args.max_epochs,
        "trials": trials,
        "best_config": best_config,
        "best_screened_mean_rank_ic": best_ic,
        "seed_validation_rank_ic": seed_rank_ic,
        "ensemble_validation_mean_rank_ic": ensemble.seed_mean_validation_rank_ic,
        "ensemble_validation_std_rank_ic": ensemble.seed_std_validation_rank_ic,
        "worst_seed": int(ensemble.worst_seed),
        "bundle_path": str(bundle.path),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", type=Path, required=True,
                        help="output root for the bundle; use a directory that cannot be "
                             "confused with a governed artifact workspace")
    parser.add_argument("--model-id", default="m0r-final-v1")
    parser.add_argument("--family", default="M0R", choices=("M0R", "M0", "M1"),
                        help="M0R/M0 are LightGBM carriers; M1 is the ResidualMLP torch "
                             "cross-section carrier")
    parser.add_argument("--factor-source", default="frozen_v2", choices=drv.FACTOR_SOURCES)
    parser.add_argument("--max-configs", type=int, default=None,
                        help="cap the config grid; M0R has a small pinned grid")
    parser.add_argument("--search-seeds", type=int, default=len(REQUIRED_SEEDS),
                        choices=range(1, len(REQUIRED_SEEDS) + 1),
                        help="seeds per config during screening (M1 only). The final fit always "
                             "uses all three required seeds so the bundle stays well-formed; "
                             "reducing this speeds up the screen at the cost of a noisier "
                             "config pick, and the report records what was used")
    parser.add_argument("--smoke", action="store_true",
                        help="one config, one screening seed, two epochs: verifies the whole "
                             "path including the bundle write, in minutes")
    parser.add_argument("--max-epochs", type=int, default=100,
                        help="accepted for parity with the driver; unused by LightGBM")
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def finish_torch(
    args: argparse.Namespace,
    frames: dict[str, pd.DataFrame],
    factor_order: tuple[str, ...],
    spans: dict[str, str],
) -> int:
    """Run the torch FINAL path and write its report in the same shape as the LightGBM one."""
    outcome = run_torch_final(args, frames, factor_order, spans)
    report = {
        "model_id": args.model_id,
        "family": args.family,
        "family_name": TORCH_FAMILY[args.family][0],
        "factor_source": args.factor_source,
        "factor_count": len(factor_order),
        "feature_order": list(factor_order),
        "spans": spans,
        "configs_searched": outcome["configs_screened"],
        "screened_seeds": outcome["screened_seeds"],
        "screening_budget_epochs": outcome["screening_budget_epochs"],
        "final_budget_epochs": args.max_epochs,
        "trials": outcome["trials"],
        "best_config": outcome["best_config"],
        "best_screened_mean_rank_ic": outcome["best_screened_mean_rank_ic"],
        "seed_validation_rank_ic": {
            str(k): v for k, v in outcome["seed_validation_rank_ic"].items()
        },
        "ensemble_validation_mean_rank_ic": outcome["ensemble_validation_mean_rank_ic"],
        "ensemble_validation_std_rank_ic": outcome["ensemble_validation_std_rank_ic"],
        "worst_seed": outcome["worst_seed"],
        "seeds": list(REQUIRED_SEEDS),
        "bundle_path": outcome["bundle_path"],
        "model_status": "OPERATIONAL_UNGATED",
        "touched_locked_test": False,
    }
    destination = args.report or (args.workspace / f"{args.model_id}-report.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    print(f"\nbundle  {outcome['bundle_path']}\nreport  {destination}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.smoke:
        args.max_configs = 1
        args.search_seeds = 1
        args.max_epochs = 2
    spec = (drv.M0R_SPEC if args.family == "M0R" else drv.M0_SPEC) \
        if args.family in ("M0", "M0R") else None

    holder = new_holder()
    resolver = SplitResolver(load_calendar(holder))
    factor_ids, version_tables = factor_source_plan(args.factor_source)
    versions = {table: current_versions(holder, table) for table in version_tables}
    factor_members = factor_members_for(factor_ids)
    factor_order = tuple(member.feature_name for member in factor_members)
    extract = extract_fn_for(args.factor_source)
    print(f"family={args.family} model_id={args.model_id} factor_source={args.factor_source} "
          f"({len(factor_ids)} factors)", flush=True)

    # train and validation only: resolving FINAL's test segment is what raises the lock
    frames = {
        "train": extract(holder, resolver, FINAL_FOLD, "train", factor_members, versions),
        "validation": extract(holder, resolver, FINAL_FOLD, "validation", factor_members, versions),
    }
    spans = _assert_pre_locked(frames)
    for role, frame in frames.items():
        print(f"{role:<11} {len(frame):>9,} rows  {spans[role]}", flush=True)

    if args.family in TORCH_FAMILY:
        return finish_torch(args, frames, factor_order, spans)

    assert spec is not None  # the dispatch above covers the torch families
    configs = spec.grid(args, args.max_configs)
    print(f"\nsearching {len(configs)} configs on validation RankIC (seed {REQUIRED_SEEDS[0]})",
          flush=True)
    trials: list[dict[str, Any]] = []
    best_config: dict[str, Any] | None = None
    best_ic = float("-inf")
    for index, payload in enumerate(configs, 1):
        model = spec.fit(payload, factor_order, frames["train"], frames["validation"],
                         seed=REQUIRED_SEEDS[0])
        predictions = drv._m0_percentile_frame(model, frames["validation"], factor_order)
        ic, days = mean_rank_ic(predictions, frames["validation"])
        trials.append({"config": payload, "validation_mean_rank_ic": ic, "valid_days": days})
        print(f"  [{index:>2}/{len(configs)}] RankIC {ic:+.6f} over {days} days  "
              f"leaves={payload.get('num_leaves')} lr={payload.get('learning_rate')}", flush=True)
        if pd.notna(ic) and ic > best_ic:
            best_ic, best_config = ic, payload
    if best_config is None:
        raise SystemExit("no config produced a finite validation RankIC")

    print(f"\nbest config RankIC {best_ic:+.6f}; refitting three seeds", flush=True)
    seed_checkpoints: dict[int, Path] = {}
    models_by_seed: dict[int, Any] = {}
    with tempfile.TemporaryDirectory() as staging:
        for seed in REQUIRED_SEEDS:
            model = spec.fit(best_config, factor_order, frames["train"], frames["validation"],
                             seed=seed)
            path = Path(staging) / f"seed-{seed}.pt"
            drv._m0_save_booster(model, path)
            seed_checkpoints[seed] = path
            models_by_seed[seed] = model
            ic = mean_rank_ic(
                drv._m0_percentile_frame(model, frames["validation"], factor_order),
                frames["validation"])[0]
            print(f"  seed {seed}: validation RankIC {ic:+.6f}", flush=True)

        ensemble = drv._m0_seed_mean_scores(models_by_seed, frames["validation"], factor_order)
        ensemble_ic, ensemble_days = mean_rank_ic(
            ensemble.rename(columns={"score": "percentile_rank"}), frames["validation"])
        print(f"  seed-mean ensemble: RankIC {ensemble_ic:+.6f} over {ensemble_days} days",
              flush=True)

        bundle = ModelArtifactBundleStore(args.workspace / "models").write(
            args.model_id,
            model_card={
                "family": spec.family_name,
                "dataset_id": f"final-{args.family.lower()}-operational",
                "model_status": "OPERATIONAL_UNGATED",
                "model_code_version": spec.model_code_version,
                "train_span": spans["train"],
                "validation_span": spans["validation"],
                "note": "trained for operational scoring; carries no G1-G6 gate evidence and "
                        "must not be represented as walk-forward-passed",
            },
            config=dict(best_config, raw_factor_count=len(factor_order)),
            signature=drv._m0_signature(best_config, len(factor_order),
                                        family_name=spec.family_name),
            feature_order=factor_order,
            preprocessing={"state_hash": "identity"},
            seed_checkpoints=seed_checkpoints,
            validation_predictions=ensemble.rename(columns={"score": "percentile_rank"}),
        )

    report = {
        "model_id": args.model_id,
        "family": spec.family,
        "family_name": spec.family_name,
        "factor_source": args.factor_source,
        "factor_count": len(factor_order),
        "feature_order": list(factor_order),
        "spans": spans,
        "configs_searched": len(configs),
        "trials": trials,
        "best_config": best_config,
        "best_validation_mean_rank_ic": best_ic,
        "ensemble_validation_mean_rank_ic": ensemble_ic,
        "ensemble_valid_days": ensemble_days,
        "seeds": list(REQUIRED_SEEDS),
        "bundle_path": str(bundle.path),
        "model_status": "OPERATIONAL_UNGATED",
        "touched_locked_test": False,
    }
    destination = args.report or (args.workspace / f"{args.model_id}-report.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    print(f"\nbundle  {bundle.path}\nreport  {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
