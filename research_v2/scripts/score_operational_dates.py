"""Score fixed dates with the operational (FINAL-window) bundles.

WHAT THIS PRODUCES

For each requested signal date, one row per universe member carrying every loaded model's
within-date percentile rank plus their equal-weight mean. Two carriers are supported, and each is
scored through the *same* code the reference training driver uses, so the numbers are comparable
to the validation predictions the bundles were selected on:

* ``M0R`` (LightGBM rank-regression): the boosters read back from the verified bundle are scored
  through ``run_real_candidate_training``'s own four-seed helpers.
* ``M1`` (ResidualMLP): the bundle is loaded by ``load_operational_bundle`` -- which refuses
  anything not marked ``OPERATIONAL_UNGATED`` and makes no promotion claim -- and scored through
  the label-free inference builder with the winsor/impute state restored from the bundle.

WHAT IT DELIBERATELY DOES NOT DO

It computes no label, no RankIC, and no return. Nothing here reads ``label_daily``; a score is a
ranking decision, and turning it into a measured outcome is a separate, explicitly-authorized
step. It also refuses to invent a universe: dates outside the published ``cn_liquid_a`` coverage
are dropped rather than back-filled, because scoring a date the universe did not cover would
produce ranks the models were never trained to produce.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "research_v2" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "research_v2" / "src"))

FROZEN_V2_PATH = (
    REPO_ROOT / "research_v2" / "artifacts" / "diagnostics" / "factor_set_frozen_v2.json"
)
FACTOR_COLUMNS = [
    "signal_date",
    "ts_code",
    "factor_id",
    "factor_version",
    "value",
    "available_date",
]


def feature_factor_pairs() -> tuple[tuple[str, str], ...]:
    """The frozen v2 ``feature_name -> factor_id`` mapping, in the training contract's order.

    Feature names are positional (``f00`` is the first frozen factor). Both the frozen JSON and
    ``walk_forward_factor_daily.FROZEN_FACTOR_SET_V2`` are read and compared, because a drift
    between them is exactly the failure this mapping exists to prevent.
    """
    import walk_forward_factor_daily as wffd

    contract_order = tuple(wffd.FROZEN_FACTOR_SET_V2)
    payload = json.loads(FROZEN_V2_PATH.read_text(encoding="utf-8"))
    declared = tuple(str(factor) for factor in payload["factors"])
    if contract_order != declared:
        raise ValueError(
            "frozen v2 factor order disagrees between the training contract "
            f"({list(contract_order)}) and factor_set_frozen_v2.json ({list(declared)})"
        )
    return tuple((f"f{index:02d}", factor) for index, factor in enumerate(contract_order))


def factor_members() -> tuple[object, ...]:
    """``FactorInput`` members for the label-free inference builder, in contract order."""
    from research_v2.datasets.builder import FactorInput

    return tuple(
        FactorInput(factor_id=factor_id, factor_version=1, feature_name=name)
        for name, factor_id in feature_factor_pairs()
    )


def missing_factor_ids(frame: pd.DataFrame, required: Sequence[str]) -> tuple[str, ...]:
    """Required factor ids with no row at all in a replayed window."""
    present = set(frame["factor_id"].astype(str).unique())
    return tuple(factor for factor in required if factor not in present)


def _rank_frame(rows: list[tuple[object, str, float]], column: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_date": [row[0] for row in rows],
            "ts_code": [row[1] for row in rows],
            column: [row[2] for row in rows],
        }
    )


def combine_model_ranks(
    ranks_by_model: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Mean the models' within-date percentile ranks, aligned on business keys.

    Two models emit their rows in score order, so combining positionally would blend different
    stocks. A key that only one model scored is an error rather than a silent inner join: it means
    the models disagree about which securities were scorable that day, and averaging the remainder
    would quietly narrow the universe.
    """
    if not ranks_by_model:
        raise ValueError("at least one model is required")
    keys = ["signal_date", "ts_code"]
    merged: pd.DataFrame | None = None
    model_columns: list[str] = []
    for model_id, frame in ranks_by_model.items():
        missing = sorted(set(keys + ["percentile_rank"]).difference(frame.columns))
        if missing:
            raise ValueError(f"model {model_id} ranks are missing columns: {missing}")
        if frame.duplicated(keys).any():
            raise ValueError(f"model {model_id} ranks contain duplicate business keys")
        column = f"{model_id}_percentile"
        current = frame.loc[:, keys + ["percentile_rank"]].rename(
            columns={"percentile_rank": column}
        )
        if merged is None:
            merged = current
        else:
            merged = merged.merge(
                current, on=keys, how="outer", validate="one_to_one", indicator=True
            )
            if not merged["_merge"].eq("both").all():
                offenders = merged.loc[~merged["_merge"].eq("both"), keys].head(5)
                raise ValueError(
                    "models scored different business keys; first offenders: "
                    f"{offenders.to_dict('records')}"
                )
            merged = merged.drop(columns=["_merge"])
        model_columns.append(column)
    if merged is None:
        raise ValueError("at least one model is required")
    merged["ensemble_percentile"] = merged.loc[:, model_columns].mean(axis=1)
    return merged.loc[:, [*keys, *model_columns, "ensemble_percentile"]].reset_index(drop=True)


def _read_bundle_family(store: object, model_id: str) -> str:
    card = store.load_model_card(model_id)
    family = card.get("family")
    if not isinstance(family, str):
        raise ValueError(f"model bundle {model_id} records no family")
    return family


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factors", type=Path, required=True, help="replayed factor parquet")
    parser.add_argument("--workspace", type=Path, required=True, help="directory holding bundles")
    parser.add_argument("--models", type=str, required=True, help="comma-separated bundle ids")
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--universe-from", choices=("db", "parquet"), default="db")
    parser.add_argument("--universe-parquet", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    if args.start > args.end:
        raise SystemExit("--start must not be after --end")
    if args.universe_from == "parquet" and args.universe_parquet is None:
        raise SystemExit("--universe-from parquet requires --universe-parquet")

    import materialize_operational_factors as operational

    from research_v2.inference.artifact_loader import load_operational_bundle
    from research_v2.inference.builder import build_label_free_inference_batch
    from research_v2.inference.predictor import predict_three_seed_ensemble
    from research_v2.training.ensemble import REQUIRED_SEEDS
    from research_v2.training.model_bundle import ModelArtifactBundleStore

    pairs = feature_factor_pairs()
    factor_order = tuple(name for name, _ in pairs)
    required_ids = tuple(factor for _, factor in pairs)

    factors = pd.read_parquet(args.factors)
    missing_columns = sorted(set(FACTOR_COLUMNS).difference(factors.columns))
    if missing_columns:
        raise SystemExit(f"factor parquet is missing columns: {missing_columns}")
    factors["signal_date"] = pd.to_datetime(factors["signal_date"]).dt.normalize().dt.date
    factors["available_date"] = pd.to_datetime(factors["available_date"]).dt.normalize().dt.date
    factors["ts_code"] = factors["ts_code"].astype(str)
    late = factors.loc[factors["available_date"] > factors["signal_date"]]
    if not late.empty:
        raise SystemExit(f"{len(late)} factor rows are published after their signal date")
    absent = missing_factor_ids(factors, required_ids)
    if absent:
        raise SystemExit(f"factor parquet does not carry the frozen v2 factors: {sorted(absent)}")
    raw_dates = sorted(set(factors["signal_date"]))
    scoring_dates = operational.window_open_dates(raw_dates, args.start, args.end)

    if args.universe_from == "db":
        import audit_formula_replay_tushare as tushare_replay
        import materialize_frozen_factor_daily as frozen_mod

        client = tushare_replay._connect()
        try:
            membership = frozen_mod.load_official_universe(client, args.start, args.end)
        finally:
            client.close()
    else:
        universe_frame = pd.read_parquet(args.universe_parquet)
        if "is_member" in universe_frame.columns:
            universe_frame = universe_frame.loc[universe_frame["is_member"].astype(bool)]
        universe_frame = universe_frame.copy()
        universe_frame["signal_date"] = (
            pd.to_datetime(universe_frame["signal_date"]).dt.normalize().dt.date
        )
        universe_frame["ts_code"] = universe_frame["ts_code"].astype(str)
        membership = {
            day: set(group["ts_code"]) for day, group in universe_frame.groupby("signal_date")
        }

    covered_dates = tuple(day for day in scoring_dates if membership.get(day))
    dropped = [day.isoformat() for day in scoring_dates if not membership.get(day)]
    if not covered_dates:
        raise SystemExit(
            f"none of {len(scoring_dates)} scoring dates have published universe membership"
        )
    if dropped:
        print(
            f"skipping {len(dropped)} dates with no universe membership: {dropped[:5]}",
            flush=True,
        )

    universe_rows = pd.DataFrame(
        [
            {"signal_date": day, "ts_code": code, "is_member": True}
            for day in covered_dates
            for code in sorted(membership[day])
        ]
    )
    store = ModelArtifactBundleStore(args.workspace)
    model_ids = tuple(part.strip() for part in args.models.split(",") if part.strip())
    if not model_ids:
        raise SystemExit("at least one model id is required")

    ranks_by_model: dict[str, pd.DataFrame] = {}
    audit: dict[str, object] = {}
    scored = set(covered_dates)
    window_factors = factors.loc[factors["signal_date"].isin(scored)]
    window_universe = universe_rows

    for model_id in model_ids:
        family = _read_bundle_family(store, model_id)
        print(f"scoring {model_id} ({family}) on {len(covered_dates)} dates", flush=True)
        if family == "RESIDUAL_MLP":
            loaded = load_operational_bundle(
                store, model_id, expected_feature_order=factor_order, device=_device(args.device)
            )
            models = {
                seed: model for seed, model in zip(REQUIRED_SEEDS, loaded.models, strict=True)
            }
            frames = []
            for day in covered_dates:
                batch = build_label_free_inference_batch(
                    window_universe,
                    window_factors,
                    factor_members=factor_members(),
                    signal_date=day,
                    preprocessor=loaded.preprocessor,
                )
                prediction = predict_three_seed_ensemble(batch, models)
                frames.append(
                    prediction.ensemble.rename(columns={"ensemble_score": "percentile_rank"})
                )
            ranks_by_model[model_id] = _rank_frame(
                [
                    (row.signal_date, str(row.ts_code), float(row.percentile_rank))
                    for row in pd.concat(frames, ignore_index=True).itertuples(index=False)
                ],
                "percentile_rank",
            )
            assert loaded.audit is not None
            audit[model_id] = {
                "family": family,
                "bundle_digest": loaded.audit.bundle_digest,
                "code_version": loaded.audit.code_version,
                "models": len(loaded.models),
                "promotion_claim": False,
            }
        elif family == "LGBM_RANKREG":
            import run_real_candidate_training as drv

            wide = _wide_factor_frame(window_factors, window_universe)
            models_by_seed = {}
            for seed in REQUIRED_SEEDS:
                models_by_seed[seed] = drv._m0_load_booster(
                    store, model_id, seed, carrier="rankreg"
                )
            seed_mean = drv._m0_seed_mean_scores(models_by_seed, wide, factor_order)
            ranks_by_model[model_id] = _rank_frame(
                [
                    (row.signal_date, str(row.ts_code), float(row.score))
                    for row in seed_mean.itertuples(index=False)
                ],
                "percentile_rank",
            )
            audit[model_id] = {
                "family": family,
                "bundle_digest": store.bundle_digest(model_id),
                "code_version": "lgbm-rankreg-1",
                "models": len(models_by_seed),
                "promotion_claim": False,
            }
        else:
            raise SystemExit(f"unsupported operational family {family!r} for bundle {model_id}")

    combined = combine_model_ranks(ranks_by_model)
    combined = combined.sort_values(
        ["signal_date", "ensemble_percentile", "ts_code"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.output, index=False)
    sidecar = {
        "scope": "OPERATIONAL_SCORES_V1",
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "signal_dates": [day.isoformat() for day in covered_dates],
        "skipped_dates": dropped,
        "factor_source": str(args.factors),
        "factor_source_sha256": hashlib.sha256(args.factors.read_bytes()).hexdigest(),
        "feature_factor_order": [[name, factor] for name, factor in pairs],
        "rows": int(len(combined)),
        "models": audit,
        "output": str(args.output),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "reads_labels": False,
    }
    sidecar_path = args.output.with_suffix(".meta.json")
    sidecar_path.write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(combined)} score rows to {args.output}", flush=True)
    print(f"wrote metadata to {sidecar_path}", flush=True)
    return 0


def _device(name: str):
    import torch

    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")
    return device


def _wide_factor_frame(factors: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long replay frame into one row per (signal_date, ts_code) with f00..f14."""
    pairs = feature_factor_pairs()
    index_by_factor = {factor: name for name, factor in pairs}
    ordered = factors.loc[factors["factor_id"].astype(str).isin(index_by_factor)]
    wide = ordered.pivot_table(
        index=["signal_date", "ts_code"],
        columns="factor_id",
        values="value",
        aggfunc="last",
    ).rename(columns=index_by_factor)
    wide = wide.reindex(columns=[name for name, _ in pairs])
    wide.columns.name = None
    keys = universe.loc[:, ["signal_date", "ts_code"]].drop_duplicates()
    joined = keys.merge(wide.reset_index(), on=["signal_date", "ts_code"], how="left")
    return joined.loc[:, ["signal_date", "ts_code", *[name for name, _ in pairs]]].reset_index(
        drop=True
    )


if __name__ == "__main__":
    raise SystemExit(main())
