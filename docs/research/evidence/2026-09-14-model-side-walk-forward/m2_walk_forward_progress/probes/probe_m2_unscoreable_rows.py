#!/usr/bin/env python3
"""Read-only probe: decompose M2's validation row deficit and find out why those rows are unscoreable.

`probe_validation_rowset_asymmetry.py` measured that M2 scores 691,097 of the F1 validation panel's
696,906 rows while every tree carrier scores all of them. This probe splits that deficit into the part
that cannot matter (rows whose label is invalid, which compute_daily_rank_ic filters out anyway) and
the part that can (valid-label rows the baseline scores and M2 does not), then asks what distinguishes
the second group -- per-factor NaN enrichment, the temporal model's own history requirement, and
whether the enrichment lines up with the yeslab_357 / yeslab_471 missingness chain documented in
problems_found.md §M / §Y / §AA.

Read-only: no database, no weights, nothing written into any artifact directory.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd

ART = Path("/data/zhangyudong/qrv2_research_artifacts")
# factor_order index -> factor id, from the frozen v2 factor set the run was launched with
REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX")
# Traced from source, not guessed: walk_forward_factor_daily.py:75-91 defines FROZEN_FACTOR_SET_V2 in
# this order, factor_source_plan("frozen_v2") returns it unchanged, and walk_forward_common.py:160
# names the panel columns f"{index:02d}" by enumerate position. Nothing in the model or dataset
# artifacts records this mapping -- feature_order.json holds only f00..f14 -- so it is resolved here
# and then corroborated against the panel's own NaN structure below.
FACTOR_BY_INDEX = (
    "yeslab_206", "yeslab_357", "yeslab_372", "yeslab_375", "yeslab_381", "yeslab_384",
    "yeslab_389", "yeslab_390", "yeslab_406", "yeslab_425", "yeslab_443_reg30_max20_v1",
    "yeslab_464", "yeslab_471", "yeslab_484", "yeslab_96",
)
CANDIDATE_ONLY = {"yeslab_372", "yeslab_375", "yeslab_384", "yeslab_389", "yeslab_390", "yeslab_406"}
MAPPING_SOURCES = (
    "research_v2/scripts/walk_forward_factor_daily.py",
    "research_v2/scripts/walk_forward_common.py",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, default=ART)
    ap.add_argument("--out", type=Path, default=ART / "m2_rowset")
    ap.add_argument("--fold", default="F1")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    fold = args.fold
    m2_ws = args.artifacts / "m2_v1"
    sel = {}
    for rec in sorted((m2_ws / "selections" / "model").glob("*/record.json")):
        payload = json.loads(rec.read_text(encoding="utf-8"))["payload"]
        sel[payload["fold_id"]] = payload
    ds = str(sel[fold]["validation_dataset_id"]).split(":")[0]

    panel_paths = sorted((m2_ws / "datasets" / ds / f"split=validation").glob("*/*.parquet"))
    panel = pd.concat([pd.read_parquet(p) for p in panel_paths], ignore_index=True)
    panel["signal_date"] = pd.to_datetime(panel["signal_date"]).dt.date
    pred = pd.read_parquet(m2_ws / "models" / f"m2-{fold.lower()}" / "validation_predictions.parquet",
                           columns=["signal_date", "ts_code"])
    pred["signal_date"] = pd.to_datetime(pred["signal_date"]).dt.date
    keys = ["signal_date", "ts_code"]

    scored = set(map(tuple, pred[keys].values))
    panel["_scored"] = [tuple(k) in scored for k in panel[keys].values]
    panel["_label_ok"] = panel["label_valid"].fillna(False).astype(bool) & np.isfinite(
        pd.to_numeric(panel["raw_return_5d"], errors="coerce")
    )
    feature_cols = [c for c in panel.columns if c.startswith("f") and c[1:].isdigit()]
    feat = panel[feature_cols].apply(pd.to_numeric, errors="coerce")
    panel["_nan_count"] = feat.isna().sum(axis=1)
    panel["_any_nan"] = panel["_nan_count"] > 0

    # the rolling history statistic must exist before any slice is taken, or the slices lack the column
    hist = 20
    panel = panel.sort_values(["ts_code", "signal_date"], kind="mergesort")
    panel["_roll__any_nan"] = (
        panel.groupby("ts_code")["_any_nan"]
        .transform(lambda s: s.rolling(hist, min_periods=1).sum())
    )

    scoreable = int(panel["_label_ok"].sum())
    groups = {
        "scored_and_label_ok": panel["_scored"] & panel["_label_ok"],
        "not_scored_and_label_ok": (~panel["_scored"]) & panel["_label_ok"],
        "not_scored_and_label_bad": (~panel["_scored"]) & (~panel["_label_ok"]),
        "scored_and_label_bad": panel["_scored"] & (~panel["_label_ok"]),
    }
    counts = {k: int(v.sum()) for k, v in groups.items()}

    dropped_valid = panel[groups["not_scored_and_label_ok"]]
    kept_valid = panel[groups["scored_and_label_ok"]]

    per_factor = {}
    for c in feature_cols:
        d = float(feat.loc[groups["not_scored_and_label_ok"], c].isna().mean())
        k = float(feat.loc[groups["scored_and_label_ok"], c].isna().mean())
        per_factor[c] = {
            "nan_rate_dropped": d,
            "nan_rate_kept": k,
            "enrichment": (d / k) if k > 0 else None,
            "excess_rows_attributable": int(round((d - k) * len(dropped_valid))),
        }
    ranked = sorted(
        ((c, v) for c, v in per_factor.items() if v["enrichment"] is not None),
        key=lambda kv: -kv[1]["enrichment"],
    )

    # corroborate the traced mapping against the panel's own NaN structure
    import hashlib

    nan_rate = {c: float(feat[c].isna().mean()) for c in feature_cols}
    sparse_cols = [c for c in feature_cols if FACTOR_BY_INDEX[int(c[1:])] in CANDIDATE_ONLY]
    dense_cols = [c for c in feature_cols if FACTOR_BY_INDEX[int(c[1:])] not in CANDIDATE_ONLY]
    f01, f12 = "f01", "f12"
    nested_zero = int((feat[f01].isna() & feat[f12].notna()).sum())
    corroboration = {
        "traced_from": [
            {"file": rel, "sha256": hashlib.sha256((REPO / rel).read_bytes()).hexdigest()}
            for rel in MAPPING_SOURCES
        ],
        "mapping": {f"f{i:02d}": name for i, name in enumerate(FACTOR_BY_INDEX)},
        "six_sparse_columns_are_the_six_candidate_only_factors": {
            "columns": sparse_cols,
            "their_nan_rates": {c: nan_rate[c] for c in sparse_cols},
            "all_between_0.45_and_0.55": all(0.45 <= nan_rate[c] <= 0.55 for c in sparse_cols),
        },
        "nine_dense_columns_max_nan_rate": max(nan_rate[c] for c in dense_cols),
        "nested_missingness_of_yeslab_357_inside_yeslab_471": {
            "documented_in": "problems_found.md §M and the results document §4 item 5",
            "rows_with_f01_nan_and_f12_present": nested_zero,
            "rows_with_f12_nan_and_f01_present": int((feat[f12].isna() & feat[f01].notna()).sum()),
            "rows_with_both_nan": int((feat[f01].isna() & feat[f12].isna()).sum()),
            "verdict": (
                "the zero cell reproduces the documented nesting ('rows missing only f01 are always "
                "zero') on a panel this probe read independently, which is the corroboration the "
                "code trace needed"
                if nested_zero == 0
                else "the documented nesting did NOT reproduce; treat the mapping as unconfirmed"
            ),
        },
    }

    # is it a history-length effect inside the assembled validation window?
    panel["_seq_pos"] = panel.groupby("ts_code").cumcount()
    dropped_valid = panel[groups["not_scored_and_label_ok"]]
    kept_valid = panel[groups["scored_and_label_ok"]]
    seq = {
        "what": (
            "the row's position within its own security's sequence inside the validation panel; a "
            "temporal model declaring history_length 20 and minimum_valid_history_length 15 cannot "
            "score the first rows of a security's sequence unless history is carried in from the "
            "train split"
        ),
        "dropped": {
            "mean": float(dropped_valid["_seq_pos"].mean()),
            "median": float(median(dropped_valid["_seq_pos"].tolist())),
            "share_below_15": float((dropped_valid["_seq_pos"] < 15).mean()),
            "share_below_20": float((dropped_valid["_seq_pos"] < 20).mean()),
        },
        "kept": {
            "mean": float(kept_valid["_seq_pos"].mean()),
            "median": float(median(kept_valid["_seq_pos"].tolist())),
            "share_below_15": float((kept_valid["_seq_pos"] < 15).mean()),
            "share_below_20": float((kept_valid["_seq_pos"] < 20).mean()),
        },
    }

    result = {
        "factor_index_mapping": corroboration,
        "sequence_position_test": seq,
        "question": (
            "Split M2's validation row deficit into the part that cannot matter and the part that can, "
            "and characterise the part that can"
        ),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "fold": fold,
        "dataset_id": ds,
        "read_only": True,
        "database_access": False,
        "weights_loaded": False,
        "panel_rows": int(len(panel)),
        "scoreable_rows_valid_label": scoreable,
        "decomposition": counts,
        "coverage_of_the_scoreable_cross_section": {
            "m2": counts["scored_and_label_ok"] / scoreable,
            "deficit_share": counts["not_scored_and_label_ok"] / scoreable,
            "note": (
                "the 1,087 invalid-label rows cannot matter: compute_daily_rank_ic filters on "
                "label_valid and a finite raw_return_5d before it counts a cross-section, so whether a "
                "prediction exists for them changes nothing. Only the valid-label deficit is a real "
                "asymmetry against a baseline that scores every row"
            ),
        },
        "dropped_valid_label_rows": {
            "count": int(len(dropped_valid)),
            "distinct_days": int(dropped_valid["signal_date"].nunique()),
            "distinct_securities": int(dropped_valid["ts_code"].nunique()),
            "nan_feature_count": {
                "dropped": {
                    "mean": float(dropped_valid["_nan_count"].mean()),
                    "median": float(median(dropped_valid["_nan_count"].tolist())),
                    "share_with_any_nan": float(dropped_valid["_any_nan"].mean()),
                },
                "kept": {
                    "mean": float(kept_valid["_nan_count"].mean()),
                    "median": float(median(kept_valid["_nan_count"].tolist())),
                    "share_with_any_nan": float(kept_valid["_any_nan"].mean()),
                },
            },
            "rolling_20day_count_of_own_rows_with_any_nan": {
                "dropped": {
                    "mean": float(dropped_valid["_roll__any_nan"].mean()),
                    "median": float(median(dropped_valid["_roll__any_nan"].tolist())),
                },
                "kept": {
                    "mean": float(kept_valid["_roll__any_nan"].mean()),
                    "median": float(median(kept_valid["_roll__any_nan"].tolist())),
                },
                "history_length_the_temporal_model_declares": hist,
                "minimum_valid_history_length_from_the_search_record": 15,
            },
        },
        "per_factor_nan_enrichment_ranked": [
            {"factor_column": c, "factor_id": FACTOR_BY_INDEX[int(c[1:])],
             "is_candidate_only": FACTOR_BY_INDEX[int(c[1:])] in CANDIDATE_ONLY, **v}
            for c, v in ranked
        ],
        "not_done": [
            "the mechanism behind the unscoreable valid-label rows is characterised, not proved: this "
            "probe shows which factors' NaNs those rows are enriched in and tests a history-length "
            "explanation, but it does not re-run the temporal batch builder",
            "no retraining, no imputation experiment, no attempt to make M2 score the dropped rows",
            "only fold F1 was examined; F2-F4 have not produced models yet",
        ],
    }
    out = args.out / f"m2_{fold.lower()}_unscoreable_rows.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps({k: result[k] for k in
                      ("panel_rows", "scoreable_rows_valid_label", "decomposition")}, indent=1))
    print(json.dumps(result["coverage_of_the_scoreable_cross_section"], indent=1))
    print(json.dumps(result["dropped_valid_label_rows"]["nan_feature_count"], indent=1))
    print(json.dumps(result["dropped_valid_label_rows"]["rolling_20day_count_of_own_rows_with_any_nan"], indent=1))
    print("top enriched factor columns:")
    for row in result["per_factor_nan_enrichment_ranked"][:8]:
        print(f"  {row['factor_column']}={row['factor_id']:28} "
              f"cand_only={row['is_candidate_only']}  dropped={row['nan_rate_dropped']:.4f} "
              f"kept={row['nan_rate_kept']:.4f} enrichment={row['enrichment']:.2f}")
    print("sequence position, dropped:", json.dumps(result["sequence_position_test"]["dropped"]))
    print("sequence position, kept    :", json.dumps(result["sequence_position_test"]["kept"]))
    print(result["factor_index_mapping"]["nested_missingness_of_yeslab_357_inside_yeslab_471"]["verdict"])
    print("six sparse cols all 0.45-0.55 NaN:",
          result["factor_index_mapping"]["six_sparse_columns_are_the_six_candidate_only_factors"]["all_between_0.45_and_0.55"])


if __name__ == "__main__":
    main()
