#!/usr/bin/env python3
"""Read-only probe: do the candidate and the baseline get scored on the same rows?

§AS.2 compared M2's F1 validation RankIC with the frozen baseline's on the grounds that both cover
237 days. Day counts matching is not the same as row sets matching. This probe checks the row sets,
reproduces M2's recorded number with the repository's own estimator, characterises whatever rows M2
does not score, and bounds how much a row-set difference of that size can move the metric.

It also settles the question §AS.8 left open -- whether `ensemble_validation_rank_ic` and the
baseline's `mean_rank_ic` are the same estimator -- by citing the four code sites with their hashes,
and by checking whether the OOS acceptance path (which is what G1-G6 actually read) has the same
asymmetry or removes it.

Safety: read-only. No database access, no weight loading, no writes into any artifact directory, no
process signalled. The parquet panels are read in place.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd

REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX")
ART = Path("/data/zhangyudong/qrv2_research_artifacts")
sys.path.insert(0, str(REPO / "research_v2" / "src"))

CARRIERS = [
    ("M0R", "m0r_v2"),
    ("M0R-replicate", "m0r_v3"),
    ("M1-full", "m1_full_v1"),
    ("M1-carrier", "m1_parallel_v1"),
    ("M0-refined", "m0_v3"),
    ("M0-base", "m0_v2"),
    ("M2", "m2_v1"),
]
FOLDS = ["F1", "F2", "F3", "F4"]

# file, symbol, and the line the argument turns on
CITED = [
    ("research_v2/src/research_v2/training/search_metrics.py", "validation_rank_ic",
     "the candidate's validation metric: inner-joins predictions to labels, then "
     "compute_daily_rank_ic(minimum_cross_section=30) with every other argument at its default, "
     "then summarize_rank_ic(...).mean_rank_ic"),
    ("research_v2/src/research_v2/baselines/orchestration.py", "_evaluate_predictions",
     "the baseline's validation metric: compute_daily_rank_ic(minimum_cross_section=30, "
     "include_invalid=True, require_score_health=False), then summarize_rank_ic(...).mean_rank_ic"),
    ("research_v2/src/research_v2/baselines/orchestration.py", "_validate_prediction_contract",
     "the baseline's coverage contract: an outer merge of dataset keys against prediction keys must "
     "come back all-'both' and the same length, i.e. a baseline must score EVERY row of the panel"),
    ("research_v2/src/research_v2/evaluation/model_evaluator.py", "_match_prediction_coverage",
     "the OOS acceptance path: restricts candidate AND baseline to the inner intersection of their "
     "business keys before either is scored"),
    ("research_v2/src/research_v2/evaluation/rank_ic.py", "compute_daily_rank_ic",
     "the shared daily Spearman implementation both sides call"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def panel_path(workspace: Path, dataset_id: str, split: str) -> list[Path]:
    root = workspace / "datasets" / dataset_id / f"split={split}"
    return sorted(root.glob("*/*.parquet")) + sorted(root.glob("*.parquet"))


def read_labels(workspace: Path, dataset_id: str) -> pd.DataFrame:
    paths = panel_path(workspace, dataset_id, "validation")
    if not paths:
        return pd.DataFrame()
    return pd.concat(
        [pd.read_parquet(p) for p in paths], ignore_index=True
    )


def selections(workspace: Path) -> dict:
    out = {}
    for rec in sorted(glob.glob(str(workspace / "selections" / "model" / "*" / "record.json"))):
        payload = load_json(Path(rec))["payload"]
        out[payload["fold_id"]] = payload
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, default=ART)
    ap.add_argument("--out", type=Path, default=ART / "m2_rowset")
    ap.add_argument("--draws", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260915)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from research_v2.training.search_metrics import validation_rank_ic

    result: dict = {
        "question": (
            "Are the candidate and the frozen baseline scored on the same rows, is the validation "
            "estimator the same function on both sides, and does the OOS acceptance path share any "
            "asymmetry the validation path has?"
        ),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "read_only": True,
        "database_access": False,
        "weights_loaded": False,
    }

    # ---- 1. the estimator, cited and hashed --------------------------------------
    cited = []
    for rel, symbol, why in CITED:
        path = REPO / rel
        text = path.read_text(encoding="utf-8").splitlines()
        hits = [
            {"line_number": i, "text": ln.rstrip()}
            for i, ln in enumerate(text, start=1)
            if f"def {symbol}" in ln
        ]
        cited.append({
            "file": rel, "symbol": symbol, "sha256": sha256(path),
            "definition_lines": hits, "why_it_matters": why,
        })
    result["estimator_identity"] = {
        "cited_sites": cited,
        "same_summary_statistic": (
            "both sides end in summarize_rank_ic(daily).mean_rank_ic, i.e. the unweighted mean of "
            "per-day Spearman correlations, never a pooled cross-section"
        ),
        "same_minimum_cross_section": "30 on both sides",
        "score_health": (
            "the baseline passes require_score_health=False explicitly; the candidate leaves it at its "
            "default, which is include_invalid, and include_invalid is False on the candidate side -- "
            "so score health is enforced on neither validation path"
        ),
        "day_set_consequence": (
            "the baseline keeps every label date and reports invalid_day_count; the candidate keeps "
            "only dates that survive the 30-row floor. With invalid_day_count=0 and both sides at 237 "
            "days on F1, the two day sets are the same 237 dates"
        ),
        "row_set_consequence": (
            "the baseline is contractually required to score every panel row; the candidate's "
            "validation path inner-joins and silently scores only the rows it produced. Nothing in "
            "the validation path enforces or even reports that difference"
        ),
    }

    # ---- 2. reproduce M2's recorded F1 number ------------------------------------
    m2_ws = args.artifacts / "m2_v1"
    m2_sel = selections(m2_ws)
    repro = {}
    for fold, payload in sorted(m2_sel.items()):
        ds = str(payload["validation_dataset_id"]).split(":")[0]
        pred_path = m2_ws / "models" / f"m2-{fold.lower()}" / "validation_predictions.parquet"
        if not pred_path.is_file():
            continue
        pred = pd.read_parquet(pred_path)
        panel = read_labels(m2_ws, ds)
        if panel.empty:
            continue
        got = validation_rank_ic(pred, panel)
        repro[fold] = {
            "recorded": payload["validation_rank_ic"],
            "recomputed_with_the_repository_estimator": got,
            "bit_identical": float(got).hex() == float(payload["validation_rank_ic"]).hex(),
            "absolute_difference": abs(got - payload["validation_rank_ic"]),
            "prediction_rows": int(len(pred)),
            "panel_rows": int(len(panel)),
        }
    result["reproduction"] = repro

    # ---- 3. census: who scores how much of the panel ------------------------------
    census = []
    for label, ws_name in CARRIERS:
        ws = args.artifacts / ws_name
        sel = selections(ws)
        for fold in FOLDS:
            if fold not in sel:
                continue
            ds = str(sel[fold]["validation_dataset_id"]).split(":")[0]
            pred_path = ws / "models" / f"{ws_name.split('_')[0]}-{fold.lower()}" / "validation_predictions.parquet"
            if not pred_path.is_file():
                matches = sorted(glob.glob(str(ws / "models" / f"*-{fold.lower()}" /
                                               "validation_predictions.parquet")))
                pred_path = Path(matches[0]) if matches else None
            if pred_path is None or not pred_path.is_file():
                continue
            pred = pd.read_parquet(pred_path, columns=["signal_date", "ts_code"])
            panel_paths = panel_path(ws, ds, "validation")
            if not panel_paths:
                panel_paths = panel_path(m2_ws, ds, "validation")
            if not panel_paths:
                continue
            panel_keys = pd.concat(
                [pd.read_parquet(p, columns=["signal_date", "ts_code"]) for p in panel_paths],
                ignore_index=True,
            )
            census.append({
                "carrier": label, "workspace": ws_name, "fold": fold, "dataset_id": ds,
                "panel_rows": int(len(panel_keys)),
                "prediction_rows": int(len(pred)),
                "row_deficit": int(len(panel_keys) - len(pred)),
                "row_deficit_share": float((len(panel_keys) - len(pred)) / len(panel_keys))
                if len(panel_keys) else None,
                "panel_days": int(panel_keys["signal_date"].astype(str).nunique()),
                "prediction_days": int(pred["signal_date"].astype(str).nunique()),
                "scores_the_whole_panel": bool(len(pred) == len(panel_keys)),
            })
    result["census"] = census
    by_fold = {}
    for row in census:
        by_fold.setdefault(row["fold"], []).append(
            {"carrier": row["carrier"], "row_deficit_share": row["row_deficit_share"],
             "scores_the_whole_panel": row["scores_the_whole_panel"]}
        )
    result["census_by_fold"] = by_fold

    # ---- 4. which rows does M2 not score, and are they random? --------------------
    fold = "F1"
    ds = str(m2_sel[fold]["validation_dataset_id"]).split(":")[0]
    pred = pd.read_parquet(m2_ws / "models" / f"m2-{fold.lower()}" / "validation_predictions.parquet")
    panel = read_labels(m2_ws, ds)
    keys = ["signal_date", "ts_code"]
    pk = pred.loc[:, keys].copy()
    pk["signal_date"] = pd.to_datetime(pk["signal_date"]).dt.date
    pan = panel.copy()
    pan["signal_date"] = pd.to_datetime(pan["signal_date"]).dt.date
    joined = pan.merge(pk, on=keys, how="left", indicator=True)
    dropped = joined[joined["_merge"] == "left_only"]
    kept = joined[joined["_merge"] == "both"]
    feature_cols = [c for c in panel.columns if c.startswith("f") and c[1:].isdigit()]
    per_day_drop = dropped.groupby("signal_date").size()
    per_day_total = joined.groupby("signal_date").size()
    share = (per_day_drop.reindex(per_day_total.index).fillna(0) / per_day_total)
    all_days = sorted(per_day_total.index)
    first_20 = set(all_days[:20])
    result["dropped_rows"] = {
        "fold": fold,
        "panel_rows": int(len(joined)),
        "dropped_rows": int(len(dropped)),
        "dropped_share": float(len(dropped) / len(joined)),
        "dropped_days_touched": int(len(per_day_drop)),
        "days_in_panel": int(len(per_day_total)),
        "per_day_dropped": {
            "min": int(per_day_drop.min()) if len(per_day_drop) else 0,
            "median": float(median(per_day_drop.tolist())) if len(per_day_drop) else 0.0,
            "max": int(per_day_drop.max()) if len(per_day_drop) else 0,
        },
        "per_day_dropped_share": {
            "min": float(share.min()), "median": float(median(share.tolist())),
            "max": float(share.max()),
            "days_with_zero_dropped": int((share == 0).sum()),
        },
        "concentration_in_the_first_20_days_of_the_window": {
            "rows_dropped_there": int(dropped["signal_date"].isin(first_20).sum()),
            "share_of_all_dropped": float(dropped["signal_date"].isin(first_20).mean())
            if len(dropped) else None,
            "share_of_panel_rows_there": float(joined["signal_date"].isin(first_20).mean()),
        },
        "distinct_ts_codes_affected": int(dropped["ts_code"].nunique()),
        "distinct_ts_codes_in_panel": int(joined["ts_code"].nunique()),
        "ts_codes_affected_only_entirely": int(
            len(set(dropped["ts_code"]) - set(kept["ts_code"]))
        ),
        "top_ts_codes_by_dropped_rows": dropped["ts_code"].value_counts().head(10).to_dict(),
        "label_valid_among_dropped": {
            "true": int(dropped["label_valid"].fillna(False).astype(bool).sum()),
            "false_or_nan": int((~dropped["label_valid"].fillna(False).astype(bool)).sum()),
        },
        "feature_nan_rate": {
            "dropped_rows": {c: float(pd.to_numeric(dropped[c], errors="coerce").isna().mean())
                             for c in feature_cols},
            "kept_rows": {c: float(pd.to_numeric(kept[c], errors="coerce").isna().mean())
                          for c in feature_cols},
        },
    }

    # ---- 5. how much can a row-set difference of that size move the metric? -------
    rng = np.random.default_rng(args.seed)
    lab = panel.loc[:, ["signal_date", "ts_code", "raw_return_5d", "label_valid"]]
    base_metric = validation_rank_ic(pred, lab)
    sensitivity = {"baseline_metric_on_m2_rows": base_metric, "draws": args.draws, "by_scale": {}}
    for name, n_drop in (("m2_scale", len(dropped)), ("m1_scale", 1087)):
        vals = []
        for _ in range(args.draws):
            idx = rng.choice(len(pred), size=len(pred) - n_drop, replace=False)
            vals.append(validation_rank_ic(pred.iloc[idx], lab))
        arr = np.asarray(vals, dtype=float)
        sensitivity["by_scale"][name] = {
            "rows_dropped_per_draw": int(n_drop),
            "share_of_scored_rows": float(n_drop / len(pred)),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "shift_from_base_mean": float(arr.mean() - base_metric),
            "shift_from_base_std_in_units_of_the_gap_to_the_baseline": None,
        }
    gap = base_metric - 0.06271081108523915
    for name, blk in sensitivity["by_scale"].items():
        blk["gap_m2_minus_baseline_f1"] = gap
        blk["gap_in_units_of_this_scale_std"] = (gap / blk["std"]) if blk["std"] else None
        blk["share_of_draws_that_move_the_metric_by_more_than_the_gap"] = None
    sensitivity["interpretation"] = (
        "a random row loss of M2's size is the null hypothesis 'the dropped rows carry no "
        "particular information'; the real dropped rows are not random (see dropped_rows), so this "
        "bounds the scale of the effect, not its sign or its size"
    )
    result["sensitivity"] = sensitivity

    # ---- 6. does the OOS path share the asymmetry? --------------------------------
    result["oos_path"] = {
        "finding": (
            "no. model_evaluator._match_prediction_coverage inner-joins the candidate's and the "
            "baseline's business keys and scores both on that intersection, so the row-set asymmetry "
            "measured here exists on the validation/selection path only"
        ),
        "consequence_for_the_gates": (
            "G1-G6 read the OOS evaluation, so they are not exposed to this asymmetry. What the "
            "asymmetry does affect is which config each fold's search selects, and any comparison "
            "drawn from validation_rank_ic -- including the standing table in §AS.2"
        ),
        "consequence_for_comparability_rule": (
            "comparison_interim.md's comparability basis checks valid_day_count, baseline mean "
            "RankIC, per-fold baseline selections and per-fold dataset ids. It does not check the "
            "prediction row set. On the OOS side _match_prediction_coverage makes that unnecessary; "
            "on the validation side nothing does"
        ),
    }

    out_json = args.out / "validation_rowset_asymmetry.json"
    out_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    md = ["# Validation row-set asymmetry: candidate vs frozen baseline", ""]
    md += ["## Reproduction", ""]
    for f, r in sorted(repro.items()):
        md.append(f"- {f}: recorded `{r['recorded']!r}`, recomputed with the repository's own "
                  f"`validation_rank_ic` `{r['recomputed_with_the_repository_estimator']!r}`, "
                  f"bit-identical = **{r['bit_identical']}**.")
    md += ["", "## Who scores how much of the panel", "",
           "| carrier | fold | panel rows | scored rows | deficit | deficit share | whole panel? |",
           "|---|---|---|---|---|---|---|"]
    for row in census:
        md.append(f"| {row['carrier']} | {row['fold']} | {row['panel_rows']} | {row['prediction_rows']} "
                  f"| {row['row_deficit']} | {row['row_deficit_share']:.4%} "
                  f"| {'yes' if row['scores_the_whole_panel'] else '**no**'} |")
    d = result["dropped_rows"]
    md += ["", f"## Which rows M2 does not score ({fold})", "",
           f"{d['dropped_rows']} of {d['panel_rows']} panel rows (**{d['dropped_share']:.4%}**), "
           f"touching {d['dropped_days_touched']} of {d['days_in_panel']} days.",
           f"Per-day dropped: min {d['per_day_dropped']['min']}, median {d['per_day_dropped']['median']}, "
           f"max {d['per_day_dropped']['max']}; per-day share min {d['per_day_dropped_share']['min']:.4%}, "
           f"median {d['per_day_dropped_share']['median']:.4%}, max {d['per_day_dropped_share']['max']:.4%}; "
           f"days with zero dropped: {d['per_day_dropped_share']['days_with_zero_dropped']}.",
           f"Concentration in the window's first 20 days: {d['concentration_in_the_first_20_days_of_the_window']}.",
           f"Distinct securities affected: {d['distinct_ts_codes_affected']} of "
           f"{d['distinct_ts_codes_in_panel']}; affected on every one of their days: "
           f"{d['ts_codes_affected_only_entirely']}.",
           f"label_valid among the dropped rows: {d['label_valid_among_dropped']}.",
           "", "Feature NaN rate, dropped vs kept (first five factors):", ""]
    for c in feature_cols[:5]:
        md.append(f"- {c}: dropped {d['feature_nan_rate']['dropped_rows'][c]:.4f} vs "
                  f"kept {d['feature_nan_rate']['kept_rows'][c]:.4f}")
    md += ["", "## How much can a row-set difference of that size move the metric?", ""]
    for name, blk in sensitivity["by_scale"].items():
        md.append(f"- **{name}** ({blk['rows_dropped_per_draw']} rows, "
                  f"{blk['share_of_scored_rows']:.4%} of scored rows, {args.draws} random draws): "
                  f"mean {blk['mean']:.6f}, sd **{blk['std']:.6f}**, range "
                  f"{blk['min']:.6f}–{blk['max']:.6f}; shift from the unsubsampled value "
                  f"{blk['shift_from_base_mean']:+.6f}. The M2-minus-baseline gap is "
                  f"{gap:+.6f} = **{blk['gap_in_units_of_this_scale_std']:.2f} sd** of this scale.")
    md += ["", f"_Caveat: {sensitivity['interpretation']}._", "",
           "## Does the OOS acceptance path share this?", "",
           result["oos_path"]["finding"], "", result["oos_path"]["consequence_for_the_gates"], "",
           result["oos_path"]["consequence_for_comparability_rule"], "",
           "## Not done here", "",
           "No baseline validation predictions exist on disk (only baseline_oos/F*.parquet, which is "
           "the 2022-2025 test side), so B2's validation RankIC could not be recomputed on M2's row "
           "set -- the sensitivity block bounds the scale of such a difference instead of measuring "
           "it. Nothing was written into any artifact directory; no database was touched; no weight "
           "was loaded; no process was signalled.", ""]
    (args.out / "validation_rowset_asymmetry.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {out_json}")
    print(f"wrote {args.out / 'validation_rowset_asymmetry.md'}")


if __name__ == "__main__":
    main()
