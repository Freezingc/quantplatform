#!/usr/bin/env python3
"""Read-only probe: M2's first completed fold (F1).

Three questions, all answerable from JSON records the training run already wrote:

  1. What did M2's 24-config grid on F1 actually say -- which axes moved the validation RankIC,
     and by how much?
  2. Where does M2's F1 winner stand against the six prior carriers and the frozen baseline on
     the *same* validation split -- and does F1 validation even predict the final OOS verdict
     in this project's own history?
  3. Do the dataset identities the M2 pre-registration demanded hold, i.e. is M2 consuming the
     same frozen per-fold datasets every prior carrier consumed?

Safety: no database access at all, no weight loading, nothing written back into any artifact
directory. Inputs are JSON records plus one parquet file read in place for a row/day count.
Outputs go to --out (default /data/zhangyudong/qrv2_research_artifacts/m2_f1_grid_and_standing).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from statistics import median
from pathlib import Path

from scipy import stats

ART = Path("/data/zhangyudong/qrv2_research_artifacts")
OUT_DEFAULT = ART / "m2_f1_grid_and_standing"

# Labels follow evidence/2026-09-14-model-side-walk-forward/comparison_interim.md's Inputs table.
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


def load_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def model_selections(workspace: Path) -> dict:
    out = {}
    for rec in sorted(glob.glob(str(workspace / "selections" / "model" / "*" / "record.json"))):
        payload = load_json(Path(rec))["payload"]
        out[payload["fold_id"]] = payload
    return out


def baseline_selections(workspace: Path) -> dict:
    out = {}
    for rec in sorted(glob.glob(str(workspace / "selections" / "baseline" / "*" / "record.json"))):
        payload = load_json(Path(rec))["payload"]
        scores = {v["baseline_id"]: v for v in payload["validation_scores"]}
        out[payload["fold_id"]] = {
            "selected_baseline_id": payload["selected_baseline_id"],
            "validation_rank_ic": scores[payload["selected_baseline_id"]]["mean_rank_ic"],
            "valid_day_count": scores[payload["selected_baseline_id"]]["valid_day_count"],
            "all_baselines": {k: v["mean_rank_ic"] for k, v in scores.items()},
            "evidence_id": payload.get("baseline_selection_id")
            or Path(rec).parent.parent.name.split(".json")[0],
        }
    return out


def grid_records(workspace: Path, fold: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(str(workspace / "search" / fold / "c*.json"))):
        rec = load_json(Path(path))
        rec["_file"] = os.path.basename(path)
        rows.append(rec)
    rows.sort(key=lambda r: r["config_index"])
    return rows


def marginal(rows: list[dict], key: str) -> dict:
    """Per-level spread of one grid axis: n, min, median, max, and which levels own the tail."""
    buckets = defaultdict(list)
    for row in rows:
        buckets[str(row["config"].get(key))].append(
            (row["result"]["ensemble_validation_rank_ic"], row["config_index"])
        )
    out = {}
    for level, items in sorted(buckets.items()):
        ics = sorted(v for v, _ in items)
        out[level] = {
            "n": len(ics),
            "min": ics[0],
            "median": float(median(ics)),
            "max": ics[-1],
            "config_indices": sorted(i for _, i in items),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, default=ART)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--fold", default="F1")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    fold = args.fold
    m2_ws = args.artifacts / "m2_v1"
    result: dict = {
        "question": (
            f"What did M2's {fold} grid say, where does its winner stand against the six prior "
            f"carriers and the frozen baseline on the same validation split, and do the "
            f"pre-registered dataset identities hold?"
        ),
        "fold": fold,
        "read_only": True,
        "database_access": False,
        "weights_loaded": False,
    }

    # ---- 1. the grid -------------------------------------------------------------
    rows = grid_records(m2_ws, fold)
    ics = [r["result"]["ensemble_validation_rank_ic"] for r in rows]
    order = sorted(range(len(rows)), key=lambda i: -ics[i])
    result["grid"] = {
        "n_configs": len(rows),
        "schema": {r["schema"] for r in rows},
        "axes": {
            key: sorted({str(r["config"].get(key)) for r in rows})
            for key in ("family", "hidden_dim", "dropout", "learning_rate")
        },
        "ranking": [
            {
                "rank": rank,
                "config_index": rows[i]["config_index"],
                "ensemble_validation_rank_ic": ics[i],
                "config": rows[i]["config"],
                "seed_validation_rank_ic": rows[i]["result"]["seed_validation_rank_ic"],
                "worst_seed": rows[i]["result"]["worst_seed"],
            }
            for rank, i in enumerate(order, start=1)
        ],
        "spread": {"min": min(ics), "median": float(median(ics)), "max": max(ics)},
        "marginals": {key: marginal(rows, key) for key in
                      ("family", "hidden_dim", "dropout", "learning_rate")},
    }

    # paired tests on the two axes that look systematic
    tests = {}
    for key in ("learning_rate", "hidden_dim", "family"):
        levels = sorted({str(r["config"].get(key)) for r in rows})
        if len(levels) != 2:
            continue
        a = [r["result"]["ensemble_validation_rank_ic"] for r in rows if str(r["config"].get(key)) == levels[0]]
        b = [r["result"]["ensemble_validation_rank_ic"] for r in rows if str(r["config"].get(key)) == levels[1]]
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        tests[key] = {
            "levels": levels,
            "median": {levels[0]: float(median(a)), levels[1]: float(median(b))},
            "median_difference_first_minus_second": float(median(a)) - float(median(b)),
            "mannwhitney_u": float(u.statistic),
            "p_two_sided": float(u.pvalue),
            "n": {levels[0]: len(a), levels[1]: len(b)},
            "separation": {
                "max_of_lower_level": max(min(a), min(b)),
                "overlap": not (max(a) < min(b) or max(b) < min(a)),
            },
        }
    result["grid"]["axis_tests"] = tests

    # seed structure: is the "worst seed" field random or systematic?
    worst = Counter(str(r["result"]["worst_seed"]) for r in rows)
    seed_ids = sorted({s for r in rows for s in r["result"]["seed_validation_rank_ic"]})
    per_seed = {
        s: {
            "median": float(median([r["result"]["seed_validation_rank_ic"][s] for r in rows])),
            "min": min(r["result"]["seed_validation_rank_ic"][s] for r in rows),
            "max": max(r["result"]["seed_validation_rank_ic"][s] for r in rows),
            "times_worst": worst.get(s, 0),
        }
        for s in seed_ids
    }
    spreads = [
        max(r["result"]["seed_validation_rank_ic"].values())
        - min(r["result"]["seed_validation_rank_ic"].values())
        for r in rows
    ]
    result["seed_structure"] = {
        "seeds": seed_ids,
        "worst_seed_census": dict(worst),
        "per_seed": per_seed,
        "within_config_seed_spread": {
            "min": min(spreads), "median": float(median(spreads)), "max": max(spreads)
        },
    }

    # ---- 2. the standing on the shared validation split --------------------------
    standing = {}
    oos = {}
    for label, ws_name in CARRIERS:
        ws = args.artifacts / ws_name
        sel = model_selections(ws)
        standing[label] = {
            "workspace": ws_name,
            "folds_present": sorted(sel),
            "validation_rank_ic": {f: sel[f]["validation_rank_ic"] for f in sorted(sel)},
            "test_dataset_id": {f: sel[f].get("test_dataset_id") for f in sorted(sel)},
            "baseline_selection_id": {f: sel[f].get("baseline_selection_id") for f in sorted(sel)},
            "model_family": {f: sel[f].get("model_family") for f in sorted(sel)},
            "grid_size_per_fold": {
                f: len(glob.glob(str(ws / "search" / f / "c*.json"))) for f in sorted(sel)
            },
        }
        report_path = ws / "report.json"
        if report_path.is_file():
            rep = load_json(report_path)
            if "oos" in rep and rep["oos"]:
                o = rep["oos"]
                oos[label] = {
                    "mean_rank_ic": o.get("mean_rank_ic"),
                    "valid_day_count": o.get("valid_day_count"),
                    "icir": o.get("icir"),
                    "hac_t_statistic": o.get("hac_t_statistic"),
                    "delta_rank_ic": o.get("delta_rank_ic"),
                    "baseline_mean_rank_ic": o.get("baseline_mean_rank_ic"),
                    "all_passed": o.get("all_passed"),
                    "gates": {k: v.get("status") for k, v in (o.get("gates") or {}).items()},
                }

    base = baseline_selections(m2_ws)
    rep = load_json(args.artifacts / "baseline_v2" / "report.json")
    from_report = {}
    for s in rep["selections"]:
        scores = {v["baseline_id"]: v for v in s["validation_scores"]}
        sel_id = s["selected_baseline_id"]
        from_report[s["fold_id"]] = {
            "selected_baseline_id": sel_id,
            "validation_rank_ic": scores[sel_id]["mean_rank_ic"],
            "valid_day_count": scores[sel_id]["valid_day_count"],
            "all_baselines": {k: v["mean_rank_ic"] for k, v in scores.items()},
            "source": "baseline_v2/report.json",
        }
    for f, v in base.items():
        v["source"] = "m2_v1/selections/baseline"
    # M2 only re-selects a baseline for folds it has reached; the frozen report covers all four.
    agree = {
        f: (base[f]["validation_rank_ic"] == from_report[f]["validation_rank_ic"])
        for f in base if f in from_report
    }
    base = {**from_report, **base}
    result["baseline"] = base
    result["baseline_source_agreement"] = {
        "folds_compared": sorted(agree),
        "identical": agree,
        "note": "M2's own baseline selection record vs the frozen baseline_v2 report, same fold",
    }

    if fold in standing.get("M2", {}).get("validation_rank_ic", {}):
        m2_ic = standing["M2"]["validation_rank_ic"][fold]
        rivals = {
            label: standing[label]["validation_rank_ic"][fold]
            for label, _ in CARRIERS
            if label != "M2" and fold in standing[label]["validation_rank_ic"]
        }
        rivals["baseline_" + base[fold]["selected_baseline_id"]] = base[fold]["validation_rank_ic"]
        ranked = sorted(rivals.items(), key=lambda kv: -kv[1])
        result["standing"] = {
            "fold": fold,
            "m2_validation_rank_ic": m2_ic,
            "field": [
                {"label": k, "validation_rank_ic": v, "m2_minus_this": m2_ic - v}
                for k, v in ranked
            ],
            "m2_position": 1 + sum(1 for _, v in ranked if v > m2_ic),
            "field_size": len(ranked) + 1,
            "m2_position_dedup": 1 + sum(
                1 for k, v in ranked if v > m2_ic and k != "M0R-replicate"
            ),
            "field_size_dedup": len([k for k, _ in ranked if k != "M0R-replicate"]) + 1,
            "best_rival": ranked[0],
            "gap_to_best_rival": m2_ic - ranked[0][1],
            "gap_to_baseline": m2_ic - base[fold]["validation_rank_ic"],
            "baseline_valid_day_count": base[fold]["valid_day_count"],
        }

    # does F1 validation predict the OOS verdict in this project's own history?
    pairs = []
    for label, _ in CARRIERS:
        if label == "M2" or label == "M0R-replicate":
            continue
        v = standing.get(label, {}).get("validation_rank_ic", {}).get("F1")
        o = oos.get(label, {}).get("mean_rank_ic")
        if v is not None and o is not None:
            pairs.append({"label": label, "f1_validation_rank_ic": v, "oos_mean_rank_ic": o})
    if len(pairs) >= 4:
        x = [p["f1_validation_rank_ic"] for p in pairs]
        y = [p["oos_mean_rank_ic"] for p in pairs]
        sp = stats.spearmanr(x, y)
        pe = stats.pearsonr(x, y)
        result["does_f1_validation_predict_oos"] = {
            "n_distinct_carriers": len(pairs),
            "note": "M0R-replicate excluded as a bit-identical duplicate of M0R; M2 excluded, no OOS yet",
            "pairs": pairs,
            "spearman_rho": float(sp.statistic),
            "spearman_p": float(sp.pvalue),
            "pearson_r": float(pe.statistic),
            "pearson_p": float(pe.pvalue),
        }

    # ---- 2b. is M2's seed sensitivity unusual? same fold, every carrier ----------
    seed_sensitivity = {}
    for label, ws_name in CARRIERS:
        ws = args.artifacts / ws_name
        grows = grid_records(ws, fold)
        if not grows:
            # older carriers keep the grid inside report.json instead of search/<fold>/
            rp = ws / "report.json"
            if rp.is_file():
                r = load_json(rp)
                for fd in r.get("folds", []):
                    if fd.get("fold_id") == fold:
                        grows = [
                            {"config_index": i, "config": sr.get("config", {}),
                             "result": sr, "schema": "from-report"}
                            for i, sr in enumerate(fd.get("search_results", []))
                        ]
        with_seeds = [g for g in grows if g.get("result", {}).get("seed_validation_rank_ic")]
        if not with_seeds:
            continue
        spreads = [
            max(g["result"]["seed_validation_rank_ic"].values())
            - min(g["result"]["seed_validation_rank_ic"].values())
            for g in with_seeds
        ]
        ens = [g["result"]["ensemble_validation_rank_ic"] for g in with_seeds]
        worst = Counter(str(g["result"].get("worst_seed")) for g in with_seeds)
        seed_sensitivity[label] = {
            "workspace": ws_name,
            "n_configs": len(with_seeds),
            "within_config_seed_spread_median": float(median(spreads)),
            "within_config_seed_spread_max": max(spreads),
            "between_config_ensemble_range": (max(ens) - min(ens)) if len(ens) > 1 else 0.0,
            "spread_over_range": (float(median(spreads)) / (max(ens) - min(ens)))
            if len(ens) > 1 and max(ens) != min(ens) else None,
            "worst_seed_census": dict(worst),
            "distinct_worst_seeds": len(worst),
        }
    result["seed_sensitivity_by_carrier"] = seed_sensitivity

    # per-seed medians for M2, and an indicative counterfactual without the bad seed
    if rows:
        seed_ids = sorted({s for r in rows for s in r["result"]["seed_validation_rank_ic"]})
        per_seed_median = {
            s: float(median([r["result"]["seed_validation_rank_ic"][s] for r in rows]))
            for s in seed_ids
        }
        bad = min(per_seed_median, key=lambda s: per_seed_median[s])
        good = max(per_seed_median, key=lambda s: per_seed_median[s])
        rest = [s for s in seed_ids if s != bad]

        def median_over_rest(row: dict) -> float:
            return float(median([row["result"]["seed_validation_rank_ic"][s] for s in rest]))

        by_rest = sorted(rows, key=median_over_rest, reverse=True)
        selected_index = result["grid"]["ranking"][0]["config_index"]
        selected_row = next(r for r in rows if r["config_index"] == selected_index)
        result["bad_seed_counterfactual"] = {
            "per_seed_median_over_the_grid": per_seed_median,
            "systematically_worst_seed": bad,
            "systematically_best_seed": good,
            "seed_effect_span": per_seed_median[good] - per_seed_median[bad],
            "between_config_ensemble_range": result["grid"]["spread"]["max"]
            - result["grid"]["spread"]["min"],
            "caveat": (
                "per-seed ICs are not the ensemble IC: the ensemble is scored on averaged "
                "predictions, so every two-seed figure below is an indicative counterfactual, "
                "not what a retrained two-seed ensemble would actually score"
            ),
            "rule": "rank configs by the median per-seed IC over the two non-worst seeds",
            "best_config_under_that_rule": {
                "config_index": by_rest[0]["config_index"],
                "config": by_rest[0]["config"],
                "median_over_remaining_seeds": median_over_rest(by_rest[0]),
                "its_three_seed_ensemble_rank": next(
                    r["rank"] for r in result["grid"]["ranking"]
                    if r["config_index"] == by_rest[0]["config_index"]
                ),
            },
            "selected_config": {
                "config_index": selected_index,
                "median_over_remaining_seeds": median_over_rest(selected_row),
                "its_ensemble_rank": 1,
            },
            "baseline_for_reference": base[fold]["validation_rank_ic"],
            "how_many_configs_clear_the_baseline_on_the_two_good_seeds": sum(
                1 for r in rows if median_over_rest(r) > base[fold]["validation_rank_ic"]
            ),
            "how_many_configs_clear_the_baseline_on_the_three_seed_ensemble": sum(
                1 for r in rows
                if r["result"]["ensemble_validation_rank_ic"] > base[fold]["validation_rank_ic"]
            ),
            "how_many_configs_clear_the_baseline_on_each_single_seed": {
                s: sum(1 for r in rows if r["result"]["seed_validation_rank_ic"][s]
                       > base[fold]["validation_rank_ic"])
                for s in seed_ids
            },
            "best_single_seed_value_anywhere_in_the_grid": {
                s: max(r["result"]["seed_validation_rank_ic"][s] for r in rows) for s in seed_ids
            },
        }

    # ---- 3. dataset identity checks ---------------------------------------------
    ident = defaultdict(dict)
    for label, _ in CARRIERS:
        for f, ds in standing[label]["test_dataset_id"].items():
            ident[f][label] = ds
    datasets_on_disk = sorted(
        d for d in os.listdir(m2_ws / "datasets") if (m2_ws / "datasets" / d).is_dir()
    )
    result["dataset_identity"] = {
        "per_fold_dataset_ids": {f: dict(v) for f, v in sorted(ident.items())},
        "all_carriers_agree_per_fold": {
            f: len(set(v.values())) == 1 for f, v in sorted(ident.items())
        },
        "m2_dataset_dirs_on_disk": datasets_on_disk,
        "m2_dirs_matching_a_known_fold_dataset": {
            d: [f for f, v in ident.items() if d in set(v.values())] for d in datasets_on_disk
        },
        "m2_baseline_selection_ids": standing["M2"]["baseline_selection_id"],
    }

    # the near-coincidence spotted while building the table, recorded at full precision
    m0r_f2 = standing["M0R"]["validation_rank_ic"].get("F2")
    base_f3 = base.get("F3", {}).get("validation_rank_ic")
    if m0r_f2 is not None and base_f3 is not None:
        result["checked_coincidence"] = {
            "what": "M0R's F2 validation RankIC vs the baseline's F3 validation RankIC look equal at 5 dp",
            "m0r_F2_validation": m0r_f2,
            "baseline_F3_selected": base.get("F3", {}).get("selected_baseline_id"),
            "baseline_F3_validation": base_f3,
            "absolute_difference": abs(m0r_f2 - base_f3),
            "same_dataset": standing["M0R"]["test_dataset_id"].get("F2")
            == ident.get("F3", {}).get("M0R"),
            "verdict": "COINCIDENCE_NOT_MISALIGNMENT" if abs(m0r_f2 - base_f3) > 0 else "IDENTICAL",
        }

    # ---- 4. what the pre-registration asked for, and what is checkable now -------
    pre_path = args.artifacts / "m2_preregistration.json"
    if pre_path.is_file():
        pre = load_json(pre_path)
        preds = pre.get("predictions", {})
        result["preregistration_status"] = {
            "registered_at_utc": pre.get("registered_at_utc"),
            "folds_expected": sorted(preds.get("per_fold", {})),
            "folds_with_a_model_selection_record": sorted(standing["M2"]["validation_rank_ic"]),
            "checkable_now": (
                "only the input half: dataset ids and the baseline selection id. The output half "
                "(945/969 valid score days, per-fold valid-day counts, per-year coverage min/median, "
                "candidate daily coverage 1.0000, G6 0.9752321981424149) needs M2's signed report, "
                "which does not exist until every fold and the OOS stage have run."
            ),
            "expected_per_fold": preds.get("per_fold"),
        }

    # the parquet is read in place for a day count only
    val_parquet = m2_ws / "models" / f"m2-{fold.lower()}" / "validation_predictions.parquet"
    if val_parquet.is_file():
        try:
            import pandas as pd

            df = pd.read_parquet(val_parquet, columns=["signal_date"])
            days = sorted({str(v)[:10] for v in df["signal_date"].astype(str)})
            result["validation_split_shape"] = {
                "path": str(val_parquet),
                "rows": int(len(df)),
                "distinct_signal_dates": len(days),
                "first": days[0],
                "last": days[-1],
                "by_year": dict(sorted(Counter(d[:4] for d in days).items())),
                "matches_baseline_valid_day_count": len(days)
                == base.get(fold, {}).get("valid_day_count"),
            }
        except Exception as exc:  # never fail the probe on an optional cross-check
            result["validation_split_shape"] = {"error": f"{type(exc).__name__}: {exc}"}

    json_path = args.out / f"m2_{fold.lower()}_grid_and_standing.json"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    # ---- markdown summary --------------------------------------------------------
    md = [f"# M2 {fold}: grid, standing, and dataset identity", ""]
    g = result["grid"]
    md += [
        f"Grid: **{g['n_configs']} configs**, axes "
        + ", ".join(f"{k}={v}" for k, v in g["axes"].items()) + ".",
        f"Ensemble validation RankIC spans **{g['spread']['min']:.6f} – {g['spread']['max']:.6f}** "
        f"(median {g['spread']['median']:.6f}).",
        "",
        "## Ranking",
        "",
        "| rank | cfg | family | hidden | dropout | lr | ens. val RankIC | worst seed |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in g["ranking"]:
        c = r["config"]
        md.append(
            f"| {r['rank']} | c{r['config_index']} | {c.get('family')} | {c.get('hidden_dim')} "
            f"| {c.get('dropout')} | {c.get('learning_rate')} | {r['ensemble_validation_rank_ic']:.6f} "
            f"| {r['worst_seed']} |"
        )
    md += ["", "## Axis tests", ""]
    for key, t in g["axis_tests"].items():
        lv = t["levels"]
        md.append(
            f"- **{key}**: median {lv[0]}={t['median'][lv[0]]:.6f} vs {lv[1]}={t['median'][lv[1]]:.6f} "
            f"(Δ={t['median_difference_first_minus_second']:+.6f}), Mann-Whitney U={t['mannwhitney_u']:.1f}, "
            f"p={t['p_two_sided']:.4f}, n={t['n'][lv[0]]}/{t['n'][lv[1]]}, "
            f"ranges {'overlap' if t['separation']['overlap'] else 'are disjoint'}."
        )
    md += ["", "## Seed structure", ""]
    ss = result["seed_structure"]
    md.append(f"Worst-seed census over {g['n_configs']} configs: **{ss['worst_seed_census']}**.")
    for s, v in ss["per_seed"].items():
        md.append(
            f"- seed {s}: median {v['median']:.6f}, range {v['min']:.6f}–{v['max']:.6f}, "
            f"worst in {v['times_worst']}/{g['n_configs']} configs."
        )
    ws_ = ss["within_config_seed_spread"]
    md.append(
        f"Within-config seed spread (max−min of the three seeds): median **{ws_['median']:.6f}**, "
        f"range {ws_['min']:.6f}–{ws_['max']:.6f}."
    )
    if "standing" in result:
        st = result["standing"]
        md += [
            "",
            f"## Standing on the shared {fold} validation split",
            "",
            f"M2 {fold} = **{st['m2_validation_rank_ic']:.6f}**, position "
            f"**{st['m2_position']} of {st['field_size']}**.",
            "",
            "| label | validation RankIC | M2 minus this |",
            "|---|---|---|",
        ]
        for row in st["field"]:
            md.append(f"| {row['label']} | {row['validation_rank_ic']:.6f} | {row['m2_minus_this']:+.6f} |")
        md.append(
            f"\nBaseline valid-day count {st['baseline_valid_day_count']}; "
            f"M2's own validation parquet has "
            f"{result.get('validation_split_shape', {}).get('distinct_signal_dates', '?')} distinct signal dates "
            f"({result.get('validation_split_shape', {}).get('first', '?')} … "
            f"{result.get('validation_split_shape', {}).get('last', '?')}), match = "
            f"**{result.get('validation_split_shape', {}).get('matches_baseline_valid_day_count')}**."
        )
    if "does_f1_validation_predict_oos" in result:
        d = result["does_f1_validation_predict_oos"]
        md += [
            "",
            "## Does F1 validation predict the OOS verdict?",
            "",
            f"Over {d['n_distinct_carriers']} distinct prior carriers: Spearman rho = "
            f"**{d['spearman_rho']:+.4f}** (p={d['spearman_p']:.4f}), Pearson r = "
            f"{d['pearson_r']:+.4f} (p={d['pearson_p']:.4f}). {d['note']}.",
            "",
            "| label | F1 validation | OOS mean RankIC | gates |",
            "|---|---|---|---|",
        ]
        for p in d["pairs"]:
            gates = oos.get(p["label"], {}).get("gates", {})
            md.append(
                f"| {p['label']} | {p['f1_validation_rank_ic']:.6f} | {p['oos_mean_rank_ic']:.6f} "
                f"| {''.join(k for k, v in gates.items() if v == 'PASS') or '-'} pass / "
                f"{','.join(k for k, v in gates.items() if v != 'PASS') or 'none'} fail |"
            )
    if "seed_sensitivity_by_carrier" in result:
        md += ["", f"## Is M2's seed sensitivity unusual? ({fold}, every carrier)", "",
               "| carrier | configs | within-config seed spread (median) | between-config ensemble range "
               "| spread / range | worst-seed census |",
               "|---|---|---|---|---|---|"]
        for label, v in sorted(result["seed_sensitivity_by_carrier"].items()):
            ratio = "-" if v["spread_over_range"] is None else f"{v['spread_over_range']:.3f}"
            md.append(
                f"| {label} | {v['n_configs']} | {v['within_config_seed_spread_median']:.6f} "
                f"| {v['between_config_ensemble_range']:.6f} | {ratio} | {v['worst_seed_census']} |"
            )
    if "bad_seed_counterfactual" in result:
        cf = result["bad_seed_counterfactual"]
        md += ["", "## Would dropping the systematically worst seed close the gap?", "",
               f"Per-seed medians over the whole grid: "
               + ", ".join(f"{s}={v:.6f}" for s, v in cf["per_seed_median_over_the_grid"].items()) + ".",
               f"Seed effect span **{cf['seed_effect_span']:.6f}** vs between-config ensemble range "
               f"**{cf['between_config_ensemble_range']:.6f}**.",
               f"Ranking instead by the median of the two non-worst seeds still selects "
               f"**c{cf['best_config_under_that_rule']['config_index']}** at "
               f"**{cf['best_config_under_that_rule']['median_over_remaining_seeds']:.6f}**, i.e. below the "
               f"baseline's {cf['baseline_for_reference']:.6f}.",
               f"Configs clearing the baseline: **{cf['how_many_configs_clear_the_baseline_on_the_three_seed_ensemble']}"
               f"/{result['grid']['n_configs']}** on the three-seed ensemble, "
               f"**{cf['how_many_configs_clear_the_baseline_on_the_two_good_seeds']}"
               f"/{result['grid']['n_configs']}** on the two non-worst seeds, and per single seed "
               f"{cf['how_many_configs_clear_the_baseline_on_each_single_seed']}.",
               f"_Caveat: {cf['caveat']}._"]
    md += ["", f"Baseline source agreement (M2's own record vs the frozen report): "
               f"**{result.get('baseline_source_agreement', {}).get('identical')}**.",
           f"M2 validation-parquet day count vs the baseline's valid_day_count: "
           f"**{result.get('validation_split_shape', {}).get('matches_baseline_valid_day_count')}**."]
    di = result["dataset_identity"]
    md += ["", "## Dataset identity", "",
           f"All carriers agree on every fold's dataset id: **{di['all_carriers_agree_per_fold']}**.",
           f"M2 dataset dirs on disk: {di['m2_dataset_dirs_on_disk']}",
           f"Mapping to known folds: {di['m2_dirs_matching_a_known_fold_dataset']}",
           f"M2 baseline selection id(s): {di['m2_baseline_selection_ids']}", ""]
    if "checked_coincidence" in result:
        cc = result["checked_coincidence"]
        md += [
            "## One coincidence, checked",
            "",
            f"{cc['what']}: M0R F2 = {cc['m0r_F2_validation']!r}, baseline F3 ({cc['baseline_F3_selected']}) "
            f"= {cc['baseline_F3_validation']!r}, |Δ| = {cc['absolute_difference']:.3e}, "
            f"same dataset = {cc['same_dataset']} → **{cc['verdict']}**.",
            "",
        ]
    md += [
        "## Not done here",
        "",
        "No OOS number for M2 exists yet, so nothing here says whether M2 passes the walk-forward; "
        "no database query was run; no weight was loaded; nothing was written into any artifact "
        "directory; the parquet was read in place for a day count only.",
        "",
    ]
    (args.out / f"m2_{fold.lower()}_grid_and_standing.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {args.out / f'm2_{fold.lower()}_grid_and_standing.md'}")


if __name__ == "__main__":
    main()
