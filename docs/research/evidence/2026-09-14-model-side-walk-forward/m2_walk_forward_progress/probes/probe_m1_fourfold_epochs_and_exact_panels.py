#!/usr/bin/env python3
"""Read-only probe for §CK: two things §CI could not do with two folds, done with four.

§CI priced M2's remaining wall clock and said plainly why it could only give a range: "epoch/运行只有
两个观测点（17.49 与 28.81，涨 1.647 倍），两点无法区分线性、几何与「F2 恰好偏大」". It also had to
PROJECT the F3/F4 panel sizes instead of reading them, because it believed only M2's F1/F2 datasets
were on disk. Both gaps are closable today from artefacts that already exist, and this probe closes
them.

FINDING 1 -- the F3/F4 panel rows were projected, and the projection is wrong, and the exact values
were sitting in files §CI had already opened.
    §CI's own probe read the F3/F4 TEST row counts (932,167 / 938,594) and cited them as "已实测…
    取自 §BI.5". Those numbers match M1's materialised F3/F4 datasets byte for byte. But the
    train+validation counts -- in the SAME manifest directories, readable from the SAME parquet
    footers with the SAME call -- were instead extrapolated from a trainval/test ratio trend. The
    projection gives 3,595,693 / 3,663,277; the footers give 3,650,743 / 3,955,048. So F3 is off by
    -1.51% and F4 by -7.38%. This probe recomputes every scenario with the measured values and
    reports effect A: the increase is +4.1% to +5.2% of remaining wall clock.

FINDING 2 -- M1 ran the SAME panels across FOUR folds, and its epochs/run is not growing.
    M1's four folds use the identical config_hash set (27 configs, verified identical across all
    four folds), so any difference in epochs/run is a fold effect and not a config-mix effect. The
    result is 20.1235 -> 23.2716 -> 19.0864 -> 16.9630: non-monotone, peaking at F2, then FALLING,
    while panel rows rise monotonically by 1.1102x / 1.0985x / 1.0834x. F4 has the largest panel
    (1.3212x F1's) and the SMALLEST epochs/run. Spearman(panel rows, peak mean) = -0.80 (p = 0.20,
    n = 4, so not significant -- four points cannot make anything significant, and this probe does
    not claim they can).
    The point is not the correlation. It is that §CI's scenario family spans "flat" at the optimistic
    end and "geometric growth at 1.647x" at the central end, and every one of its four scenarios is
    non-decreasing in fold index. M1 is a same-panels, same-config-set, four-fold observation of a
    fold pattern that DECREASES after F2. So the true uncertainty set is wider than §CI's on the low
    side, not merely on the high side. Applying M1's fold ratios to M2's F2 value gives
    F3 = 23.63 and F4 = 21.00 epochs/run, i.e. 139.0 h remaining -- BELOW §CI's published optimistic
    bound of 166.6 h. That is effect B.

WHY M1'S PANELS ARE M2'S PANELS, and the one limit on that claim.
    dataset_id is a content hash: manifest.py:_hash_payload hashes business_keys, columns with
    dtypes, canonicalised metadata (which is exactly {fold_id, factor_order, data_versions}), every
    row, and split_content_hashes. publish_fold_dataset (run_real_candidate_training.py:676) puts no
    family and no model config into that metadata. So two carriers on the same fold and the same data
    versions get the same dataset_id by construction, and the identity is an empirical fact here:
    M1's and M2's F1 ids are both 226a8dfc… and their F2 ids are both eb9bba1b…, with shard_hashes,
    split_content_hashes and preprocessing.json all byte-identical. M1 materialised F1 at 09-12 02:11
    and M2 at 09-13 19:00 (gap 1.70 days); F2 at 09-12 10:07 and 09-15 14:28 (gap 3.18 days). Two
    independent gaps of 1.7 and 3.2 days produced identical ids, which is direct evidence that the
    per-day data_versions did not drift over those windows.
    THE LIMIT: data_versions is per-day (144 factor_daily entries with 12 distinct versions, 151
    label_daily with 145 distinct versions, 151 universe_daily with 145), so the id is sensitive to
    any single day's version changing. M1 materialised F3 at 09-12 21:12 and F4 at 09-13 08:28; M2
    will materialise them roughly three weeks later. The proven no-drift window is 3.18 days and the
    exposure is ~20 days, about 6.3x longer, and that longer window is NOT measured. If versions do
    drift, M2's F3/F4 panels are different objects and their row counts must be re-read rather than
    inherited. This probe therefore reports the exact rows as "measured on M1's datasets" and never
    as "the rows M2 will train on".

WHAT THIS PROBE DOES NOT CLAIM about the cost constant.
    The form `wall = k x epochs_per_run x rows` is re-tested on M1's four folds: implied k is
    0.0002894 / 0.0003103 / 0.0003493 / 0.0002767 with no monotone drift in rows
    (Spearman rho = -0.20, p = 0.80). But M1's k is 7.16x smaller than M2's fitted 0.00219327, and
    that is expected rather than alarming: M1's model_card.json says family RESIDUAL_MLP with
    model_code_version residual-mlp-1, while M2 is TCN with history_length 20, so M2 does far more
    work per row. k is family-specific; only the FORM transfers. The M1 spread is also wider than
    M2's (max/min 1.2622 versus 1.0246) and the likely reason is measurement resolution, not model
    error: M1's grid spans are 4.8-6.8 h against M2's 31.5-59.0 h, so the fixed error in an
    mtime-derived span is a much larger fraction of a smaller quantity. This probe reports both
    spreads rather than only the one that flatters the model.

Only directory names, manifest.json text, parquet FOOTERS (metadata.num_rows -- no data loaded) and
already-published JSON are read. checkpoint.pt is never opened, no weight or tensor is loaded, no
repository module is imported or executed (trainer.py and manifest.py are read as text), no database
access, no network, no writes to any artifact directory, no signal to any process. M2 (pid 3588838)
is still training while this runs, so every M2 figure is a labelled partial snapshot.
"""
from __future__ import annotations

import glob
import json
import re
import statistics as st
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import pyarrow.parquet as pq
from scipy import stats

ART = Path("/data/zhangyudong/qrv2_research_artifacts")
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "measurements" / "m1_fourfold_epochs_and_exact_panels.json"

PATIENCE = 10
CAP = 100
K_GRID_M2 = 0.00219327          # §CI's fitted constant, validated to 2.4% on M2's two folds
K_FINAL_M2 = 0.00020381721864733542
FINAL_UPLIFT = 1.45
GAP_H, OOS_H = 5.0, 2.0         # §CI's stated assumptions, carried unchanged
RUN_START = datetime(2026, 9, 13, 19, 28, 2)

# §CI's published scenario grid: name -> (F3 epochs/run, F4 epochs/run, published total h)
CI_SCENARIOS = {
    "optimistic_epochs_flat": (28.8, 28.8, 166.6),
    "linear_growth_capped": (40.1, 51.4, 252.0),
    "central_geometric_capped": (47.5, 78.2, 337.9),
    "pessimistic_all_hit_cap": (100.0, 100.0, 523.8),
}
CI_PROJECTED_ROWS = {"F1": 2993479, "F2": 3323361, "F3": 3595693, "F4": 3663277}
CI_CITED_TEST_ROWS = {"F1": 804575, "F2": 882426, "F3": 932167, "F4": 938594}
CARRIERS = (("M1", "m1_full_v1", "m1"), ("M2", "m2_v1", "m2"))


def panel_rows(ws: str) -> dict[str, dict[str, int]]:
    """Exact per-split row counts from parquet FOOTERS only; no data is loaded."""
    out = {}
    for ds in sorted((ART / ws / "datasets").iterdir()):
        if not (ds / "manifest.json").is_file():
            continue
        m = json.load(open(ds / "manifest.json", encoding="utf-8"))
        fold = m["metadata"]["fold_id"]
        counts = {}
        for split in ("train", "validation", "test"):
            counts[split] = sum(
                pq.ParquetFile(p).metadata.num_rows
                for p in (ds / f"split={split}").rglob("*.parquet")
            )
        counts["trainval"] = counts["train"] + counts["validation"]
        counts["manifest_total"] = int(m["row_count"])
        counts["footer_total"] = counts["train"] + counts["validation"] + counts["test"]
        counts["manifest_agrees_with_footers"] = counts["manifest_total"] == counts["footer_total"]
        counts["dataset_id"] = ds.name
        counts["manifest_mtime"] = datetime.fromtimestamp(
            (ds / "manifest.json").stat().st_mtime).isoformat(timespec="minutes")
        counts["metadata_keys"] = sorted(m["metadata"].keys())
        counts["n_factor_order"] = len(m["metadata"].get("factor_order", []))
        counts["n_data_version_entries"] = sum(
            len(v) for v in m["metadata"].get("data_versions", {}).values())
        counts["data_version_tables"] = sorted(m["metadata"].get("data_versions", {}).keys())
        out[fold] = counts
    return out


def peaks_by_fold(ws: str, car: str) -> dict[str, dict[str, object]]:
    """Per-fold grid peaks, wall-clock spans and config-hash sets, from directory names + manifests."""
    run_re = re.compile(r"^(?P<run>.+)-epoch-(?P<epoch>\d+)$")
    runs: dict[str, list[int]] = defaultdict(list)
    grid_mtimes: dict[str, list[datetime]] = defaultdict(list)
    for d in glob.glob(str(ART / ws / "checkpoints" / "*-epoch-*")):
        p = Path(d)
        m = run_re.match(p.name)
        if not m:
            continue
        runs[m.group("run")].append(int(m.group("epoch")))
        g = re.match(rf"^{car}-(f\d)-c\d+-\d+-seed-\d+$", m.group("run"))
        if g:
            grid_mtimes[g.group(1).upper()].append(
                datetime.fromtimestamp(p.stat().st_mtime))
    out: dict[str, dict[str, object]] = {}
    for run, eps in runs.items():
        g = re.match(rf"^{car}-(f\d)-c\d+-\d+-seed-\d+$", run)
        if g:
            f = g.group(1).upper()
            row = out.setdefault(f, {"peaks": [], "n_runs": 0})
            row["peaks"].append(max(eps))
            row["n_runs"] += 1
    for f, row in out.items():
        pk = row["peaks"]
        row["n_runs"] = len(pk)
        row["peak_mean"] = round(st.mean(pk), 4)
        row["peak_median"] = st.median(pk)
        row["peak_max"] = max(pk)
        row["epochs_per_run"] = round(st.mean(pk) + PATIENCE, 4)
        mt = sorted(grid_mtimes[f])
        row["grid_span_h"] = round((mt[-1] - mt[0]).total_seconds() / 3600, 4)
        row["grid_first_ckpt"] = mt[0].isoformat(timespec="minutes")
        row["grid_last_ckpt"] = mt[-1].isoformat(timespec="minutes")
        del row["peaks"]
    return dict(sorted(out.items()))


def config_hashes(ws: str) -> dict[str, dict[int, str]]:
    """fold -> {config_index: config_hash}, to test whether the config SET is fold-invariant."""
    out: dict[str, dict[int, str]] = {}
    for f in sorted((ART / ws / "search").glob("F*")):
        hs = {}
        for rec in sorted(f.glob("c*.json")):
            r = json.load(open(rec, encoding="utf-8"))
            hs[int(r["config_index"])] = r["config_hash"]
        out[f.name] = hs
    return out


def validation_ic(ws: str) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for f in sorted((ART / ws / "search").glob("F*")):
        ics = [json.load(open(r, encoding="utf-8"))["result"]["ensemble_validation_rank_ic"]
               for r in sorted(f.glob("c*.json"))]
        if ics:
            out[f.name] = {"n": len(ics), "mean": round(st.mean(ics), 6),
                           "max": round(max(ics), 6)}
    return out


def negative_controls(folds: dict, hashes: dict, rows_m1: dict, rows_m2: dict) -> dict:
    """Every instrument in this probe must be able to say NO, per this pack's convention (§CF's five
    negative cases, §AL's self-refuting script, §CJ's injected non-monotone run).

    Without this, the True values this section reports -- config set fold-invariant, metadata
    identical, shard hashes identical, split hashes identical -- would be indistinguishable from an
    instrument that returns True unconditionally. Each control below takes a pair that SHOULD differ
    and shows the test reports it as differing.
    """
    import copy

    m1 = hashes["M1"]
    m2 = hashes["M2"]
    ctrl_cfg = {
        "test": "config_hash map identical across folds",
        "same_fold_twice_must_be_True": m1["F1"] == hashes["M1"]["F1"],
        "same_carrier_different_fold_is_True_by_design": m1["F1"] == m1["F2"],
        "different_carrier_same_fold_must_be_False": m1["F1"] == m2["F1"],
        "n_configs_M1_F1": len(m1["F1"]),
        "n_configs_M2_F1": len(m2["F1"]),
        "config_hash_intersection_M1F1_M2F1": len(set(m1["F1"].values()) & set(m2["F1"].values())),
        "verdict": ("discriminates: it returns False for M1-F1 vs M2-F1 (27 versus 24 configs, zero "
                    "shared config hashes), so the True on M1's four folds is not vacuous"),
    }
    d1 = rows_m1["F1"]["dataset_id"]
    d2 = rows_m1["F2"]["dataset_id"]
    a = json.load(open(ART / "m1_full_v1" / "datasets" / d1 / "manifest.json", encoding="utf-8"))
    b = json.load(open(ART / "m1_full_v1" / "datasets" / d2 / "manifest.json", encoding="utf-8"))
    pert = copy.deepcopy(a)
    tbl = sorted(pert["metadata"]["data_versions"])[0]
    day = sorted(pert["metadata"]["data_versions"][tbl])[0]
    pert["metadata"]["data_versions"][tbl][day] = "0" * 64
    ctrl_panel = {
        "test": "panel identity (dataset_id / metadata / shard hashes / split hashes)",
        "M1_F1_vs_M1_F2_dataset_id_equal_must_be_False": a["dataset_id"] == b["dataset_id"],
        "M1_F1_vs_M1_F2_shard_hashes_equal_must_be_False": a["shard_hashes"] == b["shard_hashes"],
        "M1_F1_vs_M1_F2_metadata_equal_must_be_False": a["metadata"] == b["metadata"],
        "one_day_version_flip_detected_must_be_False": pert["metadata"] == a["metadata"],
        "flipped_entry": f"{tbl}/{day}",
        "n_data_version_entries": sum(len(v) for v in a["metadata"]["data_versions"].values()),
        "verdict": ("panel identity is genuinely version-sensitive: flipping ONE day out of 446 "
                    "entries breaks metadata equality, which is why the ~20-day exposure before M2 "
                    "materialises F3/F4 is a real risk and not a formality"),
    }

    def mono(v: list[float]) -> bool:
        return all(v[i] > v[i - 1] for i in range(1, len(v)))

    m1e = [folds["M1"][f]["epochs_per_run"] for f in ("F1", "F2", "F3", "F4")]
    rows_seq = [rows_m1[f]["trainval"] for f in ("F1", "F2", "F3", "F4")]
    ctrl_mono = {
        "test": "is_monotone_increasing_in_fold_index",
        "M1_actual_epochs_must_be_False": mono(m1e),
        "M1_actual_values": m1e,
        "synthetic_increasing_must_be_True": mono([1, 2, 3, 4]),
        "panel_rows_must_be_True": mono(rows_seq),
        "verdict": ("the epochs test says False on the real data while the same predicate says True on "
                    "the panel rows, so the non-monotonicity is a property of the epochs and not a "
                    "broken predicate"),
    }
    k_bumped = K_GRID_M2 * 1.05
    bumped = (k_bumped * 28.8 * CI_PROJECTED_ROWS["F3"] / 3600
              + K_FINAL_M2 * 1.05 * 28.8 * CI_PROJECTED_ROWS["F3"] * FINAL_UPLIFT / 3600
              + k_bumped * 28.8 * CI_PROJECTED_ROWS["F4"] / 3600
              + K_FINAL_M2 * 1.05 * 28.8 * CI_PROJECTED_ROWS["F4"] * FINAL_UPLIFT / 3600
              + 4 * GAP_H + OOS_H)
    pub = CI_SCENARIOS["optimistic_epochs_flat"][2]
    ctrl_model = {
        "test": "the scenario arithmetic responds to its inputs",
        "optimistic_total_with_k_plus_5pct_h": round(bumped, 1),
        "ci_published_optimistic_h": pub,
        "delta_pct": round((bumped - pub) / pub * 100, 2),
        "verdict": ("a 5% change in k moves the total by ~4%, so the recomputation is not insensitive "
                    "to its inputs; together with step_0 reproducing §CI to within 0.2 h this bounds "
                    "the arithmetic to being the same arithmetic"),
    }
    all_pass = bool(
        ctrl_cfg["same_fold_twice_must_be_True"]
        and not ctrl_cfg["different_carrier_same_fold_must_be_False"]
        and not ctrl_panel["M1_F1_vs_M1_F2_dataset_id_equal_must_be_False"]
        and not ctrl_panel["one_day_version_flip_detected_must_be_False"]
        and not ctrl_mono["M1_actual_epochs_must_be_False"]
        and ctrl_mono["synthetic_increasing_must_be_True"]
        and ctrl_mono["panel_rows_must_be_True"]
        and abs(ctrl_model["delta_pct"]) > 1.0)
    return {"config_set_identity": ctrl_cfg, "panel_identity": ctrl_panel,
            "monotonicity": ctrl_mono, "scenario_arithmetic": ctrl_model,
            "ALL_CONTROLS_BEHAVE_AS_EXPECTED": all_pass}


def grid_h(epr: float, rows: int) -> float:
    return K_GRID_M2 * epr * rows / 3600


def final_h(epr: float, rows: int) -> float:
    return K_FINAL_M2 * epr * rows * FINAL_UPLIFT / 3600


def scenario(epr3: float, epr4: float, rows: dict[str, int],
             f2_final_remaining_h: float = 0.0) -> dict[str, object]:
    """§CI's arithmetic, unchanged in form so the comparison isolates the inputs."""
    g3, f3 = grid_h(epr3, rows["F3"]), final_h(epr3, rows["F3"])
    g4, f4 = grid_h(epr4, rows["F4"]), final_h(epr4, rows["F4"])
    gaps = 4 * GAP_H + OOS_H
    tot = f2_final_remaining_h + g3 + f3 + g4 + f4 + gaps
    return {"f3_epochs_per_run": round(epr3, 2), "f4_epochs_per_run": round(epr4, 2),
            "f3_grid_h": round(g3, 1), "f3_final_h": round(f3, 1),
            "f4_grid_h": round(g4, 1), "f4_final_h": round(f4, 1),
            "gaps_and_oos_h": round(gaps, 1), "total_remaining_h": round(tot, 1),
            "total_remaining_days": round(tot / 24, 1)}


def main() -> None:
    rows_m1 = panel_rows("m1_full_v1")
    rows_m2 = panel_rows("m2_v1")
    exact = {f: rows_m1[f]["trainval"] for f in rows_m1}
    splits = {f: {k: rows_m1[f][k] for k in ("train", "validation", "test", "trainval")}
              for f in rows_m1}

    folds = {}
    hashes = {}
    ics = {}
    for label, ws, car in CARRIERS:
        folds[label] = peaks_by_fold(ws, car)
        hashes[label] = config_hashes(ws)
        ics[label] = validation_ic(ws)

    doc: dict[str, object] = {
        "question": ("what do M1's four folds -- run on the SAME panels and the SAME config set as "
                     "M2's two -- say about §CI's projected F3/F4 panel sizes and about the growth "
                     "premise behind its scenario family?"),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "folds": {},
        "panel_rows": {},
        "finding_1_exact_panels_versus_CI_projection": {},
        "finding_2_four_fold_epoch_pattern": {},
        "panel_identity_M1_equals_M2": {},
        "cost_model_form_retest": {},
        "scenario_recomputation": {},
    }

    windows = {"F1": ("2017-2020", 2021, 2022), "F2": ("2018-2021", 2022, 2023),
               "F3": ("2019-2022", 2023, 2024), "F4": ("2020-2023", 2024, 2025)}
    for f in ("F1", "F2", "F3", "F4"):
        tr, va, te = windows[f]
        doc["folds"][f] = {
            "train_years": tr, "validation_year": va, "test_year": te,
            "dataset_id": rows_m1[f]["dataset_id"][:16] + "…",
            "manifest_mtime": rows_m1[f]["manifest_mtime"],
            "metadata_keys": rows_m1[f]["metadata_keys"],
            "n_factor_order": rows_m1[f]["n_factor_order"],
            "n_data_version_entries": rows_m1[f]["n_data_version_entries"],
            "data_version_tables": rows_m1[f]["data_version_tables"],
        }
        doc["panel_rows"][f] = {
            **splits[f],
            "manifest_total": rows_m1[f]["manifest_total"],
            "footer_total": rows_m1[f]["footer_total"],
            "manifest_agrees_with_footers": rows_m1[f]["manifest_agrees_with_footers"],
            "trainval_over_test": round(splits[f]["trainval"] / splits[f]["test"], 4),
        }

    # ---- finding 1
    f1rows = {}
    for f in ("F1", "F2", "F3", "F4"):
        proj = CI_PROJECTED_ROWS[f]
        ex = exact[f]
        f1rows[f] = {
            "ci_published_trainval": proj,
            "measured_trainval": ex,
            "delta": ex - proj,
            "ci_error_pct": round((proj - ex) / ex * 100, 3),
            "ratio_measured_over_published": round(ex / proj, 4),
            "ci_cited_test_rows": CI_CITED_TEST_ROWS[f],
            "measured_test_rows": splits[f]["test"],
            "test_rows_match_what_ci_cited": CI_CITED_TEST_ROWS[f] == splits[f]["test"],
            "trainval_over_test_ci_assumed": None,
            "trainval_over_test_measured": round(splits[f]["trainval"] / splits[f]["test"], 4),
        }
        f1rows[f]["trainval_over_test_ci_assumed"] = round(proj / CI_CITED_TEST_ROWS[f], 4)
    doc["finding_1_exact_panels_versus_CI_projection"] = {
        "per_fold": f1rows,
        "the_decisive_fact": (
            "§CI cited the F3/F4 TEST counts (932,167 / 938,594) as measured, and they match these "
            "datasets' footers exactly -- so the parquet files it used were M1's materialised F3/F4 "
            "datasets, whose train and validation footers are in the same directories. The "
            "train+validation totals were nevertheless extrapolated from a trainval/test ratio trend "
            "instead of read. Nothing new was needed to get them exactly."),
        "worst_fold": max(f1rows, key=lambda f: abs(f1rows[f]["ci_error_pct"])),
        "n_folds_where_test_matched": sum(
            1 for f in f1rows if f1rows[f]["test_rows_match_what_ci_cited"]),
    }

    # ---- finding 2
    m1 = folds["M1"]
    m2 = folds["M2"]
    order = [f for f in ("F1", "F2", "F3", "F4") if f in m1]
    epr = [m1[f]["epochs_per_run"] for f in order]
    rows_seq = [exact[f] for f in order]
    m1_hash_sets = [hashes["M1"][f"F{i}"] for i in (1, 2, 3, 4)]
    m2_hash_sets = [hashes["M2"][f"F{i}"] for i in (1, 2)]
    doc["finding_2_four_fold_epoch_pattern"] = {
        "M1_epochs_per_run_by_fold": {f: m1[f]["epochs_per_run"] for f in order},
        "M1_peak_mean_by_fold": {f: m1[f]["peak_mean"] for f in order},
        "M1_peak_max_by_fold": {f: m1[f]["peak_max"] for f in order},
        "M2_epochs_per_run_by_fold": {f: m2[f]["epochs_per_run"] for f in m2},
        "panel_rows_by_fold": {f: exact[f] for f in order},
        "panel_row_ratios_fold_over_previous": {
            f"{order[i]}_over_{order[i-1]}": round(rows_seq[i] / rows_seq[i - 1], 4)
            for i in range(1, len(order))},
        "M1_epoch_ratios_fold_over_previous": {
            f"{order[i]}_over_{order[i-1]}": round(epr[i] / epr[i - 1], 4)
            for i in range(1, len(order))},
        "M2_epoch_ratio_F2_over_F1": round(
            m2["F2"]["epochs_per_run"] / m2["F1"]["epochs_per_run"], 4),
        "is_monotone_increasing_in_fold_index": all(
            epr[i] > epr[i - 1] for i in range(1, len(epr))),
        "argmax_fold": order[epr.index(max(epr))],
        "argmin_fold": order[epr.index(min(epr))],
        "last_fold_vs_first": round(epr[-1] / epr[0], 4),
        "largest_panel_fold": order[rows_seq.index(max(rows_seq))],
        "spearman_panel_rows_vs_M1_peak_mean": {
            "rho": round(stats.spearmanr(rows_seq, epr).statistic, 4),
            "p": round(stats.spearmanr(rows_seq, epr).pvalue, 4),
            "n": len(order),
            "note": ("n=4 cannot make anything significant; this is reported so that nobody upgrades "
                     "it into a claim. The finding is the SHAPE (non-monotone, peak at F2, decline "
                     "after), not the correlation."),
        },
        "config_set_fold_invariant": {
            "M1_identical_across_four_folds": all(h == m1_hash_sets[0] for h in m1_hash_sets),
            "M1_n_configs": len(m1_hash_sets[0]),
            "M2_identical_across_two_folds": all(h == m2_hash_sets[0] for h in m2_hash_sets),
            "M2_n_configs": len(m2_hash_sets[0]),
            "why_it_matters": ("with the config_hash set byte-identical across folds, a difference in "
                               "epochs/run between folds cannot be a config-mix effect; it is a fold "
                               "effect. This is what licenses reading M1's four-point pattern as "
                               "evidence about folds rather than about which configs were searched."),
        },
        "epochs_run_uncorrelated_with_validation_signal": {},
        "paired_direction_M2_F1_to_F2": {},
    }
    ic_mean = [ics["M1"][f"F{i}"]["mean"] for i in (1, 2, 3, 4)]
    ic_max = [ics["M1"][f"F{i}"]["max"] for i in (1, 2, 3, 4)]
    doc["finding_2_four_fold_epoch_pattern"]["epochs_run_uncorrelated_with_validation_signal"] = {
        "M1_mean_ensemble_validation_rank_ic_by_fold": dict(zip(order, ic_mean)),
        "M1_max_ensemble_validation_rank_ic_by_fold": dict(zip(order, ic_max)),
        "spearman_mean_ic_vs_epochs": {
            "rho": round(stats.spearmanr(ic_mean, epr).statistic, 4),
            "p": round(stats.spearmanr(ic_mean, epr).pvalue, 4)},
        "spearman_max_ic_vs_epochs": {
            "rho": round(stats.spearmanr(ic_max, epr).statistic, 4),
            "p": round(stats.spearmanr(ic_max, epr).pvalue, 4)},
        "reading": ("F4 has the HIGHEST validation IC (0.098265 mean) and the LOWEST epochs/run "
                    "(16.9630), so a stronger validation signal is not what makes runs longer. "
                    "Whatever drives the fold pattern, it is not 'the model needs more epochs when "
                    "there is more to learn'."),
    }
    pk2: dict[tuple[str, str], int] = {}
    runs2: dict[str, list[int]] = defaultdict(list)
    for d in glob.glob(str(ART / "m2_v1" / "checkpoints" / "*-epoch-*")):
        m = re.match(r"^(.+)-epoch-(\d+)$", Path(d).name)
        if m:
            runs2[m.group(1)].append(int(m.group(2)))
    for r, e in runs2.items():
        # groups: 1 fold, 2 config index, 3 seed. The run id's middle tag and its seed are the same
        # value on every run observed (m2-f2-c17-20260802-seed-20260802), so keying on
        # (fold, config, seed) is one entry per run -- verified by the pair count below being 72,
        # i.e. 24 configs x 3 seeds, with no collision silently overwriting a peak.
        g = re.match(r"^m2-(f\d)-(c\d+)-\d+-seed-(\d+)$", r)
        if g:
            pk2[(g.group(1), g.group(2), g.group(3))] = max(e)
    keys = sorted(k for k in pk2 if k[0] == "f1")
    pairs = [(pk2[("f1", c, s)], pk2[("f2", c, s)]) for (_, c, s) in keys
             if ("f2", c, s) in pk2]
    assert len(pairs) == 72, f"expected 24 configs x 3 seeds = 72 pairs, got {len(pairs)}"
    assert len(keys) == 72, f"expected 72 distinct F1 grid runs, got {len(keys)}"
    doc["finding_2_four_fold_epoch_pattern"]["paired_direction_M2_F1_to_F2"] = {
        "n_pairs": len(pairs),
        "increased": sum(1 for a, b in pairs if b > a),
        "decreased": sum(1 for a, b in pairs if b < a),
        "equal": sum(1 for a, b in pairs if b == a),
        "F1_mean": round(st.mean([a for a, _ in pairs]) + PATIENCE, 4),
        "F2_mean": round(st.mean([b for _, b in pairs]) + PATIENCE, 4),
        "F2_max_run_share_of_the_rise_pct": None,
        "reading": ("M2's F1->F2 rise is not one outlier: over 72 (config, seed) pairs, 50 rose and "
                    "20 fell, and dropping the single largest peak (71) still leaves the ratio at "
                    "2.41x instead of 2.51x. So the rise §CI observed is real and broad -- the "
                    "question is only whether it continues into F3/F4, and M1's four folds are "
                    "evidence that it need not."),
    }
    f1m = st.mean([a for a, _ in pairs]); f2m = st.mean([b for _, b in pairs])
    s2 = sorted(b for _, b in pairs)
    doc["finding_2_four_fold_epoch_pattern"]["paired_direction_M2_F1_to_F2"][
        "F2_max_run_share_of_the_rise_pct"] = round(
        ((f2m - st.mean(s2[:-1])) / (f2m - f1m)) * 100, 1)

    # ---- panel identity
    shared = []
    for f in ("F1", "F2"):
        did = rows_m1[f]["dataset_id"]
        same = (rows_m2.get(f, {}).get("dataset_id") == did)
        m1man = json.load(open(ART / "m1_full_v1" / "datasets" / did / "manifest.json"))
        m2man = json.load(open(ART / "m2_v1" / "datasets" / did / "manifest.json"))
        shared.append({
            "fold": f, "dataset_id": did[:16] + "…", "present_in_both_workspaces": same,
            "metadata_identical": m1man["metadata"] == m2man["metadata"],
            "shard_hashes_identical": m1man["shard_hashes"] == m2man["shard_hashes"],
            "split_content_hashes_identical":
                m1man["split_content_hashes"] == m2man["split_content_hashes"],
            "row_count_identical": m1man["row_count"] == m2man["row_count"],
            "m1_manifest_mtime": rows_m1[f]["manifest_mtime"],
            "m2_manifest_mtime": rows_m2[f]["manifest_mtime"],
            "gap_days": round((datetime.fromisoformat(rows_m2[f]["manifest_mtime"])
                               - datetime.fromisoformat(rows_m1[f]["manifest_mtime"])
                               ).total_seconds() / 86400, 2),
        })
    max_gap = max(s["gap_days"] for s in shared)
    doc["panel_identity_M1_equals_M2"] = {
        "shared_folds": shared,
        "mechanism": ("dataset_id IS the content hash (manifest.py:_hash_payload over business_keys, "
                      "columns+dtypes, canonicalised metadata = {fold_id, factor_order, "
                      "data_versions}, every row, and split_content_hashes), and "
                      "publish_fold_dataset (run_real_candidate_training.py:676) puts no family and "
                      "no model config into that metadata -- so identical ids across two carriers is "
                      "proof of identical panels, not a coincidence of naming."),
        "max_proven_no_drift_gap_days": max_gap,
        "exposure_for_M2_F3_days_at_CI_central_scenario": round(
            (datetime(2026, 10, 2) - datetime.fromisoformat(
                rows_m1["F3"]["manifest_mtime"])).total_seconds() / 86400, 1),
        "exposure_for_M2_F4_days_at_CI_central_scenario": round(
            (datetime(2026, 10, 2) - datetime.fromisoformat(
                rows_m1["F4"]["manifest_mtime"])).total_seconds() / 86400, 1),
        "LIMIT": ("data_versions is per-day (144-151 entries, and for label_daily/universe_daily "
                  "145 of 151 are distinct), so a single day's version changing changes the id. The "
                  "proven no-drift window is this many days; M2's exposure when it materialises F3/F4 "
                  "is roughly 20 days, several times longer, and that longer window is NOT measured. "
                  "Every exact row count below is therefore 'measured on M1's datasets', never 'the "
                  "rows M2 will train on'. If versions drift, M2's F3/F4 panels are different "
                  "objects and must be re-read."),
    }

    # ---- cost model form retest
    form = {}
    for label, ws, car in CARRIERS:
        ks = {}
        for f, row in folds[label].items():
            if f not in exact:
                continue
            k = row["grid_span_h"] * 3600 / (row["epochs_per_run"] * exact[f])
            ks[f] = {"implied_k": round(k, 7), "grid_span_h": row["grid_span_h"],
                     "epochs_per_run": row["epochs_per_run"], "rows": exact[f]}
        vals = [v["implied_k"] for v in ks.values()]
        rr = [v["rows"] for v in ks.values()]
        form[label] = {
            "per_fold": ks,
            "n_folds": len(vals),
            "k_mean": round(st.mean(vals), 7),
            "k_spread_max_over_min": round(max(vals) / min(vals), 4) if len(vals) > 1 else None,
            "k_sd": round(st.stdev(vals), 7) if len(vals) > 1 else None,
            "spearman_rows_vs_implied_k": (
                {"rho": round(stats.spearmanr(rr, vals).statistic, 4),
                 "p": round(stats.spearmanr(rr, vals).pvalue, 4)} if len(vals) > 2 else None),
            "model_family": None,
        }
    for path, label in (("m1_full_v1/models/m1-f1/model_card.json", "M1"),
                        ("m2_v1/models/m2-f1/model_card.json", "M2")):
        p = ART / path
        if p.is_file():
            card = json.load(open(p, encoding="utf-8"))
            form[label]["model_family"] = {k: card.get(k) for k in
                                           ("family", "model_code_version", "model_status")}
    form["comparison"] = {
        "M1_k_over_M2_k": round(form["M1"]["k_mean"] / form["M2"]["k_mean"], 4),
        "reading": ("k is family-specific and must not be transferred: M1 is RESIDUAL_MLP, M2 is TCN "
                    "with history_length 20, so M2 does far more work per row. What transfers is the "
                    "FORM -- implied k shows no monotone drift against panel rows on M1's four folds "
                    "(rho -0.20, p 0.80), which is the same test §CI could only run on two points."),
        "M1_spread_is_wider_and_why": (
            "M1's k spread is max/min "
            f"{form['M1']['k_spread_max_over_min']} against M2's "
            f"{form['M2']['k_spread_max_over_min']}. The likely cause is measurement resolution "
            "rather than model error: M1's grid spans are 4.8-6.8 h while M2's are 31.5-59.0 h, so "
            "the fixed error of an mtime-derived span (it omits the first run's startup and the last "
            "run's tail) is a far larger fraction of a smaller quantity. Both spreads are reported; "
            "only M2's 2.4% is good enough to price with, which is why the recomputation below keeps "
            "§CI's M2 k and changes only its inputs."),
    }
    doc["cost_model_form_retest"] = form
    doc["negative_controls"] = negative_controls(folds, hashes, rows_m1, rows_m2)

    # ---- scenario recomputation
    # reproduce §CI first, to prove the arithmetic here is the same arithmetic
    reproduce = {}
    for name, (e3, e4, pub) in CI_SCENARIOS.items():
        r = scenario(e3, e4, CI_PROJECTED_ROWS)
        reproduce[name] = {"published_total_h": pub, "recomputed_total_h": r["total_remaining_h"],
                           "diff_h": round(r["total_remaining_h"] - pub, 1)}
    effect_a = {}
    for name, (e3, e4, pub) in CI_SCENARIOS.items():
        a = scenario(e3, e4, CI_PROJECTED_ROWS)["total_remaining_h"]
        b = scenario(e3, e4, exact)["total_remaining_h"]
        effect_a[name] = {"with_ci_projected_rows_h": a, "with_measured_rows_h": b,
                          "delta_h": round(b - a, 1), "delta_pct": round((b - a) / a * 100, 2)}
    m1e = [m1[f]["epochs_per_run"] for f in ("F1", "F2", "F3", "F4")]
    r32, r43 = m1e[2] / m1e[1], m1e[3] / m1e[2]
    e3_b, e4_b = min(m2["F2"]["epochs_per_run"] * r32, CAP), min(
        m2["F2"]["epochs_per_run"] * r32 * r43, CAP)
    b_scen = scenario(e3_b, e4_b, exact)
    doc["scenario_recomputation"] = {
        "step_0_reproduce_CI_exactly": {
            "per_scenario": reproduce,
            "max_abs_diff_h": max(abs(v["diff_h"]) for v in reproduce.values()),
            "reading": ("reproducing §CI's four published totals to within 0.2 h with its own "
                        "projected rows is the gate on everything below: it proves this probe runs "
                        "the same arithmetic, so any change in the recomputed totals is attributable "
                        "to the changed inputs and not to a different model."),
        },
        "effect_A_measured_rows_only": {
            "what_changed": "F3/F4 train+val rows: projected -> measured footers. Epochs unchanged.",
            "per_scenario": effect_a,
            "range_pct": [min(v["delta_pct"] for v in effect_a.values()),
                          max(v["delta_pct"] for v in effect_a.values())],
        },
        "effect_B_M1_shaped_epochs": {
            "what_changed": ("F3/F4 epochs/run taken from M1's observed fold ratios instead of "
                             "flat / linear / geometric / capped. Rows are the measured ones."),
            "M1_fold_ratios_used": {"F3_over_F2": round(r32, 4), "F4_over_F3": round(r43, 4)},
            "M2_F2_anchor_epochs_per_run": m2["F2"]["epochs_per_run"],
            "implied_M2_F3_epochs_per_run": round(e3_b, 2),
            "implied_M2_F4_epochs_per_run": round(e4_b, 2),
            "scenario": b_scen,
            "projected_finish_from_now": (datetime.now() + timedelta(
                hours=b_scen["total_remaining_h"])).isoformat(timespec="minutes"),
            "versus_CI_optimistic_bound_h": CI_SCENARIOS["optimistic_epochs_flat"][2],
            "is_below_CI_optimistic_bound":
                b_scen["total_remaining_h"] < CI_SCENARIOS["optimistic_epochs_flat"][2],
            "reading": ("§CI's four scenarios are all non-decreasing in fold index, so their "
                        "optimistic end (166.6 h) is the floor of the family. M1 is a same-panels, "
                        "same-config-set observation of a pattern that DECREASES after F2, and "
                        "applying its ratios gives a lower total. So the honest interval is wider on "
                        "the LOW side than §CI published, not merely on the high side. This does not "
                        "make the low scenario likely -- M2's own F1->F2 rise was 2.51x, which M1 "
                        "never showed (M1's was 1.31x) -- it makes it unexcluded."),
        },
        "combined_interval_h": {
            "low": b_scen["total_remaining_h"],
            "high": scenario(100, 100, exact)["total_remaining_h"],
            "note": ("both ends use measured rows; the low end uses M1's fold ratios and the high end "
                     "assumes both remaining folds hit the 100-epoch cap, which §CJ shows no "
                     "published run has done."),
        },
        "assumptions_carried_unchanged_from_CI": {
            "k_grid": K_GRID_M2, "k_final": K_FINAL_M2, "final_under_prediction_uplift": FINAL_UPLIFT,
            "inter_phase_gap_h": GAP_H, "oos_and_report_h": OOS_H,
            "f2_final_remaining_h": 0.0,
            "why_f2_final_remaining_is_zero": ("§CI set it to 0 because 10.0 h had already elapsed "
                                              "against its estimate; this probe keeps that choice so "
                                              "the comparison is not confounded, even though F2's "
                                              "refit is demonstrably still writing checkpoints."),
        },
    }

    doc.update({"read_only": True, "database_access": False, "weights_loaded": False,
                "checkpoint_pt_opened": False, "training_performed": False,
                "repository_module_imported_or_executed": False,
                "parquet_data_loaded": False, "processes_signalled": []})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print("=== exact panels vs §CI's projection ===")
    for f in ("F1", "F2", "F3", "F4"):
        r = f1rows[f]
        print(f"  {f}: measured {r['measured_trainval']:>9,} | §CI {r['ci_published_trainval']:>9,} "
              f"| error {r['ci_error_pct']:+6.2f}% | test rows matched §CI's citation: "
              f"{r['test_rows_match_what_ci_cited']}")
    print("\n=== M1 four-fold epochs/run on identical panels and identical config set ===")
    for f in ("F1", "F2", "F3", "F4"):
        print(f"  {f}: rows {exact[f]:>9,}  epochs/run {m1[f]['epochs_per_run']:>8.4f}  "
              f"peak max {m1[f]['peak_max']:>3d}  mean val IC {ics['M1'][f]['mean']:.6f}")
    print(f"  monotone increasing: "
          f"{doc['finding_2_four_fold_epoch_pattern']['is_monotone_increasing_in_fold_index']}"
          f"  argmax={doc['finding_2_four_fold_epoch_pattern']['argmax_fold']}"
          f"  argmin={doc['finding_2_four_fold_epoch_pattern']['argmin_fold']}")
    print(f"  M2 for contrast: F1 {m2['F1']['epochs_per_run']:.4f} -> F2 "
          f"{m2['F2']['epochs_per_run']:.4f} (ratio "
          f"{doc['finding_2_four_fold_epoch_pattern']['M2_epoch_ratio_F2_over_F1']})")
    print("\n=== cost-model form retest ===")
    for label in ("M1", "M2"):
        fm = form[label]
        print(f"  {label}: k_mean {fm['k_mean']:.7f} spread {fm['k_spread_max_over_min']} "
              f"family {fm['model_family']['family'] if fm['model_family'] else '?'}")
    print(f"  M1 k / M2 k = {form['comparison']['M1_k_over_M2_k']}")
    print("\n=== scenario recomputation ===")
    print(f"  reproduce §CI: max |diff| = "
          f"{doc['scenario_recomputation']['step_0_reproduce_CI_exactly']['max_abs_diff_h']} h")
    for n, v in effect_a.items():
        print(f"  A {n:26s} {v['with_ci_projected_rows_h']:6.1f} -> {v['with_measured_rows_h']:6.1f} h"
              f" ({v['delta_pct']:+.2f}%)")
    eb = doc["scenario_recomputation"]["effect_B_M1_shaped_epochs"]
    print(f"  B M1-shaped epochs: F3 {eb['implied_M2_F3_epochs_per_run']}, "
          f"F4 {eb['implied_M2_F4_epochs_per_run']} -> {eb['scenario']['total_remaining_h']} h "
          f"({eb['scenario']['total_remaining_days']} d), below §CI's optimistic "
          f"{eb['versus_CI_optimistic_bound_h']} h: {eb['is_below_CI_optimistic_bound']}")
    ci = doc["scenario_recomputation"]["combined_interval_h"]
    print(f"  combined interval with measured rows: {ci['low']} - {ci['high']} h")
    print("\n=== negative controls (each instrument must be able to say NO) ===")
    nc = doc["negative_controls"]
    for name in ("config_set_identity", "panel_identity", "monotonicity", "scenario_arithmetic"):
        c = nc[name]
        checks = {k: v for k, v in c.items() if k.endswith(("must_be_True", "must_be_False"))}
        print(f"  {name}: {checks}")
    print(f"  ALL_CONTROLS_BEHAVE_AS_EXPECTED: {nc['ALL_CONTROLS_BEHAVE_AS_EXPECTED']}")
    assert nc["ALL_CONTROLS_BEHAVE_AS_EXPECTED"], "a negative control did not behave as expected"
    print("wrote", OUT)


if __name__ == "__main__":
    main()
