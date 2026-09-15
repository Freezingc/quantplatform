#!/usr/bin/env python3
"""Read-only: why do the six candidate_only factors lose coverage across the OOS window?

§AN measured, from factor_daily, that the six sparse factors' coverage drifts DOWN by 0.097-0.184
across 2022-2025 while the nine dense ones hold. It recorded the drift and explicitly did not
explain it. This probe measures the same thing from the side that actually matters -- the frozen
dataset panels the models consume -- and tests three candidate explanations:

  H1  the denominator grows: the universe adds securities over time and the factor's production does
      not keep up, so coverage falls even with a flat or rising row count;
  H2  new entrants: the missing values concentrate on securities that are new to the assembled panel
      (the same population §AX showed the temporal family cannot score), which would make the drift a
      listing-cohort effect rather than a factor-production effect;
  H3  board mix: the missing values concentrate on particular boards (STAR 688/689, BSE 8xx/4xx/920,
      ChiNext 300/301), which would point at an upstream input that covers boards unevenly.

None of the three is assumed; each is measured, and the dense nine are run through the same cuts as a
control that should show nothing.

Read-only. No database access at all -- every input is a parquet panel the frozen pipeline already
published. No weights, no model, nothing written into any artifact directory.
"""
from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd

ART = Path("/data/zhangyudong/qrv2_research_artifacts")
FOLD_DATASETS = {
    "F1": "226a8dfc497c2de6b07516711f058f26a9270675916c9e67dbf7f087a07d2538",
    "F2": "eb9bba1b5a0bd66d0ad6c82536b295de1fc079aec125cb27bb68491bdb21a234",
    "F3": "9766e69f342511c5bda30188df15a7b9b17d936bdc26b9b60ba244e8e7e0c9e7",
    "F4": "6f9478d7645c8c745a73f00c03b6122e17d0c06c05cc159ece531fb00df9fb9d",
}
FOLD_TEST_YEAR = {"F1": 2022, "F2": 2023, "F3": 2024, "F4": 2025}
WORKSPACES = ("m2_v1", "m0r_v2", "m1_full_v1", "m0_v3", "m0_v2")
# walk_forward_factor_daily.py:75-91 -- FROZEN_FACTOR_SET_V2, in panel-column order
FACTOR_BY_INDEX = (
    "yeslab_206", "yeslab_357", "yeslab_372", "yeslab_375", "yeslab_381", "yeslab_384",
    "yeslab_389", "yeslab_390", "yeslab_406", "yeslab_425", "yeslab_443_reg30_max20_v1",
    "yeslab_464", "yeslab_471", "yeslab_484", "yeslab_96",
)
CANDIDATE_ONLY = {"yeslab_372", "yeslab_375", "yeslab_384", "yeslab_389", "yeslab_390", "yeslab_406"}
FCOLS = [f"f{i:02d}" for i in range(15)]


def board(ts_code: str) -> str:
    c = str(ts_code)
    if c.startswith(("688", "689")):
        return "STAR"
    if c.startswith(("8", "4", "92")):
        return "BSE"
    if c.startswith(("300", "301", "302")):
        return "ChiNext"
    if c.startswith("60"):
        return "SH_main"
    if c.startswith(("000", "001", "002", "003")):
        return "SZ_main"
    return "other"


def paths(artifacts: Path, ds: str, split: str) -> list[str]:
    for ws in WORKSPACES:
        p = sorted(glob.glob(str(artifacts / ws / "datasets" / ds / f"split={split}" / "*" / "*.parquet")))
        if p:
            return p
    raise FileNotFoundError(f"{ds} split={split}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, default=ART)
    ap.add_argument("--out", type=Path, default=ART / "sparse_coverage")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    result = {
        "question": (
            "Why do the six candidate_only factors lose coverage across the OOS window? Tested against "
            "three explanations -- a growing denominator, new-entrant securities, and board mix -- with "
            "the nine dense factors run through the same cuts as a control."
        ),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "read_only": True, "database_access": False, "weights_loaded": False,
        "source_of_truth": (
            "the frozen dataset panels each fold published, not factor_daily: this measures the values "
            "the models actually consume, which is the side the factor-set decision is about"
        ),
        "factor_column_mapping": {f"f{i:02d}": n for i, n in enumerate(FACTOR_BY_INDEX)},
        "folds": {},
    }

    for fold, ds in FOLD_DATASETS.items():
        prior = pd.concat(
            [pd.read_parquet(p, columns=["ts_code"]) for s in ("train", "validation")
             for p in paths(args.artifacts, ds, s)],
            ignore_index=True,
        )
        prior_codes = set(prior["ts_code"].unique())
        del prior

        te = pd.concat(
            [pd.read_parquet(p, columns=["signal_date", "ts_code", "label_valid", "raw_return_5d"] + FCOLS)
             for p in paths(args.artifacts, ds, "test")],
            ignore_index=True,
        )
        te["signal_date"] = pd.to_datetime(te["signal_date"]).dt.date
        te["_scoreable"] = (
            te["label_valid"].fillna(False).astype(bool)
            & np.isfinite(pd.to_numeric(te["raw_return_5d"], errors="coerce"))
        )
        te["_entrant"] = ~te["ts_code"].isin(prior_codes)
        te["_board"] = te["ts_code"].map(board)
        feat = te[FCOLS].apply(pd.to_numeric, errors="coerce")
        sc = te.loc[te["_scoreable"]]
        featsc = feat.loc[te["_scoreable"].to_numpy()]

        # H1: is the denominator growing while the numerator does not?
        per_day_rows = sc.groupby("signal_date").size()
        half = len(per_day_rows) // 2
        days_sorted = sorted(per_day_rows.index)
        first_half, second_half = days_sorted[:half], days_sorted[half:]
        h1 = {
            "scoreable_rows": int(len(sc)),
            "days": int(len(per_day_rows)),
            "rows_per_day_first_half_mean": float(per_day_rows.reindex(first_half).mean()),
            "rows_per_day_second_half_mean": float(per_day_rows.reindex(second_half).mean()),
            "denominator_growth_second_over_first": float(
                per_day_rows.reindex(second_half).mean() / per_day_rows.reindex(first_half).mean()
            ),
            "distinct_securities_first_half": int(
                sc.loc[sc["signal_date"].isin(first_half), "ts_code"].nunique()
            ),
            "distinct_securities_second_half": int(
                sc.loc[sc["signal_date"].isin(second_half), "ts_code"].nunique()
            ),
        }

        # per-factor daily coverage, and its within-year drift
        per_factor = {}
        for col in FCOLS:
            name = FACTOR_BY_INDEX[int(col[1:])]
            present = featsc[col].notna()
            daily = present.groupby(sc["signal_date"].to_numpy()).mean()
            daily = pd.Series(daily).reindex(days_sorted)
            fh = float(daily.reindex(first_half).mean())
            sh = float(daily.reindex(second_half).mean())
            per_factor[name] = {
                "column": col,
                "is_candidate_only": name in CANDIDATE_ONLY,
                "coverage_mean": float(daily.mean()),
                "coverage_median": float(daily.median()),
                "coverage_min": float(daily.min()),
                "coverage_max": float(daily.max()),
                "coverage_first_half_mean": fh,
                "coverage_second_half_mean": sh,
                "within_year_drift": sh - fh,
                "overall_missing_share": float(1.0 - present.mean()),
            }

        # H2: new-entrant enrichment
        ent = te["_entrant"].to_numpy() & te["_scoreable"].to_numpy()
        non = (~te["_entrant"]).to_numpy() & te["_scoreable"].to_numpy()
        h2 = {
            "entrant_scoreable_rows": int(ent.sum()),
            "non_entrant_scoreable_rows": int(non.sum()),
            "entrant_share_of_scoreable_rows": float(ent.sum() / max(int(te["_scoreable"].sum()), 1)),
            "missing_rate": {},
        }
        for col in FCOLS:
            name = FACTOR_BY_INDEX[int(col[1:])]
            v = feat[col].to_numpy()
            me = float(np.isnan(v[ent]).mean()) if ent.sum() else None
            mn = float(np.isnan(v[non]).mean()) if non.sum() else None
            h2["missing_rate"][name] = {
                "entrants": me, "non_entrants": mn,
                "enrichment": (me / mn) if (me is not None and mn not in (None, 0)) else None,
            }

        # H3: board mix
        h3 = {}
        for b, idx in te.groupby("_board").groups.items():
            mask = te["_scoreable"].to_numpy()[te.index.get_indexer(idx)]
            if mask.sum() < 100:
                continue
            sub = feat.loc[idx][mask]
            h3[b] = {
                "scoreable_rows": int(mask.sum()),
                "missing_rate": {
                    FACTOR_BY_INDEX[int(c[1:])]: float(sub[c].isna().mean()) for c in FCOLS
                },
            }
        board_shares = {
            b: float((te.loc[te["_scoreable"], "_board"] == b).mean()) for b in h3
        }

        result["folds"][fold] = {
            "dataset_id": ds,
            "test_year": FOLD_TEST_YEAR[fold],
            "prior_split_securities": len(prior_codes),
            "test_securities": int(te["ts_code"].nunique()),
            "H1_denominator": h1,
            "per_factor_daily_coverage": per_factor,
            "H2_new_entrants": h2,
            "H3_board_mix": {"missing_rate_by_board": h3,
                             "board_share_of_scoreable_rows": board_shares},
        }
        print(f"  {fold} ({FOLD_TEST_YEAR[fold]}): rows/day {h1['rows_per_day_first_half_mean']:.0f} -> "
              f"{h1['rows_per_day_second_half_mean']:.0f} (x{h1['denominator_growth_second_over_first']:.4f}), "
              f"entrant share {h2['entrant_share_of_scoreable_rows']:.4f}", flush=True)
        del te, feat, sc, featsc

    # ---- cross-fold summary: the drift the question is about ----------------------
    summary = {"per_factor": {}}
    for name in FACTOR_BY_INDEX:
        rows = []
        for fold in FOLD_DATASETS:
            pf = result["folds"][fold]["per_factor_daily_coverage"][name]
            rows.append({"fold": fold, "test_year": FOLD_TEST_YEAR[fold],
                         "coverage_mean": pf["coverage_mean"],
                         "within_year_drift": pf["within_year_drift"]})
        means = [r["coverage_mean"] for r in rows]
        drifts = [r["within_year_drift"] for r in rows]
        summary["per_factor"][name] = {
            "is_candidate_only": name in CANDIDATE_ONLY,
            "by_test_year": rows,
            "coverage_2022": means[0], "coverage_2025": means[-1],
            "across_year_change_2022_to_2025": means[-1] - means[0],
            "within_year_drift_median": float(median(drifts)),
            "monotone_declining_across_years": all(
                means[i] >= means[i + 1] for i in range(len(means) - 1)
            ),
        }
    sparse = [v["across_year_change_2022_to_2025"] for k, v in summary["per_factor"].items()
              if v["is_candidate_only"]]
    dense = [v["across_year_change_2022_to_2025"] for k, v in summary["per_factor"].items()
             if not v["is_candidate_only"]]
    summary["sparse_vs_dense"] = {
        "sparse_across_year_change": {"min": min(sparse), "median": float(median(sparse)), "max": max(sparse)},
        "dense_across_year_change": {"min": min(dense), "median": float(median(dense)), "max": max(dense)},
        "all_six_decline": all(x < 0 for x in sparse),
        "any_dense_declines_by_more_than_0.01": any(x < -0.01 for x in dense),
    }
    ent_share = [result["folds"][f]["H2_new_entrants"]["entrant_share_of_scoreable_rows"]
                 for f in FOLD_DATASETS]
    denom = [result["folds"][f]["H1_denominator"]["denominator_growth_second_over_first"]
             for f in FOLD_DATASETS]
    summary["explanations"] = {
        "H1_denominator_grows_within_year": {
            "second_half_over_first_half_by_fold": denom,
            "verdict": "measured per fold above; a value near 1.0 means the denominator is flat within the year",
        },
        "H2_entrant_share_of_scoreable_rows_by_fold": ent_share,
        "H3_note": "see each fold's board_share_of_scoreable_rows and missing_rate_by_board",
    }
    result["summary"] = summary

    out = args.out / "sparse_factor_coverage_in_panels.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    print(f"wrote {out}")
    print("\nacross-year change in mean daily coverage, 2022 -> 2025:")
    for name, v in summary["per_factor"].items():
        tag = "sparse" if v["is_candidate_only"] else "dense "
        print(f"  {tag} {name:28} {v['coverage_2022']:.4f} -> {v['coverage_2025']:.4f} "
              f"({v['across_year_change_2022_to_2025']:+.4f}) within-year drift med "
              f"{v['within_year_drift_median']:+.4f} monotone={v['monotone_declining_across_years']}")
    print("\n", json.dumps(summary["sparse_vs_dense"], ensure_ascii=False, indent=1))
    print("\n", json.dumps(summary["explanations"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
