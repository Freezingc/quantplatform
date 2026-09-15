#!/usr/bin/env python3
"""Read-only probe: decompose ALL FOUR folds' first-appearing securities into listing-year cohorts.

§BX established the admission lag kernel from three cohorts (2022/2023/2024) using external
`new_share` ipo_date data, and its §BX.9 recorded two things it could not do:
  1. F1's tail bucket (468 securities, the whole fold) and F2's (112) were never resolved, so F1's
     468 was not decomposed into cohort tranches.
  2. No exact 2021 non-BSE cohort count was derived, because a 2021-H1 `new_share` query succeeded on
     retry but its response was truncated in transit, and §BX refused to count a number out of a
     truncated result.

Both gaps close without any external call, because a complete vendor listing-date dump already exists
on disk from earlier read-only work:
  /data/zhangyudong/qrv2_research_artifacts/opt1_feasibility/stock_basic_full.json
  5,901 securities, fields ts_code/name/list_date/delist_date/list_status, fetched 2026-09-15 across
  all three listing states (L/D/P), so it carries 339 delisted securities and is not survivorship
  biased; zero null list_date.

Two correctness traps this probe handles explicitly, because the first run of it fell into the second:

  TRAP 1 -- cross-fold double counting. "First appearing in fold F" means the security's earliest date
  inside F's own train+validation+test concatenation falls in F's test year. Each fold has its own
  window, so a security sitting on the boundary of two ADJACENT folds can satisfy that in both. 24
  securities do (F1&F2 14, F2&F3 7, F3&F4 3; every non-adjacent pair is 0, which is what the window
  explanation predicts). Summing lag counts across folds therefore overcounts, and a cohort total built
  that way can exceed the cohort size -- which is how the first run produced a 2021 cohort at 102.7%
  cumulative admission. Every cohort cell here is built from each security's EARLIEST fold, so cells
  count distinct securities. Both matrices are emitted and the naive one is labelled DO_NOT_USE.
  This matters for §BX.4: its "283+47+5 = 335 exactly" identity was built by summing across folds, so
  it is re-examined below rather than assumed.

  TRAP 2 -- two vendor fields disagree. §BX used new_share.ipo_date; this uses stock_basic.list_date.
  The cohort sizes differ (2022: 335 vs 345, 2023: 238 vs 236, 2024: 79 vs 77, 2025: 87 vs 90) and so
  do the per-fold lag counts. Both are reported side by side rather than one silently replacing the
  other, and the kernel is checked for robustness to the choice.

Safety: reads two JSON files already on disk. No database access, no network or vendor call, no MCP,
no repository module imported or executed, no weight or checkpoint loaded, nothing written into any
artifact directory, no signal sent to any process. M2 (pid 3588838) is still training while this runs.
"""
from __future__ import annotations

import itertools
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
STOCK_BASIC = Path("/data/zhangyudong/qrv2_research_artifacts/opt1_feasibility/stock_basic_full.json")
FIRST_APP = HERE / "measurements" / "first_appearance_cause.json"
XREF = HERE / "measurements" / "ipo_first_appearance_xref.json"
OUT = HERE / "measurements" / "first_appearance_cohort_decomposition.json"

FOLDS = ("F1", "F2", "F3", "F4")
FOLD_TEST_YEAR = {"F1": 2022, "F2": 2023, "F3": 2024, "F4": 2025}
BX_PUBLISHED = {
    "F2": {"lag1y_ipo2022": 283, "ipo<=2021_or_unknown": 112},
    "F3": {"lag1y_ipo2023": 203, "lag2y_ipo2022": 47, "ipo<=2021_or_unknown": 7},
    "F4": {"lag1y_ipo2024": 63, "lag2y_ipo2023": 33, "lag3y_ipo2022": 5, "ipo<=2021_or_unknown": 8},
}
BX_NONBSE_IPO_COUNTS = {"2022": 335, "2023": 238, "2024": 79, "2025": 87}
REENTRY_LAG_YEARS = 6
FOLD_PAIRS = tuple(itertools.combinations(FOLDS, 2))


def build_matrix(table, cohort_size):
    out = {}
    for y in sorted(table):
        if y < 2018 or cohort_size[y] == 0:
            continue
        size = cohort_size[y]
        row = {"cohort_nonbj_listed": size, "admitted_by_lag": {}, "admission_rate_pct_by_lag": {}}
        for lag in sorted(table[y]):
            if lag < 0 or lag > 6:
                continue
            row["admitted_by_lag"][f"lag{lag}"] = table[y][lag]
            row["admission_rate_pct_by_lag"][f"lag{lag}"] = round(100 * table[y][lag] / size, 1)
        row["cumulative_admitted_within_panel"] = sum(
            v for k, v in row["admitted_by_lag"].items() if int(k[3:]) >= 1)
        row["cumulative_pct"] = round(100 * row["cumulative_admitted_within_panel"] / size, 1)
        row["same_year_admitted"] = table[y].get(0, 0)
        out[str(y)] = row
    return out


def main() -> None:
    sb = json.load(open(STOCK_BASIC, encoding="utf-8"))["data"]
    list_year, meta = {}, {}
    for r in sb:
        code, ld = r["ts_code"], r.get("list_date")
        if not ld:
            continue
        list_year[code] = int(ld[:4])
        meta[code] = {"name": r.get("name"), "list_date": ld, "delist_date": r.get("delist_date"),
                      "list_status": r.get("list_status"), "is_bj": code.endswith(".BJ")}
    nonbj = {c: y for c, y in list_year.items() if not meta[c]["is_bj"]}
    cohort_size = Counter(nonbj.values())

    fa = json.load(open(FIRST_APP, encoding="utf-8"))["per_fold"]
    xr = json.load(open(XREF, encoding="utf-8"))["per_fold"]
    codes_by_fold = {f: fa[f]["first_appearing_codes"] for f in FOLDS}

    # integrity: the code lists must be complete before anything is decomposed
    for f in FOLDS:
        assert len(codes_by_fold[f]) == sum(fa[f]["board_composition"].values()), f"{f}: list != board sum"
        assert len(codes_by_fold[f]) == xr[f]["n"], f"{f}: list != xref n"
        assert len(set(codes_by_fold[f])) == len(codes_by_fold[f]), f"{f}: duplicate codes within fold"

    folds_per_code = defaultdict(list)
    for f in FOLDS:
        for c in codes_by_fold[f]:
            folds_per_code[c].append(f)
    first_seen_fold = {c: sorted(fs, key=lambda f: FOLD_TEST_YEAR[f])[0] for c, fs in folds_per_code.items()}

    per_fold, cohort_lag_naive, cohort_lag_dedup = {}, defaultdict(lambda: defaultdict(int)), \
        defaultdict(lambda: defaultdict(int))
    reentries, unmatched = [], {}
    for f in FOLDS:
        ty = FOLD_TEST_YEAR[f]
        codes = codes_by_fold[f]
        lags, miss = Counter(), []
        for c in codes:
            if c not in list_year:
                miss.append(c); lags["unmatched"] += 1; continue
            lag = ty - list_year[c]
            lags[f"lag{lag}"] += 1
            cohort_lag_naive[list_year[c]][lag] += 1
            if lag >= REENTRY_LAG_YEARS:
                reentries.append({"fold": f, "test_year": ty, "ts_code": c, **meta[c], "lag_years": lag})
        unmatched[f] = miss
        per_fold[f] = {
            "test_year": ty, "n": len(codes),
            "lag_distribution": {k: lags[k] for k in sorted(lags, key=lambda s: (s == "unmatched", s))},
            "n_unmatched": len(miss), "n_same_year_admitted": lags.get("lag0", 0),
            "by_cohort_year": {str(y): lags[f"lag{ty - y}"]
                               for y in sorted({list_year[c] for c in codes if c in list_year})},
        }
    for c, fold in first_seen_fold.items():
        if c in list_year:
            cohort_lag_dedup[list_year[c]][FOLD_TEST_YEAR[fold] - list_year[c]] += 1

    overlaps = {f"{a}&{b}": sorted(set(codes_by_fold[a]) & set(codes_by_fold[b])) for a, b in FOLD_PAIRS}
    total_entries = sum(len(v) for v in codes_by_fold.values())
    matrix = build_matrix(cohort_lag_dedup, cohort_size)
    matrix_naive = build_matrix(cohort_lag_naive, cohort_size)
    for y, row in matrix.items():
        over = matrix_naive[y]["cumulative_admitted_within_panel"] - row["cumulative_admitted_within_panel"]
        if over:
            row["double_counted_if_summed_across_folds"] = over
        assert row["cumulative_pct"] <= 100.0, f"cohort {y} exceeds 100% after dedup: {row}"

    # cross-check the two vendor fields on §BX's published per-fold counts
    xcheck = {}
    for fold, pub in BX_PUBLISHED.items():
        ty = FOLD_TEST_YEAR[fold]
        got = {}
        for key, want in pub.items():
            if key == "ipo<=2021_or_unknown":
                have = sum(v for k, v in per_fold[fold]["lag_distribution"].items()
                           if k != "unmatched" and int(k[3:]) >= ty - 2021) + per_fold[fold]["n_unmatched"]
            else:
                have = per_fold[fold]["lag_distribution"].get("lag" + key.split("y_")[0][3:], 0)
            got[key] = {"bx_new_share_ipo_date": want, "this_stock_basic_list_date": have,
                        "match": want == have, "diff": have - want}
        xcheck[fold] = got

    # §BX.4's exact-335 identity, re-examined under deduplication and under the other vendor field
    def cohort_sum(y, table):
        return sum(v for lag, v in table[y].items() if lag >= 1)

    bx4 = {}
    for y in (2021, 2022, 2023, 2024):
        bx4[str(y)] = {
            "cohort_size_stock_basic": cohort_size[y],
            "cohort_size_new_share": BX_NONBSE_IPO_COUNTS.get(str(y)),
            "admitted_naive_sum_across_folds": cohort_sum(y, cohort_lag_naive),
            "admitted_dedup_earliest_fold": cohort_sum(y, cohort_lag_dedup),
            "overcount": cohort_sum(y, cohort_lag_naive) - cohort_sum(y, cohort_lag_dedup),
            "dedup_pct_of_stock_basic_cohort": round(100 * cohort_sum(y, cohort_lag_dedup) / cohort_size[y], 1),
        }
    bx4["verdict_on_the_335_identity"] = {
        "bx_claim": "283 (lag1) + 47 (lag2) + 5 (lag3) = 335 = the 2022 non-BSE IPO count, exactly",
        "naive_sum_2022": cohort_sum(2022, cohort_lag_naive),
        "dedup_sum_2022": cohort_sum(2022, cohort_lag_dedup),
        "overcount_2022": cohort_sum(2022, cohort_lag_naive) - cohort_sum(2022, cohort_lag_dedup),
        "cohort_size_this_source_2022": cohort_size[2022],
    }

    doc = {
        "question": ("§BX.9 items 1 and 2: decompose F1's 468 (and F2's 112) into listing-year cohorts "
                     "and extend the admission lag kernel to 2019-2021, using the on-disk vendor dump "
                     "instead of any new call; plus re-examine §BX.4's exact-335 identity, which summed "
                     "across folds."),
        "source": {"file": str(STOCK_BASIC), "n_securities": len(list_year), "n_non_bj": len(nonbj),
                   "n_bj": len(list_year) - len(nonbj),
                   "n_delisted_included": sum(1 for r in sb if r.get("delist_date")),
                   "list_status_counts": dict(Counter(r.get("list_status") for r in sb)),
                   "null_list_date": sum(1 for r in sb if not r.get("list_date")),
                   "fetched_by": "opt1_feasibility/fetch_stock_basic.py, 2026-09-15, all three listing "
                                 "states L/D/P, so not survivorship biased",
                   "field_used": "list_date (date trading began); §BX used new_share ipo_date"},
        "cohort_sizes_non_bj": {str(y): cohort_size[y] for y in sorted(cohort_size) if y >= 2015},
        "per_fold": per_fold,
        "cross_fold_double_counting": {
            "n_securities_in_more_than_one_fold": len(folds_per_code) - sum(
                1 for fs in folds_per_code.values() if len(fs) == 1),
            "total_fold_entries": total_entries, "distinct_securities": len(folds_per_code),
            "excess_entries": total_entries - len(folds_per_code),
            "overlapping_pairs": {k: v for k, v in overlaps.items() if v},
            "non_adjacent_pairs_all_zero": all(len(overlaps[p]) == 0 for p in ("F1&F3", "F1&F4", "F2&F4")),
            "why": ("each fold's test uses that fold's own train+val+test window, so a boundary "
                    "security can qualify in two adjacent folds"),
            "consequence": "cohort cells must come from the earliest fold; naive sums can exceed 100%",
        },
        "cross_check_vs_bx_new_share": xcheck,
        "cross_check_all_match": all(v["match"] for f in xcheck.values() for v in f.values()),
        "cohort_admission_matrix_deduplicated": matrix,
        "cohort_admission_matrix_if_summed_across_folds_DO_NOT_USE": matrix_naive,
        "bx4_identity_reexamined": bx4,
        "reentries": sorted(reentries, key=lambda r: (-r["lag_years"], r["fold"])),
        "n_reentries": len(reentries),
        "unmatched_codes": unmatched,
        "read_only": True, "database_access": False, "network_or_vendor_call": False, "mcp_used": False,
        "weights_loaded": False, "training_performed": False,
        "repository_module_imported_or_executed": False,
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
    }
    json.dump(doc, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print("source:", len(list_year), "securities;", len(nonbj), "non-BJ;",
          doc["source"]["n_delisted_included"], "delisted included; nulls",
          doc["source"]["null_list_date"])
    print("non-BJ cohort sizes:", {k: v for k, v in doc["cohort_sizes_non_bj"].items() if k >= "2019"})
    print()
    for f in FOLDS:
        p = per_fold[f]
        print(f"{f} (test {p['test_year']}, n={p['n']}, same-year={p['n_same_year_admitted']}, "
              f"unmatched={p['n_unmatched']})")
        print("   lag:", p["lag_distribution"])
        print("   by cohort:", p["by_cohort_year"])
    d = doc["cross_fold_double_counting"]
    print(f"\ndouble counting: {d['n_securities_in_more_than_one_fold']} securities in >1 fold; "
          f"{d['total_fold_entries']} entries -> {d['distinct_securities']} distinct "
          f"(excess {d['excess_entries']}); non-adjacent pairs all zero={d['non_adjacent_pairs_all_zero']}")
    for k, v in d["overlapping_pairs"].items():
        print(f"   {k}: {len(v)}")
    print("\ncross-check vs §BX new_share ipo_date: all match =", doc["cross_check_all_match"])
    for fold, got in xcheck.items():
        for key, v in got.items():
            print(f"  {'ok  ' if v['match'] else 'DIFF'} {fold} {key}: bx={v['bx_new_share_ipo_date']} "
                  f"this={v['this_stock_basic_list_date']} diff={v['diff']:+d}")
    print("\ncohort x lag admission (DEDUPLICATED, % of non-BJ cohort):")
    for y in sorted(matrix):
        if int(y) < 2019:
            continue
        m = matrix[y]
        rates = " ".join(f"{k}={v}%" for k, v in m["admission_rate_pct_by_lag"].items())
        extra = (f"  [naive sum overcounts by {m['double_counted_if_summed_across_folds']}]"
                 if "double_counted_if_summed_across_folds" in m else "")
        print(f"  {y} n={m['cohort_nonbj_listed']:4d} same-yr={m['same_year_admitted']} "
              f"cum={m['cumulative_pct']:5.1f}%{extra}  {rates}")
    print("\n§BX.4 identity re-examined:")
    for y in ("2021", "2022", "2023", "2024"):
        b = bx4[y]
        print(f"  cohort {y}: stock_basic size {b['cohort_size_stock_basic']}, new_share size "
              f"{b['cohort_size_new_share']}; admitted naive {b['admitted_naive_sum_across_folds']} "
              f"vs dedup {b['admitted_dedup_earliest_fold']} (overcount {b['overcount']}), "
              f"dedup = {b['dedup_pct_of_stock_basic_cohort']}% of cohort")
    print("  verdict:", json.dumps({k: v for k, v in bx4["verdict_on_the_335_identity"].items()
                                    if k != "bx_claim"}, ensure_ascii=False))
    print(f"\nre-entries (listed >= {REENTRY_LAG_YEARS}y before first appearance): {len(reentries)}")
    for r in doc["reentries"]:
        print(f"  {r['fold']} {r['ts_code']} {r['name']} listed {r['list_date']} lag {r['lag_years']}y "
              f"status {r['list_status']}")
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
