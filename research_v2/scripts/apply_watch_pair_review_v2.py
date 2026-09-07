"""Apply the approved watch-pair review (W1) + adjudication-4 hybrid as frozen-set v2.

Authorized 2026-09-07 by user decision (package approval):
  W1: remove yeslab_405 / yeslab_398 from the frozen set -- watch-pair review
      (watch_pair_review_v2) found both FAIL the incremental-information rule
      (pooled retention 7.6% / 20.3% given yeslab_381; 381 is dominant).
  Adjudication-4 hybrid: the six sparse factors (372/375/384/389/390/406) stay
      in the frozen set as CANDIDATE-ONLY features (model-side NaN handling;
      excluded from the linear baseline whose dropna would thin the trust
      root). The dense nine remain baseline-eligible.
  Fold structure (recorded in contract): 4y train + 1y val + 1y test, F1-F4,
      test years 2022-2025; factor backfill scope 2017-2019.

Append-only: writes a NEW frozen_date batch (2026-09-07) into factor_set_frozen
(15 ADMITTED + 2 REMOVED_WATCH_PAIR_REVIEW), never modifies the 2026-09-03 v1
batch; writes factor_set_frozen_v2.json; appends the factor_daily-lineage pair
evidence (window_id '<year>_fd') into factor_pair_redundancy (pair_class by the
established thresholds: <0.80 OK, <0.90 ELEVATED, else REDUNDANT).
Idempotent per (frozen_date/as_of_date, data_version). Dry-run unless --apply.
2026 locked-test data is never read or written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import uuid
from datetime import date
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "research_v2" / "src"))

DIAG = REPO_ROOT / "research_v2" / "artifacts" / "diagnostics"
TARGET_DATABASE = "dfs://QuantResearchV2Test"
FROZEN_DATE = date(2026, 9, 7)
V1_FROZEN_DATE = date(2026, 9, 3)
REPAIR_VARIANT_443 = "yeslab_443_reg30_max20_v1"

DENSE_BASELINE = ["yeslab_96", "yeslab_206", "yeslab_357", "yeslab_381", "yeslab_425",
                  "yeslab_443_reg30_max20_v1", "yeslab_464", "yeslab_471", "yeslab_484"]
SPARSE_CANDIDATE_ONLY = ["yeslab_372", "yeslab_375", "yeslab_384", "yeslab_389",
                         "yeslab_390", "yeslab_406"]
REMOVED = ["yeslab_398", "yeslab_405"]
FOLD_STRUCTURE = "4y_train_1y_val_1y_test_F1-F4_test2022-2025"

REVIEW_JSON = DIAG / "watch_pair_review" / "watch_pair_review_v2.json"
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
PAIRS = [("yeslab_381", "yeslab_405"), ("yeslab_381", "yeslab_398"),
         ("yeslab_381", "yeslab_443_reg30_max20_v1")]


def record_hash(rec: dict) -> str:
    payload = {k: (v.isoformat() if isinstance(v, (date, pd.Timestamp)) else v)
               for k, v in rec.items() if k != "record_hash"}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True).encode()).hexdigest()


def pair_class(rho: float) -> str:
    if rho >= 0.90:
        return "REDUNDANT"
    if rho >= 0.80:
        return "ELEVATED"
    return "OK"


def build_rows(v1_rows: pd.DataFrame, evidence_commit: str):
    admitted = sorted(DENSE_BASELINE + SPARSE_CANDIDATE_ONLY)
    watermark = hashlib.sha256(json.dumps({
        "scope": "FACTOR_SET_FROZEN_V2", "frozen_date": FROZEN_DATE.isoformat(),
        "admitted": admitted, "removed": sorted(REMOVED),
        "sparse_candidate_only": sorted(SPARSE_CANDIDATE_ONLY),
        "fold_structure": FOLD_STRUCTURE,
    }, sort_keys=True).encode()).hexdigest()
    data_version = hashlib.sha256((watermark + str(FROZEN_DATE)).encode()).hexdigest()
    run_id = str(uuid.uuid4())
    v1 = {r["factor_id"]: r for r in v1_rows.to_dict("records")}

    rows = []
    for fid in admitted:
        old = v1[fid]
        sparse = fid in SPARSE_CANDIDATE_ONLY
        monitors = {"note": "see decision sheet monitors",
                    "role": "candidate_only" if sparse else "baseline_and_candidate",
                    "baseline_eligible": not sparse,
                    "fold_structure": FOLD_STRUCTURE}
        if sparse:
            monitors["v2_reason"] = "adjudication4_hybrid: sparse coverage (up-day-defined formula), kept for candidates with native NaN handling"
        rec = {
            "frozen_date": FROZEN_DATE, "factor_id": fid, "factor_version": 1,
            "candidate_id": old["candidate_id"],
            "decision": "ADMITTED_REPAIR_VARIANT" if fid == REPAIR_VARIANT_443 else "ADMITTED",
            "grade": old["grade"],
            "monitors_json": json.dumps(monitors, ensure_ascii=False),
            "decision_point": ("D1_repair" if fid == REPAIR_VARIANT_443
                               else "A_table_A4_hybrid" if sparse else "A_table"),
            "evidence_commit": evidence_commit,
            "run_id": run_id, "data_version": data_version,
            "created_at": pd.Timestamp.utcnow().tz_localize(None),
            "source_watermark_id": watermark,
        }
        rec["record_hash"] = record_hash(rec)
        rows.append(rec)
    for fid in sorted(REMOVED):
        old = v1[fid]
        rec = {
            "frozen_date": FROZEN_DATE, "factor_id": fid, "factor_version": 1,
            "candidate_id": old["candidate_id"], "decision": "REMOVED_WATCH_PAIR_REVIEW",
            "grade": old["grade"],
            "monitors_json": json.dumps({
                "role": "removed", "baseline_eligible": False,
                "v2_reason": "W1: watch-pair review v2 FAIL (incremental retention 7.6%/20.3% given yeslab_381)",
            }, ensure_ascii=False),
            "decision_point": "W1_watch_pair_review",
            "evidence_commit": evidence_commit,
            "run_id": run_id, "data_version": data_version,
            "created_at": pd.Timestamp.utcnow().tz_localize(None),
            "source_watermark_id": watermark,
        }
        rec["record_hash"] = record_hash(rec)
        rows.append(rec)
    return rows, watermark, data_version, run_id


def build_pair_rows(review: dict):
    watermark = hashlib.sha256(json.dumps({
        "scope": "WATCH_PAIR_REVIEW_FD", "as_of": FROZEN_DATE.isoformat(),
        "pairs": [list(p) for p in PAIRS], "source": "watch_pair_review_v2.json",
        "window_suffix": "_fd",
    }, sort_keys=True).encode()).hexdigest()
    data_version = hashlib.sha256((watermark + str(FROZEN_DATE)).encode()).hexdigest()
    run_id = str(uuid.uuid4())
    rows = []
    for a, b in PAIRS:
        left, right = sorted([a, b])
        k1, k2 = f"rho__{a}__{b}", f"rho__{b}__{a}"
        rho_key = k1 if k1 in review["per_year"][str(YEARS[0])] else k2
        for y in YEARS:
            py = review["per_year"][str(y)]
            rec = {
                "as_of_date": FROZEN_DATE,
                "factor_id_left": left, "factor_id_right": right,
                "window_id": f"{y}_fd",
                "median_abs_spearman": float(py[rho_key]["median_abs_rho"]),
                "valid_day_count": int(py["days"]),
                "pair_class": pair_class(float(py[rho_key]["median_abs_rho"])),
                "run_id": run_id, "data_version": data_version,
                "created_at": pd.Timestamp.utcnow().tz_localize(None),
                "source_watermark_id": watermark,
            }
            rec["record_hash"] = record_hash(rec)
            rows.append(rec)
    return rows, watermark, data_version


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write to DB (default dry-run)")
    args = parser.parse_args(argv)

    evidence_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                     capture_output=True, text=True).stdout.strip()
    review = json.loads(REVIEW_JSON.read_text(encoding="utf-8"))

    import dolphindb as ddb
    client = ddb.session()
    client.connect("10.176.55.211", 8848, "xiahongjie", "123456")
    v1_rows = client.run(
        f"select factor_id, candidate_id, grade from loadTable('{TARGET_DATABASE}','factor_set_frozen') "
        f"where frozen_date=date('{V1_FROZEN_DATE:%Y.%m.%d}')")
    assert len(v1_rows) == 17, f"expected 17 v1 rows, got {len(v1_rows)}"

    frozen_rows, fz_watermark, fz_dv, _ = build_rows(v1_rows, evidence_commit)
    pair_rows, pr_watermark, pr_dv = build_pair_rows(review)

    print(f"v2 frozen rows: {len(frozen_rows)} (ADMITTED {len(DENSE_BASELINE) + len(SPARSE_CANDIDATE_ONLY)} "
          f"+ REMOVED {len(REMOVED)}), data_version={fz_dv[:12]}...")
    print(f"pair evidence rows: {len(pair_rows)} ({len(PAIRS)} pairs x {len(YEARS)} years, window_id '*_fd'), "
          f"data_version={pr_dv[:12]}...")
    for r in pair_rows:
        print(f"  {r['factor_id_left']} x {r['factor_id_right']} {r['window_id']}: "
              f"rho={r['median_abs_spearman']:.4f} [{r['pair_class']}] days={r['valid_day_count']}")
    if not args.apply:
        print("\nDry-run only. Re-run with --apply to write.")
        client.close()
        return 0

    from research_v2.db.migrations import MigrationRunner
    from research_v2.db.session import V2WriteSession
    session = V2WriteSession(client, target_database=TARGET_DATABASE)
    MigrationRunner(session, migration_directory=REPO_ROOT / "research_v2" / "migrations").apply()

    def staged_insert(table: str, frame: pd.DataFrame, casts: dict[str, str], guard_sql: str):
        n = int(session.query(guard_sql)["n"][0])
        if n:
            print(f"{table}: already present ({n} rows) - skipped.")
            return
        stage = f"stage_{uuid.uuid4().hex[:12]}"
        session.upload_frame(stage, frame)
        cols = ", ".join(f"{casts[c]}({c}) as {c}" for c in frame.columns)
        session.execute(
            f"typed = select {cols} from {stage}\n"
            f"tableInsert{{loadTable('{TARGET_DATABASE}','{table}')}}(typed)\n"
            f"undef(`{stage}, VAR)\nundef(`typed, VAR)")
        back = int(session.query(guard_sql)["n"][0])
        print(f"{table}: wrote {back} rows.")

    fz_frame = pd.DataFrame(frozen_rows)
    staged_insert("factor_set_frozen", fz_frame, {
        "frozen_date": "date", "factor_id": "symbol", "factor_version": "int",
        "candidate_id": "symbol", "decision": "symbol", "grade": "symbol",
        "monitors_json": "string", "decision_point": "string", "evidence_commit": "string",
        "run_id": "uuid", "data_version": "string", "created_at": "timestamp",
        "source_watermark_id": "string", "record_hash": "string",
    }, (f"select count(*) as n from loadTable('{TARGET_DATABASE}','factor_set_frozen') "
        f"where frozen_date=date('{FROZEN_DATE:%Y.%m.%d}') and data_version='{fz_dv}'"))

    pr_frame = pd.DataFrame(pair_rows)
    staged_insert("factor_pair_redundancy", pr_frame, {
        "as_of_date": "date", "factor_id_left": "symbol", "factor_id_right": "symbol",
        "window_id": "symbol", "median_abs_spearman": "double", "valid_day_count": "int",
        "pair_class": "symbol", "run_id": "uuid", "data_version": "string",
        "created_at": "timestamp", "source_watermark_id": "string", "record_hash": "string",
    }, (f"select count(*) as n from loadTable('{TARGET_DATABASE}','factor_pair_redundancy') "
        f"where as_of_date=date('{FROZEN_DATE:%Y.%m.%d}') and data_version='{pr_dv}'"))

    out = DIAG / "factor_set_frozen_v2.json"
    out.write_text(json.dumps({
        "scope": "FACTOR_SET_FROZEN_V2",
        "frozen_date": FROZEN_DATE.isoformat(),
        "supersedes": "factor_set_frozen_v1.json",
        "factors": sorted(DENSE_BASELINE + SPARSE_CANDIDATE_ONLY),
        "count": len(DENSE_BASELINE) + len(SPARSE_CANDIDATE_ONLY),
        "baseline_eligible": sorted(DENSE_BASELINE),
        "candidate_only": sorted(SPARSE_CANDIDATE_ONLY),
        "removed": {"yeslab_405": "W1: retention 7.6% given yeslab_381 (watch_pair_review_v2 FAIL)",
                    "yeslab_398": "W1: retention 20.3% given yeslab_381 (watch_pair_review_v2 FAIL)"},
        "fold_structure": FOLD_STRUCTURE,
        "folds": {"F1": {"train": "2017-2020", "val": "2021", "test": "2022"},
                  "F2": {"train": "2018-2021", "val": "2022", "test": "2023"},
                  "F3": {"train": "2019-2022", "val": "2023", "test": "2024"},
                  "F4": {"train": "2020-2023", "val": "2024", "test": "2025"}},
        "special_decisions": {"W1": "approved", "adjudication4": "hybrid_dense_baseline_sparse_candidate"},
        "data_version": fz_dv,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
