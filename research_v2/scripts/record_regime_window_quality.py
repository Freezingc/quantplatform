"""Record regime-window factor quality evidence into the V2 test database.

Loads the on-disk evidence artifacts (pre-training window comparisons for the
23 READY candidates and the formal nine, the 33-series regime redundancy
matrices, and the four-dimension quality grading) and records them in three
append-only tables created by migration 014:

  factor_window_quality   - per factor/window/universe signed IC metrics
  factor_pair_redundancy  - pairwise median abs Spearman per window
  factor_quality_grade    - four-dimension grade summary per factor

Idempotent per (as_of_date, data_version). No existing table is altered; no
locked-test data is read. Applies pending migrations first (014 onward).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from datetime import date
from pathlib import Path

import dolphindb as ddb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DIAG = REPO_ROOT / "research_v2" / "artifacts" / "diagnostics"
TARGET_DATABASE = "dfs://QuantResearchV2Test"
GRADE_RULES_VERSION = 1
WINDOW_YEARS = (2017, 2018, 2019)
VARIANTS = ("full_market", "adv20_liquid", "official_cn_liquid_a")


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def record_hash(record: dict[str, object]) -> str:
    payload = {
        key: value.isoformat() if isinstance(value, (date, pd.Timestamp)) else value
        for key, value in record.items()
        if key != "record_hash"
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True).encode()
    ).hexdigest()


def window_rows(as_of: date, run_id: str, data_version: str, watermark: str):
    rows: list[dict[str, object]] = []

    def add(factor_id, group, window_id, universe_id, days, ic, icir, share):
        rec = {
            "as_of_date": as_of,
            "factor_id": factor_id,
            "factor_version": 1,
            "factor_group": group,
            "window_id": window_id,
            "universe_id": universe_id,
            "valid_day_count": None if days is None else int(days),
            "signed_mean_rank_ic": None if ic is None else float(ic),
            "signed_icir": None if icir is None else float(icir),
            "directional_day_share": None if share is None else float(share),
            "evidence_path": "research_v2/artifacts/diagnostics",
            "run_id": run_id,
            "data_version": data_version,
            "created_at": pd.Timestamp.utcnow().tz_localize(None),
            "source_watermark_id": watermark,
        }
        rec["record_hash"] = record_hash(rec)
        rows.append(rec)

    # READY candidates, pre-training windows (comparison + robustness artifacts)
    for year in WINDOW_YEARS:
        comp = load_json(DIAG / f"historical_window_{year}_robustness_v1" / f"historical_window_{year}_comparison.json")
        rob = load_json(DIAG / f"historical_window_{year}_robustness_v1" / f"historical_window_{year}_robustness.json")
        for r in comp["rows"]:
            fid = r["factor_id"]
            rc = rob["candidates"].get(fid, {})
            for universe in VARIANTS:
                m = rc.get(universe, {})
                add(
                    fid, "READY_CANDIDATE", str(year), universe,
                    m.get("days"), r.get(f"ic_{universe}"), r.get(f"icir_{universe}"),
                    m.get("directional_day_share"),
                )

    # formal nine, pre-training windows
    for year in WINDOW_YEARS:
        rob = load_json(DIAG / "formal9_pretraining_windows_v1" / str(year) / f"historical_window_{year}_robustness.json")
        for fid, c in rob["candidates"].items():
            for universe in VARIANTS:
                m = c.get(universe, {})
                add(
                    fid, "FORMAL", str(year), universe,
                    m.get("days"), m.get("mean_rank_ic"), m.get("icir"),
                    m.get("directional_day_share"),
                )

    # train 2020-2024 + validation 2025: ICIR per factor from the grading
    # artifact (which consolidates the proofread per-candidate evidence)
    grading = load_json(DIAG / "factor_quality_grading_v1" / "factor_quality_grading_v1.json")
    for e in grading["entries"]:
        fid = e["factor_id"]
        group = e["group"]
        pt = e.get("pretrain_official_icir", {})
        add(fid, group, "TRAIN_2020_2024", "official_cn_liquid_a", None, None, e.get("train_icir"), None)
        add(fid, group, "VALIDATION_2025", "official_cn_liquid_a", None, None, e.get("validation_icir"), None)
    return rows


def redundancy_rows(as_of: date, run_id: str, data_version: str, watermark: str):
    rows: list[dict[str, object]] = []

    def pair_class(v):
        return "REDUNDANT" if v >= 0.90 else "ELEVATED" if v >= 0.80 else "OK"

    def add(left, right, window_id, v, days):
        if v is None:
            return
        rec = {
            "as_of_date": as_of,
            "factor_id_left": left,
            "factor_id_right": right,
            "window_id": window_id,
            "median_abs_spearman": float(v),
            "valid_day_count": None if days is None else int(days),
            "pair_class": pair_class(float(v)),
            "run_id": run_id,
            "data_version": data_version,
            "created_at": pd.Timestamp.utcnow().tz_localize(None),
            "source_watermark_id": watermark,
        }
        rec["record_hash"] = record_hash(rec)
        rows.append(rec)

    # pre-training windows: full 33-series pool (528 pairs each)
    for year in WINDOW_YEARS:
        d = load_json(DIAG / "pretraining_regime_redundancy_v2" / f"pretraining_redundancy_{year}.json")
        for p in d["all_pairs"]:
            add(p["factor_id_left"], p["factor_id_right"], str(year), p.get("median_abs_spearman"), p.get("valid_day_count"))

    # train/validation: formal nine pairwise matrices
    for window_id, fname in (("TRAIN_2020_2024", "pretraining_redundancy_2020.json"), ("VALIDATION_2025", "pretraining_redundancy_2025.json")):
        d = load_json(DIAG / "formal9_regime_redundancy_train_val_v1" / fname)
        for p in d["all_pairs"]:
            add(p["factor_id_left"], p["factor_id_right"], window_id, p.get("median_abs_spearman"), p.get("valid_day_count"))
    return rows


def grade_rows(as_of: date, run_id: str, data_version: str, watermark: str):
    grading = load_json(DIAG / "factor_quality_grading_v1" / "factor_quality_grading_v1.json")
    rows: list[dict[str, object]] = []
    for e in grading["entries"]:
        rec = {
            "as_of_date": as_of,
            "factor_id": e["factor_id"],
            "factor_group": e["group"],
            "grade": e["grade"],
            "total_score": None if e.get("total") is None else int(e["total"]),
            "score_train_predictive": e["scores"]["train_predictive"],
            "score_oos_validation": e["scores"]["oos_validation"],
            "score_regime_robustness": e["scores"]["regime_robustness"],
            "score_uniqueness": e["scores"]["uniqueness"],
            "train_icir": None if e.get("train_icir") is None else float(e["train_icir"]),
            "validation_icir": None if e.get("validation_icir") is None else float(e["validation_icir"]),
            "regime_min_official_icir": None if e.get("regime_min_official_icir") is None else float(e["regime_min_official_icir"]),
            "max_similarity_edge": None if (e.get("max_nine_edge") or e.get("max_intra_formal_edge_train")) is None else float(e.get("max_nine_edge") or e.get("max_intra_formal_edge_train")),
            "grade_rules_version": GRADE_RULES_VERSION,
            "evidence_path": "research_v2/artifacts/diagnostics/factor_quality_grading_v1",
            "run_id": run_id,
            "data_version": data_version,
            "created_at": pd.Timestamp.utcnow().tz_localize(None),
            "source_watermark_id": watermark,
        }
        rec["record_hash"] = record_hash(rec)
        rows.append(rec)
    return rows


def ensure_schema(client):
    import sys
    sys.path.insert(0, str(REPO_ROOT / "research_v2" / "src"))
    from research_v2.db.migrations import MigrationRunner
    from research_v2.db.session import V2WriteSession

    session = V2WriteSession(client, target_database=TARGET_DATABASE)
    runner = MigrationRunner(session, migration_directory=REPO_ROOT / "research_v2" / "migrations")
    result = runner.apply()
    executed = [m.name for m in result.executed]
    skipped = [m.name for m in result.skipped]
    return session, executed, skipped


TABLE_CASTS = {
    "factor_window_quality": {
        "as_of_date": "date", "factor_id": "symbol", "factor_version": "int",
        "factor_group": "symbol", "window_id": "symbol", "universe_id": "symbol",
        "valid_day_count": "int", "signed_mean_rank_ic": "double", "signed_icir": "double",
        "directional_day_share": "double", "evidence_path": "string", "run_id": "uuid",
        "data_version": "string", "created_at": "timestamp", "source_watermark_id": "string",
        "record_hash": "string",
    },
    "factor_pair_redundancy": {
        "as_of_date": "date", "factor_id_left": "symbol", "factor_id_right": "symbol",
        "window_id": "symbol", "median_abs_spearman": "double", "valid_day_count": "int",
        "pair_class": "symbol", "run_id": "uuid", "data_version": "string",
        "created_at": "timestamp", "source_watermark_id": "string", "record_hash": "string",
    },
    "factor_quality_grade": {
        "as_of_date": "date", "factor_id": "symbol", "factor_group": "symbol", "grade": "symbol",
        "total_score": "int", "score_train_predictive": "int", "score_oos_validation": "int",
        "score_regime_robustness": "int", "score_uniqueness": "int", "train_icir": "double",
        "validation_icir": "double", "regime_min_official_icir": "double",
        "max_similarity_edge": "double", "grade_rules_version": "int", "evidence_path": "string",
        "run_id": "uuid", "data_version": "string", "created_at": "timestamp",
        "source_watermark_id": "string", "record_hash": "string",
    },
}


def insert_table(session, table: str, frame: pd.DataFrame) -> str:
    stage = f"regime_quality_{uuid.uuid4().hex[:16]}"
    session.upload_frame(stage, frame)
    casts = TABLE_CASTS[table]
    typed = ", ".join(f"{casts[c]}({c}) as {c}" for c in frame.columns)
    script = (
        f"typed = select {typed} from {stage}\n"
        f"tableInsert{{loadTable('{TARGET_DATABASE}','{table}')}}(typed)\n"
        f"undef(`{stage}, VAR)\nundef(`typed, VAR)"
    )
    session.execute(script)
    return "published"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    parser.add_argument("--publish", action="store_true", help="actually write to the V2 test database")
    args = parser.parse_args(argv)

    descriptor = json.dumps(
        {"scope": "REGIME_WINDOW_QUALITY_RECORD_V1", "windows": list(WINDOW_YEARS) + ["TRAIN_2020_2024", "VALIDATION_2025"], "as_of": args.as_of.isoformat()},
        sort_keys=True,
    )
    watermark = hashlib.sha256(descriptor.encode()).hexdigest()
    data_version = hashlib.sha256((watermark + str(args.as_of)).encode()).hexdigest()
    run_id = str(uuid.uuid4())

    wf = pd.DataFrame(window_rows(args.as_of, run_id, data_version, watermark))
    rf = pd.DataFrame(redundancy_rows(args.as_of, run_id, data_version, watermark))
    gf = pd.DataFrame(grade_rows(args.as_of, run_id, data_version, watermark))
    print(f"built frames: factor_window_quality={len(wf)}, factor_pair_redundancy={len(rf)}, factor_quality_grade={len(gf)}")

    if not args.publish:
        print("dry-run: no database write")
        return 0

    client = ddb.session()
    client.connect("10.176.55.211", 8848, "xiahongjie", "123456")
    session, executed, skipped = ensure_schema(client)
    print(f"migrations executed: {executed}; skipped(applied): {len(skipped)}")

    for table, frame in (("factor_window_quality", wf), ("factor_pair_redundancy", rf), ("factor_quality_grade", gf)):
        existing = session.query(
            f"select count(*) as n from loadTable('{TARGET_DATABASE}','{table}') "
            f"where as_of_date=date('{args.as_of:%Y.%m.%d}') and data_version='{data_version}'"
        )
        n_existing = int(existing["n"][0]) if isinstance(existing, pd.DataFrame) and len(existing) else 0
        if n_existing:
            print(f"{table}: skipped ({n_existing} rows already present for this as_of/data_version)")
            continue
        insert_table(session, table, frame)
        back = session.query(f"select count(*) as n from loadTable('{TARGET_DATABASE}','{table}') where as_of_date=date('{args.as_of:%Y.%m.%d}')")
        print(f"{table}: inserted; rows for as_of now = {int(back['n'][0])}")
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
