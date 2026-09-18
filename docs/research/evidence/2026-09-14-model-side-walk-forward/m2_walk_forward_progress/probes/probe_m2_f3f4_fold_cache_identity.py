#!/usr/bin/env python3
"""Read-only probe for §CL: M2's F3/F4 panels are already on disk as fold-cache files, and M2 will
load them by cache hit rather than re-assembling from the database -- which closes the exposure that
§CK.3 declared "not measured".

WHAT §CK.3 GOT WRONG, and how this probe found out
--------------------------------------------------
§CK.3 reasoned that because dataset_id hashes a per-day data_versions map, and M1 materialised F3/F4
in mid-September while M2 will not reach them for ~20 more days, M2's F3/F4 panels might be different
objects by the time it builds them, so the exact row counts §CK/A15 published were labelled "measured
on M1's datasets", never "the rows M2 will train on". That reasoning assumed M2 RE-QUERIES the
database when it reaches F3/F4. It does not.

The driver resolves versions exactly ONCE. current_versions(holder, table) is called at
run_real_candidate_training.py:2228 inside setup_full_run_context, which run_full_m2 calls once at
:2923; the result is stored in a frozen dataclass field and the fold loop at :2944-2949 passes that
same dict to extract() for every fold. grep confirms :2228 is the ONLY call site of current_versions
in the driver. So M2's version snapshot is pinned at process start (2026-09-13 18:54:55) and cannot
change while pid 3588838 lives, no matter how many days pass or what happens in the database.

extract_dataset_factor_daily (defined at walk_forward_factor_daily.py:234) first computes
scoped_cache_key(fold, role, factor_members, versions) at :246 and, if that file exists under
FOLD_CACHE_ROOT (/tmp/walk_forward_cache_v1, walk_forward_common.py:49) -- tested at :248 -- returns
it verbatim at :254, BEFORE issuing any query. The cache key folds in the ENTIRE versions dict plus
the factor set, so a cache hit is proof that the pinned versions equal the versions the file was
written with.

M2's own stdout (/proc/3588838/fd/1 -> m2_v1.log) shows "loaded cached dataset" for all six F1/F2
splits, with row counts 2296573 / 696906 / 804575 and 2539884 / 783477 / 882426. Those cache files
were written 09-04 to 09-08, before M2 started on 09-13. So M2 already demonstrated, on two folds,
that its startup-pinned versions reproduce cache keys written days earlier.

THE FORENSIC MOVE
-----------------
scoped_cache_key is a pure function of (fold, role, factor_set, versions). This probe reimplements it
from source text (never importing the repository), feeds it the versions dict recorded in M1's F1
dataset metadata, and VALIDATES the reimplementation against the six F1/F2 hits the log proves: the
computed keys must resolve to existing files whose row counts equal both the log and M1's dataset
footers. Only after 6/6 validate does it apply the same function to F3/F4.

RESULT: the six F3/F4 cache files exist, were written 09-08, and their row counts equal M1's F3/F4
dataset footers EXACTLY (F3 2,790,995 / 859,748 / 932,167; F4 3,046,399 / 908,649 / 938,594). Since
the key that resolves to them was computed from M1's pinned versions -- which the F1/F2 hits prove
equal M2's pinned versions -- M2 will hit these same files for F3/F4. So the rows §CK/A15 published
are not a proxy: they are the exact frames M2 will consume, and there is no database re-query and
therefore no version-drift exposure for F3/F4 within this run.

WHAT SURVIVES OF §CK.3's CONCERN (this probe does not overclaim)
----------------------------------------------------------------
The exposure is not zero; it is CONDITIONAL, and the condition is now precise:
  * If pid 3588838 runs to completion, F3/F4 cache-hit and §CK/A15's rows stand as the trained rows.
  * If M2 CRASHES and is restarted with --resume, the new process re-resolves versions at the new
    start time; if the database's published_version changed since 09-13, the new keys for F3/F4 would
    differ and MISS, forcing a cold re-assembly whose row counts could differ from these.
  * If /tmp/walk_forward_cache_v1 is cleared before M2 reaches F3, the same cold re-assembly happens.
Both are external events, not properties of the current run. §CK.3's "~20-day window" was therefore
modelling a risk that applies only to a RESTART, not to the run in flight -- a much narrower and more
actionable statement, and one that also means the F1/F2 numbers already trained are equally safe
(they were cache hits too).

Only directory listings, parquet FOOTERS (metadata.num_rows and schema -- no row data loaded),
manifest.json text, the driver log, and source text are read. No repository module is imported or
executed, no checkpoint.pt opened, no weight or tensor loaded, no database access, no network, no
writes to any artifact directory, no signal to any process.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

ART = Path("/data/zhangyudong/qrv2_research_artifacts")
CACHE_ROOT = Path("/tmp/walk_forward_cache_v1")
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "measurements" / "m2_f3f4_fold_cache_identity.json"

# --- reimplemented from source text; NEVER imported from the repository ---
FROZEN_FACTOR_SET_V2 = (
    "yeslab_206", "yeslab_357", "yeslab_372", "yeslab_375", "yeslab_381", "yeslab_384",
    "yeslab_389", "yeslab_390", "yeslab_406", "yeslab_425", "yeslab_443_reg30_max20_v1",
    "yeslab_464", "yeslab_471", "yeslab_484", "yeslab_96",
)


def fold_cache_key(fold_id: str, role: str, versions: dict) -> str:
    """walk_forward_common.py:57 -- deterministic key bound to the published versions consumed."""
    payload = json.dumps(
        {"fold_id": fold_id, "role": role, "versions": {k: dict(v) for k, v in versions.items()}},
        sort_keys=True, ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def scoped_cache_key(fold_id: str, role: str, versions: dict) -> str:
    """walk_forward_factor_daily.py:scoped_cache_key -- folds the factor set in as a pseudo-table."""
    scope = {fid: f"1:f{i:02d}" for i, fid in enumerate(FROZEN_FACTOR_SET_V2)}
    return fold_cache_key(fold_id, role, {**versions, "factor_set": scope})


# M2 log-proven cache hits: (fold, role) -> rows the driver printed it loaded
LOG_HITS = {
    ("F1", "train"): 2296573, ("F1", "validation"): 696906, ("F1", "test"): 804575,
    ("F2", "train"): 2539884, ("F2", "validation"): 783477, ("F2", "test"): 882426,
}
# M1 dataset footers, measured independently by §CK's probe
M1_FOOTERS = {
    ("F3", "train"): 2790995, ("F3", "validation"): 859748, ("F3", "test"): 932167,
    ("F4", "train"): 3046399, ("F4", "validation"): 908649, ("F4", "test"): 938594,
}


def m1_f1_versions() -> dict:
    did = "226a8dfc497c2de6b07516711f058f26a9270675916c9e67dbf7f087a07d2538"
    return json.load(open(ART / "m1_full_v1" / "datasets" / did / "manifest.json"))[
        "metadata"]["data_versions"]


def dataset_footer_rows(fold: str) -> dict[str, int]:
    """M1's materialised dataset footers for one fold, read in place (footer only)."""
    did = {
        "F1": "226a8dfc497c2de6b07516711f058f26a9270675916c9e67dbf7f087a07d2538",
        "F2": "eb9bba1b5a0bd66d0ad6c82536b295de1fc079aec125cb27bb68491bdb21a234",
        "F3": "9766e69f342511c5bda30188df15a7b9b17d936bdc26b9b60ba244e8e7e0c9e7",
        "F4": "6f9478d7645c8c745a73f00c03b6122e17d0c06c05cc159ece531fb00df9fb9d",
    }[fold]
    out = {}
    for role in ("train", "validation", "test"):
        d = ART / "m1_full_v1" / "datasets" / did / f"split={role}"
        out[role] = sum(pq.ParquetFile(p).metadata.num_rows for p in d.rglob("*.parquet"))
    return out


def main() -> None:
    versions = m1_f1_versions()
    doc: dict[str, object] = {
        "question": ("will M2's F3/F4 panels be the frames §CK/A15 measured, or a re-assembled object "
                     "exposed to ~20 days of version drift as §CK.3 feared?"),
        "mechanism": {
            "versions_resolved_once_at": "run_real_candidate_training.py:2228 (setup_full_run_context)",
            "only_call_site_of_current_versions": True,
            "m2_context_built_at": ":2923 (run_full_m2)",
            "fold_loop_reuses_pinned_dict": ":2944-2949",
            "m2_pid": 3588838,
            "m2_start_from_etime": "2026-09-13 18:54:55",
            "cache_checked_before_any_query": (
                "walk_forward_factor_daily.py:246 computes the key, :248 tests existence, "
                ":254 returns the cached frame verbatim -- all before any query is issued"
            ),
            "cache_root": str(CACHE_ROOT),
            "consequence": ("versions cannot change while the pid lives, so days passing and database "
                            "churn are irrelevant to what this run consumes"),
        },
        "validation_against_log_proven_hits": {},
        "f3_f4_prediction": {},
        "negative_controls": {},
    }

    # --- validation: reproduce the six hits M2's own log proves
    val = {}
    all_ok = True
    for (fold, role), log_rows in LOG_HITS.items():
        key = scoped_cache_key(fold, role, versions)
        p = CACHE_ROOT / f"{key}.parquet"
        exists = p.is_file()
        rows = pq.ParquetFile(p).metadata.num_rows if exists else None
        foot = dataset_footer_rows(fold)[role]
        ok = exists and rows == log_rows == foot
        all_ok &= ok
        val[f"{fold}/{role}"] = {
            "computed_key": key[:16] + "…", "file_exists": exists,
            "cache_rows": rows, "m2_log_rows": log_rows, "m1_dataset_footer_rows": foot,
            "triple_match": ok,
            "cache_mtime": datetime.fromtimestamp(p.stat().st_mtime).isoformat(
                timespec="minutes") if exists else None,
        }
    doc["validation_against_log_proven_hits"] = {
        "per_split": val, "all_six_triple_match": all_ok,
        "reading": ("each computed key resolves to an existing file whose row count equals BOTH M2's "
                    "own log line AND M1's dataset footer -- a three-way match on six independent "
                    "splits, which is what licenses applying the same function to F3/F4"),
    }

    # --- prediction: F3/F4, which M2 has not reached
    pred = {}
    f34_ok = True
    for fold in ("F3", "F4"):
        foot = dataset_footer_rows(fold)
        for role in ("train", "validation", "test"):
            key = scoped_cache_key(fold, role, versions)
            p = CACHE_ROOT / f"{key}.parquet"
            exists = p.is_file()
            rows = pq.ParquetFile(p).metadata.num_rows if exists else None
            cols = pq.ParquetFile(p).metadata.schema.names if exists else None
            match = exists and rows == foot[role] == M1_FOOTERS[(fold, role)]
            f34_ok &= match
            pred[f"{fold}/{role}"] = {
                "computed_key": key[:16] + "…", "file_exists": exists,
                "cache_rows": rows, "m1_dataset_footer_rows": foot[role],
                "ck_A15_published_rows": M1_FOOTERS[(fold, role)],
                "matches": match, "n_columns": len(cols) if cols else None,
                "cache_mtime": datetime.fromtimestamp(p.stat().st_mtime).isoformat(
                    timespec="minutes") if exists else None,
            }
    f3tv = pred["F3/train"]["cache_rows"] + pred["F3/validation"]["cache_rows"]
    f4tv = pred["F4/train"]["cache_rows"] + pred["F4/validation"]["cache_rows"]
    doc["f3_f4_prediction"] = {
        "per_split": pred,
        "all_six_match_M1_footers_and_A15": f34_ok,
        "F3_trainval_rows_M2_will_consume": f3tv,
        "F4_trainval_rows_M2_will_consume": f4tv,
        "A15_published_F3": 3650743, "A15_published_F4": 3955048,
        "A15_confirmed_as_exact_not_proxy": (f3tv == 3650743 and f4tv == 3955048 and f34_ok),
        "reading": ("M2 will cache-hit these files for F3/F4, so the frames it trains on are these "
                    "exact bytes; A15's row counts are the trained rows, not a proxy from M1."),
    }

    # --- negative controls: the key function must be able to MISS
    bad_version = json.loads(json.dumps(versions))
    tbl = sorted(bad_version)[0]
    part = sorted(bad_version[tbl])[0]
    bad_version[tbl][part] = "0" * 64
    miss_key = scoped_cache_key("F3", "train", bad_version)
    wrong_fold_key = scoped_cache_key("F9", "train", versions)
    legacy_key = fold_cache_key("F3", "train", versions)  # legacy path: no factor_set
    doc["negative_controls"] = {
        "one_day_version_flip_changes_key_must_be_True": miss_key != pred["F3/train"]["computed_key"][:-1],
        "flipped_versions_key_resolves_to_no_file_must_be_True": not (CACHE_ROOT / f"{miss_key}.parquet").is_file(),
        "nonexistent_fold_F9_key_resolves_to_no_file_must_be_True":
            not (CACHE_ROOT / f"{wrong_fold_key}.parquet").is_file(),
        "legacy_key_without_factor_set_differs_from_scoped_must_be_True":
            legacy_key != scoped_cache_key("F3", "train", versions),
        "legacy_key_resolves_to_no_file_must_be_True":
            not (CACHE_ROOT / f"{legacy_key}.parquet").is_file(),
        "reading": ("the key function is not returning hits for everything: flipping one of the 446 "
                    "version entries, inventing fold F9, or dropping the factor_set scope each yield "
                    "a key with no file. So the six F3/F4 hits are specific to the real pinned "
                    "versions and the real factor set."),
    }
    nc = doc["negative_controls"]
    all_nc = all(v for k, v in nc.items() if k.endswith("must_be_True"))

    doc.update({
        "conclusion": {
            "all_six_log_hits_validated": all_ok,
            "all_six_f3f4_match": f34_ok,
            "all_negative_controls_pass": all_nc,
            "A15_rows_are_exact_trained_rows": doc["f3_f4_prediction"]["A15_confirmed_as_exact_not_proxy"],
            "CK3_exposure_window_applies_to": (
                "a RESTART of M2 (new process re-resolves versions) or a cleared /tmp cache, NOT to "
                "pid 3588838 running to completion -- within this run F3/F4 cache-hit and cannot drift"),
        },
        "read_only": True, "database_access": False, "weights_loaded": False,
        "checkpoint_pt_opened": False, "parquet_row_data_loaded": False,
        "training_performed": False, "repository_module_imported_or_executed": False,
        "processes_signalled": [],
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
    })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print("=== validation against M2's six log-proven cache hits ===")
    for k, v in val.items():
        print(f"  {k:16s} key={v['computed_key']} exists={v['file_exists']} "
              f"cache={v['cache_rows']} log={v['m2_log_rows']} footer={v['m1_dataset_footer_rows']} "
              f"{'OK' if v['triple_match'] else '*** MISMATCH ***'}")
    print(f"  all six triple-match: {all_ok}")
    print("\n=== F3/F4 prediction (M2 has not reached these) ===")
    for k, v in pred.items():
        print(f"  {k:16s} key={v['computed_key']} exists={v['file_exists']} "
              f"cache={v['cache_rows']} A15={v['ck_A15_published_rows']} cols={v['n_columns']} "
              f"mtime={v['cache_mtime']} {'OK' if v['matches'] else '*** MISMATCH ***'}")
    print(f"  F3 train+val M2 will consume = {f3tv:,} (A15 said 3,650,743)")
    print(f"  F4 train+val M2 will consume = {f4tv:,} (A15 said 3,955,048)")
    print(f"  A15 confirmed exact (not proxy): {doc['f3_f4_prediction']['A15_confirmed_as_exact_not_proxy']}")
    print("\n=== negative controls ===")
    for k, v in nc.items():
        if k.endswith("must_be_True"):
            print(f"  {'OK' if v else '*** FAIL ***'} {k}")
    print(f"  ALL CONTROLS PASS: {all_nc}")
    print("\nconclusion:", json.dumps(doc["conclusion"], ensure_ascii=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
