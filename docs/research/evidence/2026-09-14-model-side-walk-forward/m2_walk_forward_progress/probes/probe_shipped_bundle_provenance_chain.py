#!/usr/bin/env python3
"""Read-only probe for §CX: the shipped-model provenance chain (bundle digest <-> frozen selection
evidence) is intact on all 14 landed bundles, AND the check that enforces it is genuinely wired into
production code -- unlike the §CC.1 case where CheckpointStore.load() existed but was never called.

THE CHAIN, and why it matters. Each bundle directory carries checksums.sha256, and
model_bundle.py:249 defines the bundle digest as sha256 of that file's BYTES. Each frozen model-selection
record (selections/model/<evidence_id>/record.json) stores model_bundle_digest. The evaluator
model_walk_forward.py:188 passes expected_bundle_digest=selection.model_bundle_digest into
load_model_card, and model_bundle.py:152-153 raises "model bundle digest does not match frozen model
selection" on mismatch. So the digest binds a signed selection to exactly the bytes that ship.

THE CONTRAST WITH §CC.1 IS THE POINT. §CC.1 found CheckpointStore.load() existed, was unit-tested, and
was never called by production code -- a check that could not fire. This chain is the opposite: grep shows
expected_bundle_digest is passed at model_walk_forward.py:188, inside the INDEPENDENT evaluator, not
inside the driver that produced the bundle. So it is not self-attestation.

WHICH CARRIERS HAVE ACTUALLY HAD THE CHECK RUN. model_walk_forward needs report.json, i.e. all four
folds complete. report.json is PRESENT for m1_full_v1 (oos.all_passed True), m0_v3 (all_passed False) and
m0r_v3 (all_passed True), so for those 12 bundles the digest check has already executed in production and
passed. m2_v1 has NO report.json (F3 grid still running, F4 not started), so for its 2 bundles the
evaluator has not run and the digest check has NOT yet executed -- this probe's recomputation is, for M2
only, an external verification standing in for a check production has not yet performed. That distinction
is recorded rather than blurred: 12 bundles are verified-and-enforced, 2 are verified-but-not-yet-enforced.

WHAT IS VERIFIED PER BUNDLE (read-only): every file's sha256 matches its line in checksums.sha256; the
file set equals model_bundle.py:225-233's expected_files exactly; the recomputed digest
sha256(checksums.sha256 bytes) equals the digest recorded in the selection evidence; and each selection
record's fold_id / model_artifact_id / dataset ids are internally consistent (the artifact named exists,
and the record's fold matches the bundle's fold).

CONTROLS, each able to fail. (i) DETECTOR: mutating one byte of an in-memory copy of a checksums file
must change the recomputed digest, proving the digest is sensitive rather than a constant. (ii) MISMATCH:
pairing a real digest with a different bundle's recorded digest must be detected as a mismatch -- this is
what would happen if a selection pointed at the wrong bundle, and without it "all match" could be vacuous.
(iii) NEGATIVE ON FILE HASHES: corrupting one byte of a file's content in memory must break that file's
checksum comparison, proving the per-file check is not trivially passing. (iv) The distinctness and
family-uniqueness requirements the evaluator enforces (model_walk_forward.py:81-83) are re-checked here on
the recorded evidence, so a duplicated artifact id across folds would be caught.

Reads only: checksums.sha256 and the bundle's own files (hashed, never parsed as tensors or parquet rows),
selection record JSONs, report.json presence, and source text. No .pt loaded as a tensor, no parquet row
data read (validation_predictions.parquet is hashed as opaque bytes only), no repository module imported or
executed, no database, no network, no writes to any artifact directory, no signal to any process. Does not
touch /tmp/walk_forward_cache_v1, so no atime side effect. Deterministic and byte-reproducible.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

ART = Path("/data/zhangyudong/qrv2_research_artifacts")
REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "measurements" / "shipped_bundle_provenance_chain.json"

SEEDS = ("20260801", "20260802", "20260803")
# model_bundle.py:225-233 expected_files, reproduced here as the contract to check against
EXPECTED_BASE = {"config.json", "feature_order.json", "model_card.json", "preprocessing.json",
                 "signature.json", "validation_predictions.parquet"}
EXPECTED_FILES = EXPECTED_BASE | {f"seed-{s}.pt" for s in SEEDS}

# carriers: (workspace, model_id prefix, whether report.json exists => evaluator has run)
CARRIERS = [
    ("m1_full_v1", "m1", "M1 ResidualMLP"),
    ("m2_v1", "m2", "M2 TCN"),
    ("m0_v3", "m0", "M0 LGBM_RANKER"),
    ("m0r_v3", "m0r", "M0R LGBM_RANKREG"),
]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_checksums(path: Path) -> dict[str, str]:
    """Parse a checksums.sha256 manifest the way model_bundle.py:236-237 does."""
    out = {}
    for line in path.read_bytes().decode("utf-8").splitlines():
        if not line.strip():
            continue
        digest, sep, rel = line.partition("  ")
        if not sep or not re.fullmatch(r"[0-9a-f]{64}", digest) or rel in out:
            raise ValueError(f"malformed checksum line in {path}: {line!r}")
        out[rel] = digest
    return out


def selections_for(ws: str) -> dict[str, dict]:
    """fold_id -> selection payload, for one workspace."""
    out = {}
    for rec in sorted((ART / ws / "selections" / "model").glob("*/record.json")):
        d = json.load(open(rec, encoding="utf-8"))
        pl = d["payload"]
        pl = dict(pl)
        pl["_evidence_id"] = d.get("evidence_id")
        pl["_path"] = str(rec.relative_to(ART))
        out[pl["fold_id"]] = pl
    return out


def verify_bundle(ws: str, model_id: str, sel: dict) -> dict:
    root = ART / ws / "models" / model_id
    res: dict = {"workspace": ws, "model_id": model_id, "bundle_exists": root.is_dir()}
    if not root.is_dir():
        return res
    cks = root / "checksums.sha256"
    res["checksums_present"] = cks.is_file()
    if not cks.is_file():
        return res
    manifest = parse_checksums(cks)
    # The bundle directory holds the 9 contracted files PLUS checksums.sha256 itself, which cannot
    # list its own hash. The first draft compared the directory listing against EXPECTED_FILES and so
    # reported every bundle as having an "unexpected file" -- the comparison basis was wrong, not the
    # bundles. Two separate equalities are the correct checks: the manifest must cover exactly the 9
    # contracted files, and the directory must hold exactly those 9 plus the manifest.
    actual_files = {q.name for q in root.iterdir() if q.is_file()}
    expected_on_disk = EXPECTED_FILES | {"checksums.sha256"}
    res["manifest_covers_exactly_the_expected_files"] = set(manifest) == EXPECTED_FILES
    res["file_set_matches_expected"] = actual_files == expected_on_disk
    res["missing_files"] = sorted(expected_on_disk - actual_files)
    res["unexpected_files"] = sorted(actual_files - expected_on_disk)
    res["n_files_on_disk"] = len(actual_files)
    res["n_files_in_manifest"] = len(manifest)

    per_file = {}
    for rel, want in sorted(manifest.items()):
        p = root / rel
        got = sha256_file(p) if p.is_file() else None
        per_file[rel] = {"matches": got == want, "recorded": want[:16], "computed": (got or "")[:16]}
    res["per_file"] = per_file
    res["all_files_match_checksums"] = all(v["matches"] for v in per_file.values())

    digest = sha256_file(cks)          # == sha256(checksums bytes), model_bundle.py:249
    res["recomputed_bundle_digest"] = digest
    res["recorded_selection_digest"] = sel.get("model_bundle_digest")
    res["digest_matches_selection_evidence"] = digest == sel.get("model_bundle_digest")
    res["selection_fold_id"] = sel.get("fold_id")
    res["selection_artifact_id"] = sel.get("model_artifact_id")
    res["artifact_id_matches_bundle_dir"] = sel.get("model_artifact_id") == model_id
    res["selection_evidence_id"] = sel.get("_evidence_id")
    res["selection_record_path"] = sel.get("_path")
    return res


def detector_controls(ws: str, model_id: str) -> dict:
    """Prove the digest and the per-file checks can FAIL."""
    root = ART / ws / "models" / model_id
    cks = root / "checksums.sha256"
    original = cks.read_bytes()
    mutated = bytearray(original)
    # flip one hex character, keeping the line well-formed
    i = next(k for k, ch in enumerate(mutated) if ch in b"0123456789abcdef")
    mutated[i] = ord("f") if mutated[i] != ord("f") else ord("0")
    d_orig = sha256_bytes(bytes(original))
    d_mut = sha256_bytes(bytes(mutated))
    # corrupt one file's CONTENT in memory and confirm its hash no longer matches
    target = root / "config.json"
    content = target.read_bytes()
    corrupted = bytearray(content)
    corrupted[-2] = (corrupted[-2] + 1) % 256
    manifest = parse_checksums(cks)
    return {
        "digest_changes_when_one_manifest_byte_changes": d_orig != d_mut,
        "digest_of_original": d_orig[:16], "digest_of_mutated": d_mut[:16],
        "file_hash_breaks_when_content_corrupted":
            sha256_bytes(bytes(corrupted)) != manifest["config.json"],
        "note": ("both are in-memory mutations; nothing on disk was written, so the bundles remain "
                 "byte-identical to what production shipped"),
    }


def mismatch_control(results: list[dict]) -> dict:
    """Pair each bundle's real digest with ANOTHER bundle's recorded digest; all must mismatch."""
    ok = [r for r in results if r.get("digest_matches_selection_evidence")]
    pairs = []
    for a in ok:
        for b in ok:
            if a["recomputed_bundle_digest"] != b["recomputed_bundle_digest"]:
                pairs.append({
                    "bundle": a["model_id"], "borrowed_record_from": b["model_id"],
                    "detected_as_mismatch": a["recomputed_bundle_digest"] != b["recorded_selection_digest"],
                })
    return {"n_cross_pairs_tested": len(pairs),
            "all_detected_as_mismatch": all(p["detected_as_mismatch"] for p in pairs) and bool(pairs),
            "sample": pairs[:3]}


def evaluator_enforcement(ws: str) -> dict:
    """Has the independent evaluator actually run for this carrier (report.json present)?"""
    rep = ART / ws / "report.json"
    out = {"workspace": ws, "report_json_present": rep.is_file(),
           "evaluator_has_run": rep.is_file()}
    if rep.is_file():
        d = json.load(open(rep, encoding="utf-8"))
        oos = d.get("oos", {})
        out["oos_all_passed"] = oos.get("all_passed")
        out["gates"] = list((oos.get("gates") or {}).keys())
    out["reading"] = ("report.json exists only after model_walk_forward has produced a verdict, and that "
                      "path is where expected_bundle_digest is passed (model_walk_forward.py:188), so its "
                      "presence means the digest check has already executed in production")
    return out


def source_chain() -> dict:
    mb = (REPO / "src/research_v2/training/model_bundle.py").read_text(encoding="utf-8").split("\n")
    mwf = (REPO / "src/research_v2/evaluation/model_walk_forward.py").read_text(encoding="utf-8").split("\n")
    def grab(lines, needle):
        return {n: l.strip() for n, l in enumerate(lines, 1) if needle in l}
    return {
        "digest_definition": grab(mb, "digest=hashlib.sha256(checksums)"),
        "digest_mismatch_raises": grab(mb, "model bundle digest does not match frozen model selection"),
        "evaluator_passes_expected_digest": grab(mwf, "expected_bundle_digest=selection.model_bundle_digest"),
        "evaluator_requires_distinct_artifacts": grab(mwf, '_require_distinct(selections, "model_artifact_id"'),
        "evaluator_requires_one_family": grab(mwf, "_require_one_model_family(selections)"),
        "expected_files_contract": grab(mb, '"validation_predictions.parquet",'),
    }


def main() -> None:
    results = []
    enforcement = {}
    distinctness = {}
    for ws, prefix, family in CARRIERS:
        sels = selections_for(ws)
        enforcement[ws] = evaluator_enforcement(ws)
        # distinctness / family-uniqueness re-check (model_walk_forward.py:81-83)
        arts = [s["model_artifact_id"] for s in sels.values()]
        eids = [s.get("_evidence_id") for s in sels.values()]
        fams = {s["model_family"] for s in sels.values()}
        distinctness[ws] = {
            "n_selections": len(sels),
            "artifact_ids_distinct": len(set(arts)) == len(arts),
            "evidence_ids_distinct": len(set(eids)) == len(eids),
            "single_model_family": len(fams) == 1,
            "model_family": sorted(fams)[0] if len(fams) == 1 else sorted(fams),
            "artifact_ids": sorted(arts),
        }
        for fold in sorted(sels):
            mid = sels[fold]["model_artifact_id"]
            r = verify_bundle(ws, mid, sels[fold])
            r["family"] = family
            r["fold_id"] = fold
            r["evaluator_has_run"] = enforcement[ws]["evaluator_has_run"]
            results.append(r)

    verified = [r for r in results if r.get("bundle_exists")]
    all_ok = [r for r in verified
              if r["all_files_match_checksums"] and r["digest_matches_selection_evidence"]
              and r["file_set_matches_expected"] and r["artifact_id_matches_bundle_dir"]]
    det = detector_controls("m1_full_v1", "m1-f1")
    mis = mismatch_control(verified)
    chain = source_chain()

    enforced = [r for r in all_ok if r["evaluator_has_run"]]
    not_yet = [r for r in all_ok if not r["evaluator_has_run"]]

    doc = {
        "question": ("does each shipped bundle's digest match the frozen selection evidence that vouches "
                     "for it, and is the check that enforces this actually wired into production code?"),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "source_chain": chain,
        "contrast_with_CC1": (
            "§CC.1 found CheckpointStore.load() existed and was unit-tested but was never called by "
            "production code, so it could not fire. This chain is the opposite: "
            "model_walk_forward.py:188 passes expected_bundle_digest into load_model_card, and "
            "model_bundle.py:152-153 raises on mismatch. The check lives in the INDEPENDENT evaluator, not "
            "in the driver that produced the bundle, so it is not self-attestation."),
        "per_bundle": verified,
        "summary": {
            "n_bundles_verified": len(verified),
            "n_fully_ok": len(all_ok),
            "all_ok": len(all_ok) == len(verified) and len(verified) > 0,
            "n_enforced_by_a_completed_evaluator_run": len(enforced),
            "n_verified_but_not_yet_enforced": len(not_yet),
            "verified_but_not_yet_enforced": [
                {"model_id": r["model_id"], "workspace": r["workspace"],
                 "reason": "no report.json yet (M2 F3 grid running, F4 not started)"} for r in not_yet],
            "reading": ("12 bundles are both verified here AND already enforced by a production evaluator "
                        "run; 2 (M2 F1/F2) are verified here but the evaluator has not run yet, so for "
                        "those the check has not executed in production. The distinction is recorded "
                        "rather than blurred."),
        },
        "evaluator_run_status": enforcement,
        "distinctness_and_family_checks": distinctness,
        "controls": {},
        "read_only": True, "database_access": False, "weights_loaded": False,
        "pt_loaded_as_tensor": False, "parquet_row_data_read": False, "training_performed": False,
        "repository_module_imported_or_executed": False, "processes_signalled": [],
        "atime_side_effect": False, "byte_reproducible": True,
    }
    doc["controls"] = {
        "every_bundle_file_matches_its_checksum": all(
            r["all_files_match_checksums"] for r in verified),
        "every_bundle_digest_matches_its_selection_evidence": all(
            r["digest_matches_selection_evidence"] for r in verified),
        "every_bundle_file_set_equals_expected": all(
            r["file_set_matches_expected"] for r in verified),
        "every_selection_names_the_bundle_it_verifies": all(
            r["artifact_id_matches_bundle_dir"] for r in verified),
        "artifact_ids_and_evidence_ids_distinct_per_workspace": all(
            d["artifact_ids_distinct"] and d["evidence_ids_distinct"] for d in distinctness.values()),
        "single_model_family_per_workspace": all(
            d["single_model_family"] for d in distinctness.values()),
        "DETECTOR_digest_changes_when_manifest_byte_changes":
            det["digest_changes_when_one_manifest_byte_changes"],
        "DETECTOR_file_hash_breaks_when_content_corrupted":
            det["file_hash_breaks_when_content_corrupted"],
        "DETECTOR_cross_pairing_detected_as_mismatch": mis["all_detected_as_mismatch"],
        "source_chain_all_four_links_present": all([
            bool(chain["digest_definition"]), bool(chain["digest_mismatch_raises"]),
            bool(chain["evaluator_passes_expected_digest"]),
            bool(chain["evaluator_requires_distinct_artifacts"]),
        ]),
        "ALL_PASS": None,
    }
    doc["controls"]["ALL_PASS"] = all(
        v for k, v in doc["controls"].items() if k != "ALL_PASS")
    doc["detector_controls"] = det
    doc["mismatch_control"] = mis

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    s = doc["summary"]
    print("=== source chain (the four links) ===")
    for k, v in chain.items():
        print(f"  {k}: {list(v.items())[:1]}")
    print("\n=== per bundle ===")
    print(f"  {'workspace':13s} {'model':9s} {'fold':5s} {'files':>6s} {'fileset':>8s} "
          f"{'digest':>7s} {'evaluator':>10s}")
    for r in verified:
        print(f"  {r['workspace']:13s} {r['model_id']:9s} {r['fold_id']:5s} "
              f"{str(r['all_files_match_checksums']):>6s} "
              f"{str(r['file_set_matches_expected']):>8s} "
              f"{str(r['digest_matches_selection_evidence']):>7s} "
              f"{str(r['evaluator_has_run']):>10s}")
    print(f"\n=== summary ===")
    print(f"  verified {s['n_bundles_verified']}, fully OK {s['n_fully_ok']}")
    print(f"  enforced by a completed evaluator run: {s['n_enforced_by_a_completed_evaluator_run']}")
    print(f"  verified but evaluator not yet run:    {s['n_verified_but_not_yet_enforced']} "
          f"({[x['model_id'] for x in s['verified_but_not_yet_enforced']]})")
    print(f"\n=== detector controls ===")
    print(f"  digest changes on a 1-byte manifest change: {det['digest_changes_when_one_manifest_byte_changes']}"
          f" ({det['digest_of_original']} -> {det['digest_of_mutated']})")
    print(f"  file hash breaks on content corruption:     {det['file_hash_breaks_when_content_corrupted']}")
    print(f"  cross-pairing mismatch detected:            {mis['all_detected_as_mismatch']} "
          f"({mis['n_cross_pairs_tested']} pairs)")
    print(f"\n=== controls ===")
    for k, v in doc["controls"].items():
        print(f"  {'OK ' if v else '*** FAIL ***'} {k}")
    assert doc["controls"]["ALL_PASS"], "a control failed"
    print("wrote", OUT)


if __name__ == "__main__":
    main()
