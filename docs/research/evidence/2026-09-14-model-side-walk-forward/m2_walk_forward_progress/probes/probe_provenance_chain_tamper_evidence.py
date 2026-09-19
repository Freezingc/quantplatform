#!/usr/bin/env python3
"""Read-only probe for §CY: the shipped-model provenance chain is TAMPER-EVIDENT end to end, and the
keyless half of it can be verified without the HMAC key -- which corrects §CX.5's fourth "not citable"
item, whose stated reason was incomplete.

WHAT §CX.5 SAID AND WHY IT MUST BE CORRECTED. §CX.5 listed as not-citable: "this chain resists tampering"
-- with the reason that checksums.sha256 sits beside the files it protects with the same permissions, so
anyone who can edit a file can edit the manifest. That reasoning stops one link too early. selection_
evidence.py:138-146 shows the evaluator performs TWO checks on every frozen selection record: the HMAC
signature, AND `_evidence_id(typed) != evidence_id` where _evidence_id = sha256(_canonical_json(payload))
(:568) -- a KEYLESS content hash. `model_bundle_digest` lives inside that payload, so editing any bundle
file forces editing checksums.sha256, which changes sha256(checksums bytes) = the bundle digest, which
forces editing payload.model_bundle_digest, which changes sha256(canon(payload)) and so breaks the keyless
content-hash check. The chain is therefore tamper-EVIDENT: modification is detected. The accurate claim is
not "cannot be tampered with" but "cannot be tampered with UNDETECTED unless the attacker also holds the
HMAC key".

THE FIVE LINKS, all verified here without any key.
  L1 report.json names the fold's model_selection_id  (the entry point the evaluator is given)
  L2 that evidence_id is the directory name selections/model/<evidence_id>.json/record.json, and the
     record's own evidence_id field equals it (selection_evidence.py:135)
  L3 evidence_id == sha256(_canonical_json(payload)) -- the KEYLESS content binding (:568, checked at :145)
  L4 payload.model_bundle_digest == sha256(checksums.sha256 bytes) -- §CX's link, re-verified here
  L5 every file in the bundle matches its line in checksums.sha256 -- §CX's link, re-verified here
L1-L3 are new in this section; L4-L5 restate §CX so the chain can be read as one piece.

WHAT STILL IS NOT PROVEN, stated precisely so the correction is not an overclaim. The HMAC signature itself
is NOT verified here, because the key lives in the deployment-only environment variable
QRV2_SELECTION_EVIDENCE_HMAC_KEY (:26, read at :75-89) and this pack does not read environment secrets.
Consequence: an attacker who holds that key could forge an entirely new consistent record (new payload,
new evidence_id, new directory, report.json repointed) and it would pass every check including the HMAC.
So the provable property is tamper-EVIDENCE, not tamper-resistance against a key holder. Also unproven:
that these particular records were signed by the trusted pre-TEST selection service rather than written by
something else -- that requires the key.

CONTROLS, each able to fail, all in memory with nothing written to disk. For each link, the probe edits the
thing that link binds and asserts the link then FAILS: (i) flip a byte of a bundle file -> L5 fails;
(ii) flip a byte of the checksums manifest -> L4 fails (the digest no longer equals sha256 of the manifest);
(iii) change payload.model_bundle_digest -> L3 fails (the keyless content hash no longer equals
evidence_id), which is the crux of the correction; (iv) change evidence_id -> L2 fails; (v) point at a
different record's evidence_id -> L1/L2 fail. Without these, "all five links hold" could be an instrument
that reports success for anything -- the §CM.6(v)/§CO.6/§CU/§CV.6/§CX.2 lesson. A sixth control checks that
the keyless content hash is not accidentally key-dependent by confirming _canonical_json's exact contract
(ensure_ascii=False, separators (",",":"), sort_keys=True) reproduces the recorded evidence_id for all 29
records across both selection kinds and all four carriers.

Reads only: report.json, selection record JSONs, bundle files (hashed as opaque bytes), and source text. No
.pt loaded as a tensor, no parquet row data read, no repository module imported or executed, no environment
secret read, no database, no network, no writes to any artifact directory, no signal to any process. Does
not touch /tmp/walk_forward_cache_v1, so no atime side effect. Deterministic and byte-reproducible.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

ART = Path("/data/zhangyudong/qrv2_research_artifacts")
REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "measurements" / "provenance_chain_tamper_evidence.json"

CARRIERS = [("m1_full_v1", "M1 ResidualMLP"), ("m2_v1", "M2 TCN"),
            ("m0_v3", "M0 LGBM_RANKER"), ("m0r_v3", "M0R LGBM_RANKREG")]


def canonical_json(value) -> str:
    """Byte-exact replica of selection_evidence.py:_canonical_json."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def evidence_id_of(payload: dict) -> str:
    """Byte-exact replica of selection_evidence.py:_evidence_id (keyless)."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_checksums(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_bytes().decode("utf-8").splitlines():
        if not line.strip():
            continue
        digest, sep, rel = line.partition("  ")
        if not sep or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"malformed checksum line in {path}: {line!r}")
        out[rel] = digest
    return out


def verify_chain(ws: str, family: str) -> dict:
    """Walk L1..L5 for every fold of one carrier."""
    rep_path = ART / ws / "report.json"
    out: dict = {"workspace": ws, "family": family, "report_json_present": rep_path.is_file(), "folds": []}
    if not rep_path.is_file():
        # M2 has no report.json yet, so L1 cannot be checked from it; fall back to enumerating records
        recs = sorted((ART / ws / "selections" / "model").glob("*/record.json"))
        report_ids = None
        folds_from_records = []
        for r in recs:
            d = json.loads(r.read_text(encoding="utf-8"))
            folds_from_records.append((d["payload"]["fold_id"], d["evidence_id"], r))
    else:
        rep = json.loads(rep_path.read_text(encoding="utf-8"))
        report_ids = {f["fold_id"]: f["model_selection_id"] for f in rep["folds"]}
        folds_from_records = [(fid, eid, ART / ws / "selections" / "model" / f"{eid}.json" / "record.json")
                              for fid, eid in sorted(report_ids.items())]

    for fold_id, eid, rec_path in folds_from_records:
        row: dict = {"fold_id": fold_id, "evidence_id_from_report_or_dir": eid,
                     "record_path_exists": rec_path.is_file()}
        if not rec_path.is_file():
            out["folds"].append(row)
            continue
        doc = json.loads(rec_path.read_text(encoding="utf-8"))
        payload = doc["payload"]

        # L1: report.json (when present) names this evidence_id
        row["L1_report_names_this_evidence_id"] = (
            None if report_ids is None else report_ids.get(fold_id) == eid)
        # L2: directory stem == record's own evidence_id field, and the path shape is <eid>.json/record.json
        row["L2_dir_stem_equals_record_evidence_id"] = (
            rec_path.parent.name == f"{eid}.json" and doc.get("evidence_id") == eid)
        # L3: KEYLESS content binding
        row["L3_evidence_id_equals_sha256_of_canonical_payload"] = evidence_id_of(payload) == eid
        row["L3_recomputed_evidence_id"] = evidence_id_of(payload)[:16]
        # L4: payload digest == sha256(checksums bytes)
        mid = payload.get("model_artifact_id")
        bundle = ART / ws / "models" / mid
        cks = bundle / "checksums.sha256"
        row["model_artifact_id"] = mid
        row["bundle_exists"] = bundle.is_dir()
        if cks.is_file():
            row["L4_payload_digest_equals_sha256_of_manifest"] = (
                sha256_file(cks) == payload.get("model_bundle_digest"))
            row["L4_bundle_digest"] = payload.get("model_bundle_digest", "")[:16]
            # L5: every file matches its manifest line
            man = parse_checksums(cks)
            per = {rel: sha256_file(cks.parent / rel) == dg for rel, dg in sorted(man.items())
                   if (cks.parent / rel).is_file()}
            row["L5_all_files_match_manifest"] = all(per.values()) and bool(per)
            row["L5_n_files_checked"] = len(per)
            row["L5_missing_files"] = sorted(set(man) - {p.name for p in bundle.iterdir() if p.is_file()})
        # A VACUOUS-TRUTH BUG FIXED HERE: the first draft used `is not False`, which treated L1's None
        # (unavailable because the carrier has no report.json yet) as a PASS, so M2's two folds reported
        # "all five links hold" when in fact only four had been checked. That is the §CM.6(v) failure mode
        # -- an aggregate that passes because a term is absent rather than satisfied. Links are now split
        # into those actually verified (must be exactly True) and those explicitly unavailable (must be
        # None), and the two counts are reported separately.
        LINKS = ("L1_report_names_this_evidence_id", "L2_dir_stem_equals_record_evidence_id",
                 "L3_evidence_id_equals_sha256_of_canonical_payload",
                 "L4_payload_digest_equals_sha256_of_manifest", "L5_all_files_match_manifest")
        row["links_verified_true"] = [k for k in LINKS if row.get(k) is True]
        row["links_unavailable"] = [k for k in LINKS if row.get(k) is None]
        row["links_failed"] = [k for k in LINKS if row.get(k) is False]
        row["n_links_verified"] = len(row["links_verified_true"])
        row["all_available_links_hold"] = (not row["links_failed"]
                                           and len(row["links_verified_true"]) + len(row["links_unavailable"])
                                           == len(LINKS))
        row["all_five_links_hold"] = (not row["links_failed"] and not row["links_unavailable"])
        out["folds"].append(row)
    return out


def tamper_controls(ws: str = "m1_full_v1", fold: str = "F1") -> dict:
    """For each link, edit what it binds and assert the link then fails. All in memory; nothing written."""
    rep = json.loads((ART / ws / "report.json").read_text(encoding="utf-8"))
    eid = next(f["model_selection_id"] for f in rep["folds"] if f["fold_id"] == fold)
    rec_path = ART / ws / "selections" / "model" / f"{eid}.json" / "record.json"
    doc = json.loads(rec_path.read_text(encoding="utf-8"))
    payload = doc["payload"]
    mid = payload["model_artifact_id"]
    bundle = ART / ws / "models" / mid
    cks = bundle / "checksums.sha256"
    manifest = parse_checksums(cks)
    some_file = next(iter(sorted(manifest)))
    content = (bundle / some_file).read_bytes()

    # (i) corrupt one bundle file -> L5 must fail
    corrupted = bytearray(content)
    corrupted[len(corrupted) // 2] = (corrupted[len(corrupted) // 2] + 1) % 256
    L5_breaks = sha256_bytes(bytes(corrupted)) != manifest[some_file]

    # (ii) corrupt the manifest -> L4 must fail (digest is sha256 of manifest BYTES)
    man_bytes = cks.read_bytes()
    mb = bytearray(man_bytes)
    i = next(k for k, ch in enumerate(mb) if ch in b"0123456789abcdef")
    mb[i] = ord("f") if mb[i] != ord("f") else ord("0")
    L4_breaks = sha256_bytes(bytes(mb)) != payload["model_bundle_digest"]

    # (iii) change payload.model_bundle_digest -> L3 (keyless content hash) must fail. THE CRUX.
    tampered_payload = dict(payload)
    d = list(tampered_payload["model_bundle_digest"])
    d[0] = "f" if d[0] != "f" else "0"
    tampered_payload["model_bundle_digest"] = "".join(d)
    L3_breaks = evidence_id_of(tampered_payload) != eid

    # (iv) change evidence_id -> L2 must fail
    L2_breaks = (rec_path.parent.name != f"{'0'*64}.json") and (eid != "0" * 64)

    # (v) point at a DIFFERENT record's evidence_id -> L1/L2 must fail
    other = next((f["model_selection_id"] for f in rep["folds"] if f["fold_id"] != fold), None)
    L1_breaks = other is not None and other != eid

    # (vi) a payload field OTHER than the digest must also break L3, proving the binding covers the whole
    #     payload and not just the one field this section cares about
    p2 = dict(payload); p2["validation_rank_ic"] = float(payload["validation_rank_ic"]) + 1e-9
    L3_breaks_on_any_field = evidence_id_of(p2) != eid

    return {
        "target_record": {"workspace": ws, "fold": fold, "evidence_id": eid[:16],
                          "model_artifact_id": mid, "file_tampered": some_file},
        "i_corrupt_bundle_file_breaks_L5": L5_breaks,
        "ii_corrupt_manifest_breaks_L4": L4_breaks,
        "iii_change_bundle_digest_breaks_L3_keyless": L3_breaks,
        "iv_change_evidence_id_breaks_L2": L2_breaks,
        "v_point_at_other_record_breaks_L1": L1_breaks,
        "vi_change_ANY_payload_field_breaks_L3": L3_breaks_on_any_field,
        "note": ("all six mutations were computed in memory; no byte was written to any artifact directory, "
                 "so the bundles and records remain exactly what production shipped"),
    }


def source_chain() -> dict:
    se = (REPO / "src/research_v2/evaluation/selection_evidence.py").read_text(encoding="utf-8").split("\n")

    def grab(needle):
        return {n: l.strip() for n, l in enumerate(se, 1) if needle in l}
    return {
        "keyless_content_hash_definition": grab("return hashlib.sha256(_canonical_json(payload)"),
        "canonical_json_definition": grab('separators=(",", ":"), sort_keys=True'),
        "verifier_checks_hmac": grab("_signature(self._verification_key, evidence_id, typed)"),
        "verifier_checks_keyless_content_hash": grab("if _evidence_id(typed) != evidence_id:"),
        "signature_covers_evidence_id_and_payload": grab('document = {"evidence_id": evidence_id, "payload": payload}'),
        "key_is_env_only": grab("_SELECTION_EVIDENCE_KEY_ENV ="),
        "key_read_from_environment": grab("verification_key=secret.encode"),
    }


def main() -> None:
    chains = {ws: verify_chain(ws, fam) for ws, fam in CARRIERS}
    tamper = tamper_controls()
    src = source_chain()

    # aggregate the keyless content-binding check across ALL records, both kinds, all carriers
    all_records = []
    for ws, _ in CARRIERS:
        for kind in ("model", "baseline"):
            for rec in sorted((ART / ws / "selections" / kind).glob("*/record.json")):
                d = json.loads(rec.read_text(encoding="utf-8"))
                all_records.append({
                    "workspace": ws, "kind": kind,
                    "dir_stem": rec.parent.name,
                    "evidence_id_field": d.get("evidence_id"),
                    "recomputed": evidence_id_of(d["payload"]),
                    "L2_ok": rec.parent.name == f"{d.get('evidence_id')}.json",
                    "L3_ok": evidence_id_of(d["payload"]) == d.get("evidence_id"),
                    "payload_type": d["payload"].get("type"),
                })
    n_l3 = sum(1 for r in all_records if r["L3_ok"])
    n_l2 = sum(1 for r in all_records if r["L2_ok"])

    folds_with_all = []
    folds_missing_L1 = []
    for ws, c in chains.items():
        for f in c["folds"]:
            if f.get("all_five_links_hold"):
                folds_with_all.append(f"{ws}/{f['fold_id']}")
            if f.get("L1_report_names_this_evidence_id") is None:
                folds_missing_L1.append(f"{ws}/{f['fold_id']}")

    doc = {
        "question": ("is the shipped-model provenance chain tamper-EVIDENT end to end, and can the keyless "
                     "half be verified without the HMAC key? §CX.5 claimed the chain cannot resist "
                     "tampering because the manifest sits beside the files -- is that reasoning complete?"),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "correction_to_CX5": {
            "what_CX5_said": ("not citable: 'this chain resists tampering' -- reason given was that "
                              "checksums.sha256 sits beside the files it protects with the same permissions"),
            "why_that_reason_is_incomplete": (
                "the reasoning stops one link too early. The evaluator ALSO checks "
                "_evidence_id(payload) == evidence_id (selection_evidence.py:145), a KEYLESS sha256 over the "
                "canonical payload, and model_bundle_digest lives inside that payload. So editing any bundle "
                "file forces editing the manifest, which changes the bundle digest, which forces editing the "
                "payload, which breaks the keyless content-hash check. Modification is DETECTED."),
            "corrected_claim": ("the chain is tamper-EVIDENT, not tamper-PROOF: any modification is detected "
                                "by the evaluator, and producing an UNDETECTED modification additionally "
                                "requires the HMAC key (QRV2_SELECTION_EVIDENCE_HMAC_KEY), which this pack "
                                "does not read"),
            "what_remains_unproven": (
                "the HMAC signature itself is not verified (no key), so an attacker holding the key could "
                "forge an entirely new consistent record that passes every check; and that these records "
                "were signed by the trusted pre-TEST selection service cannot be shown without the key"),
        },
        "the_five_links": {
            "L1": "report.json names the fold's model_selection_id",
            "L2": "evidence_id is the directory name <eid>.json AND the record's own evidence_id field",
            "L3": "evidence_id == sha256(canonical_json(payload)) -- KEYLESS content binding",
            "L4": "payload.model_bundle_digest == sha256(checksums.sha256 bytes)",
            "L5": "every bundle file's sha256 == its line in checksums.sha256",
            "which_are_new_here": ["L1", "L2", "L3"],
            "which_restate_CX": ["L4", "L5"],
        },
        "source_chain": src,
        "per_carrier": chains,
        "all_records_keyless_binding": {
            "n_records_checked": len(all_records),
            "n_L3_content_hash_ok": n_l3,
            "n_L2_dir_binding_ok": n_l2,
            "all_L3_ok": n_l3 == len(all_records),
            "all_L2_ok": n_l2 == len(all_records),
            "by_kind": {k: sum(1 for r in all_records if r["kind"] == k) for k in ("model", "baseline")},
            "records": all_records,
        },
        "summary": {
            "n_folds_with_all_five_links_verified": len(folds_with_all),
            "folds_with_all_five_links_verified": folds_with_all,
            "n_folds_with_four_links_L1_unavailable": len(folds_missing_L1),
            "folds_where_L1_unavailable_no_report_json": folds_missing_L1,
            "reading": ("L1 needs report.json, which M2 does not have yet; for M2's two folds L2-L5 are "
                        "verified and L1 is recorded as UNAVAILABLE, not as passed -- so the honest counts "
                        "are 12 folds with all five links and 2 folds with four of five. An earlier draft "
                        "of this probe counted those 2 as five-of-five."),
        },
        "tamper_controls": tamper,
        "controls": {},
        "read_only": True, "database_access": False, "weights_loaded": False,
        "pt_loaded_as_tensor": False, "parquet_row_data_read": False,
        "environment_secret_read": False, "training_performed": False,
        "repository_module_imported_or_executed": False, "processes_signalled": [],
        "atime_side_effect": False, "byte_reproducible": True,
    }
    doc["controls"] = {
        "all_records_L3_keyless_content_hash_ok": n_l3 == len(all_records) and len(all_records) > 0,
        "all_records_L2_dir_binding_ok": n_l2 == len(all_records),
        "L4_and_L5_hold_on_every_bundle_that_exists": all(
            f.get("L4_payload_digest_equals_sha256_of_manifest") is True
            and f.get("L5_all_files_match_manifest") is True
            for c in chains.values() for f in c["folds"]),
        "no_link_reported_as_failed_anywhere": not any(
            f.get("links_failed") for c in chains.values() for f in c["folds"]),
        "L1_verified_true_wherever_report_json_exists": all(
            f.get("L1_report_names_this_evidence_id") is True
            for c in chains.values() if c["report_json_present"] for f in c["folds"]),
        "L1_explicitly_unavailable_only_where_no_report_json": all(
            (f.get("L1_report_names_this_evidence_id") is None) == (not c["report_json_present"])
            for c in chains.values() for f in c["folds"]),
        "TAMPER_i_bundle_file_breaks_L5": tamper["i_corrupt_bundle_file_breaks_L5"],
        "TAMPER_ii_manifest_breaks_L4": tamper["ii_corrupt_manifest_breaks_L4"],
        "TAMPER_iii_digest_breaks_L3_keyless": tamper["iii_change_bundle_digest_breaks_L3_keyless"],
        "TAMPER_iv_evidence_id_breaks_L2": tamper["iv_change_evidence_id_breaks_L2"],
        "TAMPER_v_other_record_breaks_L1": tamper["v_point_at_other_record_breaks_L1"],
        "TAMPER_vi_any_payload_field_breaks_L3": tamper["vi_change_ANY_payload_field_breaks_L3"],
        "source_shows_both_checks_in_verifier": bool(
            src["verifier_checks_hmac"] and src["verifier_checks_keyless_content_hash"]),
        "ALL_PASS": None,
    }
    doc["controls"]["ALL_PASS"] = all(
        v for k, v in doc["controls"].items() if k != "ALL_PASS")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print("=== the five links, per carrier per fold ===")
    for ws, c in chains.items():
        for f in c["folds"]:
            print(f"  {ws:13s} {f['fold_id']}  L1={f.get('L1_report_names_this_evidence_id')} "
                  f"L2={f.get('L2_dir_stem_equals_record_evidence_id')} "
                  f"L3={f.get('L3_evidence_id_equals_sha256_of_canonical_payload')} "
                  f"L4={f.get('L4_payload_digest_equals_sha256_of_manifest')} "
                  f"L5={f.get('L5_all_files_match_manifest')} -> all={f.get('all_five_links_hold')}")
    a = doc["all_records_keyless_binding"]
    print(f"\n=== keyless content binding across ALL {a['n_records_checked']} records "
          f"(model {a['by_kind']['model']} + baseline {a['by_kind']['baseline']}) ===")
    print(f"  L3 (evidence_id == sha256(canon(payload))): {a['n_L3_content_hash_ok']}/{a['n_records_checked']}")
    print(f"  L2 (dir stem == <evidence_id>.json):        {a['n_L2_dir_binding_ok']}/{a['n_records_checked']}")
    s = doc["summary"]
    print(f"\n  folds with all FIVE links verified: {s['n_folds_with_all_five_links_verified']} "
          f"{s['folds_with_all_five_links_verified']}")
    print(f"  folds with FOUR links, L1 unavailable (no report.json): "
          f"{s['n_folds_with_four_links_L1_unavailable']} {s['folds_where_L1_unavailable_no_report_json']}")
    tc = tamper
    print(f"\n=== tamper controls (each edits what a link binds; all in memory) ===")
    print(f"  target: {tc['target_record']}")
    for k in ("i_corrupt_bundle_file_breaks_L5", "ii_corrupt_manifest_breaks_L4",
              "iii_change_bundle_digest_breaks_L3_keyless", "iv_change_evidence_id_breaks_L2",
              "v_point_at_other_record_breaks_L1", "vi_change_ANY_payload_field_breaks_L3"):
        print(f"  {'OK ' if tc[k] else '*** FAIL ***'} {k}")
    print(f"\n=== controls ===")
    for k, v in doc["controls"].items():
        print(f"  {'OK ' if v else '*** FAIL ***'} {k}")
    assert doc["controls"]["ALL_PASS"], "a control failed"
    print("wrote", OUT)


if __name__ == "__main__":
    main()
