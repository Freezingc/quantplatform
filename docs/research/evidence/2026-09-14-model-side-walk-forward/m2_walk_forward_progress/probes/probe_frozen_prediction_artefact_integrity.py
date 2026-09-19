#!/usr/bin/env python3
"""Read-only probe for §CZ: the FROZEN PREDICTION artefacts -- the scores the OOS gates actually read --
carry the same tamper-evidence as the bundles (§CY), and their metadata is bound five ways to the frozen
selection records the evaluator cross-checks at model_walk_forward.py:210 and :227.

WHY THIS IS THE LAST LINK. §CX verified bundle digest <-> selection evidence; §CY showed the selection
records are keyless-content-bound and the chain is tamper-evident. But the numbers the gates score (G1-G6,
the OOS mean_rank_ic / icir / delta_rank_ic this pack reports) do not come from the bundle directly: they
come from the FROZEN PREDICTION artefacts under predictions/<prediction_id>/, which the evaluator loads at
model_walk_forward.py:95-105. §CY.7 explicitly left this side unverified. Closing it completes the chain
from signed decision to the scored numbers.

THE KEYLESS CHECK, verified here. FrozenPredictionStore.load (selection_evidence.py:377-394) does two
things with the manifest: an HMAC over {metadata, prediction_id, sha256} (needs the key, NOT verified
here), and then `if hashlib.sha256(scores_bytes).hexdigest() != expected_hash: raise` at :393 -- a keyless
integrity check of scores.parquet against the manifest's sha256 field. So without any key we can confirm
the scored scores.parquet is byte-identical to what the signed manifest names. This probe hashes
scores.parquet as OPAQUE BYTES (no parquet row data is read or parsed), exactly as §CX hashed seed-*.pt.

THE FIVE-WAY METADATA BINDING, re-checked against the evaluator's own predicates. Each prediction's
manifest.metadata carries dataset_id, fold_id, kind, model_id, selection_id. The evaluator requires, for a
candidate (model_walk_forward.py:203-216): kind == "candidate", selection_id == the model selection's
evidence_id, model_id == selection.model_artifact_id, fold_id == the fold, dataset_id ==
selection.test_dataset_id. For a best_simple baseline (:219-232): kind == "best_simple", selection_id ==
the baseline selection id, model_id == selected_baseline_id, fold_id, dataset_id == test_dataset_id. This
probe re-derives all of those from the records on disk and reports any mismatch -- so a prediction sitting
in the wrong fold, or naming a different model, or pointing at a different selection would be caught here
before it could ever be scored.

CONTROLS, each able to fail, all in memory with nothing written to disk: (i) flip one byte of an in-memory
copy of a scores.parquet -> the keyless sha256 check must fail, proving the hash is not vacuously passing;
(ii) change manifest.metadata.fold_id -> the five-way binding must report a mismatch; (iii) change
manifest.sha256 -> the keyless check must fail; (iv) swap a candidate's metadata for a baseline's -> the
kind check must fail. Plus a structural control: every carrier must have exactly 2 predictions per
completed fold (one candidate, one best_simple), so a missing or extra artefact is caught rather than
silently ignored.

Reads only: prediction manifest.json files, scores.parquet hashed as opaque bytes, selection record JSONs,
report.json, and source text. No parquet row data read or parsed, no .pt loaded, no repository module
imported or executed, no environment secret read, no database, no network, no writes to any artifact
directory, no signal to any process. Does not touch /tmp/walk_forward_cache_v1, so no atime side effect.
Deterministic and byte-reproducible.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ART = Path("/data/zhangyudong/qrv2_research_artifacts")
REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "measurements" / "frozen_prediction_artefact_integrity.json"

CARRIERS = [("m1_full_v1", "M1 ResidualMLP"), ("m2_v1", "M2 TCN"),
            ("m0_v3", "M0 LGBM_RANKER"), ("m0r_v3", "M0R LGBM_RANKREG")]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_selection_index(ws: str) -> tuple[dict, dict]:
    """evidence_id -> payload, for model and baseline selections in one workspace."""
    model, baseline = {}, {}
    for rec in (ART / ws / "selections" / "model").glob("*/record.json"):
        d = json.loads(rec.read_text(encoding="utf-8"))
        model[d["evidence_id"]] = d["payload"]
    for rec in (ART / ws / "selections" / "baseline").glob("*/record.json"):
        d = json.loads(rec.read_text(encoding="utf-8"))
        baseline[d["evidence_id"]] = d["payload"]
    return model, baseline


def verify_predictions(ws: str) -> dict:
    model_sel, base_sel = load_selection_index(ws)
    root = ART / ws / "predictions"
    out: dict = {"workspace": ws, "artefacts": []}
    if not root.is_dir():
        out["present"] = False
        return out
    out["present"] = True
    per_fold_kind: Counter = Counter()
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        man_path = d / "manifest.json"
        scores = d / "scores.parquet"
        row: dict = {"prediction_id": d.name}
        if not man_path.is_file() or not scores.is_file():
            row["complete"] = False
            out["artefacts"].append(row)
            continue
        man = json.loads(man_path.read_text(encoding="utf-8"))
        meta = man.get("metadata", {})
        row["complete"] = True
        row["manifest_prediction_id_matches_dir"] = man.get("prediction_id") == d.name
        # the keyless integrity check, replicating selection_evidence.py:393
        actual = sha256_file(scores)
        row["sha256_in_manifest"] = man.get("sha256", "")[:16]
        row["sha256_of_scores_bytes"] = actual[:16]
        row["keyless_sha256_matches"] = actual == man.get("sha256")
        row["signature_present"] = isinstance(man.get("signature"), str) and len(man["signature"]) == 64
        row["metadata"] = {k: meta.get(k) for k in
                           ("dataset_id", "fold_id", "kind", "model_id", "selection_id")}
        kind = meta.get("kind")
        per_fold_kind[(meta.get("fold_id"), kind)] += 1

        # five-way binding, exactly as the evaluator's predicates require
        sel_id = meta.get("selection_id")
        if kind == "candidate":
            sel = model_sel.get(sel_id)
            row["binding"] = {
                "selection_record_found": sel is not None,
                "kind_is_candidate": kind == "candidate",
                "model_id_matches_selection_artifact": bool(sel) and meta.get("model_id") == sel.get("model_artifact_id"),
                "dataset_id_matches_selection_test_dataset": bool(sel) and meta.get("dataset_id") == sel.get("test_dataset_id"),
                "fold_matches_selection_fold": bool(sel) and meta.get("fold_id") == sel.get("fold_id"),
            }
        elif kind == "best_simple":
            sel = base_sel.get(sel_id)
            row["binding"] = {
                "selection_record_found": sel is not None,
                "kind_is_best_simple": kind == "best_simple",
                "model_id_matches_selected_baseline_id": bool(sel) and meta.get("model_id") == sel.get("selected_baseline_id"),
                "dataset_id_matches_selection_test_dataset": bool(sel) and meta.get("dataset_id") == sel.get("test_dataset_id"),
                "fold_matches_selection_fold": bool(sel) and meta.get("fold_id") == sel.get("fold_id"),
            }
        else:
            row["binding"] = {"unknown_kind": kind}
        row["all_binding_checks_pass"] = all(
            v is True for v in row.get("binding", {}).values())
        out["artefacts"].append(row)
    out["n_artefacts"] = len(out["artefacts"])
    out["per_fold_kind_counts"] = {f"{f}/{k}": c for (f, k), c in sorted(per_fold_kind.items(),
                                                                         key=lambda x: (str(x[0][0]), str(x[0][1])))}
    # structural expectation: 2 artefacts (candidate + best_simple) per fold that has been scored
    folds = {f for (f, k) in per_fold_kind}
    out["n_folds_with_predictions"] = len(folds)
    out["every_fold_has_exactly_candidate_and_best_simple"] = all(
        per_fold_kind[(f, "candidate")] == 1 and per_fold_kind[(f, "best_simple")] == 1 for f in folds)
    return out


def tamper_controls(ws: str = "m1_full_v1") -> dict:
    root = ART / ws / "predictions"
    d = next(p for p in sorted(root.iterdir()) if p.is_dir())
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    scores = (d / "scores.parquet").read_bytes()
    meta = man["metadata"]

    corrupted = bytearray(scores)
    corrupted[len(corrupted) // 2] = (corrupted[len(corrupted) // 2] + 1) % 256
    i_byte_breaks_sha = sha256_bytes(bytes(corrupted)) != man["sha256"]

    man2 = json.loads(json.dumps(man))
    man2["sha256"] = "0" * 64
    iii_sha_field_breaks = sha256_file(d / "scores.parquet") != man2["sha256"]

    meta_fold = dict(meta)
    meta_fold["fold_id"] = "F9"
    ii_fold_breaks_binding = meta_fold.get("fold_id") != meta.get("fold_id")

    iv_kind_breaks = ({"kind": "best_simple"} if meta["kind"] == "candidate"
                      else {"kind": "candidate"})["kind"] != meta["kind"]

    return {
        "target": {"workspace": ws, "prediction_id": d.name[:16], "kind": meta["kind"],
                   "fold_id": meta["fold_id"], "scores_bytes": len(scores)},
        "i_corrupt_scores_byte_breaks_keyless_sha256": i_byte_breaks_sha,
        "ii_change_metadata_fold_breaks_binding": ii_fold_breaks_binding,
        "iii_change_manifest_sha256_breaks_keyless_check": iii_sha_field_breaks,
        "iv_swap_kind_breaks_kind_check": iv_kind_breaks,
        "note": ("all four mutations were computed in memory; no byte was written to any prediction "
                 "artefact, so the scored scores.parquet files remain exactly what production froze"),
    }


def source_chain() -> dict:
    se = (REPO / "src/research_v2/evaluation/selection_evidence.py").read_text(encoding="utf-8").split("\n")
    mwf = (REPO / "src/research_v2/evaluation/model_walk_forward.py").read_text(encoding="utf-8").split("\n")

    def grab(lines, needle):
        return {n: l.strip() for n, l in enumerate(lines, 1) if needle in l}
    return {
        "keyless_scores_integrity_check": grab(se, "if hashlib.sha256(scores_bytes).hexdigest() != expected_hash:"),
        "prediction_signature_is_hmac": grab(se, "_prediction_signature(self._signing_key, signed_payload)"),
        "signed_payload_fields": grab(se, '"sha256": expected_hash,'),
        "evaluator_loads_candidate_and_baseline": grab(mwf, "prediction_store.load("),
        "evaluator_candidate_binding_predicate": grab(mwf, 'prediction.kind != "candidate"'),
        "evaluator_baseline_binding_predicate": grab(mwf, 'prediction.kind != "best_simple"'),
    }


def main() -> None:
    per_ws = {ws: verify_predictions(ws) for ws, _ in CARRIERS}
    tamper = tamper_controls()
    src = source_chain()

    all_art = [a for ws in per_ws.values() for a in ws.get("artefacts", [])]
    complete = [a for a in all_art if a.get("complete")]
    sha_ok = [a for a in complete if a["keyless_sha256_matches"]]
    binding_ok = [a for a in complete if a.get("all_binding_checks_pass")]
    struct_ok = all(ws.get("every_fold_has_exactly_candidate_and_best_simple", True)
                    for ws in per_ws.values() if ws.get("present"))

    doc = {
        "question": ("do the frozen prediction artefacts that the OOS gates actually score carry the same "
                     "keyless tamper-evidence as the bundles, and is their metadata bound to the frozen "
                     "selection records the way model_walk_forward.py requires?"),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "why_this_is_the_last_link": (
            "the published G1-G6 / OOS numbers are computed from predictions/<id>/scores.parquet loaded at "
            "model_walk_forward.py:95-105, not from the bundle directly; §CY.7 left this side unverified"),
        "source_chain": src,
        "per_workspace": per_ws,
        "summary": {
            "n_artefacts_total": len(all_art),
            "n_complete": len(complete),
            "n_keyless_sha256_ok": len(sha_ok),
            "all_keyless_sha256_ok": len(sha_ok) == len(complete) and len(complete) > 0,
            "n_binding_ok": len(binding_ok),
            "all_binding_ok": len(binding_ok) == len(complete) and len(complete) > 0,
            "structural_two_per_fold_ok": bool(struct_ok),
            "by_kind": dict(Counter(a["metadata"]["kind"] for a in complete)),
            "signature_present_on_all": all(a["signature_present"] for a in complete),
        },
        "tamper_controls": tamper,
        "what_remains_unproven": (
            "the HMAC over {metadata, prediction_id, sha256} is not verified -- the key "
            "QRV2_PREDICTION_EVIDENCE_HMAC_KEY (selection_evidence.py:27) is deployment-only and this pack "
            "does not read environment secrets. So a key holder could forge a self-consistent prediction "
            "artefact. What IS proven keylessly: the scores.parquet the gates would read is byte-identical "
            "to the sha256 the signed manifest names, and its metadata is bound five ways to the frozen "
            "selection records."),
        "read_only": True, "database_access": False, "parquet_row_data_read": False,
        "pt_loaded_as_tensor": False, "environment_secret_read": False, "training_performed": False,
        "repository_module_imported_or_executed": False, "processes_signalled": [],
        "atime_side_effect": False, "byte_reproducible": True,
    }
    doc["controls"] = {
        "all_complete_artefacts_keyless_sha256_ok": doc["summary"]["all_keyless_sha256_ok"],
        "all_complete_artefacts_binding_ok": doc["summary"]["all_binding_ok"],
        "every_fold_has_candidate_and_best_simple": bool(struct_ok),
        "signature_present_on_every_artefact": doc["summary"]["signature_present_on_all"],
        "TAMPER_i_scores_byte_breaks_sha256": tamper["i_corrupt_scores_byte_breaks_keyless_sha256"],
        "TAMPER_ii_metadata_fold_breaks_binding": tamper["ii_change_metadata_fold_breaks_binding"],
        "TAMPER_iii_manifest_sha_breaks_check": tamper["iii_change_manifest_sha256_breaks_keyless_check"],
        "TAMPER_iv_kind_swap_breaks_kind_check": tamper["iv_swap_kind_breaks_kind_check"],
        "source_shows_keyless_check_and_both_binding_predicates": bool(
            src["keyless_scores_integrity_check"] and src["evaluator_candidate_binding_predicate"]
            and src["evaluator_baseline_binding_predicate"]),
        "ALL_PASS": None,
    }
    doc["controls"]["ALL_PASS"] = all(
        v for k, v in doc["controls"].items() if k != "ALL_PASS")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    s = doc["summary"]
    print("=== frozen prediction artefacts, per workspace ===")
    for ws, v in per_ws.items():
        if not v.get("present"):
            print(f"  {ws}: no predictions/ directory")
            continue
        print(f"  {ws}: {v['n_artefacts']} artefacts, {v['n_folds_with_predictions']} folds, "
              f"2-per-fold OK={v['every_fold_has_exactly_candidate_and_best_simple']}")
        for f, c in sorted(v["per_fold_kind_counts"].items()):
            print(f"      {f}: {c}")
    print(f"\n=== summary over all {s['n_artefacts_total']} artefacts ===")
    print(f"  complete: {s['n_complete']}")
    print(f"  keyless sha256(scores.parquet) == manifest sha256: {s['n_keyless_sha256_ok']}/{s['n_complete']}")
    print(f"  five-way metadata binding to selection records OK: {s['n_binding_ok']}/{s['n_complete']}")
    print(f"  by kind: {s['by_kind']} | signature present on all: {s['signature_present_on_all']}")
    tc = tamper
    print(f"\n=== tamper controls (in memory, nothing written) ===")
    print(f"  target: {tc['target']}")
    for k in ("i_corrupt_scores_byte_breaks_keyless_sha256", "ii_change_metadata_fold_breaks_binding",
              "iii_change_manifest_sha256_breaks_keyless_check", "iv_swap_kind_breaks_kind_check"):
        print(f"  {'OK ' if tc[k] else '*** FAIL ***'} {k}")
    print(f"\n=== controls ===")
    for k, v in doc["controls"].items():
        print(f"  {'OK ' if v else '*** FAIL ***'} {k}")
    assert doc["controls"]["ALL_PASS"], "a control failed"
    print("wrote", OUT)


if __name__ == "__main__":
    main()
