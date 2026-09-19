#!/usr/bin/env python3
"""Read-only probe for §CU: every shipped TORCH bundle is internally inconsistent -- its
validation_predictions.parquet was produced by LAST-EPOCH weights while its seed-*.pt are PEAK weights --
and this is provable from source without loading a single weight. Tree carriers are NOT affected. §CC.1
asserted this in passing; this section proves it and bounds its blast radius at zero.

THE CLAIM, AND WHY IT IS A PROOF RATHER THAN A MEASUREMENT
A shipped bundle contains both weights and a validation-prediction frame:
  * seed-{seed}.pt  -- §CT proved by sha256 these are byte-identical to the PEAK checkpoint (18/18),
    and the mechanism is trainer.py:110-116 (save only on strict improvement) +
    cross_section_search.py:206 / temporal_search.py:41 (bundle source = checkpoint_ids[-1] = last
    saved = peak) + model_bundle.py:90 (shutil.copyfile).
  * validation_predictions.parquet -- built from `predictions = percentile_frame(model, ...)`
    (cross_section_search.py:203) or `predictions = percentile_frame(model, batches.validation_batches)`
    (temporal_search.py:38), where `model` is the IN-MEMORY object at the moment fit() returned.
The question is which weights that in-memory model holds. Three facts settle it, all read from source:
  1. trainer.py never restores the best weights. `grep -n "load_state_dict|restore|best_state|
     best_weights|copy.deepcopy" trainer.py` returns NOTHING; fit() ends with
     `return TrainingResult(best_epoch, best_value, tuple(history))` (trainer.py:125) and never touches
     `model` again, so the caller's `model` is whatever the last executed epoch left in it.
  2. Early stopping runs at least `patience` non-improving epochs after the peak: trainer.py:118-120
     breaks only when `stale_epochs >= early_stopping_patience`, and stale_epochs increments on every
     epoch that does not improve (:117). So last_executed_epoch >= peak_epoch + patience.
  3. §CJ established 0 of 483 completed runs hit the cap, and §CT/§CS confirm the M2 F2 refit
     early-stopped (peak 71, cap 100), so no run was truncated -- last_executed = peak + patience
     exactly, i.e. 10 extra gradient steps for every torch run.
Therefore the parquet's predictions come from a model that took >= 10 more optimizer steps than the
weights shipped beside it. They describe DIFFERENT models. No weight needs to be loaded to know this.

BLAST RADIUS IS ZERO, and that is the part that matters. `validation_predictions.parquet` is written
(model_bundle.py:97), checksummed, and listed among expected_files (model_bundle.py:231), and one unit
test asserts the file EXISTS (test_model_artifact_bundle.py:39) -- but NO code ever reads it back. A
repo-wide grep for readers finds none in src/, scripts/, evaluation/, inference/ or gates/. In
particular load_bundle_for_evaluation rebuilds models from `io.BytesIO(bundle.files[f"seed-{seed}.pt"])`
(inference/artifact_loader.py:57) and the driver scores TEST from those rebuilt models
(run_real_candidate_training.py:2636, :3317), which is the chain §CC.2 verified. So the inconsistent
parquet is dead weight: it is shipped, checksummed and never consulted. Consequently NO published number
changes -- not OOS mean_rank_ic/icir, not G6/G4, not §BQ's reproduction, not §BR's floor sweep.

THE CONTROL THAT KEEPS THE GREP HONEST. "No readers found" is only meaningful if the same search WOULD
find a reader when one exists. So the probe runs the identical grep for `seed-` / `.pt` and must find
artifact_loader.py:57 reading it. A grep that returns nothing for both files would be vacuous; this one
returns nothing for the parquet and something for the weights, which is the asymmetry the finding rests
on. (Same discipline as §CM.6(iv)/§CP.7: a control must be able to fail.)

TREE CARRIERS ARE THE DIFFERENTIAL CONTROL, and they come out consistent. LightGBM's predict uses
`num_iteration=self.best_iteration` (lgbm_rankreg.py:314) and the saved booster.txt is itself truncated
to best_iteration (§CQ verified tree count == best_iteration, 214 << the 500 cap), so for M0/M0R the
shipped weights and the shipped validation predictions are BOTH best_iteration -- internally consistent.
The defect is specific to the torch carriers, whose trainer keeps a live mutable model and never rolls it
back. This is why the finding is stated as "torch bundles" and not "bundles".

WHAT THIS DOES NOT CLAIM. It does not quantify HOW DIFFERENT the two models' predictions are -- that
would require either loading the weights and re-scoring, or scoring the parquet against validation
labels (reading parquet row data); neither is done. It does not claim the parquet is WRONG for any
purpose it is actually used for, because it is used for none. It does not claim a fix is needed for
correctness of published results; the fix matters only if someone later starts consuming that parquet
(e.g. to report validation metrics from the bundle), at which point they would be reporting last-epoch
numbers while believing they describe the shipped peak model. Per ledger B1 no unverified magnitude is
offered.

Read-only: no repository file changed, no module imported or executed (all four source files read as
text), no .pt loaded as a tensor, no parquet opened at all (not even footers -- the parquet's existence
and row count were established in §CS, not here), no database, no network, no writes to any artifact
directory, no signal to any process. Does not touch /tmp/walk_forward_cache_v1, so no atime side effect.
Deterministic and byte-reproducible: every input is an immutable artefact or a source file, and nothing
here reads live process state.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
ART = Path("/data/zhangyudong/qrv2_research_artifacts")
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "measurements" / "bundle_validation_predictions_are_last_epoch.json"

TORCH_SOURCES = [
    "src/research_v2/training/trainer.py",
    "src/research_v2/training/cross_section_search.py",
    "src/research_v2/training/temporal_search.py",
    "src/research_v2/training/model_bundle.py",
    "src/research_v2/inference/artifact_loader.py",
    "src/research_v2/baselines/lgbm_rankreg.py",
    "scripts/run_real_candidate_training.py",
]
READER_PATTERN = re.compile(r"validation_predictions\.parquet|validation_predictions\"|files\[f?\"seed-|seed-\{seed\}\.pt|\.pt\"\]")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def grep_repo(pattern: str, globs: list[str]) -> list[str]:
    """rg-style search over the given path globs; returns 'relpath:lineno:text' hits."""
    hits = []
    for g in globs:
        for path in sorted(REPO.glob(g)):
            if not path.is_file() or path.suffix != ".py":
                continue
            try:
                for n, line in enumerate(path.read_text(encoding="utf-8", errors="replace").split("\n"), 1):
                    if re.search(pattern, line):
                        hits.append(f"{path.relative_to(REPO)}:{n}: {line.strip()[:150]}")
            except OSError:
                continue
    return hits


def trainer_never_restores_best() -> dict:
    """Fact 1: trainer.py has no weight-restoration call anywhere."""
    path = REPO / "src/research_v2/training/trainer.py"
    text = path.read_text(encoding="utf-8")
    restore_tokens = ("load_state_dict", "best_state", "best_weights", "copy.deepcopy",
                      "restore", "state_dict()")
    found = {tok: [n for n, l in enumerate(text.split("\n"), 1) if tok in l] for tok in restore_tokens}
    returns = [l.strip() for l in text.split("\n") if "return TrainingResult" in l]
    return {
        "file": str(path.relative_to(REPO)),
        "sha256": sha256_file(path)[:16],
        "restoration_tokens_found": {k: v for k, v in found.items() if v},
        "any_restoration_token_present": any(found.values()),
        "return_statement": returns,
        "conclusion": ("fit() returns TrainingResult(best_epoch, best_value, history) and never writes "
                       "back into `model`, so the caller's in-memory model is the LAST-EPOCH state"),
    }


def early_stopping_runs_patience_past_peak() -> dict:
    """Fact 2: the loop breaks only after `patience` consecutive non-improving epochs."""
    path = REPO / "src/research_v2/training/trainer.py"
    lines = path.read_text(encoding="utf-8").split("\n")
    grab = {}
    for n, l in enumerate(lines, 1):
        s = l.strip()
        if "stale_epochs += 1" in s or "stale_epochs >= self._config.early_stopping_patience" in s \
           or "stale_epochs = 0" in s or "if validation_rank_ic > best_value" in s:
            grab[n] = s
    return {"file": str(path.relative_to(REPO)), "lines": grab,
            "conclusion": ("last_executed_epoch >= peak_epoch + patience, because the loop cannot exit "
                           "until patience consecutive epochs fail to improve")}


def prediction_uses_in_memory_model() -> dict:
    """Fact 3: both torch search paths build `predictions` from the live `model`, not a reload."""
    out = {}
    for rel, needle in (("src/research_v2/training/cross_section_search.py", "percentile_frame(model"),
                        ("src/research_v2/training/temporal_search.py", "percentile_frame(model")):
        path = REPO / rel
        lines = path.read_text(encoding="utf-8").split("\n")
        hits = {n: l.strip() for n, l in enumerate(lines, 1) if needle in l}
        ckpt = {n: l.strip() for n, l in enumerate(lines, 1) if "checkpoint_ids[-1]" in l}
        out[rel] = {"prediction_line": hits, "checkpoint_source_line": ckpt,
                    "sha256": sha256_file(path)[:16]}
    return out


def classify_mention(line: str) -> str:
    """Classify one source line mentioning validation_predictions by what it actually does.

    The first draft of this classifier searched for the bare substring "validation_predictions" and
    bucketed anything unrecognised as OTHER, which produced 33 OTHER hits and tripped the control. All
    33 were the LOCAL variable `seed_validation_predictions` (an in-memory DataFrame passed toward the
    bundle write) or the keyword argument `validation_predictions=` -- neither is the shipped parquet
    file, so neither can be a reader of it. A substring search cannot tell a local variable from a file
    access, which is exactly why the control fired. The classifier is therefore keyed on the precise
    identifier and on actual file-access syntax, and READER is defined narrowly: something that takes the
    bytes/parquet file out of a bundle (files[...], read_parquet, open(...validation_predictions...)).
    """
    stripped = line.strip()
    # narrow READER forms: pulling the file out of a bundle or off disk
    if re.search(r"files\[[^\]]*validation_predictions", stripped) \
       or re.search(r"read_parquet\([^)]*validation_predictions", stripped) \
       or re.search(r"open\([^)]*validation_predictions", stripped) \
       or re.search(r"load[^=]*\([^)]*validation_predictions\.parquet", stripped):
        return "READER"
    if "validation_predictions.parquet" in stripped:
        if "temporary /" in stripped or "to_parquet" in stripped:
            return "WRITE_FILE"
        if ".is_file()" in stripped or "assert" in stripped:
            return "TEST_ASSERTS_FILE_EXISTS"
        return "FILENAME_IN_EXPECTED_LIST"
    if re.search(r"seed_validation_predictions", stripped):
        return "LOCAL_VARIABLE_seed_validation_predictions"
    if re.search(r"validation_predictions\s*[=:]", stripped):
        return "KEYWORD_ARGUMENT_OR_ANNOTATION"
    if re.search(r"validation_predictions\.(columns|sort_values|rename)|difference\(", stripped):
        return "DATAFRAME_OPERATION_BEFORE_WRITE"
    return "UNCLASSIFIED"


def parquet_reader_search() -> dict:
    """The blast-radius search, plus the control that the same search finds the WEIGHT reader.

    Reports every mention bucketed by classify_mention, so the claim "nothing reads the shipped parquet
    back" rests on an enumerated, inspectable classification rather than on an empty grep result.
    """
    globs = ["src/**/*.py", "scripts/*.py", "tests/**/*.py"]
    mentions: list[tuple[str, int, str]] = []
    for g in globs:
        for path in sorted(REPO.glob(g)):
            if not path.is_file() or path.suffix != ".py":
                continue
            try:
                for n, line in enumerate(
                        path.read_text(encoding="utf-8", errors="replace").split("\n"), 1):
                    if "validation_predictions" in line:
                        mentions.append((str(path.relative_to(REPO)), n, line.strip()))
            except OSError:
                continue
    by_role: dict[str, list[str]] = {}
    for rel, n, line in mentions:
        by_role.setdefault(classify_mention(line), []).append(f"{rel}:{n}: {line[:140]}")
    weight_reader_hits = grep_repo(r"BytesIO\(bundle\.files|files\[f?\"seed-", globs)
    return {
        "n_mentions_total": len(mentions),
        "by_role": {k: v for k, v in sorted(by_role.items())},
        "n_readers": len(by_role.get("READER", [])),
        "readers": by_role.get("READER", []),
        "any_reader_found": bool(by_role.get("READER")),
        "unclassified_needing_review": by_role.get("UNCLASSIFIED", []),
        "CONTROL_weight_reader_must_be_found": {
            "n_hits": len(weight_reader_hits), "hits": weight_reader_hits[:6],
            "found_artifact_loader_reading_seed_pt": any("artifact_loader" in h
                                                         for h in weight_reader_hits),
        },
        "classifier_self_test": classifier_self_test(),
        "conclusion": ("the shipped parquet is written once, listed once among expected files, and "
                       "existence-asserted once in a unit test; it is never read back. The weights "
                       "beside it ARE read back by artifact_loader, so this search is not vacuous."),
    }


def classifier_self_test() -> dict:
    """The classifier must call a real reader a READER, or 'no readers' proves nothing.

    Same discipline as §CM.6(iv)/§CP.7: a control must be able to fail. Each synthetic line below is the
    shape an actual consumer would take; if the classifier missed one, the headline claim would rest on
    a detector that cannot detect.
    """
    cases = {
        "bundle_files_access": 'df = pq.read_parquet(io.BytesIO(bundle.files["validation_predictions.parquet"]))',
        "read_parquet_from_bundle_path": 'pd.read_parquet(bundle.path / "validation_predictions.parquet")',
        "open_the_file": 'with open(root / "validation_predictions.parquet", "rb") as fh:',
        "files_mapping_lookup": 'raw = bundle.files["validation_predictions.parquet"]',
    }
    negatives = {
        "local_variable": "seed_validation_predictions[seed] = predictions",
        "keyword_argument": "validation_predictions=ensemble.predictions.rename(",
        "write_side": 'temporary / "validation_predictions.parquet",',
        "expected_files_list": '"validation_predictions.parquet",',
        "test_asserts_exists": 'assert (bundle.path / "validation_predictions.parquet").is_file()',
    }
    pos = {k: classify_mention(v) == "READER" for k, v in cases.items()}
    neg = {k: classify_mention(v) != "READER" for k, v in negatives.items()}
    return {"positives_detected_as_READER": pos, "all_positives_detected": all(pos.values()),
            "negatives_not_flagged_as_READER": neg, "all_negatives_clean": all(neg.values()),
            "ALL_OK": all(pos.values()) and all(neg.values())}


def tree_carrier_is_consistent() -> dict:
    """Differential control: LightGBM predicts with best_iteration and ships best_iteration trees."""
    path = REPO / "src/research_v2/baselines/lgbm_rankreg.py"
    lines = path.read_text(encoding="utf-8").split("\n")
    predict_line = {n: l.strip() for n, l in enumerate(lines, 1)
                    if "num_iteration=self.best_iteration" in l}
    best_iter = {n: l.strip() for n, l in enumerate(lines, 1)
                 if "best_iteration = getattr(booster" in l}
    return {"file": str(path.relative_to(REPO)), "predict_uses_best_iteration": predict_line,
            "best_iteration_captured_from_booster": best_iter,
            "conclusion": ("for M0/M0R both the shipped booster.txt (truncated to best_iteration, §CQ) "
                           "and the shipped validation predictions (num_iteration=best_iteration) "
                           "describe the SAME model, so tree bundles are internally consistent and the "
                           "defect is torch-specific")}


def shipped_torch_bundles() -> dict:
    """Which torch bundles exist right now, so the claim's scope is explicit and current."""
    out = {}
    for ws in ("m1_full_v1", "m2_v1"):
        root = ART / ws / "models"
        if not root.is_dir():
            continue
        for b in sorted(root.iterdir()):
            if not b.is_dir():
                continue
            pts = sorted(b.glob("seed-*.pt"))
            out[f"{ws}/{b.name}"] = {
                "n_seed_pt": len(pts),
                "has_validation_predictions_parquet": (b / "validation_predictions.parquet").is_file(),
                "files": sorted(p.name for p in b.iterdir()),
            }
    return out


def main() -> None:
    f1 = trainer_never_restores_best()
    f2 = early_stopping_runs_patience_past_peak()
    f3 = prediction_uses_in_memory_model()
    readers = parquet_reader_search()
    trees = tree_carrier_is_consistent()
    bundles = shipped_torch_bundles()

    doc = {
        "question": ("is the shipped bundle internally consistent -- do its validation_predictions and "
                     "its seed weights describe the same model? and if not, does anything consume the "
                     "inconsistent part?"),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "finding": {
            "statement": ("every shipped TORCH bundle is internally inconsistent: seed-*.pt are PEAK "
                          "weights (§CT, 18/18 byte-identical to the peak checkpoint) while "
                          "validation_predictions.parquet was produced by the LAST-EPOCH in-memory model, "
                          "which by early stopping took >= patience (10) more optimizer steps"),
            "proof_is_deductive_no_weights_loaded": True,
            "blast_radius": "ZERO -- no code reads validation_predictions.parquet back",
            "scope": "torch carriers (M1 ResidualMLP, M2 TCN); tree carriers (M0, M0R) are consistent",
            "upgrades": ("§CC.1 asserted in passing that a scalar-only fix would leave the bundle's "
                         "validation_predictions 'fighting its own weights'; this proves that fight "
                         "already exists in every shipped torch bundle, and proves it is harmless "
                         "because nothing reads the parquet"),
        },
        "fact_1_trainer_never_restores_best": f1,
        "fact_2_early_stopping_runs_patience_past_peak": f2,
        "fact_3_predictions_use_the_in_memory_model": f3,
        "blast_radius_search": readers,
        "tree_carrier_differential_control": trees,
        "shipped_torch_bundles_now": bundles,
        "source_hashes": {p: sha256_file(REPO / p)[:16] for p in TORCH_SOURCES},
        "controls": {},
        "read_only": True, "database_access": False, "weights_loaded": False,
        "pt_loaded_as_tensor": False, "checkpoint_pt_opened": False, "parquet_read": False,
        "training_performed": False, "repository_module_imported_or_executed": False,
        "processes_signalled": [], "atime_side_effect": False, "byte_reproducible": True,
    }

    # --- controls: each must be able to fail
    doc["controls"] = {
        "fact1_no_restoration_token_in_trainer": not f1["any_restoration_token_present"],
        "fact1_return_is_TrainingResult_only": any("return TrainingResult" in r for r in f1["return_statement"]),
        "fact2_patience_break_present": any("stale_epochs >=" in l for l in f2["lines"].values()),
        "fact3_both_torch_paths_predict_from_model": all(
            v["prediction_line"] for v in f3.values()),
        "fact3_both_torch_paths_source_checkpoint_from_ids_last": all(
            v["checkpoint_source_line"] for v in f3.values()),
        "parquet_has_no_reader": not readers["any_reader_found"],
        "every_mention_classified_no_unclassified": not readers["unclassified_needing_review"],
        "CONTROL_same_search_finds_the_weight_reader":
            readers["CONTROL_weight_reader_must_be_found"]["found_artifact_loader_reading_seed_pt"],
        "CONTROL_classifier_detects_a_real_reader":
            readers["classifier_self_test"]["all_positives_detected"],
        "CONTROL_classifier_does_not_overflag":
            readers["classifier_self_test"]["all_negatives_clean"],
        "CONTROL_tree_carrier_predicts_with_best_iteration":
            bool(trees["predict_uses_best_iteration"]),
        "every_shipped_torch_bundle_carries_both_parts": all(
            b["has_validation_predictions_parquet"] and b["n_seed_pt"] == 3
            for b in bundles.values()),
    }
    doc["controls"]["ALL_PASS"] = all(v for k, v in doc["controls"].items() if k != "ALL_PASS")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print("=== fact 1: trainer never restores best weights ===")
    print(f"  restoration tokens present: {f1['any_restoration_token_present']} "
          f"({f1['restoration_tokens_found'] or 'none'})")
    print(f"  return: {f1['return_statement']}")
    print("\n=== fact 2: early stopping runs >= patience past the peak ===")
    for n, l in sorted(f2["lines"].items()):
        print(f"  trainer.py:{n}: {l}")
    print("\n=== fact 3: predictions come from the in-memory model ===")
    for rel, v in f3.items():
        print(f"  {rel}")
        for n, l in v["prediction_line"].items():
            print(f"    :{n} predictions <- {l[:80]}")
        for n, l in v["checkpoint_source_line"].items():
            print(f"    :{n} checkpoint <- {l[:80]}")
    r = readers
    print("\n=== blast radius: who reads the shipped validation_predictions.parquet? ===")
    print(f"  {r['n_mentions_total']} mentions of the substring, classified by role:")
    for role, hits in r["by_role"].items():
        print(f"    {len(hits):3d}  {role}")
    print(f"  READERS found: {r['n_readers']} (must be 0)")
    st = r["classifier_self_test"]
    print(f"  classifier self-test: positives detected {st['all_positives_detected']}, "
          f"negatives clean {st['all_negatives_clean']}")
    ctrl = r["CONTROL_weight_reader_must_be_found"]
    print(f"  CONTROL - same search finds the WEIGHT reader: {ctrl['found_artifact_loader_reading_seed_pt']} "
          f"({ctrl['n_hits']} hits) -> the grep is not vacuous")
    print("\n=== differential control: tree carriers ===")
    print(f"  lgbm_rankreg predict uses best_iteration: {bool(trees['predict_uses_best_iteration'])}")
    print("\n=== shipped torch bundles now ===")
    for k, v in bundles.items():
        print(f"  {k}: {v['n_seed_pt']} seed .pt + validation_predictions.parquet="
              f"{v['has_validation_predictions_parquet']}")
    print("\n=== controls ===")
    for k, v in doc["controls"].items():
        print(f"  {'OK ' if v else '*** FAIL ***'} {k}")
    assert doc["controls"]["ALL_PASS"], "a control failed"
    print("wrote", OUT)


if __name__ == "__main__":
    main()
