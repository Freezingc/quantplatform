#!/usr/bin/env python3
"""Re-verify the evidence bundle that backs the 2026-09-10 conditional-factor review.

Run from anywhere:

    python3 docs/research/evidence/2026-09-10-conditional-factor/verify_evidence.py

It performs three checks and exits non-zero if any of them fails:

1. INTEGRITY - every vendored file still hashes to the SHA-256 recorded in
   ``provenance.json`` at the time it was copied out of the machine-local
   temporary directories and the out-of-repo artifact store.
2. NUMBERS - the figures quoted in section 1.3 of the report are recomputed
   from ``experiment/rankreg_subsets.json`` rather than trusted as text.
3. LITERATURE - every paper cited in section 5 has a versioned URL, a stated
   publication status, a stated review depth, and a content hash, so a reader
   on another machine can re-fetch the same text and confirm it.

Nothing here re-runs the original experiment. It only checks that the artifacts
committed to this branch still say what the report says they say.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent
REPORT = BUNDLE.parents[1] / "2026-09-10-conditional-factor-idea-review-v0.1.md"
TOLERANCE = 5e-7  # the report prints six decimal places

failures: list[str] = []
checks_run = 0


def record(ok: bool, label: str, detail: str = "") -> None:
    global checks_run
    checks_run += 1
    if not ok:
        failures.append(f"{label}{': ' + detail if detail else ''}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual: float, expected: float) -> bool:
    return abs(actual - expected) < TOLERANCE


def check_integrity() -> None:
    prov = json.loads((BUNDLE / "provenance.json").read_text())
    for entry in prov["files"]:
        path = BUNDLE.parents[3] / entry["repo_path"]
        if not path.exists():
            record(False, "integrity", f"missing {entry['repo_path']}")
            continue
        record(sha256(path) == entry["sha256"], "integrity",
               f"hash drift in {entry['repo_path']}")


def check_numbers() -> None:
    d = json.loads((BUNDLE / "experiment/rankreg_subsets.json").read_text())
    pooled = {k: d[k]["pooled"] for k in ("all15", "dense9", "sparse6")}

    # Section 1.3, main table: mean RankIC, delta versus the chain reference, ICIR.
    expected_pooled = {
        "dense9": (0.094058, 0.008831, 0.771237),
        "all15": (0.094376, 0.009149, 0.768173),
        "sparse6": (0.076453, -0.008774, 0.662267),
    }
    for key, (cand, delta, icir) in expected_pooled.items():
        p = pooled[key]
        record(close(p["cand"], cand), "numbers", f"{key} cand {p['cand']!r} != {cand}")
        record(close(p["delta"], delta), "numbers", f"{key} delta {p['delta']!r} != {delta}")
        record(close(p["icir"], icir), "numbers", f"{key} icir {p['icir']!r} != {icir}")

    # Both dense9 and all15 are reported over the same 945 valid days.
    record(pooled["all15"]["valid_days"] == 945 and pooled["dense9"]["valid_days"] == 945,
           "numbers", "valid_days is not 945 for both all15 and dense9")

    # Section 1.3: the marginal contribution of the six sparse factors.
    marginal = pooled["all15"]["cand"] - pooled["dense9"]["cand"]
    record(close(marginal, 0.000317821), "numbers", f"marginal {marginal!r} != 0.000317821")
    record(close(d["_verdict"]["sparse_marginal_contribution"], marginal), "numbers",
           "verdict marginal disagrees with recomputed all15-dense9")

    # Section 1.3, per-year table: all15 yearly RankIC minus dense9 yearly RankIC.
    expected_yearly = {"2022": 0.002242, "2023": 0.001549, "2024": -0.003058, "2025": 0.000662}
    for year, want in expected_yearly.items():
        got = pooled["all15"]["yearly"][year] - pooled["dense9"]["yearly"][year]
        record(close(got, want), "numbers", f"year {year} marginal {got!r} != {want}")

    # Section 6.1: sparse6 absolute RankIC is positive while its delta is negative.
    record(pooled["sparse6"]["cand"] > 0 and pooled["sparse6"]["delta"] < 0, "numbers",
           "sparse6 sign pattern changed")

    # Section 6.2: the four-configuration robustness script fits all 15 columns only,
    # so it cannot by itself establish the sparse factors' marginal contribution.
    text = (BUNDLE / "experiment/run_config_robustness.py").read_text()
    record("dense9" not in text and "sparse6" not in text, "numbers",
           "run_config_robustness.py unexpectedly contains a subset ablation")

    # Section 6.1: the shipped verdict is a fixed +/-0.001 heuristic, not a test.
    script = (BUNDLE / "experiment/run_rankreg_subsets.py").read_text()
    record("0.001" in script, "numbers", "verdict threshold no longer present in the script")


def check_literature() -> None:
    manifest = json.loads((BUNDLE / "literature/manifest.json").read_text())
    papers = manifest["papers"]
    record(len(papers) == 14, "literature", f"expected 14 papers, found {len(papers)}")
    for p in papers:
        for field in ("id", "title", "url", "publication_status", "review_depth",
                      "retrieved_on", "retrieved_content_sha256"):
            record(bool(p.get(field)), "literature", f"{p.get('id')} missing {field}")
        record(len(p.get("retrieved_content_sha256", "")) == 64, "literature",
               f"{p.get('id')} hash is not a sha256 hex digest")

    checks = json.loads((BUNDLE / "literature/url_checks.json").read_text())
    record(len(checks) == len(papers), "literature", "url check count != paper count")
    record(all(c.get("http_status") == 200 for c in checks), "literature",
           "a source URL was not reachable at delivery time")

    report_text = REPORT.read_text()
    for p in papers:
        record(p["url"] in report_text, "literature", f"{p['id']} URL not cited in the report")


def main() -> int:
    if not REPORT.exists():
        print(f"FATAL: report not found at {REPORT}")
        return 2
    check_integrity()
    check_numbers()
    check_literature()

    if failures:
        print(f"FAIL: {len(failures)} of {checks_run} checks did not hold")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"PASS: {checks_run} checks held")
    print("  integrity  - vendored artifacts match their recorded SHA-256")
    print("  numbers    - report section 1.3 recomputes from rankreg_subsets.json")
    print("  literature - 14 sources are versioned, hashed, and cited")
    return 0


if __name__ == "__main__":
    sys.exit(main())
