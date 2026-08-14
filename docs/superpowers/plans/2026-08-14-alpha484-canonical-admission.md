# Alpha484 Canonical Admission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate `yeslab_484_tushare_shsz_v1` using the full Tushare SH/SZ cross-section, then emit a read-only admission-review artifact without changing the historical factor, formal nine, database, model inputs, or locked test.

**Architecture:** The evaluator obtains full daily Tushare SH/SZ rows by open date rather than a YesLab code list, applies the existing CSAD calculator to that complete panel, and joins canonical V2 label/universe keys only after calculation. It freezes direction from training, confirms in 2025, measures coverage and formal-nine similarity, then reports a non-publishing decision.

**Tech Stack:** Python 3.11, pandas, numpy, Tushare, DolphinDB read-only sessions, pytest, Ruff.

---

### Task 1: Candidate Identity and Date Guard

**Files:**
- Create: `research_v2/scripts/evaluate_alpha484_canonical.py`
- Create: `research_v2/tests/unit/scripts/test_evaluate_alpha484_canonical.py`

- [ ] **Step 1: Write failing tests**

```python
import pytest

import evaluate_alpha484_canonical as evaluation


def test_candidate_identity_is_isolated() -> None:
    assert evaluation.CANDIDATE_ID == "yeslab_484_tushare_shsz_v1"
    assert evaluation.SOURCE_FACTOR_ID == "yeslab_484"
    assert evaluation.CANDIDATE_ID != evaluation.SOURCE_FACTOR_ID


def test_parse_args_rejects_2026() -> None:
    with pytest.raises(SystemExit):
        evaluation.parse_args(["--end", "2026-01-01", "--output", "/tmp/out"])
```

- [ ] **Step 2: Verify RED**

Run:

```bash
source /data/share/miniconda3/etc/profile.d/conda.sh
conda activate quantyd
PYTHONPATH=research_v2/scripts:research_v2/src python -m pytest -q research_v2/tests/unit/scripts/test_evaluate_alpha484_canonical.py -k 'identity or 2026'
```

Expected: import failure because the evaluator module is absent.

- [ ] **Step 3: Implement the minimal contract**

```python
CANDIDATE_ID = "yeslab_484_tushare_shsz_v1"
SOURCE_FACTOR_ID = "yeslab_484"
TRAIN_START = date(2020, 1, 2)
TRAIN_END = date(2024, 12, 31)
VALIDATION_START = date(2025, 1, 2)
VALIDATION_END = date(2025, 12, 31)
LOCKED_TEST_START = date(2026, 1, 1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=TRAIN_START)
    parser.add_argument("--end", type=date.fromisoformat, default=VALIDATION_END)
    parser.add_argument("--cache-dir", type=Path, default=Path("research_v2/artifacts/tushare_cache"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.start > args.end or args.end >= LOCKED_TEST_START:
        parser.error("evaluation range must end before the 2026 locked test")
    return args
```

Use only `REPLAY_DEFINITIONS[SOURCE_FACTOR_ID]` for the calculation and write
`CANDIDATE_ID` to all emitted evidence.

- [ ] **Step 4: Verify GREEN and commit**

Run:

```bash
PYTHONPATH=research_v2/scripts:research_v2/src python -m pytest -q research_v2/tests/unit/scripts/test_evaluate_alpha484_canonical.py -k 'identity or 2026'
ruff check research_v2/scripts/evaluate_alpha484_canonical.py research_v2/tests/unit/scripts/test_evaluate_alpha484_canonical.py
git add research_v2/scripts/evaluate_alpha484_canonical.py research_v2/tests/unit/scripts/test_evaluate_alpha484_canonical.py
git commit -m "define alpha484 canonical candidate"
```

Expected: focused tests pass and Ruff is clean before the commit.

### Task 2: Full Tushare Market Boundary

**Files:**
- Modify: `research_v2/scripts/audit_formula_replay_tushare.py`
- Modify: `research_v2/tests/unit/scripts/test_audit_formula_replay_tushare.py`
- Modify: `research_v2/scripts/evaluate_alpha484_canonical.py`
- Modify: `research_v2/tests/unit/scripts/test_evaluate_alpha484_canonical.py`

- [ ] **Step 1: Write failing market-scope tests**

```python
def test_normalize_full_tushare_market_excludes_bj() -> None:
    raw = pd.DataFrame({
        "trade_date": ["20200102", "20200102", "20200102"],
        "ts_code": ["000001.SZ", "600000.SH", "430001.BJ"],
        "open": [1.0, 2.0, 3.0], "high": [1.1, 2.1, 3.1],
        "low": [0.9, 1.9, 2.9], "close": [1.0, 2.0, 3.0],
        "pre_close": [1.0, 2.0, 3.0], "vol": [1.0, 1.0, 1.0],
        "amount": [1.0, 1.0, 1.0], "pct_chg": [0.0, 0.0, 0.0],
    })
    result = audit.normalize_full_tushare_market(raw)
    assert set(result["ts_code"]) == {"000001.SZ", "600000.SH"}


def test_validate_full_market_dates_rejects_missing_open_date() -> None:
    market = pd.DataFrame({"trade_date": [pd.Timestamp("2025-01-02")]})
    with pytest.raises(ValueError, match="full SH/SZ"):
        evaluation.validate_full_market_dates(market, (date(2025, 1, 2), date(2025, 1, 3)))
```

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=research_v2/scripts:research_v2/src python -m pytest -q research_v2/tests/unit/scripts/test_audit_formula_replay_tushare.py research_v2/tests/unit/scripts/test_evaluate_alpha484_canonical.py -k 'normalize_full or full_market_dates'
```

Expected: failure because both helpers are absent.

- [ ] **Step 3: Implement canonical full-market fetching**

Add this public helper to `audit_formula_replay_tushare.py`:

```python
def normalize_full_tushare_market(raw_market: pd.DataFrame) -> pd.DataFrame:
    codes = tuple(sorted(code for code in raw_market["ts_code"].astype(str).unique()
                         if code.endswith((".SH", ".SZ"))))
    if not codes:
        raise ValueError("Tushare daily result has no SH/SZ market rows")
    return normalize_tushare_market(raw_market, codes=codes)
```

Add `fetch_full_tushare_shsz_market`. It iterates only the open dates supplied
by `_query_open_dates`, reuses date-wise `daily_*.json` and `adj_factor_*.json`
cache records, merges adjustment factors before normalization, and fails if an
expected date has no SH/SZ rows. Do not call `_query_stock_market_codes` and do
not use label codes as CSAD peer membership.

The evaluator obtains the preheat start through `_calculator_market_start`,
fetches that one complete market panel, creates one `TushareMarketSession`, and
calls `_replay_factor` once. `validate_full_market_dates` compares normalized
trade dates to the expected open dates.

- [ ] **Step 4: Verify GREEN and commit**

Run:

```bash
PYTHONPATH=research_v2/scripts:research_v2/src python -m pytest -q research_v2/tests/unit/scripts/test_audit_formula_replay_tushare.py research_v2/tests/unit/scripts/test_evaluate_alpha484_canonical.py -k 'normalize_full or full_market_dates'
ruff check research_v2/scripts/audit_formula_replay_tushare.py research_v2/scripts/evaluate_alpha484_canonical.py
git add research_v2/scripts/audit_formula_replay_tushare.py research_v2/tests/unit/scripts/test_audit_formula_replay_tushare.py research_v2/scripts/evaluate_alpha484_canonical.py research_v2/tests/unit/scripts/test_evaluate_alpha484_canonical.py
git commit -m "fetch alpha484 canonical full market"
```

Expected: focused tests pass and Ruff is clean.

### Task 3: Coverage, Training Direction, and Admission Gates

**Files:**
- Modify: `research_v2/scripts/evaluate_alpha484_canonical.py`
- Modify: `research_v2/tests/unit/scripts/test_evaluate_alpha484_canonical.py`

- [ ] **Step 1: Write failing coverage and review tests**

```python
def test_summarize_coverage_rejects_month_below_seventy_percent() -> None:
    labels = pd.DataFrame({
        "signal_date": pd.to_datetime(["2025-01-02"] * 10 + ["2025-02-03"] * 10),
        "ts_code": [f"0000{i:02d}.SZ" for i in range(10)] * 2,
    })
    values = labels.iloc[:17].rename(columns={"signal_date": "trade_date"})
    values["replay_value"] = 1.0
    assert evaluation.summarize_coverage(values, labels)["status"] == "FAIL"


def test_build_review_never_makes_candidate_a_model_input() -> None:
    review = evaluation.build_review(
        training={"signed_mean_rank_ic": .04, "signed_icir": .60, "directional_day_share": .70, "annual_gate_years": 5},
        validation={"signed_mean_rank_ic": .03, "signed_icir": .55, "directional_day_share": .60},
        coverage={"status": "PASS"}, train_similarity={"status": "PASS"}, validation_similarity={"status": "PASS"},
    )
    assert review["status"] == "READY_FOR_EXPLICIT_ADMISSION_REVIEW"
    assert review["model_eligible"] is False
    assert review["formal_factor_set_changed"] is False
```

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=research_v2/scripts:research_v2/src python -m pytest -q research_v2/tests/unit/scripts/test_evaluate_alpha484_canonical.py -k 'coverage or build_review'
```

Expected: failure because the coverage and review functions are absent.

- [ ] **Step 3: Implement metrics and the non-publishing review**

Implement `freeze_training_direction`, `summarize_coverage`, and
`build_review`. Coverage must left-join finite candidate values to every valid
canonical SH/SZ label key and retain zero-coverage dates. It passes only at
median daily ratio >= .80, p05 daily ratio >= .70, and monthly minimum >= .70.

Use `compute_daily_rank_ic`, `summarize_rank_ic`, and
`compute_pair_daily_spearman` from `evaluate_tushare_scope_validation`.
Training freezes direction before yearly gates and 2025. Require signed mean
RankIC > .02, signed ICIR >= .50, directional share >= .55, four of five
annual training passes, and median absolute daily Spearman < .90 against all
nine formal factors in both periods.

`build_review` returns `READY_FOR_EXPLICIT_ADMISSION_REVIEW` only when every
gate passes, otherwise `REJECT_CANONICAL_GATE`. It always sets
`database_writes`, `formal_factor_set_changed`, `model_eligible`,
`model_training_started`, and `locked_test_read` to `False`.

- [ ] **Step 4: Verify GREEN and commit**

Run:

```bash
PYTHONPATH=research_v2/scripts:research_v2/src python -m pytest -q research_v2/tests/unit/scripts/test_evaluate_alpha484_canonical.py -k 'coverage or build_review'
ruff check research_v2/scripts/evaluate_alpha484_canonical.py research_v2/tests/unit/scripts/test_evaluate_alpha484_canonical.py
git add research_v2/scripts/evaluate_alpha484_canonical.py research_v2/tests/unit/scripts/test_evaluate_alpha484_canonical.py
git commit -m "gate alpha484 canonical evidence"
```

Expected: focused tests pass and Ruff is clean.

### Task 4: Run, Record, and Verify the Canonical Audit

**Files:**
- Create: `research_v2/artifacts/diagnostics/tushare_alpha484_canonical_v1/daily_rank_ic.csv`
- Create: `research_v2/artifacts/diagnostics/tushare_alpha484_canonical_v1/coverage_summary.json`
- Create: `research_v2/artifacts/diagnostics/tushare_alpha484_canonical_v1/similarity_summary.json`
- Create: `research_v2/artifacts/diagnostics/tushare_alpha484_canonical_v1/quality_summary.json`
- Create: `research_v2/artifacts/diagnostics/tushare_alpha484_canonical_v1/admission_review.json`
- Modify: `research_v2/artifacts/diagnostics/tushare_shsz_factor_audit_coverage_v1.json`

- [ ] **Step 1: Add `run_evaluation` output**

For fixed 2020–2024 and 2025 windows, load V2 latest universe/label inputs,
evaluate the full canonical candidate panel, and calculate nine same-key
formal-factor similarities. Output the five declared artifacts. The emitted
factor ID is always `yeslab_484_tushare_shsz_v1`, and old `yeslab_484` values
are never queried or overwritten as evidence.

- [ ] **Step 2: Run the bounded audit**

Run:

```bash
source /data/share/miniconda3/etc/profile.d/conda.sh
conda activate quantyd
set -a
source .env
set +a
PYTHONPATH=research_v2/scripts:research_v2/src python -u research_v2/scripts/evaluate_alpha484_canonical.py --output research_v2/artifacts/diagnostics/tushare_alpha484_canonical_v1
```

Expected: zero exit status, no database writes, and no 2026 read.

- [ ] **Step 3: Verify artifact completeness, record the ledger decision, and commit**

Assert 1,212 training RankIC rows, 243 validation rows, nine similarity edges
per period, no model-eligible output, and all safety flags false. Add an
`alpha484_tushare_shsz_canonical_v1` ledger entry without modifying the
historical Alpha484 entry or `formal_admission_factor_ids`.

Run:

```bash
python -m json.tool research_v2/artifacts/diagnostics/tushare_shsz_factor_audit_coverage_v1.json >/dev/null
git diff --check
PYTHONPATH=research_v2/scripts:research_v2/src python -m pytest -q
ruff check research_v2/scripts/audit_formula_replay_tushare.py research_v2/scripts/evaluate_alpha484_canonical.py research_v2/tests/unit/scripts/test_audit_formula_replay_tushare.py research_v2/tests/unit/scripts/test_evaluate_alpha484_canonical.py
git add -f research_v2/artifacts/diagnostics/tushare_alpha484_canonical_v1
git add research_v2/scripts/audit_formula_replay_tushare.py research_v2/scripts/evaluate_alpha484_canonical.py research_v2/tests/unit/scripts/test_audit_formula_replay_tushare.py research_v2/tests/unit/scripts/test_evaluate_alpha484_canonical.py research_v2/artifacts/diagnostics/tushare_shsz_factor_audit_coverage_v1.json
git commit -m "audit alpha484 canonical candidate"
```

Expected: full suite passes, Ruff is clean, the formal nine is unchanged, and
the candidate remains outside model inputs pending explicit admission.
