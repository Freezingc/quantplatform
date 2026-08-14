# Alpha95 Relative Amount Volatility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate 10, 20, and 40-day scale-normalized Alpha95 candidates from canonical Tushare SH/SZ data without changing any stored factor or model input.

**Architecture:** A focused, read-only evaluator will compute `STD(amount, N) / MA(amount, N)` for three windows, replay the formal nine from the same canonical market panel, and calculate RankIC plus cross-sectional similarity. Selection is based only on 2020-2024; 2025 is a single confirmation window.

**Tech Stack:** Python 3.11, pandas, numpy, DolphinDB read-only sessions, Tushare cache, pytest, ruff.

---

### Task 1: Add Formula Tests And Implementation

**Files:**
- Create: `research_v2/scripts/evaluate_alpha95_relative_amount_volatility.py`
- Create: `research_v2/tests/unit/scripts/test_evaluate_alpha95_relative_amount_volatility.py`

- [x] **Step 1: Write failing formula and leakage tests**

```python
import numpy as np
import pandas as pd


def make_market(values: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=len(values))
    return pd.DataFrame(
        {
            "trade_date": dates.repeat(2),
            "ts_code": ["000001.SZ", "600000.SH"] * len(values),
            "amount": np.repeat(values, 2),
        }
    )


def test_relative_amount_volatility_uses_complete_trailing_window() -> None:
    market = pd.DataFrame(
        {
            "trade_date": pd.date_range("2025-01-01", periods=4).repeat(2),
            "ts_code": ["000001.SZ", "600000.SH"] * 4,
            "amount": [1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0],
        }
    )
    result = evaluation.compute_relative_amount_volatility(market, window=3)
    assert result.iloc[:2].isna().all().all()
    np.testing.assert_allclose(result.iloc[2].to_numpy(), [0.5, 0.5])


def test_relative_amount_volatility_does_not_use_future_amount() -> None:
    base = make_market([1.0, 2.0, 3.0, 4.0, 5.0])
    changed = base.copy()
    changed.loc[changed["trade_date"] == pd.Timestamp("2025-01-05"), "amount"] = 999.0
    assert evaluation.compute_relative_amount_volatility(base, 3).iloc[2].equals(
        evaluation.compute_relative_amount_volatility(changed, 3).iloc[2]
    )
```

- [x] **Step 2: Run the test and verify failure**

Run: `PYTHONPATH=research_v2/scripts:research_v2/src pytest -q research_v2/tests/unit/scripts/test_evaluate_alpha95_relative_amount_volatility.py`

Expected: FAIL because the evaluator module is absent.

- [x] **Step 3: Implement the isolated formula helper**

```python
CANDIDATE_WINDOWS = (10, 20, 40)


def compute_relative_amount_volatility(market: pd.DataFrame, window: int) -> pd.DataFrame:
    required = {"trade_date", "ts_code", "amount"}
    if missing := sorted(required.difference(market.columns)):
        raise ValueError(f"market frame is missing columns: {missing}")
    amount = market.pivot(index="trade_date", columns="ts_code", values="amount").sort_index()
    deviation = amount.rolling(window=window, min_periods=window).std(ddof=1)
    baseline = amount.rolling(window=window, min_periods=window).mean()
    return deviation.div(baseline.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
```

- [x] **Step 4: Run the test and verify pass**

Run: `PYTHONPATH=research_v2/scripts:research_v2/src pytest -q research_v2/tests/unit/scripts/test_evaluate_alpha95_relative_amount_volatility.py`

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add research_v2/scripts/evaluate_alpha95_relative_amount_volatility.py research_v2/tests/unit/scripts/test_evaluate_alpha95_relative_amount_volatility.py
git commit -m "add alpha95 relative amount volatility formula"
```

### Task 2: Add Training-Only Gate And Selection Tests

**Files:**
- Modify: `research_v2/scripts/evaluate_alpha95_relative_amount_volatility.py`
- Modify: `research_v2/tests/unit/scripts/test_evaluate_alpha95_relative_amount_volatility.py`

- [x] **Step 1: Write failing training-selection tests**

```python
def test_select_training_candidate_requires_every_gate() -> None:
    rows = {
        "v10": {"signed_mean_rank_ic": 0.04, "signed_icir": 0.60, "directional_day_share": 0.70, "annual_double_gate_pass_years": 4, "max_similarity": 0.40},
        "v20": {"signed_mean_rank_ic": 0.06, "signed_icir": 0.70, "directional_day_share": 0.70, "annual_double_gate_pass_years": 3, "max_similarity": 0.30},
    }
    assert evaluation.select_training_candidate(rows) == "v10"


def test_select_training_candidate_breaks_icir_tie_by_lower_similarity() -> None:
    rows = {
        "v10": {"signed_mean_rank_ic": 0.04, "signed_icir": 0.60, "directional_day_share": 0.70, "annual_double_gate_pass_years": 4, "max_similarity": 0.30},
        "v20": {"signed_mean_rank_ic": 0.05, "signed_icir": 0.60, "directional_day_share": 0.70, "annual_double_gate_pass_years": 4, "max_similarity": 0.20},
    }
    assert evaluation.select_training_candidate(rows) == "v20"
```

- [x] **Step 2: Run the selection test and verify failure**

Run: `PYTHONPATH=research_v2/scripts:research_v2/src pytest -q research_v2/tests/unit/scripts/test_evaluate_alpha95_relative_amount_volatility.py -k select_training_candidate`

Expected: FAIL because `select_training_candidate` is undefined.

- [x] **Step 3: Implement explicit gates and selection**

```python
from collections.abc import Mapping


def passes_training_gate(row: Mapping[str, float]) -> bool:
    return (
        row["signed_mean_rank_ic"] > 0.02
        and row["signed_icir"] >= 0.50
        and row["directional_day_share"] >= 0.55
        and row["annual_double_gate_pass_years"] >= 4
        and row["max_similarity"] < 0.90
    )


def select_training_candidate(rows: Mapping[str, Mapping[str, float]]) -> str | None:
    eligible = [(candidate_id, row) for candidate_id, row in rows.items() if passes_training_gate(row)]
    if not eligible:
        return None
    return max(eligible, key=lambda item: (item[1]["signed_icir"], -item[1]["max_similarity"], item[0]))[0]
```

- [x] **Step 4: Run the focused test file**

Run: `PYTHONPATH=research_v2/scripts:research_v2/src pytest -q research_v2/tests/unit/scripts/test_evaluate_alpha95_relative_amount_volatility.py`

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add research_v2/scripts/evaluate_alpha95_relative_amount_volatility.py research_v2/tests/unit/scripts/test_evaluate_alpha95_relative_amount_volatility.py
git commit -m "add alpha95 training candidate selection"
```

### Task 3: Build Read-Only Canonical Evaluator

**Files:**
- Modify: `research_v2/scripts/evaluate_alpha95_relative_amount_volatility.py`
- Modify: `research_v2/tests/unit/scripts/test_evaluate_alpha95_relative_amount_volatility.py`

- [x] **Step 1: Write failing locked-test and review-contract tests**

```python
def test_parse_args_rejects_locked_test_dates() -> None:
    with pytest.raises(SystemExit):
        evaluation.parse_args(["--end", "2026-01-01", "--output", "/tmp/result"])


def test_build_review_rejects_validation_without_changing_formal_set() -> None:
    review = evaluation.build_review(
        selected_id="yeslab_163_relative_amount_volatility_20_v1",
        training={"selected": True},
        validation={"signed_mean_rank_ic": 0.01, "signed_icir": 0.60, "directional_day_share": 0.70, "max_similarity": 0.20},
    )
    assert review["status"] == "REJECT_VALIDATION_GATE"
    assert review["formal_factor_set_changed"] is False
    assert review["locked_test_read"] is False
```

- [x] **Step 2: Run the contract test and verify failure**

Run: `PYTHONPATH=research_v2/scripts:research_v2/src pytest -q research_v2/tests/unit/scripts/test_evaluate_alpha95_relative_amount_volatility.py -k 'parse_args or build_review'`

Expected: FAIL because parser and review builder are absent.

- [x] **Step 3: Implement the evaluator using existing replay helpers**

Create `parse_args`, `run_evaluation`, and `build_review`. Use these fixed constants:

```python
FORMAL_FACTOR_IDS = (
    "yeslab_380", "yeslab_431", "yeslab_437", "yeslab_155", "yeslab_98",
    "yeslab_94", "yeslab_376", "yeslab_396", "yeslab_442",
)
TRAIN_START = date(2020, 1, 2)
TRAIN_END = date(2024, 12, 31)
VALIDATION_START = date(2025, 1, 2)
VALIDATION_END = date(2025, 12, 31)
LOCKED_TEST_START = date(2026, 1, 1)
```

Load V2 universe/label data read-only, fetch Tushare daily amount through the cache-aware market helper, replay `FORMAL_FACTOR_IDS` with `TushareMarketSession` and `replay._replay_factor`, and compute all pairwise daily Spearman values with `compute_pair_daily_spearman`. Write `daily_rank_ic.csv`, `quality_summary.json`, `similarity_summary.json`, and `admission_review.json`. Every output records `database_writes: false`, `formal_factor_set_changed: false`, `model_training_started: false`, and `locked_test_read: false`.

- [x] **Step 4: Run unit tests and lint**

Run: `PYTHONPATH=research_v2/scripts:research_v2/src pytest -q research_v2/tests/unit/scripts/test_evaluate_alpha95_relative_amount_volatility.py`

Expected: PASS.

Run: `ruff check research_v2/scripts/evaluate_alpha95_relative_amount_volatility.py research_v2/tests/unit/scripts/test_evaluate_alpha95_relative_amount_volatility.py`

Expected: `All checks passed!`.

- [x] **Step 5: Commit**

```bash
git add research_v2/scripts/evaluate_alpha95_relative_amount_volatility.py research_v2/tests/unit/scripts/test_evaluate_alpha95_relative_amount_volatility.py
git commit -m "evaluate alpha95 relative amount volatility"
```

### Task 4: Run Research Audit And Update Ledger

**Files:**
- Create: `research_v2/artifacts/diagnostics/tushare_alpha95_relative_amount_volatility_v1/daily_rank_ic.csv`
- Create: `research_v2/artifacts/diagnostics/tushare_alpha95_relative_amount_volatility_v1/quality_summary.json`
- Create: `research_v2/artifacts/diagnostics/tushare_alpha95_relative_amount_volatility_v1/similarity_summary.json`
- Create: `research_v2/artifacts/diagnostics/tushare_alpha95_relative_amount_volatility_v1/admission_review.json`
- Modify: `research_v2/artifacts/diagnostics/tushare_shsz_factor_audit_coverage_v1.json`

- [x] **Step 1: Execute the bounded canonical run**

```bash
source /data/share/miniconda3/etc/profile.d/conda.sh
conda activate quantyd
set -a
source .env
set +a
PYTHONPATH=research_v2/scripts:research_v2/src python -u research_v2/scripts/evaluate_alpha95_relative_amount_volatility.py \
  --output research_v2/artifacts/diagnostics/tushare_alpha95_relative_amount_volatility_v1
```

Expected: an exit code of zero, three candidate summaries, a training-only selection, and a 2025 validation status.

- [x] **Step 2: Verify completeness before recording the decision**

```python
import json
from pathlib import Path


root = Path("research_v2/artifacts/diagnostics/tushare_alpha95_relative_amount_volatility_v1")
review = json.loads((root / "admission_review.json").read_text())
assert review["database_writes"] is False
assert review["formal_factor_set_changed"] is False
assert review["model_training_started"] is False
assert review["locked_test_read"] is False
assert len(review["candidate_ids"]) == 3
```

Expected: all assertions pass.

- [x] **Step 3: Update the quality ledger without admission or publication**

Add `alpha95_relative_amount_volatility_v1` to `tushare_shsz_factor_audit_coverage_v1.json` with each candidate's training and validation metrics, selected-or-rejected status, artifact paths, and all safety flags. Do not modify `formal_admission_factor_ids`.

- [ ] **Step 4: Run final checks, commit, and synchronize**

Run: `python -m json.tool research_v2/artifacts/diagnostics/tushare_shsz_factor_audit_coverage_v1.json >/dev/null && git diff --check && python -m pytest -q`

Expected: valid JSON, no whitespace errors, and a passing suite.

Force-add the four ignored artifacts, commit with `audit alpha95 relative amount volatility`, then use the authenticated GitHub Git Data API to update remote `main` with `force: false` if standard `git push` remains blocked by TLS. Verify the remote head and artifact paths after update.
