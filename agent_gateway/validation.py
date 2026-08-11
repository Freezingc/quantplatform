from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def evaluate_factor_summary(summary: dict, validation: dict | None = None) -> dict:
    """Return an actionable, QuantGPT-style factor evaluation report.

    The input is intentionally loose so it can consume rows from the existing
    factor analyzer as well as English/API-style keys.
    """
    metrics = _extract_metrics(summary)
    validation = validation or {}

    components = {
        "quality": _quality_score(metrics),
        "stability": _stability_score(metrics, validation),
        "tradability": _tradability_score(metrics),
        "cost_sensitivity": _cost_sensitivity_score(metrics),
        "out_of_sample": _out_of_sample_score(validation),
    }
    score = round(
        components["quality"] * 0.30
        + components["stability"] * 0.25
        + components["tradability"] * 0.20
        + components["cost_sensitivity"] * 0.10
        + components["out_of_sample"] * 0.15,
        2,
    )
    diagnostics = _diagnose(metrics, components, validation)
    recommendation_code = _recommendation_code(score, diagnostics)

    return {
        "score": score,
        "grade": _grade(score),
        "recommendation": _recommendation_label(recommendation_code),
        "recommendation_code": recommendation_code,
        "components": {key: round(value, 2) for key, value in components.items()},
        "diagnostic_reasons": diagnostics,
        "improvement_suggestions": _suggestions(metrics, components, validation),
        "risk_flags": _risk_flags(metrics, components, validation),
        "metrics_used": metrics,
        "threshold_notes": {
            "quality": "IC, ICIR, Sharpe, and annual return strength.",
            "stability": "ICIR, win rate, drawdown, and rolling validation.",
            "tradability": "Return, drawdown, turnover, and signal direction.",
            "cost_sensitivity": "Penalizes high turnover and weak return per turnover.",
            "out_of_sample": "Rolling validation and placebo evidence when supplied.",
        },
    }


def score_factor_summary(summary: dict, validation: dict | None = None) -> dict:
    """Backward-compatible wrapper for the unified factor evaluation layer."""
    return evaluate_factor_summary(summary, validation)


def rolling_validation(factor_long: pd.DataFrame, forward_return_col: str = "forward_return") -> dict:
    """Year-by-year rank IC validation for long factor data."""
    required = {"date", "factor", forward_return_col}
    missing = required - set(factor_long.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    df = factor_long.dropna(subset=["date", "factor", forward_return_col]).copy()
    df["date"] = pd.to_datetime(df["date"])
    rows = []
    for year, year_df in df.groupby(df["date"].dt.year):
        daily_ic = (
            year_df.groupby("date")
            .apply(lambda group: group["factor"].corr(group[forward_return_col], method="spearman"))
            .dropna()
        )
        ic_std = daily_ic.std(ddof=1)
        rows.append({
            "year": int(year),
            "rank_ic_mean": float(daily_ic.mean()) if len(daily_ic) else np.nan,
            "rank_ic_ir": float(daily_ic.mean() / ic_std) if len(daily_ic) > 1 and ic_std else np.nan,
            "days": int(len(daily_ic)),
        })
    valid = [row for row in rows if not np.isnan(row["rank_ic_mean"])]
    signs = {np.sign(row["rank_ic_mean"]) for row in valid if row["rank_ic_mean"] != 0}
    return {
        "windows": rows,
        "passed": bool(valid) and len(signs) <= 1,
        "reason": "consistent_ic_sign" if bool(valid) and len(signs) <= 1 else "unstable_or_insufficient_ic",
    }


def placebo_test(factor_long: pd.DataFrame, forward_return_col: str = "forward_return", seed: int = 7) -> dict:
    required = {"date", "factor", forward_return_col}
    missing = required - set(factor_long.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    df = factor_long.dropna(subset=["date", "factor", forward_return_col]).copy()
    if df.empty:
        return {"passed": False, "reason": "empty_data"}
    real_ic = _daily_rank_ic(df, forward_return_col).mean()
    rng = np.random.default_rng(seed)
    placebo = []
    for _ in range(20):
        shuffled = df.copy()
        shuffled["factor"] = rng.permutation(shuffled["factor"].to_numpy())
        placebo.append(_daily_rank_ic(shuffled, forward_return_col).mean())
    threshold = float(np.nanpercentile(placebo, 95))
    return {
        "real_rank_ic": float(real_ic),
        "placebo_95pct": threshold,
        "passed": bool(real_ic > threshold),
    }


def _extract_metrics(summary: dict) -> dict[str, float | None]:
    return {
        "ic_mean": _first_number(summary, ["RankIC\u5747\u503c", "rank_ic_mean", "RankIC", "IC\u5747\u503c", "ic_mean", "rank_ic"]),
        "ic_ir": _first_number(summary, ["\u5e74\u5316ICIR", "RankICIR", "ICIR", "ic_ir"]),
        "annual_return": _first_number(summary, ["\u5e74\u5316\u6536\u76ca", "annual_return_ratio", "annual_return", "return_ratio", "\u6536\u76ca\u7387"]),
        "sharpe": _first_number(summary, ["\u590f\u666e\u6bd4\u7387", "sharp_ratio", "sharpe_ratio", "sharpe"]),
        "max_drawdown": abs(_first_number(summary, ["\u6700\u5927\u56de\u64a4", "max_drawdown"], default=0.0) or 0.0),
        "win_rate": _normalize_rate(_first_number(summary, ["\u80dc\u7387", "win_percent", "win_rate"], default=None)),
        "turnover": abs(_first_number(summary, ["\u6362\u624b\u7387", "turnover"], default=0.2) or 0.0),
        "monotonicity": _first_number(summary, ["\u5355\u8c03\u6027", "monotonicity", "monotonicity_score"], default=None),
        "concentration": _first_number(summary, ["\u96c6\u4e2d\u5ea6", "concentration"], default=None),
    }


def _quality_score(metrics: dict[str, float | None]) -> float:
    ic = abs(metrics["ic_mean"] or 0.0)
    ic_ir = abs(metrics["ic_ir"] or 0.0)
    sharpe = max(metrics["sharpe"] or 0.0, 0.0)
    annual = max(metrics["annual_return"] or 0.0, 0.0)
    return (
        _clip(ic / 0.05) * 35
        + _clip(ic_ir / 1.0) * 30
        + _clip(sharpe / 1.5) * 20
        + _clip(annual / 0.20) * 15
    )


def _stability_score(metrics: dict[str, float | None], validation: dict) -> float:
    ic_ir = abs(metrics["ic_ir"] or 0.0)
    win_rate = metrics["win_rate"]
    max_dd = metrics["max_drawdown"] or 0.0
    rolling = validation.get("rolling_validation") or validation.get("rolling") or {}
    rolling_score = 70.0 if not rolling else (100.0 if rolling.get("passed") else 25.0)
    win_score = 50.0 if win_rate is None else _clip((win_rate - 0.45) / 0.20) * 100
    drawdown_score = max(0.0, 100.0 - _clip(max_dd / 0.35) * 100)
    return _clip(ic_ir / 1.0) * 35 + win_score * 0.20 + drawdown_score * 0.25 + rolling_score * 0.20


def _tradability_score(metrics: dict[str, float | None]) -> float:
    annual = max(metrics["annual_return"] or 0.0, 0.0)
    max_dd = metrics["max_drawdown"] or 0.0
    turnover = metrics["turnover"] or 0.0
    drawdown_score = max(0.0, 100.0 - _clip(max_dd / 0.30) * 100)
    direction_score = 100.0 if (metrics["ic_mean"] or 0.0) != 0 else 0.0
    return (
        _clip(annual / 0.18) * 35
        + drawdown_score * 0.30
        + _turnover_sweet_spot_score(turnover) * 0.25
        + direction_score * 0.10
    )


def _cost_sensitivity_score(metrics: dict[str, float | None]) -> float:
    turnover = metrics["turnover"] or 0.0
    annual = max(metrics["annual_return"] or 0.0, 0.0)
    if turnover <= 0:
        return 75.0
    return_efficiency = annual / max(turnover, 0.05)
    turnover_penalty = max(0.0, 100.0 - _clip(max(turnover - 0.35, 0.0) / 0.65) * 100)
    efficiency_score = _clip(return_efficiency / 0.30) * 100
    return turnover_penalty * 0.65 + efficiency_score * 0.35


def _out_of_sample_score(validation: dict) -> float:
    if not validation:
        return 50.0
    scores = []
    rolling = validation.get("rolling_validation") or validation.get("rolling")
    placebo = validation.get("placebo") or validation.get("placebo_test")
    if rolling is not None:
        scores.append(100.0 if rolling.get("passed") else 25.0)
    if placebo is not None:
        scores.append(100.0 if placebo.get("passed") else 25.0)
    return float(np.mean(scores)) if scores else 50.0


def _diagnose(metrics: dict[str, float | None], components: dict[str, float], validation: dict) -> list[str]:
    reasons: list[str] = []
    if abs(metrics["ic_mean"] or 0.0) >= 0.03:
        reasons.append("IC signal is meaningful.")
    elif abs(metrics["ic_mean"] or 0.0) < 0.01:
        reasons.append("IC is weak; signal may be noise.")

    if abs(metrics["ic_ir"] or 0.0) >= 0.8:
        reasons.append("ICIR indicates acceptable stability.")
    elif abs(metrics["ic_ir"] or 0.0) < 0.3:
        reasons.append("ICIR is low; performance is unstable.")

    if (metrics["turnover"] or 0.0) > 0.8:
        reasons.append("Turnover is high; transaction cost sensitivity is a major risk.")
    elif 0.03 <= (metrics["turnover"] or 0.0) <= 0.5:
        reasons.append("Turnover is within a practical range.")

    if (metrics["max_drawdown"] or 0.0) > 0.3:
        reasons.append("Drawdown is high; risk control or filtering is needed.")

    if components["out_of_sample"] < 50:
        reasons.append("Out-of-sample checks are weak or failed.")
    elif validation:
        reasons.append("Out-of-sample evidence supports continued research.")

    if (metrics["concentration"] or 0.0) > 0.3:
        reasons.append("Holdings or exposure concentration is high.")

    if not reasons:
        reasons.append("Metrics are mixed; continue with controlled validation.")
    return reasons


def _suggestions(metrics: dict[str, float | None], components: dict[str, float], validation: dict) -> list[str]:
    suggestions: list[str] = []
    if components["quality"] < 50:
        suggestions.append("Improve signal definition with stronger normalization, nonlinear transforms, or orthogonal fields.")
    if components["stability"] < 55:
        suggestions.append("Run yearly or rolling validation and check whether one year or market regime drives the result.")
    if components["tradability"] < 55:
        suggestions.append("Check drawdown, liquidity filters, and whether the factor only works in the long or short leg.")
    if components["cost_sensitivity"] < 55:
        suggestions.append("Reduce turnover with longer windows, decay smoothing, rank hysteresis, or rebalance constraints.")
    if components["out_of_sample"] <= 50:
        suggestions.append("Before promotion, run rolling validation and placebo tests; do not rely only on in-sample IC.")
    if (metrics["concentration"] or 0.0) > 0.3:
        suggestions.append("Add industry or market-cap neutralization and concentration constraints.")
    if not suggestions:
        suggestions.append("Candidate is strong enough for pool review; next step is production-style backtest and risk review.")
    return suggestions


def _risk_flags(metrics: dict[str, float | None], components: dict[str, float], validation: dict) -> list[str]:
    flags: list[str] = []
    if (metrics["turnover"] or 0.0) > 0.8:
        flags.append("high_turnover")
    if (metrics["max_drawdown"] or 0.0) > 0.3:
        flags.append("high_drawdown")
    if abs(metrics["ic_ir"] or 0.0) < 0.3:
        flags.append("low_stability")
    if components["out_of_sample"] < 50:
        flags.append("weak_out_of_sample")
    if (metrics["concentration"] or 0.0) > 0.3:
        flags.append("high_concentration")
    return flags


def _recommendation_code(score: float, diagnostics: list[str]) -> str:
    joined = " ".join(diagnostics)
    if score >= 75 and "major risk" not in joined and "failed" not in joined:
        return "enter_pool"
    if score >= 50:
        return "continue_optimizing"
    return "stop"


def _recommendation_label(code: str) -> str:
    labels = {
        "enter_pool": "\u9002\u5408\u5165\u6c60",
        "continue_optimizing": "\u9002\u5408\u7ee7\u7eed\u4f18\u5316",
        "stop": "\u4e0d\u5efa\u8bae\u7ee7\u7eed",
    }
    return labels[code]


def _grade(score: float) -> str:
    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    if score >= 50:
        return "C"
    return "D"


def _daily_rank_ic(df: pd.DataFrame, forward_return_col: str) -> pd.Series:
    return df.groupby("date").apply(lambda group: group["factor"].corr(group[forward_return_col], method="spearman")).dropna()


def _first_number(summary: dict, keys: list[str], default: float | None = 0.0) -> float | None:
    normalized = {_normalize_key(key): value for key, value in summary.items()}
    for key in keys:
        if key in summary and pd.notna(summary[key]):
            return _coerce_float(summary[key], default)
        normalized_key = _normalize_key(key)
        if normalized_key in normalized and pd.notna(normalized[normalized_key]):
            return _coerce_float(normalized[normalized_key], default)
    return default


def _normalize_key(value: str) -> str:
    return str(value).strip().lower().replace(" ", "").replace("_", "")


def _coerce_float(value: Any, default: float | None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out):
        return default
    return out


def _normalize_rate(value: float | None) -> float | None:
    if value is None:
        return None
    return value / 100.0 if value > 1.0 else value


def _turnover_sweet_spot_score(turnover: float) -> float:
    if 0.03 <= turnover <= 0.50:
        return 100.0
    if turnover < 0.03:
        return _clip(turnover / 0.03) * 80
    return max(0.0, 100.0 - _clip((turnover - 0.50) / 0.70) * 100)


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))
