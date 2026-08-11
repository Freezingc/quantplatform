from __future__ import annotations

import contextlib
import importlib
import inspect
import os
import pkgutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import dolphindb as ddb
import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    load_dotenv = None

from .knowledge_base import KnowledgeBase
from .validation import score_factor_summary


ROOT = Path(__file__).resolve().parents[1]
DB_ROOT = ROOT / "数据库"
EVAL_ROOT = ROOT / "因子检验"


class FactorPipelineError(RuntimeError):
    pass


def load_agent_env(env_path: str | Path | None = None) -> None:
    if load_dotenv is None:
        return
    if env_path:
        load_dotenv(Path(env_path))
        return
    load_dotenv(ROOT / ".env")


def dolphindb_config_from_env(env_path: str | Path | None = None) -> dict[str, Any]:
    load_agent_env(env_path)
    username = _first_env("DOLPHINDB_USERNAME", "dolphinDB_USERNAME")
    password = _first_env("DOLPHINDB_PASSWORD", "dolphinDB_PASSWORD")
    host = _first_env("DOLPHINDB_HOST", "dolphinDB_HOST") or "10.176.55.211"
    port = int(_first_env("DOLPHINDB_PORT", "dolphinDB_PORT") or 8848)
    if not username or not password:
        raise FactorPipelineError("缺少 DolphinDB 账号，请在 .env 中配置 DOLPHINDB_USERNAME/DOLPHINDB_PASSWORD 或 dolphinDB_USERNAME/dolphinDB_PASSWORD")
    return {"host": host, "port": port, "username": username, "password": password}


def connect_dolphindb(env_path: str | Path | None = None):
    cfg = dolphindb_config_from_env(env_path)
    session = ddb.session()
    session.connect(cfg["host"], cfg["port"], cfg["username"], cfg["password"])
    return session


def run_factor_gate(
    factor_name: str,
    method: str,
    lookback_days: int,
    formula: str = "",
    description: str = "",
    creator: str = "agent_gateway",
    category: str = "agent_generated",
    start_time: str = "2020.01.01",
    end_time: str | None = None,
    create_date: str | None = None,
    needed_factor: list[str] | None = None,
    similarity_threshold: float = 0.7,
    min_abs_ic: float = 0.01,
    min_abs_icir: float = 0.3,
    min_common_stocks: int = 50,
    min_valid_dates: int = 20,
    sample_similarity: bool = True,
    sample_days: int = 60,
    commit: bool = False,
    env_path: str | Path | None = None,
    eval_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    end_time = end_time or datetime.now().strftime("%Y.%m.%d")
    create_date = create_date or datetime.now().strftime("%Y.%m.%d")
    needed_factor = needed_factor or []
    session = connect_dolphindb(env_path)
    try:
        factor = calculate_factor(session, method, lookback_days, start_time, end_time)
        similarity = run_similarity_check(
            session,
            factor_name,
            factor,
            start_time,
            end_time,
            threshold=similarity_threshold,
            min_common_stocks=min_common_stocks,
            min_valid_dates=min_valid_dates,
            sample=sample_similarity,
            sample_days=sample_days,
        )
        factor_eval = run_ic_check(factor_name, factor, eval_params)
        gates = decide_gates(similarity, factor_eval, similarity_threshold, min_abs_ic, min_abs_icir)
        promoted = None
        if gates["passed"] and commit:
            promoted = commit_factor_to_db(
                session=session,
                factor_name=factor_name,
                factor=factor,
                creator=creator,
                create_date=create_date,
                formula=formula,
                description=description,
                lookback_days=lookback_days,
                method=method,
                needed_factor=needed_factor,
                category=category,
            )
        payload = {
            "ok": True,
            "factor_name": factor_name,
            "method": method,
            "start_time": start_time,
            "end_time": end_time,
            "commit_requested": commit,
            "promoted": promoted,
            "gates": gates,
            "similarity": similarity,
            "factor_eval": factor_eval,
            "factor_shape": list(factor.shape),
            "non_null_values": int(factor.notna().sum().sum()),
        }
        payload["knowledge_assets"] = distill_gate_knowledge(payload)
        return payload
    finally:
        with contextlib.suppress(Exception):
            session.close()


def calculate_factor(session, method: str, lookback_days: int, start_time: str, end_time: str) -> pd.DataFrame:
    calculators = discover_factor_calculators()
    if method not in calculators:
        available = ", ".join(sorted(calculators)[:50])
        raise FactorPipelineError(f"未找到因子计算器 {method}。可用示例: {available}")
    calculator = calculators[method](session, lookback_days)
    factor = calculator.calculate(start_time, end_time)
    if not isinstance(factor, pd.DataFrame):
        raise FactorPipelineError("因子计算结果不是 DataFrame")
    if factor.empty:
        raise FactorPipelineError("因子计算结果为空")
    factor.index = pd.to_datetime(factor.index)
    return factor


def discover_factor_calculators() -> dict[str, type]:
    _ensure_db_path()
    from factor_calculators.base import FactorCalculator

    calculators: dict[str, type] = {}
    package_path = DB_ROOT / "factor_calculators"
    for _, module_name, _ in pkgutil.iter_modules([str(package_path)]):
        if module_name == "base":
            continue
        try:
            module = importlib.import_module(f"factor_calculators.{module_name}")
        except Exception:
            continue
        for member_name, member in inspect.getmembers(module):
            if inspect.isclass(member) and issubclass(member, FactorCalculator) and member is not FactorCalculator:
                calculators[member_name] = member
    return calculators


def run_similarity_check(
    session,
    factor_name: str,
    factor: pd.DataFrame,
    start_time: str,
    end_time: str,
    threshold: float,
    min_common_stocks: int,
    min_valid_dates: int,
    sample: bool,
    sample_days: int,
) -> dict[str, Any]:
    _ensure_db_path()
    from get_data import DataFetcher
    from factor_similarity import compute_similarity, fetch_factor_from_db, get_all_factor_names

    fetcher = DataFetcher(s=session)
    compare_names = [name for name in get_all_factor_names(session) if name != factor_name]
    sample_dates = None
    effective_min_valid = min_valid_dates
    if sample:
        all_trade_days = fetcher.get_trade_days(exchange="SSE", start_time=start_time, end_time=end_time)
        actual_sample = min(sample_days, len(all_trade_days))
        if actual_sample > 0:
            rng = np.random.default_rng(7)
            sample_dates = pd.DatetimeIndex(sorted(rng.choice(all_trade_days, size=actual_sample, replace=False)))
            effective_min_valid = min(min_valid_dates, actual_sample)
    results = []
    skipped = []
    for existing_name in compare_names:
        try:
            existing = fetch_factor_from_db(fetcher, existing_name, start_time, end_time)
            sim, diag = compute_similarity(
                factor,
                existing,
                min_common_stocks=min_common_stocks,
                min_valid_dates=effective_min_valid,
                sample_dates=sample_dates,
            )
            if sim is None:
                skipped.append({"factor_name": existing_name, "reason": diag.get("reason", "insufficient_data"), "diag": diag})
            else:
                results.append({"factor_name": existing_name, "similarity": float(sim)})
        except Exception as exc:
            skipped.append({"factor_name": existing_name, "reason": str(exc), "diag": {}})
    results.sort(key=lambda x: x["similarity"], reverse=True)
    max_item = results[0] if results else None
    max_similarity = float(max_item["similarity"]) if max_item else 0.0
    return {
        "passed": max_similarity < threshold,
        "threshold": threshold,
        "max_similarity": max_similarity,
        "nearest_factor": max_item["factor_name"] if max_item else None,
        "top_matches": results[:20],
        "compared_count": len(results),
        "skipped_count": len(skipped),
        "skipped": skipped[:20],
    }


def run_ic_check(factor_name: str, factor: pd.DataFrame, params: dict[str, Any] | None = None) -> dict[str, Any]:
    if str(EVAL_ROOT) not in sys.path:
        sys.path.insert(0, str(EVAL_ROOT))
    from src.batch_factor_analysis_pipline import run_factor_analysis_from_files

    eval_params = params or {
        "group_num": 10,
        "factor_direction": 1,
        "benchmark": "中证500",
        "data_preprocess": True,
        "ic_horizons": [1, 3, 5, 10, 20],
    }
    with tempfile.TemporaryDirectory(prefix="agent_factor_gate_") as tmp:
        workdir = Path(tmp)
        input_dir = workdir / "factor_data_files"
        input_dir.mkdir(parents=True, exist_ok=True)
        long_df = factor.stack(dropna=False).rename("factor").reset_index()
        long_df.columns = ["trade_date", "ts_code", "factor"]
        long_df["trade_date"] = pd.to_datetime(long_df["trade_date"]).dt.strftime("%Y-%m-%d")
        long_df.to_csv(input_dir / f"factor_data_{factor_name}.csv", index=False)
        with _pushd(workdir):
            raw = run_factor_analysis_from_files(str(input_dir), params=eval_params)
    summary = _best_summary(raw)
    scored = score_factor_summary(summary) if summary else score_factor_summary({})
    return {"summary": summary, "score": scored, "raw": _jsonable_analysis(raw)}


def decide_gates(similarity: dict[str, Any], factor_eval: dict[str, Any], similarity_threshold: float, min_abs_ic: float, min_abs_icir: float) -> dict[str, Any]:
    summary = factor_eval.get("summary") or {}
    ic_mean = _first_number(summary, "IC均值", "RankIC均值", "ic_mean", "rank_ic_mean")
    icir = _first_number(summary, "年化ICIR", "ICIR", "ic_ir", "RankICIR")
    recommendation_code = (factor_eval.get("score") or {}).get("recommendation_code")
    checks = {
        "similarity": bool(similarity.get("max_similarity", 0.0) < similarity_threshold),
        "ic_mean": ic_mean is not None and abs(ic_mean) >= min_abs_ic,
        "icir": icir is not None and abs(icir) >= min_abs_icir,
        "score": recommendation_code != "stop",
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {"similarity": similarity_threshold, "min_abs_ic": min_abs_ic, "min_abs_icir": min_abs_icir},
        "metrics": {"ic_mean": ic_mean, "icir": icir, "recommendation_code": recommendation_code},
        "block_reasons": [name for name, ok in checks.items() if not ok],
    }


def commit_factor_to_db(
    session,
    factor_name: str,
    factor: pd.DataFrame,
    creator: str,
    create_date: str,
    formula: str,
    description: str,
    lookback_days: int,
    method: str,
    needed_factor: list[str],
    category: str,
) -> dict[str, Any]:
    _ensure_db_path()
    from add_factor_narrow import add2metadata, set_last_update_time, update_factor

    existing = session.run(f"""
        db = database("dfs://YesLab")
        t = loadTable(db, 'factor_metadata')
        select * from t where factor_name = '{factor_name}'
    """)
    if existing.empty:
        table_name = add2metadata(
            s=session,
            factor_name=factor_name,
            creator=creator,
            create_date=create_date,
            formula=formula,
            description=description,
            lookback_days=lookback_days,
            method=method,
            needed_factor=needed_factor,
            category=category,
        )
        metadata_created = True
    else:
        table_name = existing.iloc[0]["store_factor"]
        metadata_created = False
    update_factor(session, factor_name, factor, table_name)
    set_last_update_time(session, factor_name)
    return {"metadata_created": metadata_created, "table_name": table_name, "rows": int(factor.notna().sum().sum())}


def distill_gate_knowledge(payload: dict[str, Any]) -> dict[str, Any]:
    kb = KnowledgeBase()
    factor_name = payload["factor_name"]
    assets = {
        "similarity": kb.distill_from_event("similarity_check", {
            "factor_name": factor_name,
            "max_similarity": payload["similarity"].get("max_similarity"),
            "threshold": payload["similarity"].get("threshold"),
            "nearest_factor": payload["similarity"].get("nearest_factor"),
        }),
        "factor_eval": kb.distill_from_event("factor_eval", {
            "factor_name": factor_name,
            "summary": payload["factor_eval"].get("summary"),
            "evaluation": payload["factor_eval"].get("score"),
            "gates": payload["gates"],
        }),
    }
    return assets


def _best_summary(raw: dict[str, Any]) -> dict[str, Any]:
    full = raw.get("full_summary")
    if isinstance(full, pd.DataFrame) and not full.empty:
        return full.iloc[0].to_dict()
    ic = raw.get("ic_summary")
    if isinstance(ic, pd.DataFrame) and not ic.empty:
        ordered = ic.copy()
        if "持有期" in ordered.columns:
            one_day = ordered[ordered["持有期"].astype(str) == "1D"]
            if not one_day.empty:
                ordered = one_day
        return ordered.iloc[0].to_dict()
    return {}


def _jsonable_analysis(raw: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in raw.items():
        if isinstance(value, pd.DataFrame):
            out[key] = value.replace({np.nan: None}).to_dict(orient="records")
        else:
            out[key] = value
    return out


@contextlib.contextmanager
def _pushd(path: str | Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _ensure_db_path() -> None:
    if str(DB_ROOT) not in sys.path:
        sys.path.insert(0, str(DB_ROOT))


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _first_number(data: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = data.get(key)
        if value is None or pd.isna(value):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None
