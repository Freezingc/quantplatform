from __future__ import annotations

import tempfile
import sys
from pathlib import Path

import pandas as pd

from .dsl import ExpressionEngine
from .field_registry import FieldRegistry


def run_factor_eval_from_expression(
    expression: str,
    data: dict[str, pd.DataFrame],
    factor_name: str = "dsl_candidate",
    params: dict | None = None,
    output_root: str | Path | None = None,
) -> dict:
    """Evaluate a DSL expression through the existing factor analysis pipeline.

    This helper expects field data to be supplied as wide DataFrames. A separate
    production adapter can fetch the data from DolphinDB before calling it.
    """
    engine = ExpressionEngine(FieldRegistry.from_project())
    long_df = engine.evaluate_long(expression, data)
    workdir = Path(output_root) if output_root else Path(tempfile.mkdtemp(prefix="factor_eval_"))
    input_dir = workdir / "factor_data_files"
    input_dir.mkdir(parents=True, exist_ok=True)
    path = input_dir / f"factor_data_{factor_name}.csv"
    long_df.to_csv(path, index=False)

    factor_eval_root = Path(__file__).resolve().parents[1] / "因子检验"
    if str(factor_eval_root) not in sys.path:
        sys.path.insert(0, str(factor_eval_root))
    from src.batch_factor_analysis_pipline import run_factor_analysis_from_files

    result = run_factor_analysis_from_files(str(input_dir), params=params)
    return {"input_file": str(path), "result": result}
