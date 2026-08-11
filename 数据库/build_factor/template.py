"""Template used to render standalone factor scripts."""

FACTOR_SCRIPT_TEMPLATE = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Auto-generated factor implementation."""

import argparse
import os
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
for candidate in (CURRENT_DIR, PARENT_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:
    from build_factor.expr_parser import parse_expression  # type: ignore
    from build_factor.function_lib import *  # noqa: F401,F403
except ModuleNotFoundError:
    from expr_parser import parse_expression  # type: ignore
    from function_lib import *  # type: ignore,F401,F403

VARIABLE_PATTERN = re.compile(r"\$[A-Za-z_][A-Za-z0-9_]*")


def _build_column_lookup(columns):
    lookup = {}
    for col in columns:
        lookup[col] = col
        if isinstance(col, str):
            lookup[col.lstrip('$')] = col
        else:
            lookup[str(col)] = col
    return lookup


def _inject_dataframe_refs(expr_str, columns):
    lookup = _build_column_lookup(columns)

    def repl(match):
        symbol = match.group(0)
        candidate = lookup.get(symbol) or lookup.get(symbol.lstrip('$'))
        if candidate is None:
            raise KeyError(f"Column {symbol} not found in data file")
        return f"df[{repr(candidate)}]"

    return VARIABLE_PATTERN.sub(repl, expr_str)


def calculate_factor(expr: str, factor_name: str, data_path: str, result_path: str | None = None):
    """Compute factor values and optionally persist them to an HDF5 file."""
    df = pd.read_hdf(data_path, key='data')

    parsed = parse_expression(expr)
    parsed = _inject_dataframe_refs(parsed, df.columns)

    df[factor_name] = eval(parsed)
    series = df[factor_name].astype(np.float64)

    if result_path:
        os.makedirs(os.path.dirname(result_path) or '.', exist_ok=True)
        if os.path.exists(result_path):
            os.remove(result_path)
        series.to_hdf(result_path, key='data')
    return series


def main():
    parser = argparse.ArgumentParser(description="Run the auto-generated factor implementation.")
    parser.add_argument('--expr', default=__EXPR_LITERAL__, help='Factor expression (default: generated value)')
    parser.add_argument('--factor-name', default=__FACTOR_NAME_LITERAL__, help='Factor name (default: generated value)')
    parser.add_argument('--data-path', default=__DATA_PATH_LITERAL__, help='Path to price-volume HDF5 file')
    parser.add_argument('--result-path', default=__RESULT_PATH_LITERAL__, help='Optional output HDF5 path')
    args = parser.parse_args()

    calculate_factor(args.expr, args.factor_name, args.data_path, args.result_path)
    destination = args.result_path or 'memory'
    print(f"Factor '{'{'}args.factor_name{'}'}' computed. Result saved to {destination}.")


if __name__ == '__main__':
    main()
'''
