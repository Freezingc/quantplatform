#!/usr/bin/env python3
"""CLI for turning factor expressions into runnable scripts.

This small utility reproduces the "factor expression -> factor.py" portion of
QuantaAlpha, but in a standalone form that can be embedded in other projects.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import textwrap

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
for candidate in (CURRENT_DIR, PROJECT_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:
    from build_factor.expr_parser import parse_expression
    from build_factor.template import FACTOR_SCRIPT_TEMPLATE
except ModuleNotFoundError:
    from expr_parser import parse_expression  # type: ignore
    from template import FACTOR_SCRIPT_TEMPLATE  # type: ignore


def _json_literal(value: str) -> str:
    """Return a JSON-style literal string for injecting into template."""
    return json.dumps(value)


def build_factor_script(
    expr: str,
    factor_name: str,
    output_path: Path,
    data_path: str = 'daily_pv.h5',
    result_path: str = 'result.h5',
) -> Path:
    """Create a runnable script that computes the given expression."""
    # Validate syntax early by attempting to parse once (without columns context)
    parse_expression(expr)

    rendered = (
        FACTOR_SCRIPT_TEMPLATE
        .replace('__EXPR_LITERAL__', _json_literal(expr))
        .replace('__FACTOR_NAME_LITERAL__', _json_literal(factor_name))
        .replace('__DATA_PATH_LITERAL__', _json_literal(data_path))
        .replace('__RESULT_PATH_LITERAL__', _json_literal(result_path))
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered)
    output_path.chmod(0o755)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Build executable factor scripts from expressions.")
    parser.add_argument('expr', help="Factor expression (e.g. RANK($close))")
    parser.add_argument('--factor-name', default='custom_factor', help='Name used for column/result files')
    parser.add_argument('--output', default='factor_generated.py', help='Output script path')
    parser.add_argument('--data-path', default='daily_pv.h5', help='Default data path baked into script')
    parser.add_argument('--result-path', default='result.h5', help='Default result path baked into script')
    args = parser.parse_args()

    path = build_factor_script(
        expr=args.expr,
        factor_name=args.factor_name,
        output_path=Path(args.output),
        data_path=args.data_path,
        result_path=args.result_path,
    )
    msg = textwrap.dedent(f"""
        ✅ Factor script generated: {path}
        Run it with:
          python {path} --expr "{args.expr}" --factor-name {args.factor_name}
        """).strip()
    print(msg)


if __name__ == '__main__':
    main()
