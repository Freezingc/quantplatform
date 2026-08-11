from __future__ import annotations

import keyword
import re
from pathlib import Path

from .dsl import ExpressionEngine
from .field_registry import CALCULATOR_DIR, FieldRegistry


class CalculatorPromoter:
    """Generate production FactorCalculator drafts from validated DSL expressions."""

    def __init__(self, registry: FieldRegistry | None = None, calculator_dir: str | Path = CALCULATOR_DIR):
        self.registry = registry or FieldRegistry.from_project()
        self.engine = ExpressionEngine(self.registry)
        self.calculator_dir = Path(calculator_dir)

    def promote_to_calculator(
        self,
        factor_name: str,
        expression: str,
        creator: str = "agent_gateway",
        description: str = "Generated from DSL candidate. Review before production use.",
        lookback_days: int | None = None,
        write: bool = False,
    ) -> dict:
        validation = self.engine.validate(expression)
        if not validation.ok:
            return {"ok": False, "errors": validation.errors, "path": None, "content": None}

        class_name = _safe_class_name(factor_name)
        filename = f"{class_name}.py"
        required = validation.used_fields
        lookback = lookback_days or _infer_lookback(expression)
        content = _render_calculator(
            class_name=class_name,
            factor_name=factor_name,
            expression=expression,
            required_fields=required,
            creator=creator,
            description=description,
            lookback_days=lookback,
        )
        path = self.calculator_dir / filename
        if write:
            self.calculator_dir.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise FileExistsError(f"Calculator already exists: {path}")
            path.write_text(content, encoding="utf-8")
        return {
            "ok": True,
            "errors": [],
            "path": str(path),
            "class_name": class_name,
            "required_fields": required,
            "lookback_days": lookback,
            "content": content,
        }


def _safe_class_name(name: str) -> str:
    parts = re.split(r"[^0-9A-Za-z]+", name)
    base = "".join(part[:1].upper() + part[1:] for part in parts if part)
    if not base or base[0].isdigit() or keyword.iskeyword(base):
        base = "Generated" + base
    if not base.endswith("Calculator"):
        base += "Calculator"
    return base


def _infer_lookback(expression: str) -> int:
    windows = [int(x) for x in re.findall(r"ts_[a-z_]+\([^)]*,\s*(\d+)\s*\)", expression)]
    windows += [int(x) for x in re.findall(r"decay_linear\([^)]*,\s*(\d+)\s*\)", expression)]
    return max(windows or [1])


def _render_calculator(
    class_name: str,
    factor_name: str,
    expression: str,
    required_fields: list[str],
    creator: str,
    description: str,
    lookback_days: int,
) -> str:
    field_list = ", ".join(repr(f) for f in required_fields)
    return f'''from .base import FactorCalculator
from agent_gateway.dsl import ExpressionEngine
from agent_gateway.field_registry import FieldRegistry


class {class_name}(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = [{field_list}]
        self.expression = {expression!r}

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        field_data = {{
            field: data.xs(field, level=0)
            for field in self.required_factors
        }}
        engine = ExpressionEngine(FieldRegistry.from_project())
        return engine.evaluate_wide(self.expression, field_data)


FACTOR_CONFIG_TEMPLATE = {{
    "factor_name": {factor_name!r},
    "creator": {creator!r},
    "create_date": "",
    "formula": {expression!r},
    "description": {description!r},
    "lookback_days": {lookback_days},
    "method": {class_name!r},
    "needed_factor": [],
    "start_time": "",
    "end_time": "",
    "category": "agent_generated",
}}
'''
