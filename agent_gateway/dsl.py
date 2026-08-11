from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .field_registry import FieldRegistry
from .models import ValidationResult


class ExpressionError(ValueError):
    pass


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    min_args: int
    max_args: int
    func: Callable


class ExpressionEngine:
    """Small factor-expression DSL for research-time experiments.

    Data is represented as wide DataFrames with rows as dates and columns as
    instruments, matching the existing FactorCalculator convention.
    """

    def __init__(self, registry: FieldRegistry | None = None):
        self.registry = registry or FieldRegistry.from_project()
        self.operators = _build_operators()

    def validate(self, expression: str) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        try:
            tree = self._parse(expression)
        except Exception as exc:
            return ValidationResult(False, expression, errors=[str(exc)])

        names = sorted(_extract_names(tree))
        used_fields: list[str] = []
        missing: list[str] = []
        for name in names:
            if name in self.operators or name in {"True", "False"}:
                continue
            resolved = self.registry.resolve_name(name)
            if resolved:
                spec = self.registry.get(resolved)
                if spec and not spec.expression_enabled:
                    errors.append(f"Field is not expression-enabled: {name}")
                used_fields.append(resolved)
            else:
                missing.append(name)

        for call in _extract_calls(tree):
            spec = self.operators.get(call.name)
            if not spec:
                errors.append(f"Unsupported operator: {call.name}")
                continue
            argc = len(call.args)
            if argc < spec.min_args or argc > spec.max_args:
                errors.append(f"Operator {call.name} expects {spec.min_args}-{spec.max_args} args, got {argc}")

        if missing:
            errors.append("Missing fields: " + ", ".join(missing))
        if len(expression) > 500:
            warnings.append("Expression is longer than 500 characters; consider simplifying it.")

        return ValidationResult(
            ok=not errors,
            expression=expression,
            used_fields=sorted(set(used_fields)),
            missing_fields=missing,
            errors=errors,
            warnings=warnings,
        )

    def required_fields(self, expression: str) -> list[str]:
        result = self.validate(expression)
        return result.used_fields

    def evaluate_wide(self, expression: str, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        validation = self.validate(expression)
        if not validation.ok:
            raise ExpressionError("; ".join(validation.errors))

        canonical_data: dict[str, pd.DataFrame] = {}
        for key, value in data.items():
            resolved = self.registry.resolve_name(key) or key
            canonical_data[resolved] = _ensure_frame(value)

        missing = [field for field in validation.used_fields if field not in canonical_data]
        if missing:
            raise ExpressionError("Data missing required fields: " + ", ".join(missing))

        tree = self._parse(expression)
        result = self._eval_node(tree.body, canonical_data)
        return _ensure_frame(result)

    def evaluate_long(self, expression: str, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        wide = self.evaluate_wide(expression, data)
        long_df = (
            wide.stack(dropna=False)
            .rename("factor")
            .reset_index()
            .rename(columns={wide.index.name or "level_0": "date", wide.columns.name or "level_1": "instrument"})
        )
        long_df["date"] = pd.to_datetime(long_df["date"])
        return long_df[["instrument", "date", "factor"]]

    def _parse(self, expression: str) -> ast.Expression:
        expr = expression.strip().replace("^", "**")
        if not expr:
            raise ExpressionError("Expression is empty")
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as exc:
            raise ExpressionError(f"Syntax error: {exc.msg}") from exc
        _Guard().visit(tree)
        return tree

    def _eval_node(self, node: ast.AST, data: dict[str, pd.DataFrame]):
        if isinstance(node, ast.Name):
            resolved = self.registry.resolve_name(node.id) or node.id
            if resolved in data:
                return data[resolved]
            if node.id == "True":
                return True
            if node.id == "False":
                return False
            raise ExpressionError(f"Unknown field: {node.id}")
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool)):
            return node.value
        if isinstance(node, ast.UnaryOp):
            value = self._eval_node(node.operand, data)
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.UAdd):
                return value
            if isinstance(node.op, ast.Not):
                return ~_to_bool_frame(value)
        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left, data)
            right = self._eval_node(node.right, data)
            return _BIN_OPS[type(node.op)](left, right)
        if isinstance(node, ast.BoolOp):
            values = [self._eval_node(v, data) for v in node.values]
            out = _to_bool_frame(values[0])
            for value in values[1:]:
                if isinstance(node.op, ast.And):
                    out = out & _to_bool_frame(value)
                elif isinstance(node.op, ast.Or):
                    out = out | _to_bool_frame(value)
            return out
        if isinstance(node, ast.Compare):
            left = self._eval_node(node.left, data)
            out = None
            for op, comparator in zip(node.ops, node.comparators):
                right = self._eval_node(comparator, data)
                current = _CMP_OPS[type(op)](left, right)
                out = current if out is None else out & current
                left = right
            return out
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ExpressionError("Only simple function calls are supported")
            spec = self.operators.get(node.func.id)
            if not spec:
                raise ExpressionError(f"Unsupported operator: {node.func.id}")
            args = [self._eval_node(arg, data) for arg in node.args]
            return spec.func(*args)
        raise ExpressionError(f"Unsupported syntax: {type(node).__name__}")


class _Guard(ast.NodeVisitor):
    allowed = (
        ast.Expression, ast.Name, ast.Load, ast.Constant,
        ast.UnaryOp, ast.USub, ast.UAdd, ast.Not,
        ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
        ast.BoolOp, ast.And, ast.Or,
        ast.Compare, ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq,
        ast.Call,
    )

    def generic_visit(self, node):
        if not isinstance(node, self.allowed):
            raise ExpressionError(f"Unsupported syntax: {type(node).__name__}")
        super().generic_visit(node)


@dataclass(frozen=True)
class _Call:
    name: str
    args: tuple[ast.AST, ...]


def _extract_names(tree: ast.AST) -> set[str]:
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def _extract_calls(tree: ast.AST) -> list[_Call]:
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            calls.append(_Call(node.func.id, tuple(node.args)))
    return calls


def _ensure_frame(value):
    if isinstance(value, pd.DataFrame):
        return value.astype(float, copy=False)
    return value


def _to_bool_frame(value):
    if isinstance(value, pd.DataFrame):
        return value.fillna(False).astype(bool)
    return bool(value)


def _rank(x):
    return x.rank(axis=1, pct=True)


def _zscore(x):
    mean = x.mean(axis=1)
    std = x.std(axis=1).replace(0, np.nan)
    return x.sub(mean, axis=0).div(std, axis=0)


def _scale(x):
    lo = x.min(axis=1)
    hi = x.max(axis=1)
    return x.sub(lo, axis=0).div((hi - lo).replace(0, np.nan), axis=0)


def _ts_rank(x, n):
    n = int(n)
    return x.rolling(n, min_periods=max(1, n // 2)).rank(pct=True)


def _decay_linear(x, n):
    n = int(n)
    weights = np.arange(1, n + 1, dtype=float)
    weights = weights / weights.sum()
    return x.rolling(n, min_periods=max(1, n // 2)).apply(lambda arr: np.dot(arr, weights[-len(arr):]), raw=True)


def _where(cond, t, f):
    return t.where(_to_bool_frame(cond), f)


def _adv(volume, n):
    return volume.rolling(int(n), min_periods=max(1, int(n) // 2)).mean()


def _build_operators() -> dict[str, OperatorSpec]:
    ops = {
        "rank": OperatorSpec("rank", 1, 1, _rank),
        "zscore": OperatorSpec("zscore", 1, 1, _zscore),
        "scale": OperatorSpec("scale", 1, 1, _scale),
        "abs": OperatorSpec("abs", 1, 1, abs),
        "log": OperatorSpec("log", 1, 1, lambda x: np.log(x.replace(0, np.nan)) if isinstance(x, pd.DataFrame) else math.log(x)),
        "sqrt": OperatorSpec("sqrt", 1, 1, lambda x: np.sqrt(x.clip(lower=0)) if isinstance(x, pd.DataFrame) else math.sqrt(max(x, 0))),
        "sign": OperatorSpec("sign", 1, 1, np.sign),
        "tanh": OperatorSpec("tanh", 1, 1, np.tanh),
        "ts_mean": OperatorSpec("ts_mean", 2, 2, lambda x, n: x.rolling(int(n), min_periods=max(1, int(n) // 2)).mean()),
        "ts_std": OperatorSpec("ts_std", 2, 2, lambda x, n: x.rolling(int(n), min_periods=max(1, int(n) // 2)).std()),
        "ts_sum": OperatorSpec("ts_sum", 2, 2, lambda x, n: x.rolling(int(n), min_periods=max(1, int(n) // 2)).sum()),
        "ts_max": OperatorSpec("ts_max", 2, 2, lambda x, n: x.rolling(int(n), min_periods=max(1, int(n) // 2)).max()),
        "ts_min": OperatorSpec("ts_min", 2, 2, lambda x, n: x.rolling(int(n), min_periods=max(1, int(n) // 2)).min()),
        "ts_shift": OperatorSpec("ts_shift", 2, 2, lambda x, n: x.shift(int(n))),
        "ts_delta": OperatorSpec("ts_delta", 2, 2, lambda x, n: x - x.shift(int(n))),
        "ts_rank": OperatorSpec("ts_rank", 2, 2, _ts_rank),
        "ts_corr": OperatorSpec("ts_corr", 3, 3, lambda x, y, n: x.rolling(int(n), min_periods=max(2, int(n) // 2)).corr(y)),
        "ts_cov": OperatorSpec("ts_cov", 3, 3, lambda x, y, n: x.rolling(int(n), min_periods=max(2, int(n) // 2)).cov(y)),
        "decay_linear": OperatorSpec("decay_linear", 2, 2, _decay_linear),
        "where": OperatorSpec("where", 3, 3, _where),
        "max": OperatorSpec("max", 2, 2, lambda a, b: pd.DataFrame(np.maximum(a, b), index=a.index, columns=a.columns) if isinstance(a, pd.DataFrame) else max(a, b)),
        "min": OperatorSpec("min", 2, 2, lambda a, b: pd.DataFrame(np.minimum(a, b), index=a.index, columns=a.columns) if isinstance(a, pd.DataFrame) else min(a, b)),
        "power": OperatorSpec("power", 2, 2, lambda a, b: a ** b),
        "sign_power": OperatorSpec("sign_power", 2, 2, lambda a, b: np.sign(a) * (abs(a) ** b)),
        "adv": OperatorSpec("adv", 2, 2, _adv),
    }
    return ops


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

_CMP_OPS = {
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}
