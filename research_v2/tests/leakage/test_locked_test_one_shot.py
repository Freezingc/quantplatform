"""Canary checks for the isolated one-shot 2026 locked-test evaluator boundary."""

from __future__ import annotations

import ast
from pathlib import Path


def test_locked_test_evaluator_has_no_training_search_selection_or_label_imports() -> None:
    module = Path(__file__).parents[2] / "src/research_v2/registry/locked_test.py"
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    imported = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ] + [
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    ]
    forbidden = ("training", "search", "selection", "factors", "labels", "model_walk_forward")
    imported_segments = {segment.lower() for name in imported for segment in name.split(".")}

    assert not imported_segments.intersection(forbidden), imported
