"""The committed data dictionary must match the frozen migration contracts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

RESEARCH_ROOT = Path(__file__).resolve().parents[3]
DICTIONARY = RESEARCH_ROOT.parent / "docs" / "v2_data_dictionary.md"


def test_data_dictionary_is_in_sync_with_migrations() -> None:
    current = DICTIONARY.read_text(encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(RESEARCH_ROOT / "scripts" / "generate_data_dictionary.py")],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "wrote" in result.stdout
    regenerated = DICTIONARY.read_text(encoding="utf-8")
    try:
        assert regenerated == current, (
            "docs/v2_data_dictionary.md is stale; regenerate with "
            "scripts/generate_data_dictionary.py after migration changes"
        )
    finally:
        DICTIONARY.write_text(current, encoding="utf-8")
