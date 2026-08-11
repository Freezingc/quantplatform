from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pytest

from research_v2.db.session import (
    ReadOnlyViolation,
    TargetDatabaseViolation,
    V2WriteSession,
    YesLabReadOnlySession,
)


@dataclass
class FakeClient:
    scripts: list[str] = field(default_factory=list)
    uploads: list[dict[str, pd.DataFrame]] = field(default_factory=list)

    def run(self, script: str) -> object:
        self.scripts.append(script)
        return {"ok": True}

    def upload(self, values: dict[str, pd.DataFrame]) -> None:
        self.uploads.append(values)


@pytest.mark.parametrize(
    "script",
    [
        "append!(loadTable('dfs://YesLab', 'x'), t)",
        "dropTable(database('dfs://YesLab'), 'x')",
        "tableInsert{loadTable('dfs://YesLab', 'x')}(t)",
        "update loadTable('dfs://YesLab', 'x') set value = 0",
        "delete from loadTable('dfs://YesLab', 'x') where ts_code = `000001.SZ",
    ],
)
def test_yeslab_session_rejects_mutating_scripts(script: str) -> None:
    client = FakeClient()

    with pytest.raises(ReadOnlyViolation):
        YesLabReadOnlySession(client).query(script)

    assert client.scripts == []


def test_yeslab_session_allows_read_query_and_ignores_keywords_inside_literals() -> None:
    client = FakeClient()
    script = "select 'append! is text' as note from loadTable('dfs://YesLab', 'factor_data1')"

    result = YesLabReadOnlySession(client).query(script)

    assert result == {"ok": True}
    assert client.scripts == [script]


def test_yeslab_session_rejects_a_non_yeslab_uri() -> None:
    with pytest.raises(ReadOnlyViolation, match="YesLab"):
        YesLabReadOnlySession(FakeClient()).query(
            "select * from loadTable('dfs://QuantResearchV2', 'market_daily_clean')"
        )


def test_v2_write_session_rejects_unapproved_target_database() -> None:
    with pytest.raises(TargetDatabaseViolation):
        V2WriteSession(FakeClient(), target_database="dfs://YesLab")


def test_v2_write_session_rejects_yeslab_reference_before_execution() -> None:
    client = FakeClient()
    session = V2WriteSession(client, target_database="dfs://QuantResearchV2Test")

    with pytest.raises(TargetDatabaseViolation, match="YesLab"):
        session.execute("append!(loadTable('dfs://YesLab', 'x'), t)")

    assert client.scripts == []


def test_v2_write_session_exposes_its_approved_target_database() -> None:
    session = V2WriteSession(FakeClient(), target_database="dfs://QuantResearchV2Test")

    assert session.target_database == "dfs://QuantResearchV2Test"


def test_v2_write_session_uploads_only_safe_dataframe_variable_names() -> None:
    client = FakeClient()
    session = V2WriteSession(client, target_database="dfs://QuantResearchV2Test")
    frame = pd.DataFrame({"ts_code": ["000001.SZ"]})

    session.upload_frame("stage_market_1", frame)

    assert client.uploads[0]["stage_market_1"].equals(frame)
    with pytest.raises(TargetDatabaseViolation, match="variable"):
        session.upload_frame("stage;delete", frame)
