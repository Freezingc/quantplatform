"""Strongly typed application configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

EnvironmentName = Literal["dev", "test", "production"]
TargetDatabase = Literal["dfs://QuantResearchV2", "dfs://QuantResearchV2Test"]


class DatabaseSettings(BaseModel):
    """DolphinDB connection settings with an immutable source boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    host: str = Field(min_length=1)
    port: int = Field(gt=0, le=65535)
    username: str = Field(min_length=1)
    password: SecretStr
    source_database: Literal["dfs://YesLab"] = "dfs://YesLab"
    target_database: TargetDatabase


class AppSettings(BaseModel):
    """Validated runtime settings for one deployment environment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    environment: EnvironmentName
    dolphindb: DatabaseSettings
    artifact_root: Path
    lock_year: int = Field(ge=2026)

    @model_validator(mode="after")
    def validate_target_database(self) -> AppSettings:
        if self.environment in {"dev", "test"} and (
            self.dolphindb.target_database != "dfs://QuantResearchV2Test"
        ):
            raise ValueError("test environment must target dfs://QuantResearchV2Test")
        if self.environment == "production" and (
            self.dolphindb.target_database != "dfs://QuantResearchV2"
        ):
            raise ValueError("production environment must target dfs://QuantResearchV2")
        return self
