from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from research_v2.contracts.factor import (
    AvailabilityPolicy,
    FactorDefinition,
    FactorType,
)
from research_v2.factors.source_reader import (
    SourceFactorSchemaError,
    normalize_technical_factor_source,
)


def _definition() -> FactorDefinition:
    return FactorDefinition(
        factor_id="yeslab_1",
        factor_version=1,
        display_name="alpha",
        source_database="dfs://YesLab",
        source_table="factor_data1",
        source_value_column="value",
        source_date_column="trade_date",
        source_code_column="ts_code",
        factor_type=FactorType.TECHNICAL,
        frequency="DAILY",
        expected_direction=0,
        availability_policy=AvailabilityPolicy.SAME_DAY_CLOSE,
        max_staleness_calendar_days=0,
        unit="UNKNOWN",
        owner="research",
    )


def test_source_reader_normalizes_only_the_registered_factor_name() -> None:
    source = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "trade_date": [date(2025, 1, 2), date(2025, 1, 2)],
            "factor_name": ["alpha", "beta"],
            "value": [1.5, 9.0],
        }
    )

    normalized = normalize_technical_factor_source(
        source,
        _definition(),
        source_factor_name="alpha",
        source_watermark_id="watermark",
    )

    assert len(normalized) == 1
    assert normalized.loc[0, "factor_id"] == "yeslab_1"
    assert normalized.loc[0, "observation_date"] == date(2025, 1, 2)
    assert normalized.loc[0, "source_available_date"] == date(2025, 1, 2)
    assert normalized.loc[0, "raw_value"] == 1.5
    assert normalized.loc[0, "value_is_valid"]


def test_source_reader_marks_nonfinite_values_and_rejects_missing_contract_columns() -> None:
    source = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": [date(2025, 1, 2)],
            "factor_name": ["alpha"],
            "value": [np.inf],
        }
    )

    normalized = normalize_technical_factor_source(
        source,
        _definition(),
        source_factor_name="alpha",
        source_watermark_id="watermark",
    )

    assert not normalized.loc[0, "value_is_valid"]
    assert normalized.loc[0, "invalid_reason"] == "NONFINITE_VALUE"
    with pytest.raises(SourceFactorSchemaError, match="factor_name"):
        normalize_technical_factor_source(
            source.drop(columns="factor_name"),
            _definition(),
            source_factor_name="alpha",
            source_watermark_id="watermark",
        )


def test_source_reader_can_use_bounded_preflight_row_ids_without_hashing() -> None:
    source = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "trade_date": [date(2025, 1, 2), date(2025, 1, 2)],
            "factor_name": ["alpha", "alpha"],
            "value": [1.5, 1.5],
        }
    )

    normalized = normalize_technical_factor_source(
        source,
        _definition(),
        source_factor_name="alpha",
        source_watermark_id="watermark",
        include_source_row_id=False,
    )

    assert normalized["source_row_id"].tolist() == [
        "preflight-000000000000",
        "preflight-000000000001",
    ]
