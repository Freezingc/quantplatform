from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from research_v2.sources.tushare import (
    MissingTushareToken,
    TushareAdapter,
    TushareProClient,
    TushareSdkUnavailable,
)


class FakeTushareClient:
    def __init__(self, response: pd.DataFrame, failures: int = 0) -> None:
        self.response = response
        self.failures = failures
        self.calls = 0

    def fetch(self, endpoint: str, params: dict[str, str], token: str) -> pd.DataFrame:
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("temporary upstream failure")
        assert endpoint == "daily"
        assert params == {"start_date": "20250102", "end_date": "20250103"}
        assert token == "token-value"
        return self.response.copy()


def test_tushare_token_is_required_without_echoing_secret(tmp_path: Path) -> None:
    with pytest.raises(MissingTushareToken, match="QRV2_TUSHARE_TOKEN") as error:
        TushareAdapter.from_environment(FakeTushareClient(pd.DataFrame()), tmp_path, environ={})

    assert "token-value" not in str(error.value)


def test_tushare_adapter_retries_and_returns_cached_response(tmp_path: Path) -> None:
    client = FakeTushareClient(pd.DataFrame({"ts_code": ["000001.SZ"]}), failures=1)
    adapter = TushareAdapter.from_environment(
        client,
        tmp_path,
        environ={"QRV2_TUSHARE_TOKEN": "token-value"},
        sleep=lambda _: None,
        response_date=date(2025, 1, 4),
    )

    first = adapter.fetch_daily(date(2025, 1, 2), date(2025, 1, 3))
    second = adapter.fetch_daily(date(2025, 1, 2), date(2025, 1, 3))

    assert client.calls == 2
    assert first.equals(second)
    assert adapter.last_response_hash


def test_tushare_adapter_caches_an_explicit_secondary_endpoint(tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, str]]] = []

    class StockBasicClient:
        def fetch(self, endpoint: str, params: dict[str, str], token: str) -> pd.DataFrame:
            calls.append((endpoint, params))
            assert token == "token-value"
            return pd.DataFrame({"ts_code": ["000004.SZ"], "list_status": ["D"]})

    adapter = TushareAdapter.from_environment(
        StockBasicClient(),
        tmp_path,
        environ={"QRV2_TUSHARE_TOKEN": "token-value"},
        response_date=date(2025, 1, 4),
    )

    first = adapter.fetch_endpoint(
        "stock_basic",
        {"exchange": "", "list_status": "D", "fields": "ts_code,list_status"},
    )
    second = adapter.fetch_endpoint(
        "stock_basic",
        {"exchange": "", "list_status": "D", "fields": "ts_code,list_status"},
    )

    assert calls == [
        ("stock_basic", {"exchange": "", "list_status": "D", "fields": "ts_code,list_status"})
    ]
    assert first.equals(second)
    assert adapter.last_response_hash


def test_tushare_adapter_builds_one_auditable_historical_security_snapshot(tmp_path: Path) -> None:
    class HistoricalSecurityClient:
        def fetch(self, endpoint: str, params: dict[str, str], token: str) -> pd.DataFrame:
            assert endpoint == "stock_basic"
            assert token == "token-value"
            status = params["list_status"]
            return pd.DataFrame(
                {
                    "ts_code": [f"00000{status}.SZ"],
                    "exchange": ["SZSE"],
                    "market": ["主板"],
                    "list_date": ["20100101"],
                    "delist_date": ["" if status == "L" else "20250101"],
                    "name": [f"Name-{status}"],
                    "list_status": [status],
                }
            )

    adapter = TushareAdapter.from_environment(
        HistoricalSecurityClient(),
        tmp_path,
        environ={"QRV2_TUSHARE_TOKEN": "token-value"},
        response_date=date(2025, 1, 4),
    )

    batch = adapter.fetch_stock_basic_snapshot(as_of_date=date(2025, 1, 3))

    assert batch.source_name == "Tushare"
    assert batch.table_name == "tushare_stock_basic"
    assert batch.max_observation_date == date(2025, 1, 3)
    assert batch.source_row_count == 3
    assert set(batch.frame["list_status"]) == {"L", "D", "P"}
    assert batch.watermark_id


def test_tushare_pro_client_delegates_to_the_official_query_interface() -> None:
    calls: list[tuple[str, dict[str, str]]] = []

    class Pro:
        def query(self, endpoint: str, **params: str) -> pd.DataFrame:
            calls.append((endpoint, params))
            return pd.DataFrame({"ts_code": ["000001.SZ"]})

    module = SimpleNamespace(pro_api=lambda token: Pro())
    client = TushareProClient(module_loader=lambda _: module)

    result = client.fetch("stock_basic", {"list_status": "D"}, "token-value")

    assert result.iloc[0]["ts_code"] == "000001.SZ"
    assert calls == [("stock_basic", {"list_status": "D"})]


def test_tushare_pro_client_has_a_sanitized_missing_sdk_error() -> None:
    client = TushareProClient(
        module_loader=lambda _: (_ for _ in ()).throw(ImportError("missing package"))
    )

    with pytest.raises(TushareSdkUnavailable, match="install") as error:
        client.fetch("daily", {}, "token-value")

    assert "token-value" not in str(error.value)
