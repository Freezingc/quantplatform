"""Secondary-only Tushare adapter with token-safe retry and response caching."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping
from datetime import date
from importlib import import_module
from io import StringIO
from pathlib import Path
from time import sleep as default_sleep
from typing import Any, Protocol

import pandas as pd

from research_v2.contracts.source import SourceBatch


class MissingTushareToken(ValueError):
    """Raised when the required external-source token has not been configured."""


class TushareRequestError(RuntimeError):
    """Raised when a secondary-source request fails after bounded retries."""


class TushareSdkUnavailable(RuntimeError):
    """Raised when the optional Tushare SDK is absent from the runtime environment."""


class TushareClient(Protocol):
    """Minimal protocol implemented by a Tushare Pro client wrapper."""

    def fetch(self, endpoint: str, params: dict[str, str], token: str) -> pd.DataFrame: ...


class TushareProClient:
    """Official Tushare Pro wrapper kept behind the adapter's minimal protocol."""

    def __init__(self, module_loader: Callable[[str], Any] = import_module) -> None:
        self._module_loader = module_loader

    def fetch(self, endpoint: str, params: dict[str, str], token: str) -> pd.DataFrame:
        try:
            module = self._module_loader("tushare")
        except ImportError as error:
            raise TushareSdkUnavailable(
                "Tushare SDK is unavailable; install project dependencies"
            ) from error
        pro_client = module.pro_api(token)
        response = pro_client.query(endpoint, **params)
        if not isinstance(response, pd.DataFrame):
            raise TushareRequestError("Tushare Pro client did not return a DataFrame")
        return response.copy(deep=True)


class TushareAdapter:
    """Fetch cached Tushare data strictly for reconciliation, never primary ingestion."""

    def __init__(
        self,
        client: TushareClient,
        cache_dir: Path,
        token: str,
        *,
        max_attempts: int = 3,
        rate_limit_seconds: float = 0.25,
        sleep: Callable[[float], None] = default_sleep,
        response_date: date | None = None,
    ) -> None:
        if not token.strip():
            raise MissingTushareToken("QRV2_TUSHARE_TOKEN is required")
        if max_attempts < 1 or rate_limit_seconds < 0:
            raise ValueError("retry and rate-limit settings must be non-negative")
        self._client = client
        self._cache_dir = cache_dir
        self._token = token
        self._max_attempts = max_attempts
        self._rate_limit_seconds = rate_limit_seconds
        self._sleep = sleep
        self._response_date = response_date or date.today()
        self.last_response_hash = ""

    @classmethod
    def from_environment(
        cls,
        client: TushareClient,
        cache_dir: Path,
        *,
        environ: Mapping[str, str] | None = None,
        max_attempts: int = 3,
        rate_limit_seconds: float = 0.25,
        sleep: Callable[[float], None] = default_sleep,
        response_date: date | None = None,
    ) -> TushareAdapter:
        environment = os.environ if environ is None else environ
        token = environment.get("QRV2_TUSHARE_TOKEN", "")
        return cls(
            client,
            cache_dir,
            token,
            max_attempts=max_attempts,
            rate_limit_seconds=rate_limit_seconds,
            sleep=sleep,
            response_date=response_date,
        )

    def fetch_daily(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Fetch one cached daily range through a bounded, rate-limited request."""
        if start_date > end_date:
            raise ValueError("start_date cannot be after end_date")
        params = {
            "start_date": start_date.strftime("%Y%m%d"),
            "end_date": end_date.strftime("%Y%m%d"),
        }
        return self.fetch_endpoint("daily", params)

    def fetch_endpoint(self, endpoint: str, params: dict[str, str]) -> pd.DataFrame:
        """Fetch a cacheable secondary-source endpoint without leaking its token."""
        if not re.fullmatch(r"[a-z][a-z0-9_]*", endpoint):
            raise ValueError("endpoint must use lowercase letters, digits, and underscores")
        valid_params = all(
            isinstance(key, str) and isinstance(value, str) for key, value in params.items()
        )
        if not valid_params:
            raise ValueError("Tushare endpoint parameters must be string pairs")
        cache_path = self._cache_path(endpoint, params)
        if cache_path.exists():
            payload = cache_path.read_text(encoding="utf-8")
            self.last_response_hash = hashlib.sha256(payload.encode()).hexdigest()
            return pd.read_json(StringIO(payload), orient="table")
        last_error: Exception | None = None
        for attempt in range(self._max_attempts):
            try:
                response = self._client.fetch(endpoint, params, self._token)
                if not isinstance(response, pd.DataFrame):
                    raise TypeError("Tushare client did not return a DataFrame")
                payload = response.to_json(orient="table", date_format="iso")
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(payload, encoding="utf-8")
                self.last_response_hash = hashlib.sha256(payload.encode()).hexdigest()
                return response.copy(deep=True)
            except Exception as error:
                last_error = error
                if attempt + 1 < self._max_attempts:
                    self._sleep(self._rate_limit_seconds)
        raise TushareRequestError(
            f"Tushare daily request failed after {self._max_attempts} attempts"
        ) from last_error

    def fetch_stock_basic_snapshot(self, *, as_of_date: date) -> SourceBatch:
        """Fetch all Tushare listing states as one immutable reconciliation batch."""
        fields = "ts_code,exchange,market,list_date,delist_date,name,list_status"
        frames: list[pd.DataFrame] = []
        response_hashes: list[str] = []
        for list_status in ("L", "D", "P"):
            frame = self.fetch_endpoint(
                "stock_basic",
                {"exchange": "", "list_status": list_status, "fields": fields},
            )
            frames.append(frame)
            response_hashes.append(self.last_response_hash)
        snapshot = pd.concat(frames, ignore_index=True)
        required_columns = fields.split(",")
        missing_columns = sorted(set(required_columns).difference(snapshot.columns))
        if missing_columns:
            raise TushareRequestError(
                f"Tushare stock_basic response is missing columns: {missing_columns}"
            )
        snapshot = snapshot.loc[:, required_columns].copy()
        if snapshot["ts_code"].isna().any() or snapshot["ts_code"].duplicated().any():
            raise TushareRequestError("Tushare stock_basic response has invalid security keys")
        for column in ("list_date", "delist_date"):
            snapshot[column] = pd.to_datetime(snapshot[column], format="%Y%m%d", errors="coerce")
            snapshot[column] = snapshot[column].dt.date
        snapshot = snapshot.sort_values("ts_code", ignore_index=True)
        schema_payload = "|".join(
            f"{column}:{snapshot[column].dtype}" for column in snapshot.columns
        )
        schema_fingerprint = hashlib.sha256(schema_payload.encode()).hexdigest()
        watermark_payload = "|".join(
            [
                "Tushare",
                "stock_basic",
                as_of_date.isoformat(),
                str(len(snapshot)),
                schema_fingerprint,
                *response_hashes,
            ]
        )
        return SourceBatch(
            frame=snapshot,
            source_name="Tushare",
            table_name="tushare_stock_basic",
            watermark_id=hashlib.sha256(watermark_payload.encode()).hexdigest(),
            max_observation_date=as_of_date,
            source_row_count=len(snapshot),
            schema_fingerprint=schema_fingerprint,
            extracted_at=pd.Timestamp.now(tz="UTC").to_pydatetime(),
        )

    def _cache_path(self, endpoint: str, params: dict[str, str]) -> Path:
        canonical = json.dumps(
            {
                "endpoint": endpoint,
                "params": params,
                "response_date": self._response_date.isoformat(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        cache_key = hashlib.sha256(canonical.encode()).hexdigest()
        return self._cache_dir / f"{cache_key}.json"
