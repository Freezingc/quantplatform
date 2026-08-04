from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import run_factor_repair_overlay as runner
from run_factor_repair_overlay import (
    SOURCE_SCOPE_ID,
    SOURCE_STATUS,
    _iter_months,
    _load_market_cache,
    _load_v2_market_cache,
    _provider_for_definition,
    _source_watermark,
    main,
    parse_args,
    resolve_factor_ids,
)


class _FakeTarget:
    target_database = "dfs://QuantResearchV2Test"

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.queries: list[str] = []

    def query(self, script: str) -> pd.DataFrame:
        self.queries.append(script)
        if "factor_quality_snapshot" in script:
            return self.frame
        return pd.DataFrame()


class _FakeSource:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def query(self, script: str) -> pd.DataFrame:
        self.queries.append(script)
        if "factor_metadata" in script:
            return pd.DataFrame(
                {
                    "factor_id": [1],
                    "factor_name": ["GP_001"],
                    "creator": ["unit-test"],
                    "update_frequency": ["DAILY"],
                    "store_factor": ["factor_data1"],
                }
            )
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": [pd.Timestamp("2020-01-02")],
                "factor_name": ["GP_001"],
                "value": [1.0],
            }
        )

    def execute(self, script: str) -> None:
        raise AssertionError(f"source session write attempted: {script}")


class _EmptySource(_FakeSource):
    def query(self, script: str) -> pd.DataFrame:
        self.queries.append(script)
        if "factor_metadata" in script:
            return super().query(script)
        return pd.DataFrame(columns=["ts_code", "trade_date", "factor_name", "value"])


def test_runner_resolves_only_source_qualified_scope_rows() -> None:
    target = _FakeTarget(
        pd.DataFrame(
            {
                "factor_id": ["yeslab_1", "yeslab_2", "yeslab_3"],
                "metrics_json": [
                    json.dumps({"decision_status": SOURCE_STATUS}),
                    json.dumps({"decision_status": "SOURCE_DQ_FAIL"}),
                    json.dumps({"decision_status": SOURCE_STATUS}),
                ],
            }
        )
    )

    assert resolve_factor_ids(target) == ("yeslab_1", "yeslab_3")
    assert len(target.queries) == 1
    assert SOURCE_SCOPE_ID in target.queries[0]
    assert "locked_test_audit" not in target.queries[0]


def test_parser_defaults_to_dry_run_train_scope() -> None:
    args = parse_args(["--phase", "train", "--start", "2020-01-01", "--end", "2024-12-31"])

    assert args.phase == "train"
    assert args.start == date(2020, 1, 1)
    assert args.end == date(2024, 12, 31)
    assert args.publish is False


def test_parser_rejects_unknown_phase_and_locked_test_end_date() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--phase", "test", "--start", "2020-01-01", "--end", "2024-12-31"])

    with pytest.raises(ValueError, match="2026"):
        parse_args(["--phase", "train", "--start", "2025-01-01", "--end", "2026-01-01"])

    with pytest.raises(ValueError, match="train repair must remain"):
        parse_args(["--phase", "train", "--start", "2019-12-01", "--end", "2020-01-31"])

    with pytest.raises(ValueError, match="validation repair must remain"):
        parse_args(
            ["--phase", "validation", "--start", "2024-12-01", "--end", "2025-01-31"]
        )


def test_explicit_factor_id_must_be_in_resolved_scope() -> None:
    target = _FakeTarget(
        pd.DataFrame(
            {
                "factor_id": ["yeslab_1"],
                "metrics_json": [json.dumps({"decision_status": SOURCE_STATUS})],
            }
        )
    )
    with pytest.raises(ValueError, match="resolved source-qualified"):
        resolve_factor_ids(target, factor_id="yeslab_9")


def test_main_dry_run_reads_bounded_source_and_never_writes_source() -> None:
    target = _FakeTarget(
        pd.DataFrame(
            {
                "factor_id": ["yeslab_1"],
                "metrics_json": [json.dumps({"decision_status": SOURCE_STATUS})],
            }
        )
    )
    source = _FakeSource()
    result = main(
        ["--phase", "train", "--start", "2020-01-01", "--end", "2020-01-31"],
        source_session=source,
        target_session=target,
    )

    assert result.published is False
    assert result.counts["audit_rows"] == 1
    assert set(result.clean_batches[0].frame["source_watermark_id"]) == {
        result.identity.source_watermark_id
    }
    source_queries = "\n".join(source.queries)
    assert "trade_date between date('2020.01.01')" in source_queries
    assert "2026" not in source_queries


def test_missing_source_evidence_counts_audit_and_quarantine_rows() -> None:
    target = _FakeTarget(
        pd.DataFrame(
            {
                "factor_id": ["yeslab_1"],
                "metrics_json": [json.dumps({"decision_status": SOURCE_STATUS})],
            }
        )
    )

    result = main(
        ["--phase", "train", "--start", "2020-01-01", "--end", "2020-01-31"],
        source_session=_EmptySource(),
        target_session=target,
    )

    assert result.counts["audited"] == 1
    assert result.counts["quarantine_rows"] == 1
    assert result.counts["blocked"] == 1
    assert not result.clean_batches


def test_provider_failure_counts_quarantine_rows(monkeypatch) -> None:
    target = _FakeTarget(
        pd.DataFrame(
            {
                "factor_id": ["yeslab_1"],
                "metrics_json": [json.dumps({"decision_status": SOURCE_STATUS})],
            }
        )
    )

    def fail_overlay(*args, **kwargs):
        raise ValueError("synthetic provider failure")

    monkeypatch.setattr(runner, "build_repair_overlay", fail_overlay)
    result = main(
        ["--phase", "train", "--start", "2020-01-01", "--end", "2020-01-31"],
        source_session=_FakeSource(),
        target_session=target,
    )

    assert result.counts["audited"] == 1
    assert result.counts["quarantine_rows"] == 1
    assert result.counts["blocked"] == 1


def test_provider_evidence_watermark_changes_source_watermark() -> None:
    definition = SimpleNamespace(
        factor_id="yeslab_1",
        source_table="factor_data1",
        definition_hash="definition-a",
    )
    month = _iter_months(date(2020, 1, 1), date(2020, 1, 31))[0]
    provider_a = SimpleNamespace(
        method="KNOWN_FORMULA_RECONSTRUCTION", source_watermark_id="cache-a"
    )
    provider_b = SimpleNamespace(
        method="KNOWN_FORMULA_RECONSTRUCTION", source_watermark_id="cache-b"
    )

    assert _source_watermark(definition, month, provider_a) != _source_watermark(
        definition, month, provider_b
    )


def test_market_cache_accepts_qrv2_tushare_token_alias(tmp_path, monkeypatch) -> None:
    cache_path = tmp_path / "market.pkl"
    expected = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2020-01-01")],
            "ts_code": ["000001.SZ"],
            "close": [10.0],
            "amount": [100.0],
            "pct_chg": [0.1],
        }
    )
    expected.to_pickle(cache_path)
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    monkeypatch.setenv("QRV2_TUSHARE_TOKEN", "configured")
    monkeypatch.setenv("QRV2_TUSHARE_CACHE", str(cache_path))

    loaded = _load_market_cache()

    assert loaded is not None
    pd.testing.assert_frame_equal(loaded, expected)


def test_v2_market_fallback_derives_pct_chg_from_raw_close() -> None:
    class _MarketTarget:
        target_database = "dfs://QuantResearchV2Test"

        def __init__(self) -> None:
            self.queries: list[str] = []

        def query(self, script: str) -> pd.DataFrame:
            self.queries.append(script)
            if "published_version" in script:
                if "logical_partition='2019-12'" in script:
                    return pd.DataFrame({"data_version": ["v1"]})
                if "logical_partition='2020-01'" in script:
                    return pd.DataFrame({"data_version": ["v2"]})
                return pd.DataFrame(columns=["data_version"])
            return pd.DataFrame(
                {
                    "data_version": ["v1", "v2", "v1", "v2"],
                    "trade_date": pd.to_datetime(
                        ["2019-12-31", "2019-12-31", "2020-01-02", "2020-01-02"]
                    ),
                    "ts_code": ["000001.SZ"] * 4,
                    "raw_close": [10.0, 9.0, 8.0, 11.0],
                    "amount": [100.0, 90.0, 80.0, 120.0],
                }
            )

    target = _MarketTarget()
    month = _iter_months(date(2020, 1, 1), date(2020, 1, 31))[0]

    market = _load_v2_market_cache(target, month)

    assert market is not None
    assert market.attrs["repair_market_source"] == "v2-market-daily-clean"
    assert market.attrs["repair_market_versions"] == "2019-12:v1|2020-01:v2"
    assert market.loc[1, "close"] == 11.0
    assert market.loc[1, "pct_chg"] == 10.0
    assert {"trade_date", "ts_code", "close", "amount", "pct_chg"}.issubset(
        market.columns
    )
    pointer_queries = [query for query in target.queries if "published_version" in query]
    market_query = next(query for query in target.queries if "select trade_date" in query)
    assert any("logical_partition='2020-01'" in query for query in pointer_queries)
    assert "data_version in ['v1','v2']" in market_query
    assert "date('2019.07.05')" in market_query
    assert "date('2020.01.31')" in market_query
    assert all("2026" not in query for query in target.queries)
    provider = _provider_for_definition(SimpleNamespace(factor_id="yeslab_391"), market, month)
    assert provider.source_watermark_id.startswith("v2-market-daily-clean:")


def test_v2_market_fallback_without_published_pointer_fails_closed() -> None:
    class _NoPointerTarget:
        target_database = "dfs://QuantResearchV2Test"

        def __init__(self) -> None:
            self.queries: list[str] = []

        def query(self, script: str) -> pd.DataFrame:
            self.queries.append(script)
            return pd.DataFrame(columns=["data_version"])

    target = _NoPointerTarget()
    month = _iter_months(date(2020, 1, 1), date(2020, 1, 31))[0]

    assert _load_v2_market_cache(target, month) is None
    assert target.queries
    assert all("published_version" in query for query in target.queries)
    assert not any("select trade_date" in query for query in target.queries)


def test_known_gp_provider_retains_lookback_market_rows() -> None:
    definition = SimpleNamespace(factor_id="yeslab_391")
    month = _iter_months(date(2020, 1, 1), date(2020, 1, 31))[0]
    cache = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                ["2019-07-01", "2019-07-05", "2019-10-03", "2019-12-31", "2020-01-01"]
            ),
            "ts_code": ["000001.SZ"] * 5,
            "close": [9.0, 9.1, 9.2, 9.3, 9.4],
            "amount": [100.0] * 5,
            "pct_chg": [0.1] * 5,
        }
    )

    provider = _provider_for_definition(definition, cache, month)

    assert hasattr(provider, "market")
    retained_dates = set(pd.to_datetime(provider.market["trade_date"]).dt.date)
    assert date(2019, 10, 3) in retained_dates
    assert date(2019, 7, 5) in retained_dates
    assert date(2019, 7, 1) not in retained_dates


@pytest.mark.parametrize(
    ("factor_id", "factor_name"),
    [("yeslab_391", "GP_022"), ("yeslab_433", "GP_064")],
)
def test_known_gp_candidate_is_finite_at_month_start(
    factor_id: str, factor_name: str
) -> None:
    definition = SimpleNamespace(factor_id=factor_id)
    month = _iter_months(date(2020, 1, 1), date(2020, 1, 31))[0]
    dates = pd.bdate_range("2019-07-01", "2020-01-31")
    cache = pd.DataFrame(
        {
            "trade_date": dates,
            "ts_code": ["000001.SZ"] * len(dates),
            "close": np.linspace(10.0, 20.0, len(dates)),
            "amount": np.linspace(100.0, 200.0, len(dates)),
            "pct_chg": [0.1] * len(dates),
        }
    )
    provider = _provider_for_definition(definition, cache, month)
    candidate = provider.provide(
        pd.DataFrame(columns=["trade_date", "ts_code", "factor_name", "value"]),
        factor_name=factor_name,
    )

    assert candidate is not None
    first_day = candidate.loc[candidate["trade_date"] == date(2020, 1, 1)]
    assert not first_day.empty
    assert np.isfinite(first_day["candidate_value"].iloc[0])


def test_repair_spool_is_cleaned_when_run_fails(monkeypatch) -> None:
    class _TrackingSpool:
        name = "/tmp/qrv2-test-spool"
        cleaned = False

        def cleanup(self) -> None:
            self.cleaned = True

    spool = _TrackingSpool()

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic repair failure")

    monkeypatch.setattr(runner.tempfile, "TemporaryDirectory", lambda **_: spool)
    monkeypatch.setattr(runner, "_run_repair_with_spool", fail)

    with pytest.raises(RuntimeError, match="synthetic repair failure"):
        runner.run_repair(
            parse_args(["--start", "2020-01-01", "--end", "2020-01-31"]),
            source_session=_FakeSource(),
            target_session=_FakeTarget(
                pd.DataFrame(
                    {
                        "factor_id": ["yeslab_1"],
                        "metrics_json": [json.dumps({"decision_status": SOURCE_STATUS})],
                    }
                )
            ),
        )

    assert spool.cleaned
