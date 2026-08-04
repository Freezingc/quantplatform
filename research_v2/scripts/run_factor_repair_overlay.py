"""Run a bounded, read-only factor repair overlay audit.

The command intentionally defaults to a dry run.  It resolves the candidate
set from the source-quality snapshot, reads one source table/month group at a
time, and delegates all repair decisions to the pure overlay and persistence
contracts.  A V2 publication is only attempted after ``--publish`` is passed
explicitly and every blocker check passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import groupby
from pathlib import Path
from typing import Any, Protocol, cast

import pandas as pd

from research_v2.contracts.factor import FactorDefinition
from research_v2.db.publish import StagedBatch, V2BatchPublisher
from research_v2.db.session import V2WriteSession, YesLabReadOnlySession
from research_v2.factors.materializer import build_technical_factor_batches
from research_v2.factors.repair import RepairStatus, audit_factor_source
from research_v2.factors.repair_overlay import (
    KnownFormulaCandidateProvider,
    NullRepairProvider,
    RepairProvider,
    build_repair_overlay,
)
from research_v2.factors.repair_persistence import (
    RepairAuditRecord,
    RepairDecisionStatus,
    RepairIdentity,
    RepairQuarantineRow,
    build_repair_audit_batch,
    build_repair_identity,
    build_repair_quarantine_batch,
    repair_batch_quality_checks,
)
from research_v2.factors.source_backfill import (
    FactorMonth,
    build_factor_definitions,
)
from research_v2.runs.context import RunContext

SOURCE_SCOPE_ID = "SCREEN_TRAIN2020_2024_SOURCE_DQ_COMBINED_V1"
SOURCE_STATUS = "SOURCE_DQ_PASS_PENDING_SIMILARITY"
LOCKED_TEST_START = date(2026, 1, 1)
TRAIN_START = date(2020, 1, 1)
TRAIN_END = date(2024, 12, 31)
VALIDATION_START = date(2025, 1, 1)
VALIDATION_END = date(2025, 12, 31)
_GROUP_SIZE = 3
_FACTOR_ID_RE = re.compile(r"^yeslab_[0-9]+$")
_TABLE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_KNOWN_FORMULA_IDS = frozenset({"yeslab_391", "yeslab_433"})


class FactorRepairRunnerError(RuntimeError):
    """Raised when a repair run cannot safely proceed."""


class QuerySession(Protocol):
    def query(self, script: str) -> object: ...


@dataclass(frozen=True)
class RepairRunnerResult:
    """In-memory run manifest returned by :func:`run_repair` and ``main``."""

    phase: str
    start: date
    end: date
    factor_ids: tuple[str, ...]
    audit_batch: StagedBatch
    quarantine_batch: StagedBatch | None
    clean_batches: tuple[StagedBatch, ...]
    identity: RepairIdentity
    published: bool
    status_counts: tuple[tuple[str, int], ...]
    counts: Mapping[str, int]
    audit_batches: tuple[StagedBatch, ...] = ()
    quarantine_batches: tuple[StagedBatch, ...] = ()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI options and enforce the phase/date safety boundary."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train", "validation"), default="train")
    parser.add_argument("--start", type=date.fromisoformat)
    parser.add_argument("--end", type=date.fromisoformat)
    parser.add_argument("--factor-id", dest="factor_id")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="explicitly select the default no-write mode (the default)",
    )
    parser.add_argument("--group-size", type=int, default=_GROUP_SIZE)
    parser.add_argument("--target-database", default=os.getenv("QRV2_DDB_DATABASE", "dfs://QuantResearchV2Test"))
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)

    defaults = {
        "train": (TRAIN_START, TRAIN_END),
        "validation": (VALIDATION_START, VALIDATION_END),
    }
    default_start, default_end = defaults[str(args.phase)]
    args.start = args.start or default_start
    args.end = args.end or default_end
    if args.start > args.end:
        raise ValueError("repair start date must not be after end date")
    if args.end >= LOCKED_TEST_START:
        raise ValueError("factor repair cannot read 2026 locked-test data")
    phase_lower, phase_upper = defaults[str(args.phase)]
    if args.start < phase_lower or args.end > phase_upper:
        raise ValueError(
            f"{args.phase} repair must remain within "
            f"{phase_lower.isoformat()}..{phase_upper.isoformat()}"
        )
    if args.publish and args.dry_run:
        raise ValueError("--publish and --dry-run cannot be combined")
    if args.group_size < 1:
        raise ValueError("group size must be positive")
    if args.factor_id is not None and _FACTOR_ID_RE.fullmatch(str(args.factor_id)) is None:
        raise ValueError("factor_id must use yeslab_<integer> form")
    return args


def resolve_factor_ids(
    target: QuerySession,
    factor_id: str | None = None,
    *,
    target_database: str | None = None,
) -> tuple[str, ...]:
    """Resolve IDs exclusively from the source-qualified quality scope."""

    database_value = target_database or getattr(target, "target_database", None)
    database = str(database_value or os.getenv("QRV2_DDB_DATABASE", "dfs://QuantResearchV2Test"))
    _validate_v2_database(database)
    script = (
        "select factor_id,metrics_json from "
        f"loadTable('{database}','factor_quality_snapshot') "
        f"where scope_id='{SOURCE_SCOPE_ID}'"
    )
    result = target.query(script)
    if not isinstance(result, pd.DataFrame):
        raise FactorRepairRunnerError("source-quality scope query did not return a DataFrame")
    if not {"factor_id", "metrics_json"}.issubset(result.columns):
        raise FactorRepairRunnerError("source-quality scope is missing factor_id/metrics_json")
    resolved: set[str] = set()
    for row in result.itertuples(index=False):
        body = _metrics_body(row.metrics_json)
        if body.get("decision_status") != SOURCE_STATUS:
            continue
        normalized = _normalize_factor_id(row.factor_id)
        if normalized is not None:
            resolved.add(normalized)
    ids = tuple(sorted(resolved))
    if not ids:
        raise FactorRepairRunnerError(
            f"no factors passed {SOURCE_SCOPE_ID} with {SOURCE_STATUS}"
        )
    if factor_id is not None:
        normalized_requested = _normalize_factor_id(factor_id)
        if normalized_requested not in resolved:
            raise ValueError(
                f"factor_id {factor_id!r} is outside the resolved source-qualified factor set"
            )
        return (cast(str, normalized_requested),)
    return ids


def run_repair(
    args: argparse.Namespace,
    *,
    source_session: QuerySession,
    target_session: QuerySession,
) -> RepairRunnerResult:
    """Run the repair with guaranteed cleanup for every spool outcome."""

    overlay_spool = tempfile.TemporaryDirectory(prefix="qrv2-factor-repair-")
    try:
        return _run_repair_with_spool(
            args,
            source_session=source_session,
            target_session=target_session,
            spool_dir=Path(overlay_spool.name),
        )
    finally:
        overlay_spool.cleanup()


def _run_repair_with_spool(
    args: argparse.Namespace,
    *,
    source_session: QuerySession,
    target_session: QuerySession,
    spool_dir: Path,
) -> RepairRunnerResult:
    """Run the bounded overlay audit using injected sessions."""

    factor_ids = resolve_factor_ids(
        target_session,
        args.factor_id,
        target_database=getattr(target_session, "target_database", None),
    )
    metadata = _query_metadata(source_session, factor_ids)
    definitions = build_factor_definitions(metadata, factor_ids)
    cache = _load_market_cache()
    months = _iter_months(args.start, args.end)
    if not months:
        raise FactorRepairRunnerError("repair date range does not contain a calendar month")

    audits: list[RepairAuditRecord] = []
    quarantines: list[RepairQuarantineRow] = []
    # Keep only paths in memory during the audit pass.  The raw/clean frames
    # themselves are spooled per factor/month so a five-year run does not hold
    # the complete repaired panel in client memory.
    overlay_paths: list[tuple[FactorDefinition, FactorMonth, Path]] = []
    counts: Counter[str] = Counter()
    source_watermarks: list[str] = []

    grouped: dict[str, list[FactorDefinition]] = {}
    for definition in definitions:
        grouped.setdefault(definition.source_table, []).append(definition)
    needs_market_fallback = any(
        definition.factor_id in _KNOWN_FORMULA_IDS for definition in definitions
    )

    for month in months:
        month_cache = cache
        if month_cache is None and needs_market_fallback:
            month_cache = _load_v2_market_cache(target_session, month)
        for source_table in sorted(grouped):
            table_definitions = sorted(grouped[source_table], key=lambda item: item.factor_id)
            for offset in range(0, len(table_definitions), int(args.group_size)):
                group = tuple(table_definitions[offset : offset + int(args.group_size)])
                source = _query_source_chunk(
                    source_session,
                    source_table=source_table,
                    factor_names=tuple(item.display_name for item in group),
                    start=month.start_date,
                    end=month.end_date,
                )
                for definition in group:
                    factor_source = source.loc[
                        source["factor_name"].astype(str) == definition.display_name
                    ].copy()
                    provider = _provider_for_definition(definition, month_cache, month)
                    watermark = _source_watermark(definition, month, provider)
                    source_watermarks.append(watermark)
                    if factor_source.empty:
                        audit, quarantine_rows = _missing_source_evidence(
                            definition, month, provider, watermark
                        )
                        audits.append(audit)
                        quarantines.extend(quarantine_rows)
                        counts["blocked"] += 1
                        counts["audited"] += 1
                        counts["quarantine_rows"] += len(quarantine_rows)
                        continue
                    try:
                        overlay = build_repair_overlay(
                            factor_source,
                            provider,
                            definition=definition,
                            factor_id=definition.factor_id,
                            factor_name=definition.display_name,
                            logical_partition=month.logical_partition,
                            source_watermark_id=watermark,
                        )
                    except (TypeError, ValueError, KeyError) as error:
                        audit, quarantine_rows = _provider_failure(
                            definition,
                            month,
                            factor_source,
                            provider,
                            watermark,
                            str(error),
                        )
                        audits.append(audit)
                        quarantines.extend(quarantine_rows)
                        counts["blocked"] += 1
                        counts["audited"] += 1
                        counts["quarantine_rows"] += len(quarantine_rows)
                        continue
                    audit = _audit_record(definition, month, overlay, watermark)
                    audits.append(audit)
                    q_rows = _quarantine_rows(definition, month, overlay, audit)
                    quarantines.extend(q_rows)
                    counts["filled_rows"] += int(overlay.filled_rows)
                    counts["added_rows"] += int(overlay.added_rows)
                    counts["quarantine_rows"] += len(q_rows)
                    counts["blocked"] += int(overlay.status is RepairStatus.BLOCKED)
                    counts["audited"] += 1
                    if overlay.status is not RepairStatus.BLOCKED and not overlay.clean.empty:
                        path = spool_dir / (
                            f"{len(overlay_paths):08d}-{definition.factor_id}-"
                            f"{month.logical_partition}.pkl"
                        )
                        overlay.clean.to_pickle(path)
                        overlay_paths.append((definition, month, path))

    if not audits:
        raise FactorRepairRunnerError("repair run produced no audit rows")
    identity = build_repair_identity(audits, quarantines)
    audit_batches = tuple(
        build_repair_audit_batch(partition, identity=identity)
        for _, partition in _group_audits_by_partition(audits)
    )
    quarantine_batches = tuple(
        batch
        for _, partition in _group_quarantine_by_partition(quarantines)
        if (batch := build_repair_quarantine_batch(partition, identity=identity)) is not None
    )
    # Keep singular aliases for one-month callers while exposing all bounded
    # persistence batches for a multi-year run.
    audit_batch = audit_batches[0]
    quarantine_batch = quarantine_batches[0] if quarantine_batches else None
    context = RunContext(run_id=identity.run_id, data_version=identity.data_version)
    clean_stream = _iter_clean_batches(
        overlay_paths,
        context=context,
        identity=identity,
        counts=counts,
    )
    first_clean_batch: StagedBatch | None = None
    status_counts = tuple(sorted(Counter(str(record.decision_status) for record in audits).items()))
    counts["audit_rows"] = len(audits)
    counts["factor_count"] = len(factor_ids)
    counts["month_count"] = len(months)
    published = False
    if bool(args.publish):
        _publish(
            target_session,
            audit_batches=audit_batches,
            quarantine_batches=quarantine_batches,
            clean_batches=clean_stream,
            audits=audits,
            identity=identity,
        )
        published = True
    else:
        # Dry-run callers receive one representative batch without retaining
        # every factor/month batch; the stream has already audited all chunks.
        for batch in clean_stream:
            if first_clean_batch is None:
                first_clean_batch = batch
    return RepairRunnerResult(
        phase=str(args.phase),
        start=args.start,
        end=args.end,
        factor_ids=factor_ids,
        audit_batch=audit_batch,
        quarantine_batch=quarantine_batch,
        clean_batches=((first_clean_batch,) if first_clean_batch is not None else ()),
        identity=identity,
        published=published,
        status_counts=status_counts,
        counts=dict(counts),
        audit_batches=audit_batches,
        quarantine_batches=quarantine_batches,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    source_session: QuerySession | None = None,
    target_session: QuerySession | None = None,
) -> RepairRunnerResult:
    """CLI entry point; sessions can be injected by unit tests."""

    args = parse_args(argv)
    if source_session is None or target_session is None:
        source_session, target_session = _connect_sessions(str(args.target_database))
    result = run_repair(
        args,
        source_session=source_session,
        target_session=target_session,
    )
    mode = "published" if result.published else "dry-run"
    print(
        f"factor repair {mode}: phase={result.phase} factors={len(result.factor_ids)} "
        f"months={result.counts['month_count']} audits={result.counts['audit_rows']} "
        f"clean_rows={result.counts.get('clean_rows', 0)} "
        f"quarantine_rows={result.counts.get('quarantine_rows', 0)} "
        f"blocked={result.counts.get('blocked', 0)}",
        flush=True,
    )
    for status, count in result.status_counts:
        print(f"{status}: {count}", flush=True)
    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(
                {
                    "phase": result.phase,
                    "start": result.start.isoformat(),
                    "end": result.end.isoformat(),
                    "factor_ids": result.factor_ids,
                    "data_version": result.identity.data_version,
                    "source_watermark_id": result.identity.source_watermark_id,
                    "published": result.published,
                    "counts": result.counts,
                    "status_counts": result.status_counts,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return result


def _query_metadata(source: QuerySession, factor_ids: tuple[str, ...]) -> pd.DataFrame:
    numeric = sorted({int(item.removeprefix("yeslab_")) for item in factor_ids})
    ids = ",".join(str(item) for item in numeric)
    result = source.query(
        "select factor_id,factor_name,creator,update_frequency,store_factor "
        "from loadTable('dfs://YesLab','factor_metadata') "
        f"where factor_id in [{ids}]"
    )
    if not isinstance(result, pd.DataFrame):
        raise FactorRepairRunnerError("factor metadata query did not return a DataFrame")
    return result


def _query_source_chunk(
    source: QuerySession,
    *,
    source_table: str,
    factor_names: tuple[str, ...],
    start: date,
    end: date,
) -> pd.DataFrame:
    if _TABLE_RE.fullmatch(source_table) is None:
        raise FactorRepairRunnerError(f"unsafe YesLab source table: {source_table!r}")
    names = ",".join("'" + name.replace("'", "''") + "'" for name in factor_names)
    script = (
        "select ts_code,trade_date,factor_name,value from "
        f"loadTable('dfs://YesLab','{source_table}') "
        f"where factor_name in [{names}] "
        f"and trade_date between date('{start:%Y.%m.%d}') "
        f"and date('{end:%Y.%m.%d}')"
    )
    result = source.query(script)
    if not isinstance(result, pd.DataFrame):
        raise FactorRepairRunnerError(f"source query did not return a DataFrame: {source_table}")
    required = {"ts_code", "trade_date", "factor_name", "value"}
    if result.empty and not required.issubset(result.columns):
        return pd.DataFrame(columns=["ts_code", "trade_date", "factor_name", "value"])
    if not required.issubset(result.columns):
        missing = sorted(required.difference(result.columns))
        raise FactorRepairRunnerError(f"source query is missing columns: {missing}")
    return result.loc[:, ["ts_code", "trade_date", "factor_name", "value"]].copy()


def _provider_for_definition(
    definition: FactorDefinition,
    cache: pd.DataFrame | None,
    month: FactorMonth,
) -> RepairProvider:
    if definition.factor_id not in _KNOWN_FORMULA_IDS or cache is None:
        return NullRepairProvider()
    market = cache.copy(deep=True)
    if "trade_date" in market.columns:
        dates = pd.to_datetime(market["trade_date"], errors="coerce").dt.date
        # GP_022/GP_064 use rolling windows of roughly 60/73 trading rows;
        # 180 calendar days safely covers weekends and market holidays.
        lookback_start = month.start_date - timedelta(days=180)
        market = market.loc[(dates >= lookback_start) & (dates <= month.end_date)].copy()
    required = {"trade_date", "ts_code", "close", "amount", "pct_chg"}
    if not required.issubset(market.columns):
        return NullRepairProvider()
    watermark = hashlib.sha256(
        pd.util.hash_pandas_object(market.sort_values(sorted(required)), index=False)
        .to_numpy(dtype="uint64")
        .tobytes()
    ).hexdigest()
    source_name = str(market.attrs.get("repair_market_source", "tushare-cache"))
    return KnownFormulaCandidateProvider(market, source_watermark_id=f"{source_name}:{watermark}")


def _load_market_cache() -> pd.DataFrame | None:
    token = os.getenv("QRV2_TUSHARE_TOKEN") or os.getenv("TUSHARE_TOKEN")
    cache_name = os.getenv("QRV2_TUSHARE_CACHE") or os.getenv("TUSHARE_CACHE")
    if not token or not cache_name:
        return None
    path = Path(cache_name)
    if not path.is_file():
        return None
    try:
        if path.suffix in {".parquet", ".pq"}:
            return pd.read_parquet(path)
        if path.suffix in {".pkl", ".pickle"}:
            return pd.read_pickle(path)
        return pd.read_csv(path)
    except (OSError, ValueError, ImportError):
        return None


def _load_v2_market_cache(
    target: QuerySession,
    month: FactorMonth,
    *,
    target_database: str | None = None,
) -> pd.DataFrame | None:
    """Read one bounded V2 market chunk for formula repair warmup."""

    database_value = target_database or getattr(target, "target_database", None)
    database = str(database_value or os.getenv("QRV2_DDB_DATABASE", "dfs://QuantResearchV2Test"))
    _validate_v2_database(database)
    lookback_start = month.start_date - timedelta(days=180)
    query = (
        "select trade_date,ts_code,raw_close,amount from "
        f"loadTable('{database}','market_daily_clean') "
        f"where trade_date between date('{lookback_start:%Y.%m.%d}') "
        f"and date('{month.end_date:%Y.%m.%d}')"
    )
    try:
        result = target.query(query)
    except (RuntimeError, ValueError):
        return None
    if not isinstance(result, pd.DataFrame):
        return None
    required = {"trade_date", "ts_code", "raw_close", "amount"}
    if not required.issubset(result.columns) or result.empty:
        return None
    market = result.loc[:, ["trade_date", "ts_code", "raw_close", "amount"]].copy()
    market["trade_date"] = pd.to_datetime(market["trade_date"], errors="coerce").dt.date
    market["ts_code"] = market["ts_code"].astype(str)
    market["close"] = pd.to_numeric(market["raw_close"], errors="coerce")
    market["amount"] = pd.to_numeric(market["amount"], errors="coerce")
    market = market.dropna(subset=["trade_date", "ts_code"])
    market = market.sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)
    market["pct_chg"] = (
        market.groupby("ts_code", sort=False)["close"].pct_change() * 100.0
    ).round(12)
    market = market.sort_values(["trade_date", "ts_code"], kind="mergesort").reset_index(drop=True)
    market.attrs["repair_market_source"] = "v2-market-daily-clean"
    return market.loc[:, ["trade_date", "ts_code", "close", "amount", "pct_chg"]]


def _audit_record(
    definition: FactorDefinition,
    month: FactorMonth,
    overlay: Any,
    watermark: str,
) -> RepairAuditRecord:
    decision = overlay.decision
    source_audit = decision.source_audit
    candidate_audit = decision.candidate_audit
    method = str(overlay.repair_method)
    provider_hash = _hash_text(f"{method}|{overlay.source_watermark_id}")
    return RepairAuditRecord(
        as_of_date=month.end_date,
        logical_partition=month.logical_partition,
        factor_id=definition.factor_id,
        factor_version=definition.factor_version,
        source_table=definition.source_table,
        source_factor_name=definition.display_name,
        repair_method=method,
        source_rows=source_audit.source_rows,
        candidate_rows=candidate_audit.source_rows if candidate_audit is not None else 0,
        exact_duplicate_groups=source_audit.exact_duplicate_groups,
        exact_duplicate_rows=source_audit.exact_duplicate_rows,
        conflict_groups=decision.conflict_groups,
        nonfinite_rows=source_audit.nonfinite_rows,
        overlap_finite_rows=(
            candidate_audit.finite_rows if candidate_audit is not None else 0
        ),
        fillable_rows=decision.filled_rows + decision.added_rows,
        filled_rows=decision.filled_rows,
        added_rows=decision.added_rows,
        remaining_nonfinite_rows=decision.unresolved_rows,
        mismatch_rows=decision.mismatch_rows,
        max_abs_diff=decision.max_abs_diff,
        max_rel_diff=decision.max_rel_diff,
        pit_violation_rows=decision.pit_violation_rows,
        unresolved_rows=decision.unresolved_rows,
        decision_status=str(decision.status),
        rules_version=1,
        source_watermark_id=watermark,
        source_definition_hash=definition.definition_hash,
        provider_hash=provider_hash,
        candidate_hash=overlay.candidate_hash,
        start_date=month.start_date,
        end_date=month.end_date,
    )


def _missing_source_evidence(
    definition: FactorDefinition,
    month: FactorMonth,
    provider: RepairProvider,
    watermark: str,
) -> tuple[RepairAuditRecord, list[RepairQuarantineRow]]:
    method = str(getattr(provider, "method", provider.__class__.__name__))
    candidate_hash = _hash_text("EMPTY_CANDIDATE")
    provider_hash = _hash_text(f"{method}|{watermark}")
    audit = RepairAuditRecord(
        as_of_date=month.end_date,
        logical_partition=month.logical_partition,
        factor_id=definition.factor_id,
        factor_version=definition.factor_version,
        source_table=definition.source_table,
        source_factor_name=definition.display_name,
        repair_method=method,
        decision_status=RepairDecisionStatus.BLOCKED.value,
        unresolved_rows=1,
        remaining_nonfinite_rows=1,
        rules_version=1,
        source_watermark_id=watermark,
        source_definition_hash=definition.definition_hash,
        provider_hash=provider_hash,
        candidate_hash=candidate_hash,
        start_date=month.start_date,
        end_date=month.end_date,
    )
    row = RepairQuarantineRow(
        observation_date=month.end_date,
        ts_code="__NO_SOURCE_EVIDENCE__",
        factor_id=definition.factor_id,
        factor_version=definition.factor_version,
        factor_name=definition.display_name,
        reason="NO_SOURCE_EVIDENCE",
        source_value=None,
        candidate_value=None,
        repair_method=method,
        candidate_hash=candidate_hash,
        source_watermark_id=watermark,
        source_table=definition.source_table,
        source_definition_hash=definition.definition_hash,
        provider_hash=provider_hash,
        rules_version=1,
        start_date=month.start_date,
        end_date=month.end_date,
    )
    return audit, [row]


def _provider_failure(
    definition: FactorDefinition,
    month: FactorMonth,
    source: pd.DataFrame,
    provider: RepairProvider,
    watermark: str,
    detail: str,
) -> tuple[RepairAuditRecord, list[RepairQuarantineRow]]:
    """Turn provider/schema failures into a visible, blocking audit row."""

    try:
        source_audit = audit_factor_source(source)
    except (TypeError, ValueError, KeyError):
        source_audit = None
    method = str(getattr(provider, "method", provider.__class__.__name__))
    candidate_hash = _hash_text(f"PROVIDER_FAILURE|{detail}")
    provider_hash = _hash_text(f"{method}|{watermark}")
    audit = RepairAuditRecord(
        as_of_date=month.end_date,
        logical_partition=month.logical_partition,
        factor_id=definition.factor_id,
        factor_version=definition.factor_version,
        source_table=definition.source_table,
        source_factor_name=definition.display_name,
        repair_method=method,
        source_rows=source_audit.source_rows if source_audit is not None else len(source),
        conflict_groups=source_audit.conflict_groups if source_audit is not None else 0,
        nonfinite_rows=source_audit.nonfinite_rows if source_audit is not None else 0,
        remaining_nonfinite_rows=1,
        unresolved_rows=1,
        decision_status=RepairDecisionStatus.BLOCKED.value,
        rules_version=1,
        source_watermark_id=watermark,
        source_definition_hash=definition.definition_hash,
        provider_hash=provider_hash,
        candidate_hash=candidate_hash,
        start_date=month.start_date,
        end_date=month.end_date,
    )
    row = RepairQuarantineRow(
        observation_date=month.end_date,
        ts_code="__PROVIDER_FAILURE__",
        factor_id=definition.factor_id,
        factor_version=definition.factor_version,
        factor_name=definition.display_name,
        reason="PROVIDER_FAILURE",
        source_value=None,
        candidate_value=None,
        repair_method=method,
        candidate_hash=candidate_hash,
        source_watermark_id=watermark,
        source_table=definition.source_table,
        source_definition_hash=definition.definition_hash,
        provider_hash=provider_hash,
        rules_version=1,
        start_date=month.start_date,
        end_date=month.end_date,
    )
    return audit, [row]


def _quarantine_rows(
    definition: FactorDefinition,
    month: FactorMonth,
    overlay: Any,
    audit: RepairAuditRecord,
) -> list[RepairQuarantineRow]:
    rows: list[RepairQuarantineRow] = []
    for item in overlay.quarantine.to_dict("records"):
        observation = pd.Timestamp(item.get("trade_date")).date()
        rows.append(
            RepairQuarantineRow(
                observation_date=observation,
                ts_code=str(item.get("ts_code", "")),
                factor_id=definition.factor_id,
                factor_version=definition.factor_version,
                factor_name=definition.display_name,
                reason=str(item.get("reason", "UNSPECIFIED")),
                source_value=_finite_or_none(item.get("source_value")),
                candidate_value=_finite_or_none(item.get("candidate_value")),
                repair_method=audit.repair_method,
                candidate_hash=overlay.candidate_hash,
                source_watermark_id=audit.source_watermark_id,
                source_table=definition.source_table,
                source_definition_hash=definition.definition_hash,
                provider_hash=audit.provider_hash,
                rules_version=1,
                start_date=month.start_date,
                end_date=month.end_date,
            )
        )
    return rows


def _iter_clean_batches(
    overlay_paths: Iterable[tuple[FactorDefinition, FactorMonth, Path]],
    *,
    context: RunContext,
    identity: RepairIdentity,
    counts: Counter[str],
) -> Iterable[StagedBatch]:
    """Materialize clean chunks lazily from the bounded overlay spool."""

    for definition, _month, path in overlay_paths:
        frame = pd.read_pickle(path)
        materialized = build_technical_factor_batches(
            frame,
            definition,
            source_factor_name=definition.display_name,
            # The persistence identity is the authoritative watermark for the
            # repaired clean batch.  Recomputing a factor-local watermark here
            # would make clean lineage disagree with audit/quarantine lineage.
            source_watermark_id=identity.source_watermark_id,
            context=context,
            record_hash_strategy="fast",
            include_lineage=True,
        )
        counts["clean_batches"] += 1
        counts["clean_rows"] += len(materialized.clean_batch.frame)
        if materialized.quarantine_batch is not None:
            counts["source_quarantine_rows"] += len(materialized.quarantine_batch.frame)
        yield materialized.clean_batch


def _publish(
    target: QuerySession,
    *,
    audit_batches: tuple[StagedBatch, ...],
    quarantine_batches: tuple[StagedBatch, ...],
    clean_batches: Iterable[StagedBatch],
    audits: list[RepairAuditRecord],
    identity: RepairIdentity,
) -> None:
    blocked = [
        record
        for record in audits
        if record.decision_status == RepairDecisionStatus.BLOCKED.value
    ]
    checks = repair_batch_quality_checks(audits, identity=identity)
    if blocked or any(str(check.status) != "PASS" for check in checks):
        raise FactorRepairRunnerError("repair is blocked; no repaired batch was published")
    publisher = V2BatchPublisher(cast(Any, target))
    for batch in audit_batches:
        publisher.publish(batch, checks)
    for batch in quarantine_batches:
        publisher.publish(batch, checks)
    for _, partition_batches in groupby(
        clean_batches, key=lambda batch: batch.logical_partition
    ):
        publisher.publish_chunk_stream(partition_batches, checks)


def _group_audits_by_partition(
    audits: Iterable[RepairAuditRecord],
) -> tuple[tuple[str, tuple[RepairAuditRecord, ...]], ...]:
    grouped: dict[str, list[RepairAuditRecord]] = {}
    for record in audits:
        grouped.setdefault(record.logical_partition, []).append(record)
    return tuple(
        (partition, tuple(rows)) for partition, rows in sorted(grouped.items())
    )


def _group_quarantine_by_partition(
    rows: Iterable[RepairQuarantineRow],
) -> tuple[tuple[str, tuple[RepairQuarantineRow, ...]], ...]:
    grouped: dict[str, list[RepairQuarantineRow]] = {}
    for row in rows:
        partition = row.observation_date.strftime("%Y-%m")
        grouped.setdefault(partition, []).append(row)
    return tuple((partition, tuple(items)) for partition, items in sorted(grouped.items()))


def _connect_sessions(target_database: str) -> tuple[YesLabReadOnlySession, V2WriteSession]:
    import dolphindb as ddb

    client = ddb.session()
    client.connect(
        os.getenv("QRV2_DDB_HOST", "10.176.55.211"),
        int(os.getenv("QRV2_DDB_PORT", "8848")),
        os.getenv("QRV2_DDB_USERNAME", "xiahongjie"),
        os.getenv("QRV2_DDB_PASSWORD", "123456"),
    )
    return YesLabReadOnlySession(client), V2WriteSession(client, target_database=target_database)


def _iter_months(start: date, end: date) -> tuple[FactorMonth, ...]:
    periods = pd.period_range(start=start, end=end, freq="M")
    months: list[FactorMonth] = []
    for period in periods:
        lower = max(start, period.start_time.date())
        upper = min(end, period.end_time.date())
        months.append(FactorMonth(str(period), lower, upper))
    return tuple(months)


def _metrics_body(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _normalize_factor_id(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    if text.lower() in {"nan", "nat", "none"}:
        return None
    if text.isdigit():
        text = f"yeslab_{text}"
    return text if _FACTOR_ID_RE.fullmatch(text) else None


def _source_watermark(
    definition: FactorDefinition,
    month: FactorMonth,
    provider: RepairProvider | None,
) -> str:
    method = str(getattr(provider, "method", "NO_EVIDENCE_BACKFILL"))
    provider_watermark = str(getattr(provider, "source_watermark_id", ""))
    return _hash_text(
        "|".join(
            (
                SOURCE_SCOPE_ID,
                definition.factor_id,
                definition.source_table,
                month.logical_partition,
                definition.definition_hash,
                method,
                provider_watermark,
            )
        )
    )


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _finite_or_none(value: object) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else None


def _validate_v2_database(database: str) -> None:
    if database not in {"dfs://QuantResearchV2", "dfs://QuantResearchV2Test"}:
        raise ValueError("target database must be an approved V2 database")


if __name__ == "__main__":
    main()
