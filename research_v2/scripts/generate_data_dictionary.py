"""Render the V2 data dictionary from the frozen migration contracts.

The ``.dos`` migrations are the authoritative schema source; this script parses
table DDL, partition/sort columns, and forward-only ``addColumn`` migrations,
then writes a markdown data dictionary.  Regenerate after any migration change
so the dictionary cannot drift from the deployed contracts.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH_ROOT / "src"))

from research_v2.db.migrations import discover_migrations  # noqa: E402

_TABLE_DDL = re.compile(
    r"(?P<variable>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*table\(\s*1:0\s*,\s*"
    r"`(?P<columns>[A-Za-z_][A-Za-z0-9_`]*),\s*\[(?P<types>[^\]]*)\]",
    re.DOTALL,
)
_CREATE_TABLE = re.compile(
    r"createPartitionedTable\(\s*table=(?P<variable>[A-Za-z_][A-Za-z0-9_]*),"
    r"\s*tableName=`(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?:,\s*partitionColumns=`(?P<partition>[A-Za-z_][A-Za-z0-9_]*))?"
    r"(?:,\s*sortColumns=`(?P<sort>[A-Za-z_][A-Za-z0-9_`]*))?",
    re.DOTALL,
)
_LOAD_TABLE_VARIABLE = re.compile(
    r"(?P<variable>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*loadTable\(\s*targetDatabase\s*,"
    r"\s*`(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\)"
)
_ADD_COLUMN = re.compile(
    r"addColumn\(\s*(?P<variable>[A-Za-z_][A-Za-z0-9_]*)\s*,"
    r'\s*"(?P<column>[A-Za-z_][A-Za-z0-9_]*)"\s*,\s*(?P<type>[A-Z]+)\s*\)'
)

# Producer lineage and business keys recorded from the materializer contracts.
TABLE_LINEAGE = {
    "schema_migration": ("db.migrations runner", "audit of applied migrations"),
    "migration_lock": ("db.migrations runner", "advisory lock for migration runs"),
    "published_version": ("db.publish publisher", "visibility pointer per object/partition"),
    "pipeline_run": ("runs.audit pipeline_run context manager", "run lifecycle audit"),
    "source_watermark": ("backfill/source adapters", "source observation watermarks"),
    "security_master_scd": ("materializers.security_master", "point-in-time security SCD"),
    "market_daily_clean": ("materializers.market", "cleaned daily market + history features"),
    "market_quarantine": ("materializers.market (quarantine)", "invalid adjustment-factor audit"),
    "market_security_quarantine": (
        "materializers.market (quarantine)",
        "market rows without active SCD records",
    ),
    "tradability_daily": ("materializers.tradability", "signal-date tradability rules"),
    "universe_snapshot": ("materializers.universe", "versioned universe membership"),
    "label_daily": ("materializers.labels", "primary forward-open 5d labels"),
    "factor_daily": ("factor admission pipeline", "admitted factor values"),
    "factor_evaluation_daily": ("materializers.factor_evaluation", "factor RankIC daily"),
    "factor_evaluation_summary": ("materializers.factor_evaluation", "factor RankIC summary"),
    "factor_repair": ("factor repair pipeline", "repair overlay rows"),
    "factor_repair_candidates": ("factor repair pipeline", "repair candidate audit"),
    "factor_quality_latest": ("factor quality latest pipeline", "latest quality snapshot"),
    "factor_window_quality": ("scripts.record_regime_window_quality", "signed IC metrics per factor/window/universe (pre-training + train + validation)"),
    "factor_pair_redundancy": ("scripts.record_regime_window_quality", "pairwise median abs Spearman redundancy per window"),
    "factor_quality_grade": ("scripts.record_regime_window_quality", "four-dimension quality grade per factor"),
    "fundamental_pit_snapshot": ("fundamental snapshot pipeline", "fundamental PIT data"),
    "model_oos_rank_ic_daily": ("materializers.oos", "model OOS RankIC daily evidence"),
    "model_oos_rank_ic_summary": ("materializers.oos", "model OOS RankIC summary evidence"),
    "active_model_pointer": (
        "registry.active_pointer",
        "consumer active model pointer + switch audit",
    ),
    "data_quality_result": ("quality.engine DQ run", "blocking DQ check outcomes"),
    "dataset_manifest": ("datasets.manifest", "content-addressed dataset identity"),
    "experiment_run": ("training.search experiment audit", "trial lease/audit records"),
    "factor_observation_clean": ("factor materialization", "cleaned admitted factor observations"),
    "factor_observation_clean_latest": ("compact pipeline", "latest compact factor observations"),
    "factor_quality_snapshot": ("factor quality pipeline", "factor quality evidence snapshot"),
    "factor_quarantine": ("factor admission quarantine", "rejected factor rows audit"),
    "factor_rank_ic_daily": ("materializers.factor_evaluation", "factor RankIC daily"),
    "factor_rank_ic_summary": ("materializers.factor_evaluation", "factor RankIC summary"),
    "factor_registry": ("factor admission registry", "admitted factor identities/versions"),
    "factor_repair_audit": ("factor repair pipeline", "repair action audit"),
    "factor_repair_candidate": ("factor repair pipeline", "repair candidates"),
    "factor_repair_quarantine": ("factor repair pipeline", "repair quarantine"),
    "factor_set": ("factor set freezing", "frozen factor sets"),
    "factor_set_member": ("factor set freezing", "factor set membership"),
    "factor_similarity": ("candidate pair similarity", "factor pair similarity evidence"),
    "label_daily_latest": ("compact pipeline", "latest compact labels"),
    "latest_manifest": ("compact pipeline", "latest compact manifests"),
    "locked_test_audit": ("registry.locked_test", "2026 locked test audit trail"),
    "metric_daily": ("evaluation metric materializer", "daily evaluation metrics"),
    "metric_summary": ("evaluation metric materializer", "summary evaluation metrics"),
    "model_registry": ("registry.model_registry", "model lifecycle status machine"),
    "prediction_daily": ("inference.publisher", "published daily predictions"),
    "source_reconciliation": (
        "sources reconciliation (Tushare)",
        "cross-source reconciliation records",
    ),
    "split_definition": ("datasets.splits", "frozen split definitions"),
    "tradability_daily_latest": ("compact pipeline", "latest compact tradability"),
    "universe_daily": ("universe materialization", "daily universe membership"),
    "universe_daily_latest": ("compact pipeline", "latest compact universe"),
    "universe_definition": ("universe definitions", "immutable universe rule versions"),
}
BUSINESS_KEYS = {
    "schema_migration": ("schema_version",),
    "published_version": ("object_name", "logical_partition", "published_at"),
    "security_master_scd": ("ts_code", "valid_from"),
    "market_daily_clean": ("trade_date", "ts_code"),
    "tradability_daily": ("trade_date", "ts_code"),
    "universe_snapshot": ("signal_date", "universe_id", "universe_version", "ts_code"),
    "label_daily": ("signal_date", "ts_code", "label_id", "label_version"),
    "model_oos_rank_ic_daily": ("signal_date", "evaluation_id"),
}


def main() -> int:
    migrations = discover_migrations(RESEARCH_ROOT / "migrations")
    tables: dict[str, dict] = {}
    variable_tables: dict[str, str] = {}
    for migration in migrations:
        variables: dict[str, list[tuple[str, str]]] = {}
        for match in _TABLE_DDL.finditer(migration.script):
            columns = [c for c in match.group("columns").split("`") if c]
            types = [t.strip() for t in match.group("types").split(",") if t.strip()]
            if len(columns) != len(types):
                raise ValueError(
                    f"migration {migration.version:03d} variable {match.group('variable')}"
                    f" has {len(columns)} columns but {len(types)} types"
                )
            variables[match.group("variable")] = list(zip(columns, types, strict=True))
        for match in _CREATE_TABLE.finditer(migration.script):
            variable = match.group("variable")
            if variable not in variables:
                raise ValueError(
                    f"migration {migration.version:03d} creates {match.group('name')}"
                    f" from unknown variable {variable}"
                )
            tables[match.group("name")] = {
                "columns": variables[variable],
                "partition": match.group("partition") or "-",
                "sort": (match.group("sort") or "-").replace("`", ", "),
                "migration": migration.version,
            }
            variable_tables[variable] = match.group("name")
        for match in _LOAD_TABLE_VARIABLE.finditer(migration.script):
            variable_tables[match.group("variable")] = match.group("name")
        for match in _ADD_COLUMN.finditer(migration.script):
            table_name = variable_tables.get(match.group("variable"))
            if table_name is None or table_name not in tables:
                raise ValueError(
                    f"migration {migration.version:03d} addColumn targets unknown"
                    f" variable {match.group('variable')}"
                )
            tables[table_name]["columns"] = tables[table_name]["columns"] + [
                (match.group("column"), match.group("type"))
            ]

    lines = [
        "# V2 数据字典（自动生成）",
        "",
        "由 `scripts/generate_data_dictionary.py` 从 `research_v2/migrations/*.dos`",
        "冻结合约渲染。迁移变更后必须重新生成本文件。",
        "",
        f"迁移版本：001–{migrations[-1].version:03d}；表数量：{len(tables)}。",
        "",
    ]
    for name in sorted(tables):
        spec = tables[name]
        producer, description = TABLE_LINEAGE.get(name, ("（见迁移脚本）", "-"))
        keys = ", ".join(BUSINESS_KEYS.get(name, ("-")))
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"- 来源：迁移 {spec['migration']:03d}；生产者：{producer}")
        lines.append(f"- 说明：{description}")
        lines.append(f"- 业务键：`{keys}`")
        lines.append(f"- 分区列：`{spec['partition']}`；排序列：`{spec['sort']}`")
        lines.append("")
        lines.append("| 列名 | 类型 |")
        lines.append("|---|---|")
        for column, column_type in spec["columns"]:
            lines.append(f"| {column} | {column_type} |")
        lines.append("")

    output = RESEARCH_ROOT.parent / "docs" / "v2_data_dictionary.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {output} ({len(tables)} tables)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
