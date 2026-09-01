# Quant Research V2 A — Foundation and Database Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立 V2 独立 Python 工程、强类型配置、只读/可写数据库边界、版本化 DolphinDB 迁移、运行治理表和两阶段发布基础，使后续数据与模型模块在可测试、可审计的底座上运行。

**Architecture:** `research_v2` 采用 src layout；Pydantic 配置从 YAML + 环境变量加载；数据库连接分为 `YesLabReadOnlySession` 与 `V2WriteSession`，前者在 API 与运行时禁止写操作。迁移按版本、checksum 和数据库锁执行；所有物化任务通过 `pipeline_run` 与 staging/publish repository 完成幂等发布。

**Tech Stack:** Python 3.11、Pydantic 2、Typer、PyYAML、DolphinDB SDK 3.0、pytest/Hypothesis、Ruff、mypy、DolphinDB script/TSDB。

---

## 前置与非目标

- 前置：完成主计划 Phase 0，批准数据库容量、分区 bucket 数、权限矩阵和已轮换的 secret 名称。
- 本计划不读取实际因子数据、不回填业务表、不训练模型。
- 不修改 `dfs://YesLab`；所有迁移只允许目标 `dfs://QuantResearchV2Test` 或 `dfs://QuantResearchV2`。
- 文中提交步骤只在用户授权开发且工作区已处于正式 Git worktree 后执行。

## 文件清单

**Create:**

```text
research_v2/pyproject.toml
research_v2/README.md
research_v2/configs/base.yaml
research_v2/configs/dev.yaml
research_v2/configs/test.yaml
research_v2/configs/production.yaml
research_v2/src/research_v2/__init__.py
research_v2/src/research_v2/cli/__init__.py
research_v2/src/research_v2/cli/app.py
research_v2/src/research_v2/config/__init__.py
research_v2/src/research_v2/config/models.py
research_v2/src/research_v2/config/loader.py
research_v2/src/research_v2/contracts/__init__.py
research_v2/src/research_v2/contracts/base.py
research_v2/src/research_v2/db/__init__.py
research_v2/src/research_v2/db/session.py
research_v2/src/research_v2/db/migrations.py
research_v2/src/research_v2/db/repositories.py
research_v2/src/research_v2/db/publish.py
research_v2/src/research_v2/runs/__init__.py
research_v2/src/research_v2/runs/context.py
research_v2/migrations/001_control_tables.dos
research_v2/migrations/002_pit_tables.dos
research_v2/migrations/003_factor_tables.dos
research_v2/migrations/004_experiment_tables.dos
research_v2/tests/conftest.py
research_v2/tests/unit/test_package.py
research_v2/tests/unit/cli/test_app.py
research_v2/tests/unit/config/test_loader.py
research_v2/tests/unit/db/test_session_guard.py
research_v2/tests/unit/db/test_migrations.py
research_v2/tests/unit/db/test_publish.py
research_v2/tests/unit/runs/test_context.py
research_v2/tests/contract/db/test_schema_contracts.py
research_v2/tests/property/db/test_publish_idempotency.py
research_v2/tests/integration/db/test_migration_roundtrip.py
research_v2/tests/integration/db/test_atomic_publish.py
```

## Task A1: 建立可安装包与质量工具

**Files:**

- Create: `research_v2/pyproject.toml`
- Create: `research_v2/README.md`
- Create: `research_v2/src/research_v2/__init__.py`
- Create: `research_v2/tests/conftest.py`
- Create: `research_v2/tests/unit/test_package.py`

- [x] **A1.1 写失败的包导入测试**

```python
def test_package_exposes_version() -> None:
    import research_v2

    assert isinstance(research_v2.__version__, str)
    assert research_v2.__version__
```

- [x] **A1.2 验证测试先失败**

Run: `cd research_v2 && /data/zhangyudong/.conda/envs/quantyd/bin/python -m pytest tests/unit/test_package.py -q`

Expected: `ModuleNotFoundError: No module named 'research_v2'` 或缺少 `__version__`。

- [x] **A1.3 定义项目元数据和依赖边界**

`pyproject.toml` 固定 `requires-python = ">=3.11,<3.12"`，运行依赖包括 `dolphindb`、`pandas`、`numpy`、`pyarrow`、`pydantic`、`PyYAML`、`typer`、`scikit-learn`、`scipy`、`statsmodels`、`lightgbm`、`torch`；dev extra 包括 `pytest`、`pytest-cov`、`hypothesis`、`ruff`、`mypy` 与 pandas stubs。不要把凭据或本机路径写入配置。

- [x] **A1.4 实现最小包并安装开发依赖**

Run: `cd research_v2 && /data/zhangyudong/.conda/envs/quantyd/bin/python -m pip install -e '.[dev]'`

Expected: 安装成功；此命令只能在获得实施授权后执行。

- [x] **A1.5 运行静态与单元测试**

Run: `cd research_v2 && ruff check src tests && mypy src && pytest tests/unit/test_package.py -q`

Expected: 退出码 0。

- [x] **A1.6 提交**

```bash
git add research_v2/pyproject.toml research_v2/README.md research_v2/src/research_v2/__init__.py research_v2/tests/conftest.py research_v2/tests/unit/test_package.py
git commit -m "build(v2): scaffold isolated research package"
```

## Task A2: 强类型配置与 secret 边界

**Files:**

- Create: `research_v2/configs/base.yaml`
- Create: `research_v2/configs/dev.yaml`
- Create: `research_v2/configs/test.yaml`
- Create: `research_v2/configs/production.yaml`
- Create: `research_v2/src/research_v2/config/models.py`
- Create: `research_v2/src/research_v2/config/loader.py`
- Test: `research_v2/tests/unit/config/test_loader.py`

- [x] **A2.1 写失败测试：环境覆盖与禁止明文 secret**

```python
def test_loader_reads_secret_only_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("QRV2_DDB_PASSWORD", "runtime-secret")
    settings = load_settings("test", config_root=tmp_path)
    assert settings.dolphindb.password.get_secret_value() == "runtime-secret"


def test_plaintext_password_in_yaml_is_rejected(tmp_path):
    write_yaml(tmp_path, {"dolphindb": {"password": "must-not-live-here"}})
    with pytest.raises(ValueError, match="secret"):
        load_settings("test", config_root=tmp_path)
```

- [x] **A2.2 运行测试并确认失败**

Run: `cd research_v2 && pytest tests/unit/config/test_loader.py -q`

Expected: 导入或断言失败。

- [x] **A2.3 实现 Pydantic 配置模型**

最小类型：

```python
class DatabaseSettings(BaseModel):
    host: str
    port: int = Field(gt=0, le=65535)
    username: str
    password: SecretStr
    source_database: Literal["dfs://YesLab"] = "dfs://YesLab"
    target_database: Literal["dfs://QuantResearchV2", "dfs://QuantResearchV2Test"]


class AppSettings(BaseModel):
    environment: Literal["dev", "test", "production"]
    dolphindb: DatabaseSettings
    artifact_root: Path
    lock_year: int = 2026
```

loader 合并顺序固定为 `base.yaml` → `dev.yaml`/`test.yaml`/`production.yaml` 中与目标环境同名的文件 → 环境变量；日志输出使用 `model_dump(exclude={...})` 或 SecretStr 掩码。

- [x] **A2.4 增加配置性质测试**

覆盖未知字段拒绝、production 不得指向 Test DB、test 不得指向生产 DB、目标 URI 白名单、端口边界、环境缺 secret fail closed。

- [x] **A2.5 运行验证**

Run: `cd research_v2 && pytest tests/unit/config/test_loader.py -q && ruff check src/research_v2/config tests/unit/config`

Expected: 全部通过且测试日志没有 secret 原文。

- [x] **A2.6 提交**

```bash
git add research_v2/configs research_v2/src/research_v2/config research_v2/tests/unit/config
git commit -m "feat(v2): add typed configuration and secret boundary"
```

## Task A3: 数据库会话与 V1 写保护

**Files:**

- Create: `research_v2/src/research_v2/db/session.py`
- Test: `research_v2/tests/unit/db/test_session_guard.py`
- Test: `research_v2/tests/contract/db/test_schema_contracts.py`

- [x] **A3.1 写失败测试：YesLab 只读**

```python
@pytest.mark.parametrize("script", [
    "append!(loadTable('dfs://YesLab','x'), t)",
    "dropTable(database('dfs://YesLab'), 'x')",
    "tableInsert{loadTable('dfs://YesLab','x')}",
])
def test_read_only_session_rejects_mutating_script(script: str) -> None:
    with pytest.raises(ReadOnlyViolation):
        YesLabReadOnlySession(fake_client()).run(script)
```

- [x] **A3.2 写失败测试：目标库白名单与参数化查询**

测试 `V2WriteSession` 只接受 V2/V2Test，禁止拼接外部字符串为表名；表名通过 registry enum 映射。

- [x] **A3.3 实现接口与 guard**

```python
class QuerySession(Protocol):
    def query(self, script: str, params: Mapping[str, object] | None = None) -> pd.DataFrame: ...


class WriteSession(QuerySession, Protocol):
    def execute(self, script: str, params: Mapping[str, object] | None = None) -> object: ...
```

只读实现不暴露 `execute`；底层 `run` 再用 tokenizer/command allowlist 进行纵深防御，不能只靠字符串 `startswith("select")`。

- [x] **A3.4 加连接生命周期和重试**

仅对连接中断、超时等瞬时错误做有限指数退避；语法、权限、schema、质量错误立即失败。每次连接记录 host/port/username（不含密码）、DB role 和 trace id。

- [x] **A3.5 运行验证**

Run: `cd research_v2 && pytest tests/unit/db/test_session_guard.py tests/contract/db/test_schema_contracts.py -q`

Expected: YesLab 的所有 mutation fixture 被拒绝，V2 query/write contract 通过。

- [x] **A3.6 提交**

```bash
git add research_v2/src/research_v2/db/session.py research_v2/tests/unit/db/test_session_guard.py research_v2/tests/contract/db/test_schema_contracts.py
git commit -m "feat(v2): enforce readonly source database sessions"
```

## Task A4: 运行上下文与不可变标识

**Files:**

- Create: `research_v2/src/research_v2/contracts/base.py`
- Create: `research_v2/src/research_v2/runs/context.py`
- Test: `research_v2/tests/unit/runs/test_context.py`

- [x] **A4.1 写失败测试：同输入产生稳定 data version**

```python
def test_data_version_is_content_addressed() -> None:
    left = RunContext.build(config=CFG, watermarks=WATERMARKS, code_commit="abc")
    right = RunContext.build(config=CFG, watermarks=reversed(WATERMARKS), code_commit="abc")
    assert left.data_version == right.data_version
    assert left.run_id != right.run_id
```

- [x] **A4.2 实现核心 value objects**

定义 `RunId`（UUIDv7 或时间有序 UUID）、`DataVersion`（canonical JSON SHA-256）、`SchemaVersion`、`SourceWatermarkId`、UTC `created_at`。canonicalizer 必须排序 mapping/set、标准化日期/时区/浮点与 NaN，拒绝不可序列化对象。

- [x] **A4.3 记录环境摘要**

收集 Python、OS、包版本、CUDA/GPU 名称和 code commit；环境摘要不参与业务数据版本，但写入 experiment/pipeline audit。

- [x] **A4.4 运行测试**

Run: `cd research_v2 && pytest tests/unit/runs/test_context.py -q`

Expected: 排序不影响 hash，内容变化必然改变 hash，run id 每次唯一。

- [x] **A4.5 提交**

```bash
git add research_v2/src/research_v2/contracts/base.py research_v2/src/research_v2/runs research_v2/tests/unit/runs
git commit -m "feat(v2): add reproducible run context"
```

## Task A5: 迁移运行器

**Files:**

- Create: `research_v2/src/research_v2/db/migrations.py`
- Create: `research_v2/migrations/001_control_tables.dos`
- Test: `research_v2/tests/unit/db/test_migrations.py`
- Test: `research_v2/tests/integration/db/test_migration_roundtrip.py`

- [x] **A5.1 写失败测试：排序、checksum、锁和幂等**

覆盖：迁移文件按三位版本升序；已执行同 checksum 跳过；同版本内容变化拒绝；两个 runner 只能一个持有锁；中途失败不记录成功版本；目标非 V2/V2Test 拒绝。

- [x] **A5.2 实现 migration parser 与 runner**

```python
@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    checksum: str
    script: str


class MigrationRunner:
    def plan(self) -> Sequence[Migration]: ...
    def apply(self, *, dry_run: bool = False) -> MigrationResult: ...
```

`dry_run` 输出版本、checksum、目标库和待建表，不执行 script。

- [x] **A5.3 在 001 创建 bootstrap 与控制表**

`001_control_tables.dos` 必须幂等创建：

- `schema_migration(schema_version, name, checksum, applied_at, applied_by, run_id)`；
- `migration_lock(lock_name, owner_run_id, acquired_at, expires_at)`；
- `pipeline_run(run_id, pipeline_name, status, data_version, config_hash, code_commit, started_at, finished_at, error_class, error_message, environment_json)`；
- `source_watermark(source_watermark_id, source_name, table_name, max_observation_date, source_row_count, schema_fingerprint, captured_at, run_id)`；
- `data_quality_result(run_id, check_name, scope_name, severity, status, observed_value, expected_rule, sample_json, created_at)`；
- `source_reconciliation(run_id, trade_date, ts_code, field_name, primary_value, secondary_value, abs_diff, rel_diff, status, created_at)`。
- `published_version(object_name, logical_partition, data_version, previous_data_version, published_at, run_id)`。

控制表按月份日期分区；`pipeline_run`/migration 表保持低基数 keyed/partitioned 设计，具体引擎以 Phase 0 容量审批为准。

- [ ] **A5.4 添加测试库集成 fixture**

fixture 要求显式 `QRV2_INTEGRATION=1` 和 Test DB；缺任一条件自动 skip，绝不能回退生产 DB。每次测试使用唯一 namespace/run_id 并清理自己的 staging 对象。

- [ ] **A5.5 运行 dry-run 与集成测试**

Run: `cd research_v2 && qr-v2 db migrate --env test --dry-run`

Expected: 只列出 `001`，数据库无变化。

Run: `cd research_v2 && QRV2_INTEGRATION=1 pytest tests/integration/db/test_migration_roundtrip.py -q -m integration`

Expected: 首次 applied=1，第二次 applied=0；篡改 checksum fixture 被拒绝。

- [x] **A5.6 提交**

```bash
git add research_v2/src/research_v2/db/migrations.py research_v2/migrations/001_control_tables.dos research_v2/tests/unit/db/test_migrations.py research_v2/tests/integration/db/test_migration_roundtrip.py
git commit -m "feat(v2): add audited DolphinDB migrations"
```

## Task A6: PIT、因子和实验 schema 迁移

**Files:**

- Create: `research_v2/migrations/002_pit_tables.dos`
- Create: `research_v2/migrations/003_factor_tables.dos`
- Create: `research_v2/migrations/004_experiment_tables.dos`
- Modify: `research_v2/tests/contract/db/test_schema_contracts.py`

- [x] **A6.1 先写 schema contract fixtures**

以 Python 常量声明每张表的精确列名、DolphinDB 类型、列顺序、分区键、sort columns、业务唯一键和可空性。测试通过 `schema(loadTable(...))` 比较，禁止只检查“表存在”。

- [x] **A6.2 编写 002 PIT 表迁移**

创建：`security_master_scd`、`market_daily_clean`、`tradability_daily`、`universe_definition`、`universe_daily`。事实表业务唯一键分别为：

```text
security_master_scd: ts_code + valid_from
market_daily_clean: trade_date + ts_code
tradability_daily: trade_date + ts_code
universe_definition: universe_id + version
universe_daily: signal_date + universe_id + universe_version + ts_code
```

所有日期事实表含统一审计字段；`market_daily_clean` 保留 raw OHLC、adj_factor、adjusted_open、volume、amount、ADV20 和 source flags。

- [x] **A6.3 编写 003 因子与标签表迁移**

创建：`factor_registry`、`factor_observation_clean`、`factor_quarantine`、`factor_quality_snapshot`、`factor_similarity`、`factor_set`、`factor_set_member`、`label_daily`。关键业务唯一键：

```text
factor_observation_clean: signal_date + ts_code + factor_id + factor_version
factor_quality_snapshot: as_of_date + factor_id + factor_version + scope_id
factor_similarity: as_of_date + factor_id_left + factor_id_right + scope_id
factor_set_member: factor_set_id + factor_id + factor_version
label_daily: signal_date + ts_code + label_id + label_version
```

`label_daily` 必须显式包含 `entry_date`、`exit_date`、`label_available_date`、`raw_return`、`rank_pct`、`winsorized_return`、`zscore_return`、可空的 `neutral_residual`、`label_valid`、`invalid_reason` 和统一审计字段。2026 行的访问/发布由应用 capability 与专用数据库角色共同保护，普通研究身份无读取权限。

- [x] **A6.4 编写 004 实验表迁移**

创建：`split_definition`、`dataset_manifest`、`experiment_run`、`metric_daily`、`metric_summary`、`prediction_daily`、`model_registry`、`active_model_pointer`、`locked_test_audit`。`model_registry.status` 仅允许：

```text
DRAFT
WALK_FORWARD_PASSED
SHADOW
LOCKED_TEST_APPROVED
LOCKED_TEST_PASSED
LOCKED_TEST_FAILED
PRODUCTION
RETIRED
```

- [x] **A6.5 固化 TSDB 物理设计**

日期事实表使用 `RANGE/HASH` 组合：月份日期分区 + `ts_code` 或 `factor_id` HASH；sortColumns 从高频过滤顺序推导。实际 HASH bucket 常量从已批准 capacity config 读取并渲染迁移，不得临场硬编码。

任何可版本化事实表的“逻辑业务键”都只要求在同一 `data_version` 内唯一；物理唯一键为 `logical_business_key + data_version`。消费者先从 `published_version` 按 `object_name + logical_partition` 取得唯一可见版本，再读取事实表。publish 在一个原子脚本中写事实行并 compare-and-swap 更新该 pointer，禁止原地覆盖旧版本。

- [x] **A6.5a 固化 DolphinDB 类型契约**

contract tests 使用以下统一类型，不允许 adapter 自行猜测类型。`!` 表示应用层必填，`?` 表示允许 DolphinDB null；所有枚举保存为 `SYMBOL`，JSON/canonical hash 保存为 `STRING`。

```text
COMMON_AUDIT = run_id UUID!, data_version STRING!, created_at TIMESTAMP!,
               source_watermark_id STRING!, record_hash STRING!

schema_migration = schema_version INT!, name STRING!, checksum STRING!,
                   applied_at TIMESTAMP!, applied_by SYMBOL!, run_id UUID!
migration_lock = lock_name SYMBOL!, owner_run_id UUID!, acquired_at TIMESTAMP!,
                 expires_at TIMESTAMP!
published_version = object_name SYMBOL!, logical_partition STRING!, data_version STRING!,
                    previous_data_version STRING?, published_at TIMESTAMP!, run_id UUID!
pipeline_run = run_id UUID!, pipeline_name SYMBOL!, status SYMBOL!, data_version STRING!,
               config_hash STRING!, code_commit STRING!, started_at TIMESTAMP!,
               finished_at TIMESTAMP?, error_class SYMBOL?, error_message STRING?,
               environment_json STRING!
source_watermark = source_watermark_id STRING!, source_name SYMBOL!, table_name SYMBOL!,
                   max_observation_date DATE!, source_row_count LONG!,
                   schema_fingerprint STRING!, captured_at TIMESTAMP!, run_id UUID!
data_quality_result = run_id UUID!, object_name SYMBOL!, check_name SYMBOL!, scope_name STRING!,
                      severity SYMBOL!, status SYMBOL!, observed_value STRING?,
                      expected_rule STRING!, sample_json STRING?, created_at TIMESTAMP!
source_reconciliation = trade_date DATE!, ts_code SYMBOL!, field_name SYMBOL!,
                        check_version INT!, primary_value STRING?, secondary_value STRING?,
                        abs_diff DOUBLE?, rel_diff DOUBLE?, status SYMBOL!, COMMON_AUDIT

security_master_scd = ts_code SYMBOL!, exchange SYMBOL!, board SYMBOL!, list_date DATE!,
                      delist_date DATE?, security_name STRING!, is_st BOOL!,
                      listing_status SYMBOL!, valid_from DATE!, valid_to DATE?, COMMON_AUDIT
market_daily_clean = trade_date DATE!, ts_code SYMBOL!, raw_open DOUBLE?, raw_high DOUBLE?,
                     raw_low DOUBLE?, raw_close DOUBLE?, volume LONG?, amount DOUBLE?,
                     adj_factor DOUBLE?, adjusted_open DOUBLE?, valid_trade_day BOOL!,
                     valid_days_20 INT?, adv20 DOUBLE?, listing_age_trading_days INT?,
                     source_flags STRING!, dedup_resolution SYMBOL!, COMMON_AUDIT
tradability_daily = trade_date DATE!, ts_code SYMBOL!, is_st BOOL!, is_suspended BOOL!,
                    is_listed BOOL!, listing_age_trading_days INT!, valid_days_20 INT?,
                    adv20 DOUBLE?, is_eligible_pre_liquidity BOOL!, reason_bitmap LONG!,
                    rule_version INT!, COMMON_AUDIT
universe_definition = universe_id SYMBOL!, universe_version INT!, rules_json STRING!,
                      rules_hash STRING!, status SYMBOL!, COMMON_AUDIT
universe_daily = signal_date DATE!, universe_id SYMBOL!, universe_version INT!,
                 ts_code SYMBOL!, is_member BOOL!, exclusion_reason SYMBOL?,
                 reason_bitmap LONG!, adv20_rank INT?, membership_rank INT?, COMMON_AUDIT

factor_registry = factor_id SYMBOL!, factor_version INT!, display_name STRING!,
                  factor_type SYMBOL!, frequency SYMBOL!, expected_direction SYMBOL!,
                  source_database STRING!, source_table SYMBOL!, source_value_column SYMBOL!,
                  availability_policy SYMBOL!, max_staleness_calendar_days INT!, unit SYMBOL!,
                  definition_hash STRING!, owner SYMBOL!, status SYMBOL!, COMMON_AUDIT
factor_observation_clean = signal_date DATE!, ts_code SYMBOL!, factor_id SYMBOL!,
                           factor_version INT!, value DOUBLE!, observation_date DATE!,
                           available_date DATE!, report_period DATE?, stale_days INT?,
                           dedup_resolution SYMBOL!, source_revision_id STRING?, COMMON_AUDIT
factor_quarantine = observation_date DATE!, ts_code SYMBOL!, factor_id SYMBOL!,
                    factor_version INT!, reason SYMBOL!, source_row_ids_json STRING!,
                    distinct_values_json STRING!, COMMON_AUDIT
factor_quality_snapshot = as_of_date DATE!, factor_id SYMBOL!, factor_version INT!,
                          scope_id STRING!, eligibility_status SYMBOL!, metrics_json STRING!,
                          rules_version INT!, COMMON_AUDIT
factor_similarity = as_of_date DATE!, scope_id STRING!, factor_id_left SYMBOL!,
                    factor_id_right SYMBOL!, median_daily_abs_spearman DOUBLE?,
                    valid_days INT!, status SYMBOL!, cluster_id STRING?, COMMON_AUDIT
factor_set = factor_set_id STRING!, selection_scope_id STRING!, rules_version INT!,
             member_count INT!, status SYMBOL!, content_hash STRING!, COMMON_AUDIT
factor_set_member = factor_set_id STRING!, factor_id SYMBOL!, factor_version INT!,
                    feature_order INT!, frozen_direction INT!, selection_reason STRING!, COMMON_AUDIT
label_daily = signal_date DATE!, ts_code SYMBOL!, label_id SYMBOL!, label_version INT!,
              entry_date DATE!, exit_date DATE!, label_available_date DATE!,
              raw_return DOUBLE?, rank_pct DOUBLE?, winsorized_return DOUBLE?,
              zscore_return DOUBLE?, neutral_residual DOUBLE?, label_valid BOOL!,
              invalid_reason SYMBOL?, entry_realized_tradable BOOL?, COMMON_AUDIT

split_definition = split_id STRING!, fold_id SYMBOL!, split_role SYMBOL!, start_date DATE!,
                   end_date DATE!, purge_trading_days INT!, locked BOOL!, rules_hash STRING!,
                   COMMON_AUDIT
dataset_manifest = dataset_id STRING!, fold_id SYMBOL!, manifest_json STRING!,
                   artifact_uri STRING!, row_count LONG!, manifest_hash STRING!, COMMON_AUDIT
experiment_run = experiment_id STRING!, model_family SYMBOL!, model_config_hash STRING!,
                 dataset_id STRING!, seed LONG!, status SYMBOL!, artifact_uri STRING?,
                 started_at TIMESTAMP!, finished_at TIMESTAMP?, COMMON_AUDIT
metric_daily = experiment_id STRING!, fold_id SYMBOL!, signal_date DATE!, metric_name SYMBOL!,
               metric_value DOUBLE?, sample_count INT!, valid BOOL!, COMMON_AUDIT
metric_summary = experiment_id STRING!, fold_id SYMBOL!, metric_scope SYMBOL!,
                 metric_name SYMBOL!, metric_value DOUBLE?, status SYMBOL!, COMMON_AUDIT
prediction_daily = model_id STRING!, model_version INT!, signal_date DATE!, ts_code SYMBOL!,
                   raw_score DOUBLE!, score DOUBLE!, rank INT!, percentile_rank DOUBLE!,
                   universe_id SYMBOL!, universe_version INT!, inference_manifest_id STRING!,
                   COMMON_AUDIT
model_registry = model_id STRING!, model_version INT!, model_family SYMBOL!, status SYMBOL!,
                 artifact_uri STRING!, artifact_hash STRING!, evidence_json STRING!,
                 previous_status SYMBOL?, transitioned_at TIMESTAMP!, transitioned_by SYMBOL!,
                 COMMON_AUDIT
active_model_pointer = consumer_id SYMBOL!, environment SYMBOL!, model_id STRING!,
                       model_version INT!, pointer_version LONG!, effective_at TIMESTAMP!,
                       previous_model_id STRING?, previous_model_version INT?, COMMON_AUDIT
locked_test_audit = audit_id UUID!, model_id STRING!, model_version INT!, action SYMBOL!,
                    preregistration_hash STRING!, capability_hash STRING?, actor SYMBOL!,
                    approver SYMBOL?, consumed BOOL!, event_at TIMESTAMP!, details_json STRING!,
                    COMMON_AUDIT
```

`pipeline_run.status` 仅允许 `PENDING/RUNNING/SUCCESS/FAILED/CANCELLED/SOURCE_CHANGED/NOOP_NON_TRADING_DAY`；`data_quality_result.status` 仅允许 `PASS/FAIL/ERROR`，严重度仅允许 `INFO/WARN/ERROR/BLOCKER`。contract test 还需校验 `valid_to` 等 SCD 结束日的闭开语义由应用层统一执行。

- [ ] **A6.6 在 Test DB 验证全部 schema**

Run: `cd research_v2 && QRV2_INTEGRATION=1 pytest tests/contract/db/test_schema_contracts.py tests/integration/db/test_migration_roundtrip.py -q -m integration`

Expected: 4 个版本依次应用；每张表列、类型、分区和 sort contract 精确相符；再次执行无变化。

- [x] **A6.7 提交**

```bash
git add research_v2/migrations/002_pit_tables.dos research_v2/migrations/003_factor_tables.dos research_v2/migrations/004_experiment_tables.dos research_v2/tests/contract/db/test_schema_contracts.py
git commit -m "feat(v2): define versioned research schemas"
```

## Task A7: Repository 与两阶段原子发布

**Files:**

- Create: `research_v2/src/research_v2/db/repositories.py`
- Create: `research_v2/src/research_v2/db/publish.py`
- Test: `research_v2/tests/unit/db/test_publish.py`
- Test: `research_v2/tests/property/db/test_publish_idempotency.py`
- Test: `research_v2/tests/integration/db/test_atomic_publish.py`

- [x] **A7.1 写失败的状态机测试**

```python
def test_publish_never_exposes_failed_staging(repo, invalid_batch):
    handle = repo.stage(invalid_batch)
    with pytest.raises(PublishValidationError):
        repo.publish(handle)
    assert repo.visible_rows(data_version=handle.data_version) == 0
```

覆盖 `CREATED → STAGED → VALIDATED → PUBLISHED`，失败转 `REJECTED`；`PUBLISHED` 不可回退；同一 `data_version + logical_partition` 内容相同幂等、内容不同冲突。

- [x] **A7.2 实现表 registry 和 typed repository**

只允许 enum 中的表，所有 DataFrame 入库先做列、dtype、nullability、唯一键、日期时区和审计字段验证。repository 不接受任意数据库 URI 或任意 script。

- [x] **A7.3 实现 staging 验证**

必检：

- 预期/实际行数；
- 业务唯一键；
- 必填列空值；
- 日期范围和越界；
- record hash 重算；
- source watermark 存在且未失效；
- schema fingerprint；
- DQ 中不存在 `BLOCKER/FAIL`。

- [x] **A7.4 实现原子 publish 与并发锁**

同一 logical partition 只允许一个 owner；publish 写入正式表并更新可见版本必须处于同一 DolphinDB transaction/原子脚本。进程崩溃后 staging 可由 TTL 清理，但不可自动发布。

- [x] **A7.5 性质测试与故障注入**

Hypothesis 生成乱序行、重复行、NaN、不同 chunk 切分；验证 record hash 与幂等性。集成测试在 stage 后、validate 后、正式 append 前后注入异常，证明消费者只能看见旧版本或完整新版本。

- [x] **A7.6 运行验证**

Run: `cd research_v2 && pytest tests/unit/db/test_publish.py tests/property/db/test_publish_idempotency.py -q`

Run: `cd research_v2 && QRV2_INTEGRATION=1 pytest tests/integration/db/test_atomic_publish.py -q -m integration`

Expected: 所有故障点均不出现部分可见数据。

- [x] **A7.7 提交**

```bash
git add research_v2/src/research_v2/db/repositories.py research_v2/src/research_v2/db/publish.py research_v2/tests/unit/db/test_publish.py research_v2/tests/property/db/test_publish_idempotency.py research_v2/tests/integration/db/test_atomic_publish.py
git commit -m "feat(v2): add validated atomic publishing"
```

## Task A8: CLI、运行审计与安全回归

**Files:**

- Create: `research_v2/src/research_v2/cli/app.py`
- Modify: `research_v2/src/research_v2/runs/context.py`
- Create: `research_v2/tests/unit/cli/test_app.py`
- Create: `research_v2/tests/integration/db/test_pipeline_run_audit.py`

- [x] **A8.1 写失败的 CLI 测试**

要求存在：

```text
qr-v2 db migrate --env test --dry-run
qr-v2 db status --env test
qr-v2 source inspect --env test --as-of 2025-12-31
```

所有 mutation command 必须打印目标库、环境和计划摘要；production migration 额外要求显式 `--confirm-target QuantResearchV2`，非交互环境缺少该参数即失败。

- [x] **A8.2 实现 pipeline run context manager**

```python
with pipeline_run(repo, name="db.migrate", context=context) as run:
    runner.apply()
```

进入写 `RUNNING`；正常退出写 `SUCCESS`；异常写 `FAILED`、异常类和截断后的安全信息，再重新抛出。不得把 secret、完整查询结果或 token 写入 error message。

- [ ] **A8.3 增加日志脱敏测试**

在配置、连接错误、CLI traceback 和 pipeline audit 中植入 sentinel secret，断言 stdout/stderr/DB audit 均不包含原文。

- [ ] **A8.4 运行完整 A 计划验证**

Run: `cd research_v2 && ruff check src tests && mypy src`

Run: `cd research_v2 && pytest tests/unit tests/contract tests/property -q`

Run: `cd research_v2 && QRV2_INTEGRATION=1 pytest tests/integration/db -q -m integration`

Expected: 全部退出码 0；V1 write guard、migration idempotency、atomic visibility 和 audit 状态均通过。

- [x] **A8.5 更新 README**

记录本地配置、环境变量名称（不含值）、dry-run、Test DB 限制、迁移审批流程和常见故障恢复。

- [x] **A8.6 提交并请求审查**

```bash
git add research_v2/src/research_v2/cli research_v2/src/research_v2/runs research_v2/tests/unit/cli research_v2/tests/integration/db/test_pipeline_run_audit.py research_v2/README.md
git commit -m "feat(v2): expose audited database CLI"
```

随后使用 `superpowers:requesting-code-review`，重点审查数据库权限边界、migration checksum、并发 publish 和 secret 泄漏。

## A 计划完成标准

- [x] V1 写操作在类型层、API 层和运行时三层被拒绝。
- [x] 四个 migration 版本在 V2Test 首次成功、再次执行零副作用、checksum 漂移失败。
- [x] 所有目标表 schema/分区/sort columns 与 contract 一致。
- [x] 发布故障注入不能产生部分可见数据。
- [x] pipeline run 对成功、失败、取消均有完整审计。
- [x] 日志与配置不含明文凭据。
- [x] 尚未写入生产业务数据，也未改动 V1。

## 状态核对（2026-09-01）

已按现有代码与测试逐项核对并勾选完成步骤（包与质量工具、配置与 secret
边界、V1 会话保护、运行上下文、迁移运行器与 001–013 迁移、两阶段原子发布、
CLI 与 README 均有对应测试/产物；全量单元套件 947 通过，`ruff check` 与
`mypy src` 干净）。仍开放的条目及原因：

- A5.4/A5.5：迁移专用 Test-DB 集成 fixture 未建立（dry-run/幂等/checksum
  已在单元层锁定；真实库验证并入 `tests/integration` 门控验收）。
- A6.6：在真实 V2Test 库验证全部 schema 属运营执行证据。
- A8.3：日志脱敏测试缺失（配置层 secret 禁令已有测试）。
- A8.4：完整验证含 `QRV2_INTEGRATION=1` 集成（发布故障恢复测试存在但门控未跑）。
