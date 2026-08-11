# Agent Gateway

这个包为现有量化平台增加一层面向 Agent 的研究入口，不替换当前生产环境里的因子更新流程。

核心定位：

```text
URL / PDF / 自然语言因子想法
  -> 抽取因子逻辑
  -> 参考数据库已有字段生成候选表达式
  -> 快速验证和评分
  -> 沉淀研究知识
  -> 好因子生成 FactorCalculator 草稿
  -> 人工 review 后进入生产更新链路
```

## 主要能力

- 从 DolphinDB 字段、每日更新脚本、因子元数据和已有 `factor_calculators` 声明中构建字段目录。
- 从 `数据库/daily_task/update_*.py` 这类脚本生成 Agent 可读的数据库字段参考文档。
- 从文本、URL 或 PDF 资料中抽取候选因子。
- 只使用字段目录中允许的字段生成和校验 DSL 表达式。
- 将 DSL 表达式作为研究层临时因子进行计算和验证。
- 用统一评估层为因子检验结果打分：
  质量、稳定性、可交易性、成本敏感度、样本外可信度。
- 在挖因子、因子相似度检验、因子检验之后，把研究经验沉淀为 `rules`、`findings`、`failures`。
- 将通过校验的表达式生成可 review 的 `FactorCalculator` 草稿。

## CLI 示例

```bash
python -m agent_gateway.cli list-fields
python -m agent_gateway.cli generate "5 day reversal factor with volume confirmation"
python -m agent_gateway.cli validate "rank(-1 * ts_delta(close, 5) / ts_shift(close, 5))"
python -m agent_gateway.cli score "{\"ic_mean\":0.04,\"ic_ir\":0.8,\"annual_return\":0.12,\"turnover\":0.2}" --validation-json "{\"rolling_validation\":{\"passed\":true}}"
python -m agent_gateway.cli knowledge similarity_check "{\"factor_name\":\"Demo\",\"max_similarity\":0.91,\"threshold\":0.8}"
python -m agent_gateway.cli search-knowledge high_turnover
python -m agent_gateway.cli promote ReportReversal "rank(-1 * ts_delta(close, 5) / ts_shift(close, 5))"
python -m agent_gateway.cli gate-factor DemoFactor DemoFactorCalculator --lookback-days 20 --start-time 2023.01.01
python -m agent_gateway.generate_field_reference
```

`promote` 默认只生成草稿内容，不会写入生产目录。只有人工确认后再加 `--write`，它才会在生产 `factor_calculators` 目录下创建新的 calculator 文件。

`gate-factor` 默认是 dry-run：连接 DolphinDB、计算候选因子、做相似度检验和 IC 检验，并把结果沉淀到研究知识库，但不会入库。只有显式加 `--commit`，且相似度、IC、ICIR 等 gate 全部通过后，才会调用现有 `数据库/add_factor_narrow.py` 逻辑写入 `factor_metadata` 和因子窄表。

数据库连接优先读取仓库根目录 `.env`：

```text
DOLPHINDB_HOST=10.176.55.211
DOLPHINDB_PORT=8848
DOLPHINDB_USERNAME=your_username
DOLPHINDB_PASSWORD=your_password
```

也兼容旧变量名 `dolphinDB_USERNAME`、`dolphinDB_PASSWORD`。

## 统一因子评估输出

`score_factor` 和 `score_factor_summary` 不只返回原始指标，而是返回一个可直接辅助决策的评估报告：

- `score`：综合评分。
- `grade`：A/B/C/D 等级。
- `recommendation`：中文建议，例如适合入池、适合继续优化、不建议继续。
- `recommendation_code`：给程序使用的稳定代码：
  `enter_pool`、`continue_optimizing`、`stop`。
- `components`：质量、稳定性、可交易性、成本敏感度、样本外可信度五个分项。
- `diagnostic_reasons`：为什么给出这个判断。
- `improvement_suggestions`：下一步研究建议。
- `risk_flags`：高换手、低稳定性、高回撤、高集中、样本外较弱等风险标签。

评分器同时支持现有因子检验里常见的中文指标名，以及 API/脚本中更方便使用的英文指标名，例如：

- `ic_mean`
- `ic_ir`
- `annual_return`
- `sharpe`
- `max_drawdown`
- `win_rate`
- `turnover`

## 研究知识库

研究事件会被沉淀成结构化资产，默认保存在：

```text
agent_gateway_runs/knowledge_base/
  assets.jsonl
  rules/
  findings/
  failures/
```

这层不是普通日志，而是为了保存可复用的研究经验：

- `rules`：已经通过评分门槛、值得复用或进入候选池 review 的经验。
- `findings`：有研究价值但还需要进一步验证的观察。
- `failures`：弱信号、高相似、高成本、高风险、字段缺失等不应反复踩坑的失败经验。

以下流程会自动沉淀知识资产：

- `ingest_factor_source`
- `generate_factor_expressions`
- `score_factor`

对于包外部的检查结果，例如你们已有的因子相似度脚本，可以手动调用：

```bash
python -m agent_gateway.cli knowledge similarity_check "{\"factor_name\":\"Demo\",\"max_similarity\":0.91,\"threshold\":0.8}"
```

Claude Code 可用的 MCP 工具：

- `distill_research_knowledge`
- `list_research_knowledge`
- `search_research_knowledge`

## API

API 层依赖 `fastapi` 和 `uvicorn`。

启动方式：

```bash
uvicorn agent_gateway.api:app --host 127.0.0.1 --port 8010
```

核心路由：

- `GET /api/v1/fields`
- `POST /api/v1/factor-sources/ingest`
- `POST /api/v1/factors/generate-expressions`
- `POST /api/v1/expressions/validate`
- `POST /api/v1/factors/promote-to-calculator`
- `POST /api/v1/factors/score`
- `POST /api/v1/factors/gate`
- `POST /api/v1/knowledge`
- `GET /api/v1/knowledge`
- `GET /api/v1/knowledge/search`

## Claude Code MCP

在 Claude Code 使用的 Python 环境里安装 MCP 依赖：

```bash
pip install mcp
```

仓库根目录下已经包含 `.mcp.json`。在服务器上需要检查并修改其中的：

- `command`：Claude Code 要调用的 Python 解释器绝对路径。
- `cwd`：服务器上的 `Quant-Platform-Prod` 仓库绝对路径。

修改后，在仓库根目录启动 Claude Code，然后运行：

```text
/mcp
```

预期能看到的 server 名称：

```text
quant-platform-agent
```

## Claude Code 里可以这样说

- 列出 `stock_factor_daily` 里可以用于成交量类表达式的字段。
- 读取字段参考文档，并总结 `financial_metrics_table` 中有哪些字段。
- 根据这段研报描述生成 5 个候选 DSL 表达式。
- 校验 `rank(volume_ratio)` 是否能在字段目录下通过。
- 评估这份因子检验结果，判断是否适合入池。
- 把这次相似度检验结果沉淀到研究知识库。
- 在生成新因子之前，先搜索过去关于 `high_turnover` 的失败经验。
- 为 `rank(-1 * ts_delta(close, 5) / ts_shift(close, 5))` 生成 calculator 草稿，先不要写文件。




