# Quant Research V2 F — Inference, Monitoring, and Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将通过历史门槛的 V2 模型安全运行于每日无标签 SHADOW 推理，建立水位驱动、fail-closed 的发布与监控，完成 V1/V2 并行、独立兼容导出、active pointer 切换/回滚，并在合法时间对 2026 执行一次性 locked test。

**Architecture:** 日频 DAG 只有在交易日历、行情、证券状态、股票池和全部冻结因子水位对齐时才构建 inference batch；加载 artifact 前校验 manifest/signature/hash，输出经两阶段发布到 `prediction_daily`。2026 影子阶段只监控数据与分数，不计算/展示已实现收益或 RankIC。生产消费者通过 compare-and-swap 的 active model pointer 读取，V1 产物与路径保持不变。

**Tech Stack:** Python 3.11、DolphinDB SDK、PyTorch、Pandas/PyArrow、Pydantic、Typer、HTML/JSON reporting、pytest/Hypothesis；外部告警通过可插拔 notifier interface。

---

## 前置与非目标

- 工程实施前置：A–E 的代码、schema 与测试已完成；F 的管线可先用 V2Test fixture 验证。正式运营前置：至少一个复杂候选模型处于 `WALK_FORWARD_PASSED`；若没有任何模型通过，只完成 F 的工程验收，不开始正式 SHADOW 计时，也不改变模型状态。
- 本计划不修改/覆盖 V1 数据、模型、CSV、PKL、定时任务或数据库表。
- 2026 年未结束且最后 T+6 标签未成熟前，不计算 2026 RankIC/ICIR/PnL，不以影子结果调参。
- SHADOW 发布不代表交易授权；生产切换需 locked test 通过和独立审批。
- notifier 默认只记录本地审计；接入企业微信、邮件等外部系统需另行提供 endpoint/权限。

## 文件清单

**Create:**

```text
research_v2/configs/inference/daily.yaml
research_v2/configs/monitoring/default.yaml
research_v2/configs/export/v1_compatible.yaml
research_v2/src/research_v2/inference/artifact_loader.py
research_v2/src/research_v2/inference/builder.py
research_v2/src/research_v2/inference/predictor.py
research_v2/src/research_v2/inference/publisher.py
research_v2/src/research_v2/inference/daily_pipeline.py
research_v2/src/research_v2/monitoring/watermarks.py
research_v2/src/research_v2/monitoring/data_health.py
research_v2/src/research_v2/monitoring/drift.py
research_v2/src/research_v2/monitoring/score_health.py
research_v2/src/research_v2/monitoring/shadow.py
research_v2/src/research_v2/monitoring/notifier.py
research_v2/src/research_v2/registry/active_pointer.py
research_v2/src/research_v2/registry/locked_test.py
research_v2/src/research_v2/export/compatibility.py
research_v2/src/research_v2/reporting/daily.py
research_v2/src/research_v2/reporting/shadow.py
research_v2/src/research_v2/cli/inference.py
research_v2/src/research_v2/cli/export.py
research_v2/runbooks/daily-inference.md
research_v2/runbooks/source-delay.md
research_v2/runbooks/model-rollback.md
research_v2/runbooks/locked-test-2026.md
research_v2/runbooks/disaster-recovery.md
research_v2/tests/unit/inference/test_artifact_loader.py
research_v2/tests/unit/inference/test_builder.py
research_v2/tests/unit/inference/test_predictor.py
research_v2/tests/unit/inference/test_publisher.py
research_v2/tests/unit/inference/test_daily_pipeline.py
research_v2/tests/unit/monitoring/test_watermarks.py
research_v2/tests/unit/monitoring/test_data_health.py
research_v2/tests/unit/monitoring/test_drift.py
research_v2/tests/unit/monitoring/test_score_health.py
research_v2/tests/unit/monitoring/test_shadow.py
research_v2/tests/unit/registry/test_active_pointer.py
research_v2/tests/unit/registry/test_locked_test.py
research_v2/tests/unit/export/test_compatibility.py
research_v2/tests/property/inference/test_daily_idempotency.py
research_v2/tests/leakage/test_shadow_has_no_label_access.py
research_v2/tests/leakage/test_locked_test_one_shot.py
research_v2/tests/golden/data/expected_export.csv
research_v2/tests/golden/data/expected_export.pkl.sha256
research_v2/tests/golden/test_compatibility_export.py
research_v2/tests/integration/inference/test_daily_shadow_pipeline.py
research_v2/tests/integration/inference/test_publish_failure_recovery.py
research_v2/tests/integration/registry/test_pointer_rollback.py
research_v2/tests/integration/registry/test_locked_test_workflow.py
research_v2/tests/integration/test_v1_v2_parallel_acceptance.py
```

## Task F1: 安全加载冻结 artifact

**Files:**

- Create: `research_v2/src/research_v2/inference/artifact_loader.py`
- Test: `research_v2/tests/unit/inference/test_artifact_loader.py`

- [ ] **F1.1 写失败测试：任一 hash/签名不符即拒绝**

逐项篡改 config、feature order、preprocessor、任一 seed state dict、signature、dataset id、code/model version，loader 必须在构建模型前失败。缺一个 seed 也失败，不能改为双 seed ensemble。

- [ ] **F1.2 实现 allowlisted state_dict loader**

根据 `model_family + model_code_version` 从本地 registry 映射已知类；只用 `torch.load(..., weights_only=True, map_location="cpu")` 加载 tensor state dict，不执行 pickle 任意对象/动态 import 路径。先在 CPU 校验 key/shape/dtype，再移动到明确 device。

- [ ] **F1.3 校验生产契约**

要求模型状态至少 `WALK_FORWARD_PASSED`；factor set `FROZEN`；feature order/缺失 indicator 与 preprocessor 完全匹配；universe/label/split 版本与 model card 一致；推理不要求 label table 水位。

- [ ] **F1.4 写加载审计**

记录 model id、artifact hashes、device、代码版本、加载时间和调用 run id；不记录 state tensor 内容或 secret。

- [ ] **F1.5 运行测试并提交**

Run: `cd research_v2 && pytest tests/unit/inference/test_artifact_loader.py -q`

Expected: 正常 bundle 加载；所有 tamper/missing/version mismatch fail closed。

```bash
git add research_v2/src/research_v2/inference/artifact_loader.py research_v2/tests/unit/inference/test_artifact_loader.py
git commit -m "feat(v2): verify immutable inference artifacts"
```

## Task F2: 水位驱动的无标签 Inference Builder

**Files:**

- Create: `research_v2/configs/inference/daily.yaml`
- Create: `research_v2/src/research_v2/monitoring/watermarks.py`
- Create: `research_v2/src/research_v2/inference/builder.py`
- Test: `research_v2/tests/unit/monitoring/test_watermarks.py`
- Test: `research_v2/tests/unit/inference/test_builder.py`
- Test: `research_v2/tests/leakage/test_shadow_has_no_label_access.py`

- [ ] **F2.1 固化每日依赖水位**

对 signal date T 要求：`trade_calendar` 确认 T 开市；security master、market clean、tradability、universe 和 factor set 中每个 required factor 均有覆盖 T 的 `PUBLISHED` watermark；运行开始/结束 fingerprint 一致。label 不在依赖清单中。

- [ ] **F2.2 写缺水位/错版本失败测试**

任一源落后 T、data version 未发布、factor version 不同、DQ 有 BLOCKER、source 正在变更或 universe 为空，都不得 build/publish。周末/休市返回 `NOOP_NON_TRADING_DAY`，不是失败。

- [ ] **F2.3 实现无标签 anchor join**

只以 T 日已发布 universe members 为 anchor，连接 T 日 factor observations，应用 artifact 内冻结 preprocessor；不 import label repository、不生成 target 列。代码依赖测试检查 inference package 不引用 `labels`/evaluation unlocked APIs。

- [ ] **F2.4 固化 batch health 门槛**

BLOCKER：业务键重复、PIT `available_date>T`、required feature 列缺失、finite row coverage <95%、universe size <80% 或 >120% 的近 20 个有效日中位数。每因子 coverage 低于训练参考 p01 再减 10 个百分点记 ERROR；单一 factor ERROR 不自动阻断有 missing mask 的模型，但进入报告。

- [ ] **F2.5 生成 inference manifest**

保存 signal date、model/factor set/universe ids、输入 data versions/watermarks、行/列/null/coverage、preprocessor hash 和 batch shard hash；明确 `contains_labels=false`。同输入生成同 manifest id。

- [ ] **F2.6 运行测试并提交**

Run: `cd research_v2 && pytest tests/unit/monitoring/test_watermarks.py tests/unit/inference/test_builder.py tests/leakage/test_shadow_has_no_label_access.py -q`

Expected: 完整水位成功，所有缺失/未来/label access canary 失败关闭。

```bash
git add research_v2/configs/inference/daily.yaml research_v2/src/research_v2/monitoring/watermarks.py research_v2/src/research_v2/inference/builder.py research_v2/tests/unit/monitoring/test_watermarks.py research_v2/tests/unit/inference/test_builder.py research_v2/tests/leakage/test_shadow_has_no_label_access.py
git commit -m "feat(v2): build watermark-locked inference batches"
```

## Task F3: 预测、score 健康与两阶段发布

**Files:**

- Create: `research_v2/src/research_v2/inference/predictor.py`
- Create: `research_v2/src/research_v2/inference/publisher.py`
- Create: `research_v2/src/research_v2/monitoring/score_health.py`
- Test: `research_v2/tests/unit/inference/test_predictor.py`
- Test: `research_v2/tests/unit/inference/test_publisher.py`
- Test: `research_v2/tests/unit/monitoring/test_score_health.py`
- Test: `research_v2/tests/property/inference/test_daily_idempotency.py`

- [ ] **F3.1 写 3-seed ensemble 预测测试**

每 seed 在 T 日有效股票内转 percentile rank，最终 score 三 seed 等权；输入股票 permutation 后按 key 恢复结果一致。任一 seed 输出 NaN/inf、代码缺失/新增、shape 错误，整日失败。

- [ ] **F3.2 实现 score health blocking checks**

发布要求 finite score coverage ≥95%、至少 10 个 unique scores、score std >1e-6、所有 universe key 最多一行；top/bottom 1% 不得全部由同一 missingness pattern 组成。另报告 seed pair Spearman，任一 pair <0.20 记 ERROR 但不单独阻断。

- [ ] **F3.3 实现稳定排名与输出契约**

`prediction_daily` 每行：`signal_date, ts_code, model_id, model_version, raw_score, score, rank, percentile_rank, universe_id, universe_version, inference_manifest_id` 加统一审计字段。rank 排序固定 `score DESC, ts_code ASC`；相同 score 使用相同 average percentile rank，但展示 rank 按代码稳定。

- [ ] **F3.4 使用 A 计划 staging/publish**

先写 staging，验证行数等于 healthy prediction keys、唯一键、hash、水位和 DQ，再原子发布。logical key=`signal_date+model_id+model_version`；相同输入幂等，不同内容同 key 冲突且需新 run/data version，不能覆盖。

- [ ] **F3.5 性质和故障测试**

Hypothesis 改变行顺序/chunk/device，结果按 tolerance 与 key 一致；在每个 stage/publish 故障点注入异常，消费者只能看到零行或整日完整行。

- [ ] **F3.6 运行测试并提交**

Run: `cd research_v2 && pytest tests/unit/inference/test_predictor.py tests/unit/inference/test_publisher.py tests/unit/monitoring/test_score_health.py tests/property/inference/test_daily_idempotency.py -q`

Expected: ensemble、稳定 rank、health gate、幂等和原子性通过。

```bash
git add research_v2/src/research_v2/inference/predictor.py research_v2/src/research_v2/inference/publisher.py research_v2/src/research_v2/monitoring/score_health.py research_v2/tests/unit/inference/test_predictor.py research_v2/tests/unit/inference/test_publisher.py research_v2/tests/unit/monitoring/test_score_health.py research_v2/tests/property/inference/test_daily_idempotency.py
git commit -m "feat(v2): publish healthy daily predictions atomically"
```

## Task F4: 每日 DAG、重试和运行报告

**Files:**

- Create: `research_v2/src/research_v2/inference/daily_pipeline.py`
- Create: `research_v2/src/research_v2/reporting/daily.py`
- Create: `research_v2/src/research_v2/cli/inference.py`
- Modify: `research_v2/src/research_v2/cli/app.py`
- Test: `research_v2/tests/unit/inference/test_daily_pipeline.py`
- Test: `research_v2/tests/integration/inference/test_daily_shadow_pipeline.py`
- Test: `research_v2/tests/integration/inference/test_publish_failure_recovery.py`

- [ ] **F4.1 固化 DAG 与状态**

```text
calendar check
→ capture source watermarks
→ materialize/verify market
→ materialize/verify universe
→ materialize/verify governed factors
→ build label-free inference manifest
→ load artifact
→ predict three seeds
→ score/data health
→ atomic publish
→ monitoring/report/notify
```

状态 `PENDING/RUNNING/SUCCESS/FAILED/NOOP_NON_TRADING_DAY`；前置失败使步骤状态为 `SKIPPED_DEPENDENCY_FAILED`，pipeline 总状态仍为 `FAILED`。

- [ ] **F4.2 定义调度与时效目标**

默认交易日 19:30 Asia/Shanghai 启动，20:00 前完成 prediction publish。调度器只触发 CLI；业务正确性仍由 calendar/watermark guard 决定。源未齐可在 19:35、19:45、20:00 重试三次，之后失败并告警，绝不发布前一日数据冒充当日。

- [ ] **F4.3 定义 CLI**

```text
qr-v2 inference daily --signal-date YYYY-MM-DD --model-id sha256:... --mode shadow
qr-v2 inference status --signal-date YYYY-MM-DD --model-id sha256:...
qr-v2 inference retry --run-id UUID --from-step watermark
```

retry 复用原配置/model id，重新捕获水位并生成新 run id；已成功发布的同内容步骤幂等跳过。`--mode production` 只接受 active pointer 指向且状态 PRODUCTION 的模型。

- [ ] **F4.4 实现每日 HTML/JSON 报告**

包含各步骤耗时/状态、水位、universe/feature coverage、score 分布、seed agreement、prediction 行数、artifact/data hashes、告警和发布版本；不包含标签、收益、RankIC 或持仓建议。

- [ ] **F4.5 集成测试**

在 V2Test 模拟正常日、休市、源迟到后恢复、源永不齐、factor DQ blocker、GPU 失败、publish 中断、相同日重跑。断言只有完整成功日可见，失败日无陈旧替代。

- [ ] **F4.6 运行测试并提交**

Run: `cd research_v2 && pytest tests/unit/inference/test_daily_pipeline.py -q`

Run: `cd research_v2 && QRV2_INTEGRATION=1 pytest tests/integration/inference -q -m integration`

Expected: DAG、重试、状态、幂等和失败恢复通过。

```bash
git add research_v2/src/research_v2/inference/daily_pipeline.py research_v2/src/research_v2/reporting/daily.py research_v2/src/research_v2/cli/inference.py research_v2/src/research_v2/cli/app.py research_v2/tests/unit/inference/test_daily_pipeline.py research_v2/tests/integration/inference
git commit -m "feat(v2): orchestrate fail-closed daily inference"
```

## Task F5: 数据/分数漂移与 notifier

**Files:**

- Create: `research_v2/configs/monitoring/default.yaml`
- Create: `research_v2/src/research_v2/monitoring/data_health.py`
- Create: `research_v2/src/research_v2/monitoring/drift.py`
- Create: `research_v2/src/research_v2/monitoring/notifier.py`
- Test: `research_v2/tests/unit/monitoring/test_data_health.py`
- Test: `research_v2/tests/unit/monitoring/test_drift.py`

- [ ] **F5.1 固化 reference distributions**

从 FINAL train 2020–2024 保存每因子 winsorized transformed decile bins、missing rate、median/IQR；从 2025 validation 保存 ensemble score deciles/coverage。reference artifact 与 model id 一起 hash/冻结。

- [ ] **F5.2 实现 PSI 与分布健康**

每日和滚动 20 日计算 feature/score PSI，bin 零频加 epsilon=`1e-6` 并归一化。阈值：`PSI <0.10 INFO`、`0.10–0.25 WARN`、`>0.25 ERROR`。另监控 missing rate 相对 reference 增加、极值比例、universe size、score std/unique、seed agreement。

- [ ] **F5.3 区分阻断与调查告警**

业务键/PIT/finite/水位/coverage blocker 会阻断 publish；单纯 PSI ERROR 不自动阻断当日 SHADOW/PRODUCTION prediction，但连续 3 个有效日 ERROR 触发人工 review，禁止新的模型晋级/active pointer 切换，现有 active model 保持不变。

- [ ] **F5.4 实现 notifier interface 与去重**

`AuditNotifier` 默认写 pipeline/DQ audit；外部 notifier 只接受脱敏 structured payload。相同 `alert_key+signal_date+severity` 只发送一次；恢复时发送 RESOLVED。网络通知失败不改变 prediction 发布结果，但记独立 ERROR。

- [ ] **F5.5 运行测试并提交**

Run: `cd research_v2 && pytest tests/unit/monitoring/test_data_health.py tests/unit/monitoring/test_drift.py -q`

Expected: PSI 参考值、阈值边界、连续日状态、blocker 区分和告警去重通过。

```bash
git add research_v2/configs/monitoring/default.yaml research_v2/src/research_v2/monitoring/data_health.py research_v2/src/research_v2/monitoring/drift.py research_v2/src/research_v2/monitoring/notifier.py research_v2/tests/unit/monitoring/test_data_health.py research_v2/tests/unit/monitoring/test_drift.py
git commit -m "feat(v2): monitor inference data and score drift"
```

## Task F6: SHADOW 部署审批、60 日资格与报告

**Files:**

- Create: `research_v2/src/research_v2/monitoring/shadow.py`
- Create: `research_v2/src/research_v2/reporting/shadow.py`
- Test: `research_v2/tests/unit/monitoring/test_shadow.py`

- [ ] **F6.1 在首个正式影子日前完成部署审批**

人工核验历史 G1–G6、artifact、推理 manifest contract、V2Test 演练和回滚准备，随后执行 `WALK_FORWARD_PASSED → SHADOW`。只有状态为 `SHADOW` 且 shadow pointer 指向该模型的成功日才计入 60 日；Test 环境 smoke run 不计。

- [ ] **F6.2 定义有效影子日**

只有当日为交易日、pipeline `SUCCESS`、prediction 完整发布、无 BLOCKER、报告 artifact 可验证时计 1 个 valid shadow day。失败/休市/补跑失败不计；之后成功补跑可计入对应 signal date 一次。

- [ ] **F6.3 写 60 日边界测试**

59 日不能进入下一状态，60 日可申请；日期必须不同且按日历递增。中间 WARN 不清零，BLOCKER 不计数；连续 3 日 PSI ERROR 使资格 `REVIEW_REQUIRED`，即使数量达到 60 也不能批准。

- [ ] **F6.4 实现影子报告**

报告 valid/invalid dates、成功率、延迟 SLO、universe/feature coverage、score/seed/drift、事件与恢复、artifact/model/data versions。对于 2026，性能栏固定显示 `LOCKED_NOT_COMPUTED`，代码不查询 label/paper PnL。

- [ ] **F6.5 生成资格证据但不自动进入 locked 状态**

达到 60 日且无未解决 ERROR 后生成 `shadow_evidence.json` 并保持 `SHADOW`。进入 `LOCKED_TEST_APPROVED` 还必须等 F9 的年份成熟、预注册和双人审批；系统不因计满天数自动 transition。evidence hash 进入后续 registry event。

- [ ] **F6.6 运行测试并提交**

Run: `cd research_v2 && pytest tests/unit/monitoring/test_shadow.py -q`

Expected: 有效日计数、错误阻断、无 label access 和证据 hash 通过。

```bash
git add research_v2/src/research_v2/monitoring/shadow.py research_v2/src/research_v2/reporting/shadow.py research_v2/tests/unit/monitoring/test_shadow.py
git commit -m "feat(v2): qualify sixty-day shadow operation"
```

## Task F7: 独立兼容导出，不覆盖 V1

**Files:**

- Create: `research_v2/configs/export/v1_compatible.yaml`
- Create: `research_v2/src/research_v2/export/compatibility.py`
- Create: `research_v2/src/research_v2/cli/export.py`
- Test: `research_v2/tests/unit/export/test_compatibility.py`
- Test: `research_v2/tests/golden/data/expected_export.csv`
- Test: `research_v2/tests/golden/data/expected_export.pkl.sha256`
- Test: `research_v2/tests/golden/test_compatibility_export.py`

- [ ] **F7.1 从 V1 消费者审计冻结导出 contract**

在 Phase 0 读取现有消费者所需列名/dtype/index/order，写显式 mapping；V2 canonical prediction 不迁就旧列。配置必须列出每个输出列来源与转换，未知列失败。

- [ ] **F7.2 固定独立路径**

```text
artifacts/exports/v2/{model_id}/{signal_date}/predictions.csv
artifacts/exports/v2/{model_id}/{signal_date}/predictions.pkl
artifacts/exports/v2/{model_id}/{signal_date}/manifest.json
```

exporter 使用 exclusive create + 临时目录原子 rename；解析后的任何路径若落入 V1 输出根即 `UnsafeExportPath`。默认不创建/更新 latest symlink。

- [ ] **F7.3 实现 CSV/PKL 确定性**

排序、列顺序、float precision、日期格式、pickle protocol 与 index 由 contract 固定；manifest 保存 source prediction data version、row count 和文件 SHA-256。PKL 只用于受控兼容，不作为 V2 canonical artifact。

- [ ] **F7.4 黄金文件测试**

CSV 与批准文本逐字节比较；PKL 反序列化后的 schema/values 精确比较并验证文件 hash。测试将 V1 path 设为哨兵只读目录，确保未写入。

- [ ] **F7.5 定义命令与提交**

```text
qr-v2 export compatibility --signal-date YYYY-MM-DD --model-id sha256:... --format csv,pkl
qr-v2 export verify --manifest artifacts/exports/v2/.../manifest.json
```

Run: `cd research_v2 && pytest tests/unit/export/test_compatibility.py tests/golden/test_compatibility_export.py -q`

Expected: contract、确定性、原子写与 V1 path guard 通过。

```bash
git add research_v2/configs/export/v1_compatible.yaml research_v2/src/research_v2/export research_v2/src/research_v2/cli/export.py research_v2/tests/unit/export research_v2/tests/golden
git commit -m "feat(v2): export isolated V1-compatible predictions"
```

## Task F8: Active model pointer、切换与回滚

**Files:**

- Create: `research_v2/src/research_v2/registry/active_pointer.py`
- Modify: `research_v2/src/research_v2/cli/registry.py`
- Test: `research_v2/tests/unit/registry/test_active_pointer.py`
- Test: `research_v2/tests/integration/registry/test_pointer_rollback.py`

- [ ] **F8.1 写 compare-and-swap 测试**

pointer key=`consumer_id+environment`；更新必须提交 expected current model/version。两个并发切换只能一个成功；每次 append event 保存 old/new、actor、reason、evidence、effective_at 和 run id。

- [ ] **F8.2 强制状态和证据 guard**

production pointer 只接受 `PRODUCTION` 模型；shadow pointer 只接受至少 `WALK_FORWARD_PASSED` 且有 shadow deployment approval 的模型。存在未解决 blocker/连续漂移 review 时拒绝切换。

- [ ] **F8.3 实现切换 dry-run**

```text
qr-v2 registry pointer plan --consumer paper-sim --model-id sha256:...
qr-v2 registry pointer switch --consumer paper-sim --model-id sha256:... --expected-current sha256:...
qr-v2 registry pointer rollback --consumer paper-sim --event-id UUID
```

plan 输出 schema/artifact/最近 prediction/兼容导出/状态检查，不改 pointer。switch/rollback 均要求明确 environment 和二次确认生产目标。

- [ ] **F8.4 实现回滚**

rollback 是新的 CAS event，指向上一个仍可验证的 model id；不删除失败模型/预测/审计。若旧 artifact 不完整或状态 RETIRED，拒绝并要求选择其他已批准版本。

- [ ] **F8.5 V2Test 演练**

模拟 A→B 成功、并发 C 冲突、B→A 回滚、artifact 损坏回滚拒绝、V1 消费者未配置时零影响。验证读取者在任一时刻只见一个完整 pointer。

- [ ] **F8.6 运行测试并提交**

Run: `cd research_v2 && pytest tests/unit/registry/test_active_pointer.py -q`

Run: `cd research_v2 && QRV2_INTEGRATION=1 pytest tests/integration/registry/test_pointer_rollback.py -q -m integration`

Expected: CAS、证据 guard、原子可见和回滚审计通过。

```bash
git add research_v2/src/research_v2/registry/active_pointer.py research_v2/src/research_v2/cli/registry.py research_v2/tests/unit/registry/test_active_pointer.py research_v2/tests/integration/registry/test_pointer_rollback.py
git commit -m "feat(v2): switch and rollback models atomically"
```

## Task F9: 2026 一次性 locked test

**Files:**

- Create: `research_v2/src/research_v2/registry/locked_test.py`
- Create: `research_v2/tests/unit/registry/test_locked_test.py`
- Create: `research_v2/tests/leakage/test_locked_test_one_shot.py`
- Create: `research_v2/tests/integration/registry/test_locked_test_workflow.py`
- Create: `research_v2/runbooks/locked-test-2026.md`

- [ ] **F9.1 在解锁前冻结 preregistration**

最迟在 locked evaluation 前保存不可变 `locked_test_preregistration.json`：model id/3 seed artifacts、FINAL train/validation manifests、factor set/preprocessor、2026 universe/label definition、评价代码 hash、G1–G6、命令和预期操作者角色。不得预填结果。

- [ ] **F9.2 写时间成熟 guard**

只有当前日期晚于 2026 最后交易日的第 6 个后续交易日，且该 exit date 的 clean adjusted open 水位已稳定/对账/DQ 通过，才允许申请 unlock。系统从交易日历计算日期，不用自然日常量。

- [ ] **F9.3 实现双人/一次性审批记录与状态转换**

申请人与批准人必须不同；审批先验证 `shadow_evidence.json` 和当前状态 `SHADOW`，再执行 `SHADOW → LOCKED_TEST_APPROVED`，并生成绑定 preregistration hash、model id、label watermark 和截止时间的 one-shot capability。使用后立即标记 consumed；过期、hash 改变、第二次调用都失败并写 audit。

- [ ] **F9.4 只在专用进程解锁 label repository**

专用 evaluator 只接受状态 `LOCKED_TEST_APPROVED`，构建 FINAL 2026 test manifest，加载事先冻结模型，禁止 trainer/search/selector import 或调用；完成后 capability 无论成功失败都 consumed。进程 stdout 只在最终报告生成后公开 aggregate metrics，避免中途人为终止再调参。

- [ ] **F9.5 运行预注册的 G1–G6**

历史 2020–2025 门槛证据保持不变；2026 locked test 额外要求 RankIC≥0.03、HAC t-stat≥2.0、相对 2026 fold-matched best simple baseline ΔRankIC≥0.005、score not collapsed。结果不满足任一项转 `LOCKED_TEST_FAILED`；满足则 `LOCKED_TEST_PASSED`，再经独立生产审批转 `PRODUCTION`。

- [ ] **F9.6 防止“失败后用 2026 再优化”**

一旦任一候选读取 2026 label，该 model lineage 标记 `LOCKED_EXPOSED`；其后用 2026 调整的后代不得把 2026 再当 locked test，只能等待新的未来完全样本外年度。所有 read audit 保留。

- [ ] **F9.7 一次性/故障测试**

覆盖过早、标签未成熟、同人审批、token 过期、prereg hash 改变、正常成功、评价中崩溃、第二次调用、失败后状态。崩溃也 consume capability，不能“重试直到满意”；只有基础设施故障且证明任何 aggregate/row-level label 未暴露，才由独立审计决定新 capability。

- [ ] **F9.8 运行测试并提交**

Run: `cd research_v2 && pytest tests/unit/registry/test_locked_test.py tests/leakage/test_locked_test_one_shot.py -q`

Run: `cd research_v2 && QRV2_INTEGRATION=1 pytest tests/integration/registry/test_locked_test_workflow.py -q -m integration`

Expected: 时间、双人、one-shot、无训练 API、状态机和失败污染测试通过。

```bash
git add research_v2/src/research_v2/registry/locked_test.py research_v2/tests/unit/registry/test_locked_test.py research_v2/tests/leakage/test_locked_test_one_shot.py research_v2/tests/integration/registry/test_locked_test_workflow.py research_v2/runbooks/locked-test-2026.md
git commit -m "feat(v2): enforce one-shot 2026 locked evaluation"
```

## Task F10: Runbooks、灾备与 V1/V2 并行验收

**Files:**

- Create: `research_v2/runbooks/daily-inference.md`
- Create: `research_v2/runbooks/source-delay.md`
- Create: `research_v2/runbooks/model-rollback.md`
- Create: `research_v2/runbooks/disaster-recovery.md`
- Create: `research_v2/tests/integration/test_v1_v2_parallel_acceptance.py`
- Create: `research_v2/runbooks/drill-shadow-switch-rollback.md`
- Create: `research_v2/tests/integration/test_shadow_switch_rollback_drill.py`

- [x] **F10.1 编写每日操作 runbook**

写出 owner、19:30 触发/20:00 SLO、成功检查、失败分类、三次水位重试、手工 rerun 命令、禁止操作、报告位置和升级路径。每条命令包含 dry-run/目标环境检查。

- [x] **F10.2 编写 source delay 与 DQ 故障 runbook**

明确：源迟到不发布；schema drift 停止并更新 contract；重复冲突隔离；不复制昨日预测；Tushare 只用于确认差异；恢复后以原 signal date 新 run 补跑。

- [x] **F10.3 编写回滚与灾备 runbook**

覆盖 pointer 回滚、artifact 校验失败、V2 DB 暂不可用、GPU 不可用、磁盘不足、staging 清理、审计恢复。备份/恢复演练只操作 V2Test 快照；生产灾备方案审批后另排窗口。

- [x] **F10.4 做 V1/V2 平行不干扰测试**

执行完整 V2Test 日流程前后，对已批准 V1 schema fingerprints 和 sentinel 输出 hashes 比较，必须完全不变；网络 mock 断言没有任何 V1 mutation call。V2 exporter 仅写独立目录。

- [x] **F10.5 影子切换/回滚桌面演练**

由研究、数据、运行三角色按照 runbook 完成：源迟到 → 恢复补跑 → 60 日 evidence → shadow pointer 切换 → 注入 score collapse → 回滚。记录每一步耗时、命令、结果和改进项。

- [x] **F10.6 运行完整 F 计划验证**

Run: `cd research_v2 && ruff check src tests && mypy src`

Run: `cd research_v2 && pytest tests/unit/inference tests/unit/monitoring tests/unit/registry tests/unit/export tests/leakage/test_shadow_has_no_label_access.py tests/leakage/test_locked_test_one_shot.py tests/golden/test_compatibility_export.py -q`

Run: `cd research_v2 && QRV2_INTEGRATION=1 pytest tests/integration/registry tests/integration/test_v1_v2_parallel_acceptance.py tests/integration/test_shadow_switch_rollback_drill.py -q -m integration`

Expected: 全部退出码 0；V1 fingerprints/hashes 无变化；2026 label 仅 one-shot workflow 可访问。

- [x] **F10.7 提交并请求最终审查**

```bash
git add .gitignore research_v2/runbooks research_v2/tests/integration
git commit -m "docs(v2): complete inference and recovery runbooks"
```

随后先使用 `superpowers:requesting-code-review`，再使用 `superpowers:verification-before-completion`；只有命令输出证明确实通过，才可声明平台实施完成。

## F 计划完成标准

- [ ] 每个交易日只在所有非标签水位和 DQ 通过后发布完整预测，失败时无陈旧/部分数据。
- [ ] 2026 SHADOW 至少 60 个有效日，期间无任何 label/RankIC/PnL 访问。
- [ ] 兼容 CSV/PKL 仅写 V2 独立路径，V1 schema/文件/hash 不变。
- [ ] active pointer 切换/回滚原子、可审计且受模型状态约束。
- [ ] 2026 locked test 只有成熟后、双人审批、预注册、one-shot 路径可执行。
- [ ] locked test 通过前模型不进入 PRODUCTION；失败 lineage 不得重复利用 2026 作为样本外。
- [ ] 日常、迟到、回滚、灾备 runbooks 经桌面演练验证。

注：F10 已完成；以上总体验收项待 F4 的每日 CLI、运行报告、属性测试和 inference 集成测试补齐后再逐项勾选。
