# 因子数据修复阶段报告

## 当前阶段

本阶段只执行因子源数据质量修复的只读审计和本地单元测试，不训练模型，不读取 2026 锁定测试数据，也没有向现有 V2 表写入数据。

已完成的实现提交：

- `15cc535`–`4a9933c`：源键审计、修复提供器和隔离规则；
- `262c440`–`ce1498f`：迁移 009 的修复审计/隔离表；
- `1b880ec`：修复批次身份、质量门禁和指纹校验；
- `eee253f`–`4ea8637`：TRAIN/VALIDATION 干运行器、流式发布和向量化重复键审计；
- `f31f941`–`3eb40ba`：修复前后质量比较报告。

## 真实数据库 smoke 结果

命令：

```text
python research_v2/scripts/run_factor_repair_overlay.py \
  --phase train --start 2020-01-01 --end 2020-01-31 \
  --factor-id yeslab_391 --dry-run \
  --manifest research_v2/artifacts/diagnostics/train_factor_repair_smoke_manifest.json
```

结果：

| 项目 | 值 |
| --- | ---: |
| 因子 | `yeslab_391` / `GP_022` |
| 审计月份 | `2020-01` |
| clean 行数 | 56,920 |
| 隔离行数 | 38,344 |
| 填充值 | 0 |
| 阻断因子数 | 0 |
| 发布 | `false` |
| 审计状态 | `REPAIR_AUDIT_PASS_WITH_UNRESOLVED_QUARANTINE` |

完整 manifest 保存在
[`train_factor_repair_smoke_manifest.json`](../../research_v2/artifacts/diagnostics/train_factor_repair_smoke_manifest.json)。

该结果说明当前运行环境没有配置可用的 `QRV2_TUSHARE_TOKEN` 和本地行情缓存，因此 GP 候选补值没有启用；现有源数据中的非有限/未解析行进入隔离区。即使没有 `BLOCKED` 状态，`PASS_WITH_UNRESOLVED_QUARANTINE` 也会被发布质量门禁拒绝，不能绕过。

## 全量运行处置

曾启动 2020–2024 TRAIN 全量只读干运行。由于完整因子/月分块在当前数据量下超过 30 分钟，任务被安全中止；堆栈显示耗时集中在重复键审计，未发生写入，且没有生成全量 manifest。随后已将重复键审计改为向量化分组统计，并通过 4,000 组回归测试验证语义不变。

在行情候选缓存配置完成、且 smoke 结果先达到 `REPAIR_AUDIT_PASS` 或 `REPAIRED_PENDING_QUALITY_RECHECK` 后，再恢复全量 TRAIN 分段干运行。任何隔离行、冲突键或候选重叠不一致都会继续阻止发布。

## 验证结果

- 全量研究测试：`356 passed`；
- Ruff：`All checks passed`；
- mypy：`research_v2` 124 个源文件及比较脚本均无错误；
- 没有执行模型训练，没有发布修复版本，没有查询 2026 数据。

