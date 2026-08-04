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

该结果最初是在没有配置 `QRV2_TUSHARE_TOKEN` 和本地行情缓存的情况下执行的；随后已增加只读的 V2 `market_daily_clean` 回退，并按预热区间每个逻辑月解析 `published_version`。对于 2020-01，V2 行情表本身从 2020-01 才开始，GP_022 滚动窗口没有足够历史行情，因此 38,344 行被正确标记为 `NO_FINITE_CANDIDATE`。即使没有 `BLOCKED` 状态，`PASS_WITH_UNRESOLVED_QUARANTINE` 也会被发布质量门禁拒绝，不能绕过。

## V2 行情回退与公式一致性 smoke

后续 smoke 使用同一只读 runner、无外部 Tushare token：

| 窗口 | 结果 | clean 行 | 隔离行 | 主要原因 |
| --- | --- | ---: | ---: | --- |
| 2020-01 / `yeslab_391` GP_022 | `PASS_WITH_UNRESOLVED_QUARANTINE` | 56,920 | 38,344 | `NO_FINITE_CANDIDATE` |
| 2020-12 / `yeslab_391` GP_022 | `REPAIR_BLOCKED` | 0 | 1 | `CANDIDATE_OVERLAP_MISMATCH` |

2020-12 有足够行情预热，公式候选可以计算，但与源因子重叠值不一致；修复器拒绝覆盖并记录 `mismatch_rows=1`。这不是可以忽略的工程错误，而是需要重新核对 GP_022 原始计算所用行情字段、复权口径或公式版本的质量问题。

在应用 4 位小数的 `pct_chg` 映射后，2020-12 的 180,334 个有限重叠键中仍有 112 个超过当前重叠门槛，其中 18 个相对误差超过 `1e-4`，最大相对误差约为 `0.266763`。大部分差异已被精度规则消除，但尾部异常仍然存在；当前重叠门槛为 `abs_tol=1e-8`、`rel_tol=1e-6`，因此不能用放宽阈值掩盖尾部异常。

V2 `market_daily_clean` 当前可提供 `raw_close` 和 `amount`，但没有 `pre_close`/`pct_chg` 原始字段；用相邻 `raw_close` 推导的百分比变化不足以证明 GP_022/GP_064 的公式完全一致。要继续补值，需要提供包含精确 `pct_chg`（最好同时包含 `pre_close`）的 Tushare 缓存，或建立经过审计的行情字段映射；不修改既有表结构，也不直接降低重叠门槛。

当前已有 47 个 source-qualified 因子的数据库快照均为
`SOURCE_DQ_PASS_PENDING_SIMILARITY`：全范围有限值比例为 `0.737690–0.849923`，其中 38 个因子低于 0.80；值冲突键为 `0`，但 27 个因子的重复行比例为 `0.50`。这些重复行会进入确定性去重/隔离流程，不能直接当作无问题数据。PIT 键覆盖的 P05 比例最低为 `1.253870`，反映了重复输入对原始覆盖统计的抬升，不能替代去重后的实际覆盖率。

对应 manifest：

- [`2020-01 smoke v6`](../../research_v2/artifacts/diagnostics/train_factor_repair_smoke_jan2020_v6_manifest.json)
- [`2020-12 smoke v3`](../../research_v2/artifacts/diagnostics/train_factor_repair_smoke_dec2020_v3_manifest.json)

## 全量运行处置

曾启动 2020–2024 TRAIN 全量只读干运行。由于完整因子/月分块在当前数据量下超过 30 分钟，任务被安全中止；堆栈显示耗时集中在重复键审计，未发生写入，且没有生成全量 manifest。随后已将重复键审计改为向量化分组统计，并通过 4,000 组回归测试验证语义不变。

在 2020-12 的公式重叠不一致被解释并重新验证、且 smoke 结果达到 `REPAIR_AUDIT_PASS` 或 `REPAIRED_PENDING_QUALITY_RECHECK` 后，再恢复全量 TRAIN 分段干运行。任何隔离行、冲突键或候选重叠不一致都会继续阻止发布。

## 验证结果

- 全量研究测试：`356 passed`；
- Ruff：`All checks passed`；
- mypy：`research_v2` 124 个源文件及比较脚本均无错误；
- 没有执行模型训练，没有发布修复版本，没有查询 2026 数据。
