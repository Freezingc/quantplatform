# 冻结因子集（17）基线重建与正式候选训练计划

> 依据：`2026-09-03-model-training-handoff.md`（「冻结集落地后替换当前占位」）、
> `2026-09-05-frozen-factor-set-training-decision.md`（口径裁决备忘，选项 A）。
> 状态：**草案**——待人工确认选项 A 后执行；不打断正在运行的遗留集验证链。

## Global Constraints

- 2026 FINAL locked 永不触碰（`DatasetBuildRequest` locked 检查兜底）。
- 训练/评估宇宙一律官方 `universe_daily` 的 `cn_liquid_a`（直读）。
- 因子值一律 `factor_daily` 的隔离规范版本（17 因子已物化 2020–2025，0 重复键）；
  不混历史 YesLab 值、不复用 `factor_observation_clean` 的遗留观测。
- 遗留 14 因子链（重跑 #3 → M1–M3）不中断；其产物定性为「管线机制验证」，
  不作为正式业绩证据；本计划的基线是新信任根。
- 缓存隔离已核验：`fold_cache_key` 含表名版本映射，`factor_daily` 路径与
  `factor_observation_clean` 路径键空间不相交（`walk_forward_common.py:57`）。
- 门控 G1–G6、E7 预算（≤30 configs）、3 种子统一契约、判定器模式全部沿用。

## Task 1: factor_daily 版本化装配路径

**Files:**
- Modify: `research_v2/scripts/walk_forward_common.py`（新增装配源，不改动现有函数契约）
- 可能新增: `research_v2/scripts/publish_factor_daily_versions.py`（登记脚本）
- Test: `research_v2/tests/unit/scripts/test_walk_forward_factor_daily.py`

**Interfaces:**
- Consumes: `factor_daily`（schema 见迁移 016：signal_date/ts_code/factor_id/
  factor_version/universe_id/value/run_id/data_version/record_hash）；
  `published_version`（object_name='factor_daily'，按 signal_date 月分区登记
  data_version——物化器目前未登记，需补登记步骤，沿用 A 计划 staging/publish 契约）。
- Produces: 与现有 `extract_dataset` 同契约的折数据帧（factor 列序由冻结清单决定）。

- [ ] 1.1 失败测试：无 published_version 登记或缺分区 → 装配 fail-closed 拒绝。
- [ ] 1.2 失败测试：行级 data_version 与登记版本不符 → 拒绝（防未登记数据混入）。
- [ ] 1.3 实现装配源 + 版本过滤（复用 `filter_current` 模式）；宇宙按日对齐
      `universe_id='cn_liquid_a'`。
- [ ] 1.4 确定性测试：同版本快照两次装配逐位一致；缓存键与旧路径不相交。
- [ ] 1.5 运行测试并提交。

## Task 2: 冻结集常量接线

**Files:**
- Modify: `research_v2/scripts/walk_forward_common.py`（`WALK_FORWARD_FACTOR_IDS`
  替换为冻结 17 清单，或引入 `FROZEN_FACTOR_SET_V1` 并按运行模式选择——实现时二选一，
  保持遗留链可复现为前提）
- Test: 常量测试 + `raw_factor_count_min` 兼容性核验

- [ ] 2.1 冻结 17 清单常量（含 `yeslab_443_reg30_max20_v1` 与物化键名映射核验）。
- [ ] 2.2 `configs/models/*.yaml` 的 `raw_factor_count_min: 15` 与 17 因子兼容
      （17 ≥ 15，无需改 yaml；测试固化该事实，消除候选计划的既有张力条款）。
- [ ] 2.3 遗留链复现性保护：criterion-303 复现命令仍指向旧常量路径（若 2 采用
      模式选择方案）；提交前重跑一致性核验器确认。

## Task 3: 冻结集基线 walk-forward（新信任根）

**Files:** 复用 `scripts/run_real_walk_forward.py`（装配源切换后）
- [ ] 3.1 全量运行 B0–B3 × F01–F06（缓存命中后装配 ~2 分钟；训练时长参照遗留链 ~15h）。
- [ ] 3.2 判定渲染：`summarize_real_walk_forward.py` →
      `docs/reports/<date>-frozen-set-baseline-walk-forward.md`（新命名，不占用
      criterion-303 名号；303 保留为遗留集证据）。
- [ ] 3.3 折匹配基线预测工件 + 扩展选拔字段（invalid_day_count/unselected_reasons）
      随报告签发（现有代码已含）。

## Task 4: 核验器参考更新

- [ ] 4.1 `verify_baseline_rerun_consistency.py --reference` 指向新冻结集基线报告
      （新信任根）；候选消费 schema 预检门无需改动（同报告 schema）。
- [ ] 4.2 复现 runbook 增补「冻结集基线」章节与命令。

## Task 5: 正式候选序列（M1→M2→M3）

- [ ] 5.1 以冻结集基线为 `--baseline-artifacts/--baseline-report` 输入，复用现有
      候选驱动与顺序守卫（`--prior-report` 证据链）重跑 M1→M2→M3。
- [ ] 5.2 判定归档 + 对比文档（复用两个渲染器）+ 主计划 10.2 更新。
- [ ] 5.3 遗留链结果在对比文档中并列呈现并明确标注「遗留集管线验证」。

## Self-Review 记录

- 依据链完整：09-03 交接指令 → 冻结契约 → 物化完成 → 本计划；决策备忘选项 A 获批后生效。
- 缓存/键空间隔离、版本绑定、2026 锁定兜底均已按现状核验（见 Global Constraints）。
- 与遗留链无写冲突：新装配源、新常量（或模式选择）、新报告命名；遗留链工件与
  criterion-303 归档不动。
- 已知开放项：体制感知加权（09-03 交接 §4，D3=a）在基线重建后的模型阶段处理，
  先经 OOS 验证再固化，不在本计划前置内。
