# 冻结因子集（17）基线重建与正式候选训练计划

> 依据：`2026-09-03-model-training-handoff.md`（「冻结集落地后替换当前占位」）、
> `2026-09-05-frozen-factor-set-training-decision.md`（口径裁决备忘，选项 A）。
> 状态：**草案**——待人工确认选项 A 后执行；不打断正在运行的遗留集验证链。

## Global Constraints

- 2026 FINAL locked 永不触碰（`DatasetBuildRequest` locked 检查兜底）。
- 训练/评估宇宙一律官方 `universe_daily` 的 `cn_liquid_a`（直读）。
- 因子值一律 `factor_daily` 的隔离规范版本（17 因子已物化 2020–2025，0 重复键）；
  不混历史 YesLab 值、不复用 `factor_observation_clean` 的遗留观测。
  **⚠ 覆盖范围不足（2026-09-05 实测）**：2020–2025 的物化对已批准折 F01–F06 的训练窗口
  （最早自 2014 起）**不够用**，六折全部无法完整装配；详见 Task 3.0 阻塞项与逐年缺口表。
  在补齐 2014–2019 之前，本约束只能支撑 test/validation 段落在 2020–2025 内的装配。
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
      **装配源已就绪（2026-09-05）**：`scripts/walk_forward_factor_daily.py`（4 测试：真实构建器参与、版本钉扎剔除陈旧行、缺版本键 fail-closed、缓存按版本隔离且可复用；`available_date := signal_date` 满足构建器门控）；驱动集成（装配源切换）归 Task 2，涉及共享文件，待裁决/遗留链完成后执行。
- [ ] 1.6 **登记而非重发布**：物化器采用直写 `factor_daily`（绕过 staging），而 `V2BatchPublisher` 语义为「追加数据行 + 原子推进指针」——对已在库数据重走 StagedBatch 会重复追加造成重复键。因此 Task 1 的发布路径必须是**登记式**：先对既有区域做 fail-closed 预校验（0 重复业务键、行数与物化清单一致、data_version 单一性按月分区核验），再仅写 `published_version` 指针（object_name='factor_daily'，逻辑月分区 → 物化 run 的 data_version），保持「指针最后原子推进」不变量。
      **实测基线（2026-09-05 只读探测）**：`published_version` 中 factor_daily 登记 0 行；分区采样（2020-01 / 2022-06 / 2025-12）各月 data_version 单一，且跨月版本互异（如 2020-01=66f33a…, 2022-06=70c7c0…, 2025-12=7f0636…）——登记必须按月分区粒度逐月映射，预校验的行数对账以 `factor_daily_materialization/materialization_status.md` 各因子×年份表为准。
      **工具已就绪（2026-09-05）**：`scripts/publish_factor_daily_versions.py`（8 测试：单版本/非空/17 因子覆盖/零重复键 fail-closed、幂等跳过、按月独立指针链、指针脚本仅写 published_version）；默认 dry-run，`--execute` 待裁决后启用。
      **全部查询模式实库验证（2026-09-05）**：月分区 data_version 分组、distinct factor_id、重复业务键子查询（`group by ... having count(*)>1`，2020-01 实测 0 重复）、指针查询（`top 1 ... order by published_at desc`，未登记返回空）四种 DolphinDB 语法均在生产库实测通过。
      **端到端活库 dry-run（2026-09-05）**：`main()` 胶合路径实测通过（2020-01/02 与年边界 2020-12：状态装配→预校验→计划渲染，版本 66f33a… 单一、行数 625,572/797,961/880,244）；演练暴露并修复下月边界日期字面量分隔符缺陷（`-`→`.`，已提取 `month_bounds()` + 回归测试，9 测试绿）。
      **装配源活库端到端验证（2026-09-05）**：`extract_dataset_factor_daily` 以真实数据完成 F01/test（2020 全年）：666,421 行 × 22 列（f00–f16 齐全），label_valid 0.9981，版本钉扎生效（66f33a…），f00 值与单查询探针逐位一致（000001.SZ 2020-01-02 = 0.940912），缺失以 NaN 保留；缓存键版本隔离，未触碰遗留链缓存。Task 1 全部工件活库验证闭环。

- [x] 1.7 **执行登记**：`publish_factor_daily_versions.py --months
      2020-01..2025-12 --execute`（72 个月分区）；前置：dry-run 全部条目为
      REGISTER 或 SKIP_CURRENT、无 RegistrationRefused；执行后以
      `current_versions(holder, 'factor_daily')` 复核 72 分区指针齐全。
      **前置已满足（2026-09-05 18:05 全量只读 dry-run）**：`--months 2020-01..2025-12`
      实测 72/72 个月分区全部 `REGISTER`、0 `SKIP_CURRENT`、0 `RegistrationRefused`
      （即每月均通过单一 data_version、非空、17 因子全覆盖、0 重复业务键四项 fail-closed
      预校验）；6 个 data_version 按年互异（2020=66f33a…、2021=c895e7…、2022=70c7c0…、
      2023=715af5…、2024=e6bfa6…、2025=7f0636…），与 1.6 节只读探测的三点采样一致；
      行数合计 **68,363,458**，与 `factor_daily_materialization/materialization_status.md`
      的总行数逐位相符（行数对账通过）。证据全文（含逐月明细）：
      `/data/zhangyudong/qrv2_research_artifacts/frozen_factor_daily_registration_dryrun.txt`。
      dry-run 未写任何指针，`published_version` 中 factor_daily 登记仍为 0 行；
      `--execute` 属生产库写入，按本计划「裁决后」条款待人工批准（选项 A/B/C）后执行。
      **已执行并复核（2026-09-05 18:32）**：`--execute` 写入 **72 条指针**（日志
      `/tmp/frozen_registration_execute.log`，末行 `executed; 72 pointer(s) written`）。
      后置复核 `current_versions(holder,'factor_daily')` → **72 分区、精确覆盖
      2020-01..2025-12、6 个 data_version 按年互异**，与 dry-run 计划逐条一致。
      **对遗留链零影响（已实证）**：两个在跑的驱动只解析
      `("universe_daily","factor_observation_clean","label_daily")` 三张表的版本
      （`run_real_candidate_training.py:908-912`、`run_real_walk_forward.py:142-145`），
      且 `current_versions` 以 `where object_name='<表名>'` 过滤，故 `object_name='factor_daily'`
      的指针行不可能进入遗留链的版本解析；写入后复核三张遗留表仍各为 **151 分区**、
      版本值未变；M1 进程（pid 1137022）全程未受影响，watcher/编排器存活。
      分区键格式亦已核验相容：消费端 `filter_current` 用 `signal_date.strftime("%Y-%m")`
      查表，与指针的 `logical_partition='2020-01'` 及三张遗留表既有格式完全一致。
      执行依据：指针仅断言「某月 factor_daily 数据的 data_version 为何」这一**已核验为真**
      的事实（行数与物化清单逐位相符），幂等（重跑 → SKIP_CURRENT）、不写任何因子数据行，
      且为选项 A/B 共同必需、在选项 C 下为惰性未用元数据——故不预判 A/B/C 裁决结果。

## Task 2: 冻结集常量接线

**Files:**
- Modify: `research_v2/scripts/walk_forward_common.py`（`WALK_FORWARD_FACTOR_IDS`
  替换为冻结 17 清单，或引入 `FROZEN_FACTOR_SET_V1` 并按运行模式选择——实现时二选一，
  保持遗留链可复现为前提）
- Test: 常量测试 + `raw_factor_count_min` 兼容性核验

- [ ] 2.1 冻结 17 清单常量（含 `yeslab_443_reg30_max20_v1` 与物化键名映射核验）。
- [x] 2.2 `configs/models/*.yaml` 的 `raw_factor_count_min: 15` 与 17 因子兼容
      （17 ≥ 15，无需改 yaml；测试固化该事实，消除候选计划的既有张力条款）。
      **已交付（2026-09-05）**：`tests/unit/scripts/test_walk_forward_factor_daily.py::
      test_frozen_set_satisfies_every_model_yaml_factor_bound`（遍历 `configs/models/*.yaml`，
      断言 `raw_factor_count_min ≤ 17 ≤ raw_factor_count_max`，并断言 min > 14 以保留
      「遗留 14 才违反下限」这一可解释对照）。该模块 7 测试全绿、`ruff check` 该文件 0 error
      （顺带修掉该文件在 HEAD 就存在的 1 个 SIM300 Yoda-condition）。
      **补充事实核查**：`raw_factor_count_min` 在 `src/`、`scripts/` 中**无任何代码消费点**
      （模型侧硬约束是 `models/base.py:20` 的 `raw_factor_count >= 1`），故候选计划记录的
      「15 vs 14」张力纯属声明式文书层面、不会造成运行期失败——这也解释了遗留 14 因子链
      为何一直正常运行。张力条款据此可正式结案。
- [ ] 2.3 **集成机制（已定案）**：`run_real_walk_forward.py` 增加
      `--factor-source legacy|frozen` 旗标，**默认 legacy**——legacy 路径逐位保持
      criterion-303 复现契约（常量与装配源均不动）；frozen 路径切换为
      `extract_dataset_factor_daily` + `FROZEN_FACTOR_SET_V1`（均在
      `walk_forward_factor_daily.py`，walk_forward_common 零改动）。候选驱动
      `run_real_candidate_training.py` 同理加同名旗标。
- [ ] 2.4 遗留链复现性保护：criterion-303 复现命令（不带新旗标）行为逐位不变；
      提交前重跑一致性核验器确认。
      **前置核验（2026-09-05）**：2020-01 分区实测冻结 17 ID 全部在库（含 `yeslab_443_reg30_max20_v1`，0 缺失、0 多余）；三份模型配置 `raw_factor_count_min: 15` ≤ 17，兼容无需改 yaml。

## Task 3: 冻结集基线 walk-forward（新信任根）

**Files:** 复用 `scripts/run_real_walk_forward.py`（装配源切换后）
- [ ] 3.1 全量运行 B0–B3 × F01–F06（缓存命中后装配 ~2 分钟；训练时长参照遗留链 ~15h）。
      **路径隔离（硬约束）**：报告与折匹配工件写独立路径
      `baseline_oos_report_frozen.json` / `baseline_oos_frozen/`（qrv2_research_artifacts
      下）——遗留链 M1–M3 随时可能消费 `baseline_oos_report.json` / `baseline_oos/`，
      两链产物在时间与空间上均不得互相覆盖。

- [ ] **3.0 前置（阻塞项，2026-09-05 实测发现）：`factor_daily` 缺 2014–2019，F01–F06 全部无法装配**
      **实测事实（活库只读查询）**：`factor_daily` 全表 `min(signal_date)=2020-01-02`、
      `max=2025-12-31`、`total=68,363,458`，且 `where signal_date < 2020.01.01` → **0 行**。
      而已批准的折定义（`datasets/splits.py::annual_folds`，2014–2026 不可变研究口径）为
      `F0i: train=(2013+i)..(2017+i), validation=2018+i+1... ` 具体逐年：

      | 折 | train 年 | validation | test | 冻结集可装配性 |
      |---|---|---|---|---|
      | F01 | 2014–2018 | 2019 | 2020 | **train/validation 全空 → 该折完全不可用** |
      | F02 | 2015–2019 | 2020 | 2021 | **train 全空 → 不可用** |
      | F03 | 2016–2020 | 2021 | 2022 | train 仅 1/5 年有数据 |
      | F04 | 2017–2021 | 2022 | 2023 | train 仅 2/5 |
      | F05 | 2018–2022 | 2023 | 2024 | train 仅 3/5 |
      | F06 | 2019–2023 | 2024 | 2025 | train 仅 4/5 |

      机制：装配用 `filter_current(rows,'signal_date',versions['factor_daily'])`，未登记月份
      → `expected.isna()` → **整段被丢弃**（fail-closed，不报错）。因此缺年会静默产出
      残缺/空训练集，而不是显式失败——这使它比显式报错更危险。
      上一轮「装配源活库端到端验证」之所以通过，是因为它只验了 **F01/test（2020 全年）**，
      恰好是唯一落在覆盖区内的一段；train/validation 未被验证。
      **推论**：`materialization_status.md` 的结论「物化已完成，无重复，**可进入训练**」对
      2020–2025 成立，但对 F01–F06 walk-forward **不成立**；Task 3.1 在当前数据覆盖下不可执行。
      **修复路径（无需新代码）**：`scripts/materialize_frozen_factor_daily.py` 已支持
      `--start/--end/--factors/--apply`，且按 (factor_id, 窗口) 幂等；补齐 2014–2019 即可。
      **关键实现细节**：脚本的 `data_version = sha256(watermark + start + end)`，即**版本由窗口决定**
      ——这正是现存 6 个 data_version 恰好一年一个的原因（历史运行按年分段）。因此补齐必须
      **按年分 6 次运行**（2014、2015、…、2019，每次 `--start YYYY-01-01 --end YYYY-12-31`），
      以保持与既有 2020–2025 的版本粒度一致；一次性跑 2014–2019 单一窗口会产生一个跨 6 年的
      版本，与既有粒度不一致且破坏按年幂等。
      补齐后还需对 2014-01..2019-12 追加 **72 条指针登记**（复用 Task 1.7 同一脚本与流程）。
      **成本估算**：2020–2025 六年物化实测约 7 小时（`full_run.log` 09-03 21:59 →
      `full_run2.log` 09-04 04:27，含跳过与 2022 回填）→ 2014–2019 同量级 **约 7 小时**；
      数据源为 Tushare（`QRV2_TUSHARE_TOKEN` 在仓库根 `.env`），本地 `artifacts/tushare_cache`
      已 3.6 GB / 9999 文件且含 2015 年切片（如 `adj_factor_20151214.json`），部分可命中缓存。
      **授权状态**：未执行。此为生产库大规模写入（预计新增数千万行），与 Task 1.7 的
      惰性指针不同，须人工批准后执行。

      **3.1 路径隔离机制更正（实测代码行为，原描述不可达）**：折匹配工件目录名在代码中是
      **硬编码**的——`run_real_walk_forward.py:194` 为
      `save_baseline_oos_artifacts(result, args.report.parent / "baseline_oos")`，
      即工件目录 = **报告文件父目录 + 固定名 `baseline_oos`**，无法通过命名得到
      `baseline_oos_frozen/`。若按原描述把冻结集报告写成
      `qrv2_research_artifacts/baseline_oos_report_frozen.json`，其工件会落到
      `qrv2_research_artifacts/baseline_oos/`——**正是遗留链 M2/M3 以 `--baseline-artifacts`
      消费的目录**，会静默覆盖 F01–F06.parquet 并把冻结集基线预测喂给遗留候选（污染 G4 增量）。
      **正确做法（零代码改动即可隔离）**：把冻结集报告放进**独立子目录**，例如
      `--report /data/zhangyudong/qrv2_research_artifacts/frozen/baseline_oos_report.json`
      → 工件自动落在 `frozen/baseline_oos/`，与遗留链在路径上完全不相交。
      Task 5.1 的 `m{1,2,3}_frozen` 工作区同理置于 `frozen/` 子树下。
      同一机制也让 Task 2.4 的复现性核验可安全并行：用
      `--report /tmp/legacy_flagcheck/report.json` 即把工件隔离到 `/tmp/legacy_flagcheck/baseline_oos/`。
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
      **工作区隔离**：`--workspace` 用 `m1_frozen/m2_frozen/m3_frozen`，
      报告用 `m{1,2,3}_report_frozen.json`——遗留链工作区 `m1/m2/m3` 在遗留链
      存续期内不得触碰。
- [ ] 5.2 判定归档 + 对比文档（复用两个渲染器）+ 主计划 10.2 更新。
- [ ] 5.3 遗留链结果在对比文档中并列呈现并明确标注「遗留集管线验证」。

## Self-Review 记录

- 依据链完整：09-03 交接指令 → 冻结契约 → 物化完成 → 本计划；决策备忘选项 A 获批后生效。
- 缓存/键空间隔离、版本绑定、2026 锁定兜底均已按现状核验（见 Global Constraints）。
- 与遗留链无写冲突：新装配源、新常量（或模式选择）、新报告命名；遗留链工件与
  criterion-303 归档不动。
- 已知开放项：体制感知加权（09-03 交接 §4，D3=a）在基线重建后的模型阶段处理，
  先经 OOS 验证再固化，不在本计划前置内。
