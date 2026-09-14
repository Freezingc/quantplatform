# 2026-09-14：M1 全网格通过、链式监视器自动发射了 M2、而 M2 的首折遥测把它的工期从 6.2 天改成约 10–11 天

本篇是**当前状态**文档，取代 `2026-09-12-m1-full-grid-launch-and-measured-cost-model.md`
里所有关于"运行到哪了 / ETA 是多少"的部分（那份的成本模型推导与验收定义仍然有效，
本篇不重复）。读这一份就知道现在该做什么。

---

## 1. 一句话状态

**M1 全网格已通过并归档；M2 自 09-13 18:54:56 起在跑，预计 09-23 至 09-25 完成；M3 未发射。**
仓库干净、分支 `codex/publish-quant-platform`、HEAD `ca69e17`。

## 2. M1 全网格：结果与证据

`m1_full_v1`，09-12 02:05 发射 → 09-13 17:42 完成，**41.0 小时**、41 路、钉 1 线程、
四折各搜满 **27/27** 格、**日志 0 错误**。

| 门 | 观测值 | 阈值 | |
|---|---|---|---|
| G1 mean RankIC | 0.09305385234488889 | 0.030 | PASS |
| G2 HAC t | 12.47214210303097 | 2.0 | PASS |
| G3 正年占比 | 1.0 | 0.70 | PASS |
| **G4 Δ RankIC** | **+0.007826986071187494** | 0.005 | **PASS** |
| G5 最大年占比 | 0.28362613392846053 | 0.50 | PASS |
| G6 有效日占比 | 0.9752321981424149 | 0.95 | PASS |

`valid_day_count 945`、`baseline_mean_rank_ic 0.0852268662737014`（与其它候选 16 位全同）、
`icir 0.749371`；逐年 2022 0.091083 / 2023 0.073011 / 2024 0.105018 / 2025 0.101157。

**四折胜者（逐折独立选择）**：F1 c15 `128/3/0.0`（验证 0.060023）、F2 c14 `128/2/0.2`（0.094376）、
F3 c12 `128/2/0.0`（0.085111）、F4 c16 `128/3/0.1`（0.101731）。
**四个胜者全是 `hidden_dim 128`**，而载体钉死的 c0 是 `hidden 64 / blocks 1`。

**三项验收全过**（都在 09-13 18:00–18:55 由链式监视器自动执行，日志
`m1_m2_chain_watcher.log`）：

* **门**：`family=M1`、`mode=full`、`oos.all_passed=true` ✓
* **验收 C′**：四折胜出格的最终拟合检查点与搜索阶段检查点**逐字节相同，54/54** ✓
* **验收 A**：同工作区 `--resume` 重跑（18:00:15→18:54:56，四折整折采纳、零次拟合），
  两份报告 **md5 逐字节相同 `600a1fd945324a2bfa2af8de9b5ed954`** ✓
  ——**这是该判据第一次在 27 格规模上被验证**（此前五份逐字节相同的报告全来自单配置的 M0R）。

**链条最后一环也已验证**：`load_walk_forward_evidence` 接受 `m1_full_v1/promotion_evidence.json`
（sha256 `e941274d44922d2a118a70cac34dc528…`），**registry 独立重算的 G1–G6 与报告逐位相同**。

**已归档**：`docs/reports/2026-09-13-m1-full-grid-real-candidate-walk-forward.md`
（与 `m1_full_v1/verdict.md` md5 全同 `f31293f38389f5c52c0d4ec2efbe614e`）。
计划 **3.4 与 3.5 已勾选**并附证据。

**M1 仍未注册进 registry**（`model_registry/models/` 下只有 M0R 一条）。注册是治理动作，未代为执行。

## 3. 候选排名（全部同基线、同 945 日集，可直接相减）

| 排名 | 候选 | 搜索格数 | Δ RankIC (G4) | 含验证集选择偏差？ |
|---|---|---|---|---|
| **1** | **M0R**（LightGBM rankreg，网格钉死） | 1 | **+0.009149** | **否** |
| 2 | M1 全网格（ResidualMLP） | 27 | +0.007827 | **是** |
| 3 | M1 单配置载体（已被上行取代） | 1 | +0.005845 | 否 |
| — | M0 `refined`（LambdaRank） | 24 | +0.004202 | 是 → **FAIL** |
| — | M0 `base`（LambdaRank） | 24 | +0.003640 | 是 → **FAIL** |

**全网格把 M1 的 G4 提高了 34%**（余量从 +17% 到 +57%），**值得跑**；
但**排名没有翻转，M0R 仍第一**。而且差距比表面的 0.001322 更大：
M0R 不含选择偏差、M1 含，所以"M1 只差 0.0013"应读成"带偏差仍差 0.0013"。
完整分析见 `docs/reports/2026-09-12-candidate-standings-comparison.md`。

## 4. M2：已发射，而首折遥测把工期改了

**发射**：09-13 18:54:56，由监视器在 M1 通过全部门与两项验收后自动执行
（内存 511.2 GiB ≥ 480.5 门槛）。24 路、钉 1 线程、`--worker-memory-gb 17.1`、
`--prior-report M1=m1_full_v1/report.json`（全网格那份，不是载体那份）、`--resume`、
新工作区 `m2_v1`。pid 3588838。

**首折遥测两项都读到了，都比规划值差**：

1. **降级 = 1.437（中位），不是外推的 1.18。** 24 个配置全部落在 **1.332–1.583**。
   M2 在 24 路下的降级比 M1 在 **41 路**下的（1.132）还差，尽管并发度只有 M1 的 59%——
   因为 M2 的 worker 载荷是 M1 的 **11 倍**（4.16 GiB 批次张量对 0.381 GiB），
   24 个 worker 同时流式读写约 100 GiB 张量，**访存带宽而非核数才是自变量**。
2. **配置级粒度下，一轮 = `3 × 最慢格`，不是 `3 × 网格均值`。** 最慢格是 **TCN/128**：
   21–22 epoch × 约 2400 s ≈ **14.1 h/次拟合**（c18 seed 20260801 实测 19:03→09:15 = 14.2 h，
   与 21 × 2408 s = 14.05 h 吻合）。**单折搜索 = 3 × 14.1 ≈ 42.3 h**，
   而计划按网格均值算的是 24.4 h → **低估 1.74 倍**
   （分解：降级 1.22× × 最慢格对均值 1.43×）。

**修正后的工期**：单折 ≈ 49.9 h（42.3 搜索 + 7.3 最终拟合 + 0.3 面板重建），
四折按实测行数比 1.000/1.110/1.220/1.321 折算 = **232 h ≈ 9.7 天 → 09-23 11:00 前后**；
计入 M1 学到的"按行数折算低估约 15%"则 **267 h ≈ 11.1 天 → 09-24 22:00 前后**。
**规划口径应为约 10–11 天，此前文档写的 6.17 天低估约 60%。**

**内存已核实不是瓶颈**：父进程实测 RSS **14.74 GiB**（低于文档假设的 21.99），
四折的护栏复查全部 `fits=True`、F4 余量 12.5 GiB；
worker 实测 RSS 均值 12.46 GiB、**最大 17.05 GiB**，
与探针在生产路径上测的 **17.06 GiB** 相差 0.01 GiB —— **17.1 的申报值是准的**。

**没有免费的加速**：makespan ≈ `ceil(72/W) × 最慢单次拟合 × deg(W)`，
W=24 给 3 × 1.437 = 4.31，W=16 给约 6.5，W=12 给约 7.5 —— **24 路已是最优**。
唯一真杠杆是那 4.16 GiB 的批次张量（惰性构造/按需生成/连续内存视图），
它同时降低内存与访存争用；但要改一个已测试的契约、且需重启 M2（损失已跑的 18 h），
**属所有者裁决，未动**。

## 5. 待办（按顺序）

### 4.1 M2 之后也已经自动化了：两个新监视器（2026-09-14 14:47 起在跑）

M1→M2 那一环的监视器已经跑完并成功，所以同样的模式延伸到了后面两环。
**现在有三个进程在跑**：M2 驱动（pid 3588838）、链式监视器、逐折预检监视器。

**(1) `m2_m3_chain_watcher.py`（pid 188088）——M2 通过后自动发射 M3。**
与 M1→M2 那个同构，但**多了一道更强的检查**，并且**每个检查都做过负向对照**：

| 阶段 | 内容 | 失败时 |
|---|---|---|
| 1 wait | `m2_v1/report.json` 存在、稳定、M2 驱动已退出 | 继续等 |
| 2 gates | `family=M2`、`mode=full`、`oos.all_passed=true` | 中止，不发 M3 |
| **3 registry** | **`load_walk_forward_evidence` 从签名的 `promotion_evidence.json` 独立重算 G1–G6，六道必须与报告逐位相同** | 中止 |
| 4 C′ | 四折胜出格的最终拟合检查点与搜索阶段逐字节相同 | 中止 |
| 5 A | `--resume` 重跑，报告 md5 逐字节相同 | 中止 |
| 6 memory | `MemAvailable ≥ 140 GiB`（M3 48 路 × 2.5 GiB + 0.6 = 120.6，/0.9 = 134，留余量取 140） | 等 30 分钟后中止 |
| 7 launch | M3：48 路、钉 1 线程、`--prior-report M1=m1_full_v1/report.json --prior-report M2=m2_v1/report.json` | — |

第 3 阶段是新增的，理由是它比验收 A 更贴近 M3 真正依赖的东西：
M3 的前置只需要"M2 过了门"，而**门数值能否从签名证据独立重算出来**正是这件事的实质；
A 测的是可复现性（也重要，但那是另一回事）。
**两个方向都验过**：对 `m1_full_v1` 的真实证据返回 True 并给出 sha256 `e941274d44922d2a118a70ca`；
把 G4 的观测值改成 0.123456 后返回 False 并精确指出
`G4: registry PASS/0.007826986071187494 vs report PASS/0.123456`。

C′ 还带一条**防空通过**的守卫：找不到可比对的 (搜索, 最终) 检查点对时**报失败而不是报 0/0 相同**。
这条守卫在开发时就抓到了我自己的参数化 bug（见下）。

**(2) `m2_fold_preflight_watcher.py`（pid 206286）——M2 每折一完成就跑一次评价预检。**
动机是时间差：M2 的评价阶段在四折全跑完后**只做一次**（约 09-23），
而**五个阻塞项里的五个都只在评价阶段发作**；但每折的 bundle 与签名预测**一折一落地就存在**。
**M2 又是这条边界上风险最高的家族**：它是时序的，而且 card 的家族**由胜者决定**
（`m2_card_family(winning) = f"TEMPORAL_{winning}"`，GRU 或 TCN 在搜索跑完前不可知）——
那正是 blocker 4（card family 不在评价器白名单里）与 blocker 6（loader 没有 `TEMPORAL_TCN`
条目、TCN 赢了也无法服务）出事的位置。
**折 1 发现是可修的一天，折 4 之后发现是白跑的八天。**

它照抄评价器的逐折路径（`_require_verified_model_bundle`、两个预测工件的验签与
`_require_candidate_evidence` / `_require_best_simple_evidence`、`load_split` +
`_labels_with_lineage`、`_validate_oos_frame` × 3），**并加了 M1 预检不需要的一项**：
card 的家族必须同时在评价器的 `_MODEL_CARD_FAMILIES[CARRIER]` 白名单里、
且 `artifact_loader` 能用它构造出模型（构造出的 `model_family` 还要等于 card 上写的那个）。

**这个脚本已在 M1 的四折上端到端验证过**（`--workspace m1_full_v1 --carrier M1 --prefix m1 --once`）：
**四折全部 PASSED、无告警**，包括 loader 真的构造出了模型
（`ResidualMLP, 70657 params`，与 F2 胜者 c14 `hidden 128/blocks 2` 的参数量吻合）、
冻结 config 与搜索胜者的比较键为 `['dropout','hidden_dim','num_blocks']` 且无失配。
顺带核实了一个此前没记的事实：**F4 的 best_simple 是 B3，而 F1–F3 是 B2**（逐折基线选择不同）。

**开发中它先抓到了我自己的两个 bug，都值得记**：

* **参数化漏了一处**：`preflight()` 里写死了 `_MODEL_CARD_FAMILIES.get("M2")`，
  而我用 `--carrier M1` 测试，于是它拿 M1 的 `RESIDUAL_MLP` 去比 M2 的白名单并**正确报警**。
  报警是对的、参数化是错的——已改成用 `CARRIER`。
  （这次误报顺带暴露了一个有用事实：`_MODEL_CARD_FAMILIES["M2"]` 实际是
  `{'M2','TEMPORAL_GRU','TEMPORAL_TCN'}`，**载体名自身也在白名单里**，
  因为它是用 `{carrier: {carrier, *names} …}` 构造的。）
* **"冻结 config == 搜索胜者"这个比较对 M2 会误判**：`torch_bundle_config`
  会**剥掉**架构 config dataclass 没声明的字段（M2 剥掉 `family` 与 `learning_rate`），
  所以 bundle 的 config 只有 `{raw_factor_count, hidden_dim, dropout}`，
  而搜索记录里的 config 有 `{family, hidden_dim, dropout, learning_rate}`。
  直接比会失败。已改成**只比冻结 config 自己的键、要求交集非空（防空通过）、
  并另外通过 card 的家族来校验胜者的架构选择**（`TEMPORAL_{winner.family}` 必须等于 card family）。
  实测确认：`TemporalGRUConfig`/`TemporalTCNConfig` 的字段都是
  `raw_factor_count/hidden_dim/dropout`，与 loader 的 `model_config_fields` 逐项相同。

**两个监视器都在 git 树外**（`/data/zhangyudong/qrv2_research_artifacts/`），
因为它们含有本次运行专属的硬编码（pid 3588838、工作区路径）。
日志：`m2_m3_chain_watcher.log`、`m2_fold_preflight.log`；
告警文件：`m2_m3_chain_watcher.ABORT`、`m2_fold_preflight.ALARM`、
成功标记 `m2_m3_chain_watcher.LAUNCHED`。
**接手时先看这三个文件是否存在**，它们比读日志快。

### 5.1 原待办清单

1. **等 M2**（预计 09-23 至 09-25）。监控：`ls m2_v1/search/F1 | wc -l` 应到 24；
   日志 0 错误；worker RSS 最大值不超 17.1 GiB；每折的护栏复查不被拒。
2. **M2 跑完后**：做与 M1 相同的三项验收（门 / C′ / A）+ §4.2 判定归档。
   预检脚本 `/tmp/verify_f1_bundle_loads.py`（已支持 `--fold`）可逐折提前跑，
   **但注意它对 M2 需要改两处**：模型工厂是 `_m2_model_factory`、
   card 的 `family` 会是 `TEMPORAL_GRU` 或 `TEMPORAL_TCN`（由胜者决定，不是固定值）。
3. **M3**：发射包已备好（48 路、1.02–1.18 天、内存已实测确认够），
   但**工期同样要按 M2 的教训重估**——M3 是静态家族、载荷小，
   降级应接近 M1 的 1.132 而不是 M2 的 1.437，所以它受"最慢格"影响的方式与 M2 不同
   （48 路 = 16 配置 × 3 seed 恰好一轮种子级，**每 worker 只跑 1 次拟合**，
   所以是 `1 × 最慢格`，不是 `3 × 最慢格`）。
4. **两条被推迟的驱动改动**（现在 M2 在跑，窗口又关上了，要等 M2 结束）：
   fold-cache 临时文件名竞态、`--folds` 死参数应改为拒绝非默认值。
   两者都不是 M2 的前置（缓存对这四折已全热、M2 不传 `--folds`）。

   **【2026-09-14 15:20 更新：两条都已实现、测好、变异检验过，放在独立 worktree 的分支上等合并】**
   既然 M2 还要跑约 9 天、期间不能动主工作树的驱动（spawn 每折从磁盘重导入），
   就用 `git worktree` 把它们做在了主树之外：

   * **worktree**：`/data/zhangyudong/qrv2_worktrees/deferred-driver-fixes`
   * **分支**：`codex/deferred-driver-fixes`，从 `9e0b2be` 起，两个提交：
     * `4f44e12` **缓存竞态**：新增 `write_fold_cache_atomic(cache_path, dataset)`，
       临时名带上 **pid + 8 位随机片段**（pid 分隔并发进程，随机片段分隔同进程内同键的两次写入、
       并抗 31 天不重启主机上的 pid 复用）；失败时删掉自己的临时文件（旧版不删，
       每次中断都在那个已有 6 GiB 的目录里留一个永不读取也永不清理的文件）。
       两处重复的模式（`walk_forward_common.py`、`walk_forward_factor_daily.py`）都改为走同一个助手。
     * `1dbc36c` **`--folds`**：新增 `_requested_folds(args)`，只接受全集（任意顺序），
       其余一律 `SystemExit`，理由写进消息（`folds_complete` / `WALK_FORWARD_PASSED`）。
       **校验放在 `parse_args` 而不是 `main()`**——`parse_args` 是唯一入口，
       放在那里就不可能被人重排到几小时拟合之后。
   * **主工作树未被触碰，已验证**：`git status` 干净、HEAD 仍是 `9e0b2be`、
     且 `grep -c "_requested_folds\|write_fold_cache_atomic"` 在主树的两个驱动文件里都是 **0**。
   * **门禁（在 worktree 内实跑）**：`ruff` 全清（src、tests 与三个被改脚本）、
     `mypy src` 170 文件清、`pytest tests -q` **2127 passed / 46 skipped**（基线 2107 / 45，
     即净增 21 个测试、0 失败）。
   * **变异检验（每个修复都做了，且失败信息指向"为什么要紧"）**：
     * 缓存：把临时名改回 `with_suffix(".parquet.tmp")` → **3 个测试失败**，
       关键那条报 `two writers computed the same temporary path and would interleave`
       并列出两个相同路径。
     * `--folds` 变异一：删掉 `parse_args` 里的调用 → 端到端与 wiring 两条测试失败。
     * `--folds` 变异二：**把调用挪到 `main()`**（功能等价、但可被重排）→ 同样两条失败。
       这条更隐蔽的退化正是 wiring 测试存在的理由。
   * **合并时机**：M2 与 M3 都跑完之后（约 09-25 前后）。合并只需把分支并入
     `codex/publish-quant-platform`；届时**必须重跑一次全门禁**，因为主线在这期间可能已前进。
     **不要在 M2 或 M3 在飞时合并**——那会改动主树的驱动，而下一折的 worker 会读到新代码。
5. **5.1 / 5.2 / 5.3**：5.1 要 M1–M3 全齐；5.2 要 5.1；
   5.3 的 M0R 实例已做（`docs/reports/2026-09-12-m0r-shadow-approval-preconditions.md`），
   而 **M1 通过已使"复杂候选"的解释分歧不再阻塞**——见该文档 §4 的 09-14 更新。

## 6. 红线（不变）

零 DB 写入；不碰 `materialization_status.{json,md}`、`/tmp/frozen_chain_hold.txt`、2026 locked 段；
`docs/research/` 保持未跟踪（**已加进本地 `.git/info/exclude`**，因为我两次用 `git add -A docs/`
误提交过它——规则写在文档里不管用，得去掉机制）；
推送用 `https_proxy=http://127.0.0.1:7890 git push origin codex/publish-quant-platform`；
**M2 在跑期间不要编辑 `research_v2/scripts/run_real_candidate_training.py`**
（spawn 每折从磁盘重新导入）；治理决定归所有者。

---

## 7. 【2026-09-14 16:15 增补】M2 运行 21 小时的遥测、一个会让下一个人算错 ETA 的陷阱、两条被推翻的判断

本节是对第 4 节与第 5 节的**运行中实测增补**，不改变任何结论；
但其中 §7.2 那条陷阱**已经让我自己算错过一次 ETA**，所以必须写下来。

### 7.1 遥测（16:10 实测，M2 已运行 21h16m，pid 3588838 存活）

* **F1 搜索 13/24 格已落记录，日志 0 错误**（日志里 24 条含 "error" 的行全是 torch 的
  CUDA 初始化警告，见 §7.3）。已落盘的 13 格是 c0–c5、c8、c9、c11、c13、c15、c16、c17。
* **剩下 11 格正好是网格的重的那一半**：GRU-128 ×3（c6/c7/c10）、TCN-64 ×2（c12/c14）、
  **TCN-128 ×6（c18–c23 全在里）**。其中 6 格已进入第 3 个（最后一个）seed，5 格在第 2 个。
* **worker RSS 最大 16.77 GiB**（申报 17.1），24 个 worker 全在；`MemAvailable` 约 196 GiB；
  swap 8 GiB 已用满但 `si/so = 0`（静止，没有在抖）。
* **无 `*.ABORT` / `*.ALARM`**；两个监视器都在，且 **PPID = 1、自有会话、无 TTY**
  （`setsid` 生效），所以它们不随任何终端会话消失——这一条本轮专门核过，
  因为一个会在九天里悄悄死掉的监视器比没有监视器更坏。
* **监视器与驱动的产物路径已对齐**：驱动写 `model_id = f"m2-{fold_id.lower()}"`
  （`run_real_candidate_training.py` 第 2969 行），预检监视器轮询
  `m2_v1/models/{prefix}-{fold}/checksums.sha256`、`prefix="m2"`——**逐字符相符**。
* **F1 搜索的收口时刻**：按第 4 节那条实测（TCN/128 单次拟合 14.1 h、三个 seed 串行）
  推算 c18 的第三 seed 约在 **09-15 11:30–13:30** 结束，即 **F1 搜索 ≈ 42 h**，
  与第 4 节的 42.3 h 吻合。**整折（含 7.3 h 最终拟合与 OOS）约 09-15 21:00**，
  四折 **09-23 至 09-25 的口径不变**。
* F1 已落盘 13 格的验证集前三：TCN-64/0.2/3e-4 = 0.059239、GRU-64/0.0/3e-4 = 0.057970、
  GRU-64/0.1/3e-4 = 0.056430；**lr 3e-4 占前九名中七席**。
  **这只是验证集、且 6 个 TCN-128 全未出，F1 胜者未定**，不可与任何 G4 相比。

### 7.2 陷阱：不要用"相邻两个**改进**检查点的时间差"推每次拟合的墙钟时长

我本轮这么做了，得到"F1 搜索今晚收口、M2 约 09-20 完成"的结论，**错了 3–4 天**。
原因是检查点只在**验证分提升**时落盘，而早停 patience 是 10：
一次拟合在最优 epoch 之后还要**空跑至多 10 个 epoch**。
TCN/128 的改进检查点跨度是 ep1→11、约 6.7 h，而真实单次拟合是 **21–22 epoch、14.1 h**——
**尾巴占了约 40%**，而且它恰好落在最慢的那一格上，也就是决定 makespan 的那一格。

正确口径（与第 4 节一致，此处只是把错误写法明确禁掉）：
**每 epoch 成本 = 同一 (config, seed) 相邻两个 epoch 检查点的时间差**，
**单次拟合 = 每 epoch 成本 ×（最优 epoch + patience）**，
**一轮 = 3 × 最慢格**。检查点的 mtime 只能当**下界**用。

### 7.3 两条被推翻的判断（都写在这里，免得下一个人重新推一遍）

1. **"GPU 不可用是最大提速杠杆"——错。** 主机确实坏了：`nvidia-smi` 报
   `Failed to initialize NVML: Driver/library version mismatch`（NVML 库 580.178），
   `torch 2.6.0+cu124` 的 `cuda.is_available()` 为 **False**，每个 worker 都打一条 Error 804 警告。
   **但驱动在四个训练点都硬编码 `device=torch.device("cpu")`**
   （第 597 / 919 / 2630 / 3315 行），**框架没有 CUDA 训练路径**，所以修 GPU 对本管线**零收益**。
   而且修它要重载驱动或重启主机，会直接杀掉在跑的 M2。
   **若将来真要上 GPU**：`training/runner.py` 第 52 行记着 device 是**影响数值**的
   （cpu 与 cuda 的摘要不同），换设备等于换数值契约——验收 C′ 的逐字节比对、
   复现审计、以及运行时描述里的 `cuda_available` / `cuda_device_names` 全要重做，那是一次新的谱系。
2. **"批张量优化必须在 M3 之前决定"——错。** 那个 4.16 GiB 的批次张量是
   **M2 时序族独有**的（`history_length=20` 的窗口批）；M3 是静态族、
   走 `build_static_training_batches`、载荷属 M1 量级（申报 2.5 GiB/worker、实测增量 +0.32 GiB）。
   所以**这个决定的窗口不在 M3 之前**，它只对未来某次 M2 重跑有意义，可以一直推迟到 M3 之后。

### 7.4 一个外部事实：本机约 90 GiB 被另一个用户的常驻进程占着

4 个属于 `zhangzhenye`（conda 环境 `trellis2`）的 `multiprocessing.spawn` 进程已运行
**27 天 20 小时**，RSS 约 22.1 / 23.9 / 21.0 / 23.7 GiB，CPU 0.3–0.4%（空转但不放内存）。
主机 629 GiB、当时 `MemAvailable` 约 196 GiB。
**含义**：第 4 节发射时算的"511.2 GiB 可用"里，这 90 GiB 是被别人占走的，
所以实际余量比申报口径薄。**不属于本项目，未动**；
若将来某次运行报内存不足，这是第一个要查的地方。
M3 的门槛是 `MemAvailable ≥ 140 GiB`，在 M2 退出后绰绰有余，**不构成阻塞**。

### 7.5 新增交付物：模型侧结果的文字记录已推到研究分支（供论文撰写）

所有者要求在**另一台服务器**上用实验结果辅助论文撰写，只要文字记录、不要权重。
已按其指定推到 `codex/research-conditional-factor-review` 的 `docs/research/` 下
（提交 `2cb14de`，已推送成功，远端 tip 与本地一致）：

* `docs/research/2026-09-14-model-side-walk-forward-results-v0.1.md`——自包含叙述：
  实验协议（15 因子 frozen_v2 / 基线只见 9 个稠密因子、四折年度滚动与逐折行数与 dataset id、
  三 seed 百分位均值集成、六道门定义）、五个已完成运行的完整 G1–G6、逐年 RankIC、
  逐折胜者、已有的复现证据，以及**写论文时必须主动披露的七条方法学限制**。
* `docs/research/evidence/2026-09-14-model-side-walk-forward/`——机读 JSON
  （由随包的**只读**提取器 `extract_from_reports.py` 从签名报告生成，不是手抄）、
  `problems_found.md`（问题记录）、以及被引用原件的**逐字节副本** + `MANIFEST.json`
  （记源路径、sha256、取值时的 git HEAD）。复制原件是必要的：研究分支比本分支落后约 95 个提交，
  那些报告在它上面并不存在。
* **没有任何权重、检查点、parquet 或数据库导出**；`promotion_evidence.json` 只记哈希不复制。

其中三条是**纠正**而不是复述，值得单独点出来，因为论文里最容易写错：

1. **基线看不到那 6 个稀疏因子是设计**（它们按公式在约一半交易日上是 NaN，稀疏性本身即信号），
   所以候选的信息集**严格包含**基线的——G4 是"相对诚实的简单参照的改进"，
   **不是**信息集匹配的对照实验。
2. **G4 是在 945 日交集上、与"在该日集上重算的基线"相减**：候选报告里的
   `baseline_mean_rank_ic = 0.0852268662737014`（945 日），
   而基线自己报告的是 `0.08705554536301544`（969 日）——**两个数都对，口径不同**。
   且 `945 / 969 = 0.9752321981424149`，**正是每个候选的 G6 观测值**，
   差的 24 天就是稀疏因子无法打分的那些天。
3. **2023 年候选弱于基线**（候选 0.0696–0.0730 对基线 0.0781），池化后才为正。
   G3/G5 都过，但这是应当主动披露的年份异质性，不要只报池化数。

**红线未被触碰**：主工作树的 `docs/research/` **仍然是未跟踪的**
（`.git/info/exclude` 第 7 行那条规则对**所有** worktree 生效，所以在研究分支的
worktree 里也必须用 `git add -f`；本次提交只包含 `docs/research/` 下的 30 个文件，
已用 `git diff --cached --name-only | grep -v '^docs/research/'` 验证为空）。
研究工作树在 `/data/zhangyudong/qrv2_worktrees/research-conditional-factor-review`，
**主树未被切换分支、未被编辑**（HEAD 仍是 `c490353`、`git status` 干净），
所以在跑的 M2 不受影响。

---

## 8. 【2026-09-14 16:45 增补】Task 5.1 的渲染器已经写好并合进主线，M2 未被触碰

M2 还要跑约九天，九天里唯一能真正推进的是**不依赖 M2 结果**的那部分工作。
本轮做了两件，都已推送。

### 8.1 `summarize_candidate_comparison.py`：把 Task 5.1 从"拼表"变成"重跑一条命令"

计划 Task 5.1 要的东西（各候选门控结果、相对基线与相对彼此的增量、
最优候选的年度分布与最差 seed、无候选通过时的明确记录）全部已存在于签名报告里，
缺的是一个**不会做错减法**的渲染器。已新增
`research_v2/scripts/summarize_candidate_comparison.py` + 26 个测试（提交 `39ee96a`，
分支 `codex/candidate-comparison-renderer` 已 ff 合并、worktree 已清理）。

两道守卫是重点，**都做过变异检验（即证明它们能失败）**：

* **可比性分组**：按 `(valid_day_count, baseline_mean_rank_ic, 逐折冻结基线选择, 逐折 dataset id)`
  分组，只有最大组进入两两增量矩阵，其余**列出理由且不给数字**。
  这正是"945 日的候选之间可以相减、而基线自己的 969 日数不可以"的那个条件。
* **共享基线恒等式**：组内必须满足 `delta(a) − delta(b) == mean(a) − mean(b)`（校验到 1e-12），
  否则该格印 `IDENTITY_VIOLATION` 而不是数字。

**写测试时抓到了渲染器自己的一个真空通过 bug**：`gates_passed` 原写作
`all_passed and all(item.satisfied for item in self.gates)`，而**空元组上的 `all()` 为真**，
于是一个根本没有任何门记录的候选会被渲染成 `WALK_FORWARD_PASSED`——
与链式监视器为 C′ 专门拒绝的"0/0 相同"是同一个形状。现已要求门记录必须覆盖 G1–G6。

**还有一条测试自身是真空的，也是变异检验暴露的**：断言整篇文档里出现
`IDENTITY_VIOLATION` 在守卫被删掉后**依然通过**，因为矩阵下方那段解释文字里就写着这个词。
已改成只断言矩阵那一节、并断言整行的精确内容。

**刻意不做的**：不含时间戳与主机名（渲染是输入的纯函数，同输入两次 md5 全同，已实测
`c79e5545ebad6d36c3e935935176f3b7`）；不重算任何指标；
**不发明"基线在候选日集上的逐年值"**（签名报告里没有这个量，见研究分支问题记录 J 节）；
不印 OOS 口径的最差 seed（报告只有验证口径的逐 seed 值）。

**门禁（在该 worktree 内实跑）**：`ruff` 全清（scripts、src、tests）、
`mypy src` 170 文件清、`pytest tests -q` **2132 passed / 46 skipped / 0 failed**
（基线 2107 / 45，即净增 26 个测试；46 与 `deferred-driver-fixes` worktree 当时的 skip 数一致，
多出的那一个 skip 是 `test_set_attention.py` 的 CUDA 容量测试，因本机 GPU 不可用而跳过）。

**合并安全性已逐项核过**（这是本轮唯一动主树的操作）：合并前记录
`run_real_candidate_training.py` 的 sha256 前 16 位 `33742880f314ca21` 与 mtime
`2026-09-12 00:50:33.890987021`，`git merge --ff-only` 之后**两者逐字符未变**，
分支相对主线的 diff 恰好是 2 个新文件、1111 行插入、0 行删除；
M2 驱动 pid 3588838 存活、25 个子进程数不变、`git status` 干净。

---

## 10. 【2026-09-14 18:50 增补】逐年增量补上了，**它推翻了两处已写进文档的说法**

新增只读脚本 `research_v2/scripts/analyze_yearly_baseline_deltas.py`（提交 `af75449`，
17 个单元测试，分支 worktree 已 ff 合并并清理），
判定文档 `docs/reports/2026-09-14-yearly-deltas-and-the-24-unevaluable-days.md`。
**门禁**：ruff 清、`mypy src` 170 文件清、`pytest tests -q` **2149 passed / 46 skipped / 0 failed**
（基线 2132/46，净增 17）。合并前后 `run_real_candidate_training.py` 的 sha256 与 mtime
逐字符未变（`33742880f314ca21` / `2026-09-12 00:50:33.890987021`），M2 未受影响
（pid 3588838 存活 23h49m、F1 15/24、无告警）。

**闭合的缺口**：候选报告只有自己的逐年值（945 日）与**池化**的基线均值（同 945 日），
而基线自己的报告是 969 日口径，所以"逐年增量"此前**任何工件里都没有**。
做法是重放评价器自己的基线分支（它为了产出那个池化数本来就构造了基线的**逐日**帧），
多调一次 `summarize_rank_ic_by_year`。**四道验收全部逐位通过才输出**：
A 复现签名报告的池化与四个逐年值；B 复现评价器自己的池化基线均值；
C 逐年增量按日加权重建公布的池化 delta（1e-12）；
D `945/969` 重建报告里的 G6（969 由 `_valid_label_day_count(labels, 30)` **测得**）。

**推翻之一：「2023 年候选弱于基线」是错的**（这句话我昨天写进了研究分支的记录）。
同日集重算：**M0R +0.003192、M1 全网格 +0.003205**，两者**四年全为正**；
只有已被取代的 M1 载体是 −0.000179。错因是**跨日集相减**——
正是 `summarize_candidate_comparison.py` 的可比性守卫在代码里挡住的那个形状，
**同一个错误在文档里发生了一次**。

**推翻之二：「G6 那 24 天来自 6 个稀疏因子无法打分」不成立。** 实测：
24 天 = `2022-07-29` + 2023 的 23 天（`04-28`、`05-10/11/12`、`05-16`、`05-19`、`06-21`、
`06-26～06-30`、`07-05～07-07`、`07-10～07-14`、`07-17～07-19`）；
那些天候选**有 3,003–3,255 行带有限分数**（有效标签行 3,355–3,623），
覆盖率 **0.8951 / 0.8989**，而 `ScoreHealthEvidence.passes()` 要求 **`0.90 <= coverage <= 1.0`**
——**是以 0.001–0.005 之差擦到硬地板之下**（30 行的截面阈值远未触及）。
**同一标签帧下、看不到那 6 个稀疏因子的 best_simple 基线在这 24 天同样不可评、逐日相同**，
所以与稀疏因子无关，也**不构成对候选有利的筛选**。
真正未闭合的是：那些日期上约 10% 的有效标签行为什么没有有限分数——
**属因子/数据可用性侧，已定位到具体日期，建议移交因子会话**。

**敏感性（`--sensitivity-no-score-health`）**：关掉地板、日集回到 969，
M0R 池化 delta `+0.009149 → +0.009542`、M1 全网格 `+0.007827 → +0.008194`，
**四年仍全为正、排名不变**——**G4 的通过不依赖这块地板，且地板对候选偏保守**。

**一次独立的逐位印证（值得单独记）**：关掉地板后，在**候选工作区里**算出的基线是
`0.08705554536301544`（969 日）与逐年
`0.08118566121294693 / 0.07806425016339068 / 0.09060417297861999 / 0.09832154338626425`，
与**基线自己那份签名报告**（`baseline_v2/report.json`，09-08 生成）**五个数逐位相同**
（`repr()` 比对）。所以 **969 与 945 的差异 100% 来自 0.90 覆盖率地板**，
不来自数据集/标签/行集的任何其它差别——**我此前"基线用的是另一个数据集所以日集不同"的
猜测也被这次比对否掉了**。

**研究分支同步更新**（`codex/research-conditional-factor-review` 提交 `e67a25e`，已推送）：
更正记录写在文档开头、§1.3 与 §2.1 原位改写（新增 §2.1.1 逐年增量表与 §2.1.2 敏感性表）、
`problems_found.md` 的 J 节标记为已闭合、
三组输出复制到 `evidence/…/yearly_baseline_deltas/`。
**M2 出报告后应对它重跑这个脚本**：若 M2 的不可评日集与 M0R/M1 不同，
"同 945 日集"这个可比性前提就要重新检验（对比渲染器的守卫会自动把它挡在增量矩阵外）。

---

## 11. 【2026-09-14 19:40 增补】§10 里那 24 天的归属被再次改准：**是基线一侧，不是候选**

§10 写的机制（"那些天候选有 3,003–3,255 行带有限分数、覆盖率 0.8951/0.8989"）**读错了主语**。
`evaluate_frozen_oos` 在算逐日 RankIC 之前会先调 `_match_prediction_coverage`，
**把候选与基线都限制到两者键的交集**，所以那份诊断描述的是**交集**，不是候选。
提交 `78216af` 把两侧分开测（新增 `coverage_attribution`，4 个新测试，
`pytest tests -q` **2153 passed / 46 skipped**，ruff 与 `mypy src` 全清）：

| 年 | 候选覆盖率中位数 | 基线（=交集）中位数 | 交集最小值 | 跌破 0.90 | 其中候选单独本可通过 | 以 <0.005 之差通过 |
|---|---|---|---|---|---|---|
| 2022 | **1.0000** | 0.9254 | 0.8951 | 1 | **1** | 0 |
| 2023 | **1.0000** | 0.9093 | 0.8903 | 23 | **23** | **42** |
| 2024 | **1.0000** | 0.9924 | 0.9711 | 0 | 0 | 0 |
| 2025 | **1.0000** | 0.9859 | 0.9809 | 0 | 0 | 0 |

**候选四年逐日覆盖率恒为 1.0000、跌破地板的天数为 0**；
而 `best_simple` 基线工件的键集**恰好等于"9 个稠密因子全非空"的行集**
（整折集合相等、对称差为 0：F1 两边都是 747,461、F2 两边都是 820,829），
占有效标签截面的 0.89–0.99。**24 天全部是基线一侧拖下去的**
（`days_dropped_by_the_baseline_side = 24 / days_pair_below_floor = 24`）。
**所以 G6 = 945/969 度量的是基线工件的稠密因子完整性，不是候选的打分能力。**

两个后果：

1. **2023 年在刀刃上**：219 个可评日里 **42 天（19%）以不到 0.005 的余量通过 0.90 地板**
   （最险的 `2023-05-17`、`2023-07-04` 都是 0.9001）。地板挪 0.005，2023 的样本就会大幅变动。
2. **地板加在交集上，于是基线的行覆盖率决定了候选的可评日集**。这不是 bug
   （配对指标必须取交集），但足够反直觉，已连同三种处置写进
   `docs/reports/2026-09-14-yearly-deltas-and-the-24-unevaluable-days.md` §4.2：
   维持现状（当前，且已证明对候选偏保守）／重新发布覆盖率完整的冻结基线／
   把地板只加在候选一侧。**后两种都会改动每一个已公布的门值、等于开新谱系，属所有者裁决，未做。**

顺带核清：`m0r_v2` 与 `m1_full_v1` 里的 `best_simple` 分数 parquet **逐字节相同**，
且与 `baseline_v2/baseline_oos/F1.parquet` 的 **747,461 行分数逐位相同**；
文件 sha 不同只是因为已发布形态把 lineage 从列移进了签名 manifest、
并把 `dataset_id` 从人工标签 `real-f1-v1-test` 重绑到内容寻址 id。

**仍未闭合的只剩再上一层**（移交因子会话，已定位到日期与因子）：
为什么在 `2022-07-29` 与 2023 年那 23 天，约 7–10% 的行有稠密因子缺失
（`2023-07-05` 的 3,627 个有效标签行上，9 个稠密因子 NaN 份额 0.0008–0.1020，
`yeslab_471` 0.1020、`yeslab_357` 0.0976 最高；缺失高度重叠，故"至少一个缺失"只占 7.1%）。

**研究分支已同步**（`codex/research-conditional-factor-review` 提交 `38843a7`，已推送）：
更正记录升到 v0.1.2、§1.3 与 §4 原位改写、`problems_found.md` 新增 K 节（设计观察）、
`yearly_baseline_deltas/` 换成带 `coverage_attribution` 的新输出、
并把主分支那份方法文档复制进 `vendored/`（该分支落后主线，不复制就读不到）。

**本轮所有改动都不影响在跑的 M2**：`run_real_candidate_training.py` 的 sha256
（`33742880f314ca21`）与 mtime（`2026-09-12 00:50:33.890987021`）在两次 ff 合并前后逐字符未变，
M2 驱动 pid 3588838 存活（19:40 时 24h45m、F1 16/24、无告警），
两个看守都在（链式 320418、逐折预检 215628，PPID 均为 1）。

---

## 12. 【2026-09-14 19:45 增补】M3 也有逐折预检了：第三个看守已在跑（pid 459020），而它需要先学会"等一个还不存在的驱动"

### 12.1 缺口

逐折预检是 M2 这条边界上最值钱的一道保险（六个阻塞点里五个只在评价阶段发作，
而每折的 bundle 与签名预测**一折一落地就存在**：折 1 发现是可修的一天，折 4 之后发现是白跑的八天）。
但那个实例是**按 M2 参数化的**，而 M3 同样有暴露面——
blocker 5（bundle config 字段）打的正是 **M2 与 M3 两家**。
M3 约 09-24 由链式看守自动发射、约一天跑完，**在它跑完之前没有任何逐折检查**。

直接把同一个脚本用 `--carrier M3` 起一份也不行，有两处会当场出错：

1. **驱动识别是硬编码 pid**（`M2_PID = 3588838`）+ `"--family M2"`，
   而 M3 的 pid 要等链式看守发射时才存在；
2. 因此它会**立刻**走进"驱动没了"的分支，为一次**根本还没发射**的运行写下 ALARM——
   而一条假告警的代价是让人从此不看告警文件。

### 12.2 改动（`m2_fold_preflight_watcher.py`，改前备份 `/tmp/m2_fold_preflight_watcher.py.bak-20260914`，sha256 前缀 `3b0d41051c285439`）

* **按命令行发现驱动**：`find_driver(family)` 扫 `/proc`；
  `is_driver(cmd, family)` 要求 cmdline 同时含 `run_real_candidate_training.py` 与 `--family <F>`，
  **并且第一个 token 含 `python`**。最后这条不是装饰：链式看守是通过
  `bash -c "cd …; setsid nohup env … python scripts/run_real_candidate_training.py --family M3 …"` 发射的，
  那一瞬间存在一个**cmdline 里两根针都有、但其实只是 shell** 的进程；
  信它就会报告一个从未 exec 的驱动。
* **钉 pid 仍然支持**（`--driver-pid`），`--discover-driver` 才切到发现模式；
  `driver_alive()` 两条路都走同一个 `is_driver`，所以"pid 被复用"这个九天窗口里的老问题照旧被挡住。
* **等驱动阶段**：`--wait-for-driver-hours`（M3 实例给 336 h = 14 天，覆盖 M2 的 09-23～25 收工再加余量）。
  等待期间不会写 ALARM；超过期限才写，且消息里明说"若链式看守已 ABORT，以它的理由为准"。
  睡眠按剩余期限截断，不会因为一个 poll 间隔把 14 天的预算超成 14 天零 5 分钟。
* **停止标记**：`--stop-marker`（M3 实例指向 `m2_m3_chain_watcher.ABORT`）。
  该文件一出现就**安静退出、不叠加第二条告警**——失败已经有人报告了，
  再报一次只会让两个文件互相掩盖时间线。
* 告警文案改为用 `CARRIER` 与 `WS.name`（原来写死 "M2 driver exited" 与 `m2_v1.log`）；
  新增 `--poll-seconds` 只为测试能快速跑完。
* **第 1–6 段检查逻辑一行未改**：与备份逐行 diff，被删的只有 `driver_alive` 的旧实现、
  `global` 那一行、启动日志那一行、以及写死 M2 的告警文案。

### 12.3 两个方向都验过（`test_fold_preflight_watcher_driver_discovery.py`，**23/23 通过**，可重跑）

* `is_driver`：接受真正的解释器命令行；**拒绝那个 launcher shell**；拒绝错的 family；拒绝空串与无关进程。
* **活体对照**：`find_driver("M2")` 与一份**独立实现**的 `/proc` 扫描结果一致（当前 = 3588838）；
  `find_driver("M3")` 与 `find_driver("M9")` 都返回 None。
* 钉 pid 的四种组合：活 pid + 对 family → True；死 pid → False；
  **活 pid + 错 family → False**；不钉 pid 走发现 → True/False 各自正确。
* **等驱动阶段端到端**（临时工作区、真进程）：驱动始终不出现 → 退出码 1、写 ALARM、
  文案是 `M3 driver never appeared`、日志证明它**先等了**（`WAIT: no M3 driver yet`）而不是立刻告警；
  预置 stop-marker → **退出码 0、不写 ALARM**、日志写明 `STOP:` 与理由。
* `--once` 指向一个还不存在的工作区 → 不崩、不误告警。

### 12.4 第三个看守已在跑

```
pid 459020  PPID 1  自有会话  无 TTY
--workspace …/m3_v1 --carrier M3 --prefix m3 --discover-driver
--wait-for-driver-hours 336 --stop-marker …/m2_m3_chain_watcher.ABORT
--log …/m3_fold_preflight.log --alarm …/m3_fold_preflight.ALARM
```

日志现在是两行：`watcher start … driver=discovered wait_h=336.0 stop_marker=…` 与
`WAIT: no M3 driver yet; waiting up to 336 h for one`。**这就是它该有的样子**，不是故障。

**接手时要看的告警/标记文件从三个变成五个**：
`m2_m3_chain_watcher.{ABORT,LAUNCHED}`、`m2_fold_preflight.ALARM`、
**`m3_fold_preflight.ALARM`**（新）、以及 M3 发射后的 `m3_v1.log`。
判断看守是否在跑用 `pgrep -af "chain_watcher.py|preflight_watcher.py"`，不要信文档里的 pid。

**一个命名上的坦白**：这个脚本现在对 M1/M2/M3 都通用，文件名却还叫
`m2_fold_preflight_watcher.py`。**故意没改名**——正在跑的 M2 实例、本文 §4.1 与
研究分支的记录都按这个路径引用它，在多日运行中途改名换来的整洁不值那个混淆。
M3 实例跑的是**同一个文件**，只是参数不同。

### 12.5 M2 状态（19:41 实测）

驱动存活 1d00h46m，**F1 18/24**（今天一路 13 → 15 → 16 → 17 → 18，剩下的全是 TCN-128 那一档），
无 `*.ABORT` / `*.ALARM`，ETA 口径不变（09-23 至 09-25）。

---

## 13. 【2026-09-14 20:45 增补】那 24 天的成因查到了：**`yeslab_357` 与 `yeslab_471` 对科创板证券成对缺失，止于 2023 年 9 月**

§11 把 24 天归到了基线一侧，但没回答"基线为什么少那些行"。本轮回答它，
新增只读脚本 `research_v2/scripts/analyze_dense_factor_coverage.py` + 15 个测试
（提交 `f96c4b2`，报告 `docs/reports/2026-09-14-dense-factor-coverage-gap-and-the-24-days.md`）。
**只读已发布的 dataset parquet：不连数据库、不读权重、不碰预测工件，四个测试年 21 秒跑完。**

**结论是一个等式，不是一段叙述**：

> 被丢掉的日子 = **稠密不完整份额 > 10% 的日子** = 2022 年 1 天 + 2023 年 23 天 + 0 + 0 = **24 天**，
> 与逐日帧里 `rank_ic_valid=False` 的日子**逐日相同**。

（0.90 覆盖率地板与 10% 不完整份额本来就是同一件事的两种说法；
2023 年的**中位数是 9.07%**，所以整年都贴着这条线——这就是 §11 里"42 天以 <0.005 通过"的来源。）

| 年 | 有效标签行 | 稠密不完整行 | 份额 | 受影响证券 / 全部 | **科创板受影响 / 在架** | 份额 >10% 的天数 |
|---|---|---|---|---|---|---|
| 2022 | 803,572 | 57,051 | 7.10% | 688 / 4,342 | **326 / 326（100%）** | **1** |
| 2023 | 881,863 | 61,554 | 6.98% | 686 / 4,714 | **427 / 457（93%）** | **23** |
| 2024 | 931,411 | 7,666 | 0.82% | 319 / 4,882 | 26 / 490 | 0 |
| 2025 | 936,317 | 13,332 | 1.42% | 331 / 4,911 | 48 / 543 | 0 |

**缺的是哪两个因子**：`f01 = yeslab_357` 与 `f12 = yeslab_471`，
2022 年 43,740 行、2023 年 52,400 行同时缺这两个（第二名的组合只有 6,261 / 4,273 行）。
**而且是嵌套的**：`只缺 f01` 的行数**恒为 0**（2022 与 2023 都是），
即 f01 的缺失集是 f12 的子集——**两者共享同一个上游依赖，修一个等于修两个**。
缺失率逐年：`yeslab_357` **0.0548 → 0.0598 → 0.0007 → 0.0005**；
`yeslab_471` 0.0605 → 0.0630 → 0.0033 → 0.0053。
2024/2025 残留的是另一对（`f00 = yeslab_206` 与 `f10 = yeslab_443_reg30_max20_v1`，同样嵌套），
份额 0.8–1.4%，**从未足以丢掉任何一天**。

**缺的是哪些证券**：几乎全是**科创板 688/689**。2023 年 686 只受影响证券里 427 只是科创板
（当年在架 457 只，**93%**）；2022 年是 **326/326 = 100%**。
**不是偶发**：2023 年有 **565 只**在 ≥20 个不同交易日上不完整，
`689009.SH` **四年里每个交易日都缺**（242/242/242/243），说明 10 月那次好转没有覆盖到它。

**时间断点不是数据版本造成的**：逐月看，2023 年 1–9 月不完整份额在 7.4%–10.0%，
**10 月 0.54%、11 月 0.64%、12 月 0.72%**；而 dataset manifest 里
**`factor_daily` 2023 全年的 data_version 是同一个**（`715af5a14a4b…`）。
所以不是"后来的分区用了更新版本"，而是**同一版本内因子值本身在 10 月起了变化**。

**移交因子会话的问题（具体到可接手）**：
`yeslab_357` 与 `yeslab_471` 为什么在 **2022-01 至 2023-09** 对几乎全部**科创板**证券无值，
而自 **2023-10** 起有值？在同一 data_version 内出现这种断点，
需要确认是上游覆盖变化、可得性规则（上市时长/财报期/样本资格）变化，还是发布拼接方式造成的。
**本轮只测已发布面板，没有读因子生产代码、没有连数据库。**

### 13.1 一个会误导人的运行方式（记下来免得下次白查）

**从仓库根目录跑 `pytest research_v2/tests` 会让 `tests/unit/training/test_parallel_search.py`
的 4 个测试失败**（那些测试 spawn worker，依赖 suite 文档规定的 rootdir/cwd）。
**从 `research_v2/` 里跑则全绿**：本轮实测 `2168 passed / 46 skipped / 0 failed`
（基线 2153/46，净增 15）。这是**调用方式的问题，不是回归**——
同一棵树在正确 cwd 下是绿的，`ruff` 与 `mypy src`（170 文件）也都清。

### 13.2 状态

主分支 `f96c4b2`、研究分支 `938cebd`（v0.1.3，含 `dense_factor_coverage/` 与两份新报告的副本），
均已推送。`run_real_candidate_training.py` 的 sha256（`33742880f314ca21`）与 mtime 逐字符未变；
M2 驱动存活 1d01h26m、F1 18/24、无告警；三个看守都在（320418 / 215628 / 459020，PPID 均为 1）。
临时 worktree `dense-coverage` 已清理，现存两个：`deferred-driver-fixes`（等 M3 后合并）
与 `research-conditional-factor-review`（论文用）。

---

## 14. 【2026-09-14 21:00 增补】复现审计补上了缺的那一环：两个领先候选**四折全部逐位相同**

§D（研究分支问题记录）里那条"签名链证明的是**归属**、不是**复现**"一直缺后半段的证据：
复现审计此前只在 `m1_parallel_v1`（M1 载体，09-11）上跑过，
而**被引用得最多的两个运行从未审计**。本轮用既有工具
`research_v2/scripts/verify_prediction_reproduction.py` 补齐（只读、不连数据库）：

| 工作区 | 家族 | 复核行数 | identical / max abs diff | registry 状态（机器读出） | 打分线程 |
|---|---|---|---|---|---|
| `m0r_v2` | LGBM_RANKREG | **3,557,762** | 四折全 **true / 0.0** | **`WALK_FORWARD_PASSED`** | 52（宿主默认；LightGBM 无 torch 检查点可钉） |
| `m1_full_v1` | RESIDUAL_MLP | **3,553,163** | 四折全 **true / 0.0** | **`UNREGISTERED`** | **1**（该运行自己记录的值） |

报告 `docs/reports/2026-09-14-prediction-reproduction-audits.md`（提交 `5d8837d`），
原始日志 `$A/reproduction_audit_{m0r_v2,m1_full_v1}.log`，
JSON 摘要也复制进了研究分支的 `reproduction_audits/`（提交 `4186baf`，已推送）。
两个工作区并行跑约 **18 分钟**。

**三点必须一起转述，否则"0.0"会被误读**：

1. **审计把自己钉到该运行记录的线程数**（`recorded_run_threads()` 从工作区自己的检查点 manifest 读，
   混合多个值直接拒绝）。已训练的 torch bundle 前向**不是线程稳定**的：
   同一份 `m1_v2` F1 bundle 在 1/2/4/8 线程下逐位相同，**16 线程差 4.805844e-05、52 线程差 4.923683e-05**，
   量级是**近似并列股票的秩反转**（会改选股名次），不是末位舍入。
2. **它对任何差异都抛错并非零退出**（`_compare_scores` → `ReproductionError`），
   `"reproduced": true` 只在四折全同时才会被写出，所以**可以直接当门禁用**。
3. **三者不可互相替代**：复现审计答"存下来的分数是否确由这个 bundle 算出"；
   验收 C′ 答"这个 bundle 是否由那次搜索选出"（检查点逐字节比对）；
   验收 A 答"这次运行能否从头重跑"（`--resume` 报告逐字节相同）。

**给下一班的建议（本轮未做）**：M2 每折落地后跑一次
`verify_prediction_reproduction.py --workspace $A/m2_v1 --fold F1`（`--fold` 可重复、
不需要 report、不查 registry，正是为在跑的运行设计的）。
**逐折预检看守不重算分数**（它查证据链、card family 白名单、loader 可构造性、冻结 config 与胜者一致），
所以复现审计补的是它不覆盖的那一块；而 M2 是时序家族、TEST 面板要从三个已发布分片重建 20 日窗口，
**是最值得逐折提前验的一个**。

### 14.1 红线文件的完整性快照（本轮所有操作之后核过）

| 文件 | sha256 前 16 位 | mtime |
|---|---|---|
| `research_v2/artifacts/diagnostics/factor_daily_materialization/materialization_status.json` | `44ab75f6d073f9d9` | 2026-09-07 16:36 |
| `…/materialization_status.md` | `fdea7ba855bdcd6c` | 2026-09-07 16:36 |
| `/tmp/frozen_chain_hold.txt` | `a47a7fe690126116` | 2026-09-05 21:56 |
| `research_v2/scripts/run_real_candidate_training.py` | `33742880f314ca21` | 2026-09-12 00:50:33.890987021 |

四个都早于今天所有操作、`git status` 干净，即**本轮零改动**。
（`/tmp/frozen_chain_hold.txt` 本身值得一读：它记的正是 2014 年 `dropna(subset=全部 17 因子)`
只剩 **11.6%** 覆盖率而暂停冻结链的决定——与今天测到的"稠密因子完整性决定可评日集"是同一类问题，
只是当年更极端。）

### 14.2 状态（21:00 实测）

M2 驱动 pid 3588838 存活 1d02h、**F1 18/24**、无 `*.ABORT` / `*.ALARM`；
三个看守都在且 PPID=1（链式 320418、M2 逐折 215628、M3 逐折 459020）。
主分支 `5d8837d`、研究分支 `4186baf`，均已推送。ETA 口径不变（09-23 至 09-25）。

---

## 15. 【2026-09-14 21:05 增补】逐折预检里加进了**复现审计**，两个看守已用新代码重启（**pid 又变了**）

### 15.1 为什么加

§14 说清了"归属 vs 复现"的分工：逐折预检查的是**归属**
（bundle 验签、card family 白名单、loader 可构造、冻结 config 与搜索胜者一致、三个帧的验签与校验），
**它一行分数都不重算**。而 M2 恰恰是复现路径上最复杂的一家：
时序家族的 TEST 分数来自 **20 日窗口面板**，审计要从三个已发布分片把它重建出来
（`_recompute_temporal_torch`），这条路**从未在真实数据上跑过一折**。
按本文一开始就写下的那句标准——**折 1 发现是可修的一天，折 4 之后发现是白跑的八天**——
这一步应该在折 1 就做，而不是等到 M2 的评价阶段（约 09-23）。

所以逐折预检现在在**每折的证据检查全过之后**，追加一次
`verify_prediction_reproduction.py --workspace … --fold <F>`（`--fold` 模式正是为在跑的运行设计的：
从该折签名选择取 bundle 与 TEST 数据集，不需要 report、不查 registry）。

### 15.2 实现上的两个决定

* **子进程，不是 import**：审计在**进程启动时**钉 torch 的 intra/inter-op 线程数，
  而 inter-op **一个进程只能配一次**；看守自己 `import torch`，
  在同进程里跑审计就钉不住了（§14 已实测：已训练 bundle 在 16/52 线程下差 4.8e-05 / 4.9e-05，
  是秩反转）。子进程同时把它的内存与看守隔开。
* **凡不是真通过，一律告警**：非零退出、退出 0 但**没有 JSON 摘要**、摘要**不是合法 JSON**、
  摘要里 **folds 为空**（审计了什么也没比——与 C′ 的"0/0 相同"同一个形状）、
  任何一折 `identical=false`、以及**超时**（默认 3600 s，消息里明写"超时**不是**通过，
  该折分数处于未验证状态"）。脚本缺失也告警。

### 15.3 两个方向都验过（`test_fold_preflight_reproduction_audit.py`，**21/21 通过**，可重跑）

用一个假的审计 CLI 演全部八种结局：干净通过（返回 True、不写告警、日志带行数与线程数）；
退出 0 但报差异；退出 1；退出 0 无摘要；退出 0 摘要非法 JSON；退出 0 但 folds 为空；
超时（把 `AUDIT_TIMEOUT_S` 调到 2 s、假脚本睡 30 s）；脚本不存在。
**七种非通过结局全部落 ALARM 并返回 False**，且每条告警文案都写明"为什么这要紧"。
测试把看守的 `REPO / WS / KEY_PATH / LOG / ALARM` 重指到临时目录，
**不碰真实 `m2_v1`、不碰真实审计 CLI、不连数据库**。

### 15.4 两个看守已重启，**pid 变了**

改前备份 `/tmp/m2_fold_preflight_watcher.py.bak-20260914b`（sha256 前缀 `31bbdcee76f3b669`）；
与备份逐行 diff，**被删的只有两行**（旧的 PREFLIGHT COMPLETE 日志行与旧的启动日志行），
其余全是新增；已有的证据检查逻辑一行未改。旧日志留档为
`m2_fold_preflight.log.pre-restart-20260914` 与 `m3_fold_preflight.log.pre-restart-20260914`。

| 看守 | 旧 pid | **新 pid** | 启动日志里的关键项 |
|---|---|---|---|
| 链式（M2→M3） | — | **320418**（未重启） | `WAIT: for m2_v1/report.json …` |
| M2 逐折预检 | 215628 | **529186** | `carrier=M2 driver=3588838 reproduction_audit=True audit_timeout=3600` |
| M3 逐折预检 | 459020 | **529385** | `carrier=M3 driver=discovered wait_h=336.0 stop_marker=…ABORT reproduction_audit=True` |

三者 **PPID 均为 1**、自有会话、无 TTY，无 `*.ALARM` / `*.ABORT`。
**本文里出现过的所有看守 pid（188088、206286、213719、215628、459020）现在全部作废**，
以 `pgrep -af "chain_watcher.py|preflight_watcher.py"` 为准。

### 15.5 代价与状态

每折多花约 5 分钟（时序折要重建窗口面板并重算数据集哈希），
相对 M2 单折约 50 小时的工期可以忽略；审计进程峰值内存与一个 worker 同量级，
而它跑的时候该折的 24 路搜索已经结束，内存正处在最宽松的时刻。

M2 本身未受影响：驱动 pid 3588838 存活 1d02h06m、F1 **18/24**、
`run_real_candidate_training.py` 的 sha256 与 mtime 仍逐字符未变。
**F1 落地时将第一次同时得到两件事**：逐折证据预检，与**时序家族在真实数据上的第一次复现审计**。

---

## 16. 【2026-09-14 21:40 增补】新的审计阶段已在**真实签名工件**上端到端验过：M1 四折全过

§15 的测试用的是假审计 CLI，它证明的是"看守对各种结局的反应正确"，
**没有**证明"看守与真审计接起来能跑通"（最可能出错的地方是 JSON 摘要的形状：
`--fold` 模式的输出与整跑模式并不显然相同）。所以又做了一次真件端到端：

```
m2_fold_preflight_watcher.py --workspace …/m1_full_v1 --carrier M1 --prefix m1 --once \
    --log /tmp/m1_preflight_e2e.log --alarm /tmp/m1_preflight_e2e.ALARM
```

**结果：四折全部 `PREFLIGHT COMPLETE`，末行 `4 fold(s) checked; alarm file absent - clean`，
全程 21:05:54 → 21:35:59（30 分钟），未写任何 ALARM。**逐折审计明细：

| 折 | 审计起止 | 耗时 | 复核行数 | max abs diff | 打分线程 |
|---|---|---|---|---|---|
| F1 | 21:09:45 → 21:13:09 | 3m24s | 803,572 | **0.0** | 1（requested 1） |
| F2 | 21:16:27 → 21:20:10 | 3m43s | 881,863 | **0.0** | 1（requested 1） |
| F3 | 21:23:46 → 21:27:45 | 3m59s | 931,411 | **0.0** | 1（requested 1） |
| F4 | 21:31:33 → 21:35:59 | 4m26s | 936,317 | **0.0** | 1（requested 1） |

行数与 §14 的整跑审计逐折相同，`--fold` 模式因此与整跑模式**互相印证**。
顺带确认了 `--fold` 模式的摘要形状（`{"folds": [...], "reproduced", "scored_under_torch_threads",
"threads_requested"}`，每个 fold 记录含 `identical` / `row_count` / `max_absolute_difference`）
与看守解析器所设一致——**这正是最可能静默出错的地方**：
若键名不符，`folds` 里的 `identical` 会取到 `None`，`not None` 为真，
于是"全部通过"会被读成"全部不通过"并误告警。
日志已归档为 `$A/m1_full_v1_preflight_e2e_with_audit.log`。

### 16.1 两个为 M2 准备的量化事实

* **单折审计耗时随行数增长**（3m24s → 4m26s，803k → 936k 行），静态家族约 **4 分钟/折**。
  M2 是时序家族，TEST 面板要从三个已发布分片重建 20 日窗口，**预计更久**；
  超时上限设的是 3600 s，余量足够。
* **审计子进程峰值 RSS 实测 5.6 GiB**（静态家族，采样 240 s 取最大）。
  它会与**下一折的 24 路搜索并行**运行，而驱动在折初的内存护栏要求
  `24 × 17.1 + parent ≈ 425 GiB` 可用（F1 折初实测可用 496.5 GiB，余量约 70 GiB），
  所以即使时序审计的峰值是静态的数倍，也**不会把下一折的护栏挤到拒绝**。
  这一条是算出来的余量，不是实测的时序峰值——**M2 F1 落地时应顺手记一次实际峰值**。

### 16.2 时序复现路径的既有覆盖

`tests/unit/inference/test_reproduction.py` 在当前树上 **74 passed**（6.0 s），
其中包含时序专项：`test_the_temporal_path_closes_on_a_real_bundle`（真 TemporalGRU bundle +
真已发布折）、`test_the_temporal_row_set_is_the_labelled_subset`、
以及 TEMPORAL_GRU / TEMPORAL_TCN 两家都必须路由到时序打分器的那条参数化测试
（路由错到静态打分器不会大声失败，只会用错误的窗口给分）。
所以**明天 M2 F1 的审计不是"第一次有任何覆盖"，而是第一次遇到真实数据**。

### 16.3 状态（21:37 实测）

M2 驱动 pid 3588838 存活 **1d02h42m**、F1 **18/24**、无 `*.ABORT` / `*.ALARM`；
三个看守都在且 PPID=1（链式 320418、M2 逐折 529186、M3 逐折 529385）。
主分支本节提交后为最新；`run_real_candidate_training.py` 的 sha256（`33742880f314ca21`）
与 mtime（`2026-09-12 00:50:33.890987021`）自始至终未变。

---

## 17. 【2026-09-14 22:00 增补】计划 §5.3 的 M1 实例已产出，而它带出一条**服务侧的硬前置**

### 17.1 Task 5.3 的 M1 实例（提交 `fffa973`）

`docs/reports/2026-09-14-m1-full-grid-shadow-approval-preconditions.md`。
**它不重复 M0R 前置清单里与模型无关的部分**（F6.1 五项框架、批准后生效的六条约束、
"复杂候选"的解释分歧都在那份里），只写 M1 特有的：

1. **M1 比 M0R 多一道前置：先注册。** F6.1 的 `WALK_FORWARD_PASSED → SHADOW`
   以"已在册"为前提，而 **M1 不在册**（本轮由复现审计**机器读出** `registry_status=UNREGISTERED`，
   `model_registry/models/` 下仍只有 M0R 的 `32315c89…`）。
   注册所需证据已齐且已被独立验证（`promotion_evidence.json` sha256 `e941274d…`，
   registry 重算的 G1–G6 与报告逐位相同）。注册哪一份 M1（载体 vs 全网格）仍是所有者裁决。
2. **四折是四个不同架构**（`128/3/0.0`、`128/2/0.2`、`128/2/0.0`、`128/3/0.1`）
   却共用**一个 model_id**，而 shadow pointer 指向的是 `model_id + model_version`
   （`registry/active_pointer.py` 的 `ActiveModelPointer`），**不是某个折的 bundle**。
   所以"某个打分日由哪一折的 bundle 出分"这个规则**目前没有被明确写下**，
   本文不代为设定，但把它列为批准前必须明确的项——对 M1 比对 M0R 要紧，
   因为选错折不只是数值差异，而是**用了另一个模型**。
3. **选择偏差使"选 M1 还是 M0R"不是中性的**：M0R +0.009149（1 格、无偏差）
   对 M1 +0.007827（27 格、含偏差），真实差距大于表面的 0.001322。

### 17.2 顺带测出来的一条新缺陷：**服务侧不钉 torch 线程数，而工件里也没有这个数**

复现审计能报 `max_absolute_difference=0.0`，靠的是**把自己钉到该运行记录的线程数**
（`recorded_run_threads()` 从**工作区的检查点 manifest** 读）。但是：

* **card 只有四个字段**（`dataset_id` / `family` / `model_status` / `model_code_version`，
  已核 `m1-f1/model_card.json`），bundle 的其余文件也不含线程数——
  **离开那个工作区，重算出 0.0 的条件就丢了**；
* **服务侧一行线程配置都没有**：`src/research_v2/inference/` 下只有 `reproduction.py`
  与线程有关（且是**读**不是设）；`torch_scoring.py` / `predictor.py` / `daily_pipeline.py`
  与 `configs/inference/daily.yaml`、`configs/production.yaml`、`configs/base.yaml`
  都没有 `set_num_threads` / `configure_torch_threads`（逐文件核过）；
* 而**已训练的 torch bundle 前向不是线程稳定的**（1/2/4/8 逐位相同，
  **16 线程差 4.805844e-05、52 线程差 4.923683e-05**，是近似并列股票的秩反转）。

**推论**：把 torch 候选（M1，将来 M2/M3）放进影子、在默认 52 线程宿主上打分，
分数会与被审计、被签名的不一致，**而现有任何一道门都不会发现**
（PSI 比分布、监控比数据与分数健康，没有一道会把影子分数与 bundle 重算值逐行对比）。
**这不是 M1 独有的问题，是所有 torch 候选的**；M0R 是 LightGBM，本轮审计在宿主默认 52 线程下
仍然逐位相同，所以它不受这条影响。

**处置**（两层，都已成文，**都没有动代码**）：

* **今天就能做、不需改代码**：影子/生产打分进程启动时把 intra 与 inter 都钉到 **1**
  并留痕——已写进 M1 前置清单 §3.1，**批准影子前必须落在运行手册里**。
* **长期修法已成设计稿**：`docs/superpowers/specs/2026-09-14-serving-thread-contract-design.md`
  （提交 `981a48a`）。形状受三条约束决定：不能让现有 bundle 失效
  （`artifact_loader` 里已有同一类教训的原话："requiring the field would refuse every existing
  bundle, including the promoted M0R"）、合并窗口在 M3 之后、
  以及沿用仓库既有的 `unverified_bindings` 做法（**说出没校验什么，而不是悄悄跳过**）。
  设计稿含逐文件改动清单、6 条测试（每条都要变异检验）、验收标准
  （其中硬的一条：改动后 `m0r_v2` 与 `m1_full_v1` 的四折审计仍须 0.0），
  以及明确的"本设计不做什么"。

### 17.3 对合并窗口的影响：**待合并的驱动改动从两条变成三条**

`codex/deferred-driver-fixes` 的两条（fold-cache 临时名竞态 `4f44e12`、
`--folds` 死参数 `1dbc36c`）之外，再加上 §17.2 的 card 线程字段。
三者都改 `run_real_candidate_training.py`，**必须等 M2 与 M3 都跑完后同一批合并、合并后重跑全门禁**。
**本轮已核实合并窗口是干净的**：自 `9e0b2be` 以来主线只动了
4 份文档 + 3 个新分析脚本 + 3 个新测试文件，与那两条修复触及的 5 个文件
（驱动、`walk_forward_common.py`、`walk_forward_factor_daily.py` 及两个测试文件）
**交集为空**（`comm -12` 实测为空），所以不存在冲突。

### 17.4 状态（22:00 实测）

M2 驱动 pid 3588838 存活约 1d03h、F1 **18/24**、无 `*.ABORT` / `*.ALARM`；
三个看守都在且 PPID=1（链式 320418、M2 逐折 529186、M3 逐折 529385）。
主分支 `981a48a`、研究分支 `26be7e9`，均已推送；
研究分支的 `problems_found.md` D 节已追加这条服务侧前置（因为它是"结果可验证"这个主张的边界条件）。
`run_real_candidate_training.py` 的 sha256 与 mtime 仍未变。

---

## 18. 【2026-09-14 23:00 增补】§17.2 那条缺陷已经修好了，放在分支上等合并窗口

**分支 `codex/serving-thread-contract`（提交 `b70ad8a`，未合并）**，
worktree 在 `/data/zhangyudong/qrv2_worktrees/serving-thread-contract`。
设计与实现结果都在 `docs/superpowers/specs/2026-09-14-serving-thread-contract-design.md`
（§1–§6 是设计，**§7 是实现结果、门禁与变异检验**）。改动 6 个文件、+492/−29 行。

**三部分**：

1. **card 带上线程数**：`torch_model_card(..., torch_intraop_threads)` 是**必填关键字**
   （省略即 `TypeError`——可选参数会让"忘了传"重新变成静默缺口），写入前校验为正整数。
   取值来自新的 `_card_torch_threads()`：并行运行用 `execution.torch_threads`
   （驱动本来就强制非空、worker 就是按它钉的），串行运行用实测的宿主值——
   **两条路都与检查点 manifest 记录的是同一个数**，所以第 3 部分的一致性校验才有意义。
2. **服务侧不许猜**：`load_inference_artifact(..., serving_torch_threads=None)`，
   card 优先、调用方补位；两者都有且不一致 → **拒绝**（不调和）；
   torch 家族两者都缺 → **拒绝加载**并在消息里给出两条补救路径。
   钉线程时若 inter-op 池已启动（`RuntimeError`），**比对当前线程数、不一致就报错**，不静默放过。
   **评价侧刻意不拒绝**：`load_bundle_for_evaluation` 照旧加载旧 bundle，
   只把该绑定记进 `unverified_bindings`——复现审计走这条路，拒绝就等于让全部历史工件不可审计。
3. **审计侧交叉校验**：`_require_card_threads_to_agree_with_the_run()` 在**整跑与 `--fold` 两个入口**
   都调用；card 无字段则跳过（历史工件）、字段畸形则拒绝、与检查点记录不一致则拒绝。

**门禁**：`ruff` 清、`mypy src` 170 文件清、`pytest tests -q` **2187 passed / 46 skipped / 0 failed**
（基线 2168/46，净增 19）。**设计里最硬的那条验收是重测的、不是假定的**：
用改动后的代码重跑两个领先候选的整跑复现审计，
`m0r_v2` 与 `m1_full_v1` **四折仍全部 `identical=true`、`max_absolute_difference=0.0`**
（3,557,762 行与 3,553,163 行）。日志存为
`$A/reproduction_audit_{m0r_v2,m1_full_v1}.post_thread_contract.log`。

**五处守卫全部做过变异检验**，其中一处是变异检验救回来的：
最初那 5 条 card/检查点测试**直接调用私有助手**，把两个入口的调用删掉后它们**全部照旧通过**
（守卫存在但没接线）。补了一条走 `verify_fold_predictions_reproduce` **入口**的接线测试
（给 stub 真实 bundle 摘要以通过冻结折校验），外加它的反向对照
（card 与检查点一致时审计必须**越过**这道检查、因别的原因失败）。
其余四处变异分别被：服务侧拒绝分支、`unverified_bindings` 里的该绑定（3 条测试）、
`_card_torch_threads` 的申报值优先、`card_torch_threads` 的严格类型校验抓到。

### 18.1 合并窗口：现在是**三条**驱动改动一起合，且已核实互不重叠

| 分支 | 改驱动的哪里 |
|---|---|
| `codex/deferred-driver-fixes`（`4f44e12`、`1dbc36c`） | fold-cache 原子写入助手；`parse_args` 里的 `_requested_folds` |
| `codex/serving-thread-contract`（`b70ad8a`） | `torch_model_card` 与它的四个调用点、`_card_torch_threads` |

**文件集重叠但函数不重叠**（都动 `run_real_candidate_training.py`），
合并顺序建议先 `deferred-driver-fixes` 再本分支，合并后**必须重跑全门禁**。
**窗口仍是 M2 与 M3 都跑完之后**（链式看守在 M2 验收后立即发射 M3，中间没有安全窗口）。

### 18.2 一条要写进 M2 / M3 判定文档的事实

合并**只影响此后新写的 bundle**：**M0R / M1 / M2 / M3 的 card 都是 4 字段**，
它们的服务线程数只能来自各自工作区的检查点 manifest（都是 **1**），
或按 §17.2 的临时措施在部署侧钉住。别让下一个人以为工件里应该有这个字段。

### 18.3 状态（23:00 实测）

M2 驱动 pid 3588838 存活 **1d04h**、**F1 18/24**；剩下的 6 格都在**第 3 个 seed** 上
（22:42 仍在落检查点：`m2-f1-c18-20260803-…-epoch-3`、22:37 `c19-…-epoch-7`），
worker 100% CPU、RSS 约 17 GiB，无 `*.ABORT` / `*.ALARM`。
三个看守都在且 PPID=1（320418 / 529186 / 529385）。
主工作树未被本轮任何改动触碰：`run_real_candidate_training.py` 的 sha256 仍是
`33742880f314ca21`、mtime 仍是 `2026-09-12 00:50:33.890987021`。
另外核过一件事：**报告与晋级证据里不含任何 git 提交号**
（只有 `model_code_version` / `data_version` / `schema_version` 这类逻辑版本），
所以主线前进不会让验收 A 的"逐字节相同"失效。

### 8.2 研究分支：已渲染的真实对比 + 一处对自己 v0.1 的更正

`codex/research-conditional-factor-review`（提交 `25b11ce`，已推送）新增
`docs/research/evidence/2026-09-14-model-side-walk-forward/comparison_interim.md`——
上面那个渲染器对**六个已完成运行 + 基线**的真实输出
（可比性组 6/6、领先候选 M0R G4 +0.009149、M0R 对 M1 全网格 +0.001322、
M0R 的最差 seed 是 F1 的 20260802 @ 0.067647 验证 RankIC）。
M2/M3 落地后同一条命令补两个 `--report` 就是 Task 5.1 的那份文档。

**同时更正了 v0.1 里的一处比较**：原文说"候选相对基线的增量在 2023 年是负的"，
拿的是候选的逐年值（**945 日集**）对基线的逐年值（**969 日集**）——
**日集不同，严格来说不能逐年相减**，而且签名工件里**没有**"基线在候选日集上的逐年值"。
现在写成能主张的部分：日集差异在池化口径下值 **−0.0018**
（`0.08705554536301544` → `0.0852268662737014`），而 2023 的表面差是 **−0.0051**，
约为其 **2.8 倍**，所以**符号大概率稳健、精确幅度不可得**；
要闭合它得从已签名的预测工件把基线在 945 日集上重算一遍（只读、可行、属新增分析，
**本轮未做，留给所有者决定**）。

### 8.3 M2 状态（16:40 实测，未受本轮任何操作影响）

驱动存活 21h47m，F1 仍 **13/24**（剩下的正是 TCN-128 那一档最慢的格子，
检查点持续落盘，最近一次 16:38），worker RSS 最大 16.70 GiB，无 `*.ABORT` / `*.ALARM`。
**ETA 口径不变：09-23 至 09-25。**

---

## 9. 【2026-09-14 17:00 增补】链式看守补上了"发射后不验证"的缺口，看守已用新代码重启（**pid 变了**）

### 9.1 缺口：`launch_m3()` 把驱动扔到后台，于是 bash 的退出码只证明"shell 接受了这行命令"

原来的第 7 段是 `launch_m3(); log("done: M3 launched"); return 0`，
而 `launch_m3` 内部是 `setsid nohup env … python … &` 加一句 `echo "M3 launched pid $!"`。
**后台化的命令无论随后是否立刻死掉，bash 都返回 0**，所以这三种情况在过去都会被记成成功：
顺序守卫（`--prior-report` 不合格）当场拒绝、第一次数据集加载崩掉、HMAC key 环境变量没传进去。
三者都会留下 `LAUNCHED` 标记、看守退出码 0、日志写着"done"，
而**M3 其实一秒都没跑**——按 M3 约一天的工期，这个谎要过一天以上才可能被拆穿。
M1→M2 那一环是同一个形状（`m1_m2_chain_watcher.log` 里 18:54:56 两行连在一起），
只是那次发射真的成功了，所以缺口没有暴露。

### 9.2 补的两段

* **第 8 段 `verify_launch(pid)`**：默认 900 s 的沉降窗口内每 30 s 轮询，要求
  ①该 pid 的 **cmdline 同时含** `run_real_candidate_training.py` 与 `--family M3`
  （pid 九天里会被复用，所以只认 cmdline，不认"有这个 pid"），
  ②`m3_v1/` 工作区已被创建（驱动很早就建它，是"真的在干活"的正向信号）。
  两者缺一即 `abort`，并把 `m3_v1.log` 的尾部 900 字符写进 ABORT 文件。
  **一个已跑完并留下 report 的驱动算成功**，不算发射失败。
* **第 9 段 `watch_until_terminal(pid)`**：每 300 s 轮询，直到
  `m3_v1/report.json` 存在且 mtime 连续两次不变、且驱动已退出（→ 成功，并把该报告的
  `family / mode / all_passed / delta_rank_ic / 未通过的门` 摘要写进日志），
  或驱动在没有报告的情况下消失（→ `abort`，附日志尾部、提示查 dmesg 是否 OOM、
  并明确写"不要盲目重发：已完成折的搜索记录逐格落盘，`--resume` 会采纳而不是重训"）。
  **这一段守的是另一件事**：M3 要跑约一天，第 20 小时死掉不该等到第 200 小时才被发现。

`driver_alive()` 改为委托新的 `process_is(pid, *needles)`（同一份实现、两处调用，
不是复制一遍），`launch_m3()` 改为返回解析出的 pid，解析不出来就 `abort`
（并写明"先读日志、先 pgrep，**不要再发一个 M3**"）。
**第 1–6 段一行未改**：与备份逐行 diff，被删除的行只有 `driver_alive` 的四行旧实现、
`launch_m3` 的旧签名、以及 `main()` 里那两行旧的收尾（备份在
`/tmp/m2_m3_chain_watcher.py.bak-20260914`，改前 sha256 前缀 `c8e131543c98145f029e`）。

### 9.3 两个方向都验过，而且验证过程抓到了新代码自己的一个 bug

测试脚本留在 `/data/zhangyudong/qrv2_research_artifacts/test_m2_m3_chain_watcher_launch_checks.py`
（与被测脚本同目录，可重跑；**26/26 通过**）。它把看守模块的 `M3WS / M3REPORT / M3LOG`
重指到临时目录，用一个 cmdline 携带同样两根针的假驱动来演，
**不碰真实的 `m3_v1`、不碰在跑的 M2、不碰数据库**。
负向对照包括：活着但从不建工作区、启动即退出（并验证 ABORT 里带日志尾部）、
**pid 活着但 cmdline 不是 M3 驱动**（pid 复用守卫）、死了且没有报告。
正向对照包括：`driver_alive()` 对**真实在跑的 M2 驱动**返回 True，
以及"报告已存在且驱动已退出"被判成功。

**测试第一轮就抓到两个真问题，都是我自己刚写的代码**：

1. **fork/exec 竞态**：`verify_launch` 原本第一次轮询就下结论，
   而子进程在 fork 之后、exec 之前显示的仍是**父进程的 cmdline**，
   于是"活着的驱动"被判成"没启动"。已改成与第 1 段同一套办法：
   先 `sleep(min(poll, 5))`，并**要求连续两次轮询失败**才认定驱动已死。
   这不是纯测试问题——真实发射走 `setsid → nohup → env → python` 这条 exec 链，
   同样有一个窗口；九天之后的一次误判 ABORT 会让 M3 白白无人看守。
2. **"报告出现即成功"是错的**：驱动写完报告还要做收尾，
   所以 `watch_until_terminal` 要求 **mtime 连续两次不变且驱动已退出**；
   测试用一个"写完报告后继续赖着"的假驱动 + 线程 join 超时证明它**不会提前返回**，
   再杀掉假驱动证明它随后**会**返回成功。

### 9.4 看守已用新代码重启，**pid 从 213719 变成 320418**

重启前确认它仍停在 WAIT（自 14:54:35 起日志无新行），旧日志已留档为
`m2_m3_chain_watcher.log.pre-restart-20260914`。重启后：
新 pid **320418**、**PPID = 1**、自有会话、无 TTY（`setsid` 生效，不随任何终端消失），
日志两行 `watcher start (M2 -> M3)` / `WAIT: …`（16:55:15），无 `*.ABORT`。
逐折预检看守未改动、未重启，仍是 **pid 215628**。
**接手时注意**：本文 §4.1 里写的 pid 188088 与 §7.1 里的 213719 都已作废，以 320418 为准；
判断看守是否在跑请用 `pgrep -af m2_m3_chain_watcher.py`，不要信文档里的数字。
M2 本身未受任何影响：驱动 pid 3588838 存活 22h00m、F1 仍 13/24、无告警。

## 19. 【2026-09-15 01:20 增补】那 24 天的根因已闭合（**不在因子生产端**），顺着它测出一道结构性门缺口；一个分母缺陷已修在分支上等合并窗口

本轮全部**只读**排查 + 一个**未合并**的修复分支。M2 未被触碰：驱动 pid 3588838 存活，
三个看守都在，无 `*.ABORT` / `*.ALARM`。

### 19.1 根因：重放所需的 PIT 行业标签在 2023-09-25 之前没有科创板成员（推翻 §13 的移交方向）

§13 把 24 个不可评 OOS 日追到 `yeslab_357` / `yeslab_471` 对科创板成对缺失，
但把成因判给"因子会话"。**这个移交方向是错的**，四段链路在同一天上对齐后矛盾点自己浮出来了：

| 2023-07-05 | `yeslab_357` | `yeslab_471` | 对照 `yeslab_381`（同表、同一次物化） |
|---|---|---|---|
| 上游 YesLab 存量 | 科创板 588 只、**540 只有值**、无 inf | 611 只、527 只有值 | 606 只、539 只有值 |
| `factor_daily`（2026-09-03 物化） | 3,274 行、**科创板 0** | 3,258 行、**科创板 0** | 3,625 行、**科创板 318 全有值** |
| published pin vs manifest | 一致、单一 data_version | 一致 | 一致 |
| 已发布面板 | 报缺 354 行（318 科创板） | 报缺 370 行 | 报缺 3 行（0 科创板） |

逐代码对齐：面板报缺的 354 个代码里 **353 个上游当天有非空值，含全部 318 个科创板**；
2022-07-29 是 **283/283，含全部 181 个**。**上游不缺。**

根因在 `scripts/materialize_frozen_factor_daily.py`：冻结链**不抄 YesLab 存量值，而是从 Tushare 行情重放公式**，
且只给 `PIT_FACTORS = {"yeslab_357", "yeslab_363", "yeslab_471"}` 传 PIT 行业提供方
（`VersionedPITIndustryProvider`，`PIT_CUTOVER = 2021-12-10`，之后走 SW2021）。
而那份 SW2021 成分缓存（31 个文件、7,733 行）里，**科创板 688/689 最早的 `in_date` 是 2023-09-25**
（559 只同日入册），非科创板可追到 1990-12-10。没行业标签 → 重放算不出值 → `factor_daily` 没行。
这也解释了 §13 里那两个怪事：**不同表、不同所有主的两个因子逐月缺口行数完全相同**
（0 / 0 / 1,286 / 5,522），而**同表对照因子全程满覆盖**（6,596 / 6,458 / 6,230 / 5,539）——对照因子不取行业标签。

两处算术闭合：① 2023-09-25 是周一，9 月 25–28 共 **4 个交易日**，**1,286 ÷ 4 = 321.5 行/日**，
10 月 **5,522 ÷ 17 = 324.8 行/日**；② `factor_rank_ic_daily` 是 **2026-08-02**（早于 9-03 物化）算的，
那天 `yeslab_357` 的 `sample_count` = **3,626**（对照 3,622、当日有效标签行 3,627），
**8 月时截面还是完整的**，缺口与 9-03 那次重放同时出现——这排除了"上游后来才回补"。
`689009.SH` 四年全缺也由此解释：它在行业缓存里**完全不存在**。

另排除三条（都是实测）：版本钉住过期（pin == manifest、单一版本）；
修复路径（`factor_repair_audit` / `factor_repair_quarantine` **全表 0 行**）；
隔离路径（`factor_quarantine` 536,627 行但**只覆盖 2025-01-02..27**，对 2022/2023 沉默，**不能当证据**——
排查中一度把它当证据用，已撤回）。

**可复现性警告**：数据商重述过行业历史（今天查 `688002.SH`，带 `src=SW2021` 返回空、
不带过滤返回 `in_date=2026-08-21` / `801080.SI`，与缓存的 `801740.SI` / `2023-09-25` 不符），
所以 32 份缓存原件已随证据包逐字节复制。

**移交对象因此从因子所有者改为平台/重放口径的所有者**，要决定的只有一件事：
要不要为这两个因子补 2023-09-25 之前的科创板行业历史。补则 `factor_daily` 变 →
所有已签名报告与门值都变 → 新谱系。本轮**未做任何此类改动**，按"维持现状 + 论文披露"记录
（已测得该排除对候选偏保守）。

**更正落地**：`docs/reports/2026-09-14-dense-factor-coverage-gap-and-the-24-days.md` 已加更正横幅、
新增 **§4.1**、§5 移交改向（主树提交 `6a3cc42`，已推送）。
证据包在分支 `codex/research-conditional-factor-review` 的
`docs/research/evidence/2026-09-14-model-side-walk-forward/star_industry_coverage/`
（7 个只读探针 + 11 份测量 + 32 份缓存原件 + 全部 sha256；提交 `2352e0d`、`fd01356`）。

### 19.2 顺着根因测出的结构性缺口：**没有任何一道在用的覆盖率门会拦下那 24 天**

把冻结 **v2 的 15 个因子**（M0R/M1/M2/M3 实际消费的集合；此前两台仪器都只为 v1 的 17 个、
且只在训练窗口跑过）放在**实际消费的 `factor_daily`** 上、在 **OOS 窗口 2022–2025** 筛查，
并把**两台在用仪器的阈值都应用到同一组计数**上：

**9 个稠密因子全部 PASS**，含 `yeslab_357`（median 0.9992 / p05 **0.9131** / min 0.9024）与
`yeslab_471`（median 0.9936 / p05 **0.9091** / min **0.8980**）。`expected_day_count = 969`，
**与 G6 = 945/969 的分母逐日相同**。

论证只要三个数字：471 的 **p05 = 0.9091**；面板实测 2023 年九因子交集覆盖率中位数
**1 − 0.0907 = 0.9093**；地板是 **0.90**（`gates.py:35`）。前两者差 **0.0002**——
因为 357 的缺失集嵌套在 471 里，**交集覆盖率就等于 471 自己的覆盖率**。
**一个因子的第 5 百分位坐在 0.909，正好产出 24 个跌破 0.90 的日子；而 p05 门只要求 0.30（活体）/ 0.70（快照）。**
median 门更松（两者都 ≥0.993），因为**中位数天生看不见"某类证券整段没有"**。
这不是阈值调错：**被度量的量（单因子分位数）与被约束的量（九因子交集逐日值）不是同一个**。

能拦下它的门（未实现，属所有者决定）：对**实际进入基线工件的那个交集**，
用评价器同样的分母与同一条 0.90 地板逐日判定，并把"跌破天数"作为**准入**产物。

同批还测出：**6 个稀疏因子（372/375/384/389/390/406）在 OOS 窗口上两台仪器都 FAIL**
（median 0.46–0.49、p05 0.11–0.30、最差单日 0.019）。它们不在稠密 9 里，基线工件不受影响，
但**候选消费全部 15 个**（冻结链不 `dropna`），即候选特征矩阵有 6 列在 OOS 窗口中位覆盖率不到一半。
成因早已查明（`frozen_sparse_factor_root_cause.txt`，2026-09-05：重放路径下 GP 公式的固有逐日行为），
**本轮补的是它在 OOS 窗口上的首次持久化测量**。

证据：同分支 `coverage_gates_vs_the_intersection_floor/`（提交 `471e02c`）。

### 19.3 顺带查出的分母缺陷：**已修在分支上，等合并窗口**

`universe_daily` 保留每一次发布的版本，而**同一天可以有多个版本都带 `is_member=1`**：
OOS 窗口 969 天里 **20 天**如此，最坏 2025-01-07 是 **19,603 行 对 钉住版本 3,915 = 5.007×**
（2025-01-09 四个版本为 3,923+3,923+7,846+3,923，其中 `f0a6c677…` **自身每键还重复两次**）。
**冻结链安全**：装配对三张表都走 `filter_current`，钉住版本给 3,923，与已发布面板当天 **3,923 行逐行相同**。
**治理脚本不安全**：`screen_factor_daily_coverage.py` 文档里写明的 C6.3 分母就是不带版本过滤的
`count(is_member=1)`，于是它报出的 `minimum_coverage_ratio` 对 15 个因子一律 ≈ **0.197–0.200**（= 1/5），
换成钉住分母后是 **0.90–0.998**。**判定没有翻转**（两台仪器都不读 min），
但此前那次 v1 训练窗口筛查用的是 `2020-01-01..2024-12-31`，**含 2024-12-25..31 这五天**，min 同样被污染；
那份结果从未持久化，所以没有错数字进过库。

**修复分支**：`codex/coverage-screen-denominator`，worktree `/data/zhangyudong/qrv2_worktrees/coverage-screen-denominator`，
从 `6a3cc42` 起。**未合并**。改动只在 `scripts/screen_factor_daily_coverage.py` 与其测试：

* 分母改为**按 published_version 解析 + 按 `(signal_date, ts_code)` 去重**；分子同样按 pin 解析
  （`factor_daily` 也会累积版本，不解析就可能读出 >1.0 的覆盖率——持久化快照里 357 的 median 1.0016 就是这种形状）；
* 重复键检查**限制在已发布版本内**：原内层分组不含 `data_version`，
  于是"每版本各一行"会被读成冲突，**一次正常重发布就能让因子背上 `DUPLICATE_BUSINESS_KEYS` 而出局**；
* **fail-closed**：某天有行、但所在分区没有任何已发布版本 → 抛 `UnpublishedPartitionError`，
  而不是把那天悄悄从窗口里删掉（`screen_factor` 不会把 universe 未覆盖的天算作因子的失分，
  静默跳过等于偷偷缩短窗口）；
* 诊断里同时报**旧口径**（`legacy_denominator`）：最坏放大 **5.0072**、20 天受影响、
  逐日 legacy_rows 对 published_members，这样读旧报告的人能对得上。

**门禁（在 worktree 内实跑）**：`ruff` 全清（脚本、测试与 `src`）；
`mypy scripts/screen_factor_daily_coverage.py` 回到基线
（只剩 `walk_forward_common.py` 那条既有的 dolphindb stub 报错，与主树逐字相同）；
`pytest tests -q` **2176 passed / 46 skipped，161.34s**
（主树 `6a3cc42` 收集 2214，本分支收集 2222 = **+8 个新测试**，0 失败）。
**变异检验 5 处全部被杀**：分母用回 raw（2 个测试死）、内层去掉 `ts_code`（去重测试死）、
重复键查询去掉版本过滤（版本范围测试死）、fail-closed 改成静默跳过（fail-closed 测试死）、
删掉 legacy 诊断（legacy 测试死）。
**真实库端到端复验**：修好的脚本跑 `2022-01-01..2025-12-31` 得 **PASS 11/17**、
失败者正是那 6 个稀疏因子，稠密因子 min = 0.9737 / 0.9024 / 0.8980 / 0.9980，
与独立探针逐位相同；`factor_daily` 侧 `keys_with_more_than_one_version = 0`
（即分子在该窗口从未被污染，被污染的只有分母）。

**合并窗口**：与 `codex/deferred-driver-fixes`（`1dbc36c`）、`codex/serving-thread-contract`（`b70ad8a`）
同一批，**M2 与 M3 都跑完之后**；届时必须重跑全门禁（主线会前进）。
**不要在 M2/M3 在飞时合并**——虽然本脚本不在驱动的导入路径上（看守也不调用它，
只有 `preflight_frozen_baseline.py` 以子进程方式跑它、且只取退出码），但同一批一起合更省事、也更安全。
**退出码不变的证据**：门只看 median / p05 / 月度 / day_ratio / dups，不看 min；
两种分母下 PASS/FAIL 名单逐因子相同（实测）。

### 19.4 M2 状态（本轮实测，非估计）

F1 **19/24**（`c21` 于 09-15 00:51 落地），剩 `c18/c19/c20/c22/c23` 在第三个 seed 上；
worker 全部满载（6 个 100% CPU）。逐格完成时间线显示最慢的一格从 09-14 19:40 到 09-15 00:51
（310 分钟）才落地，所以 **F1 收口时间以"最慢格"为准，不要用平均**。
整体 ETA 维持 **09-23 至 09-25**（§7 的成本模型），未因本轮任何操作改变。

## 20. 【2026-09-15 02:00 增补】§19.2 缺的那台仪器已经做出来并验证：**只读发布表就能预测那 24 天，24/24 逐日命中**

§19.2 说明在用的门**结构上看不见**交集地板（单因子 median/p05 对九因子交集的逐日 0.90 地板），
但只停在"该有一道门"。本轮把它做成了仪器，并且**用真实运行验证过**，不是自证。

**新分支**：`codex/dense-intersection-screen`，worktree `/data/zhangyudong/qrv2_worktrees/dense-intersection-screen`，
从 `25c3315` 起，提交 **`c162073`**，**未合并**（与 `deferred-driver-fixes` / `serving-thread-contract` /
`coverage-screen-denominator` 同一批，等 M2 与 M3 都跑完）。新增两个文件：
`research_v2/scripts/screen_dense_intersection_coverage.py` 与其 13 个单元测试。**没有改动任何既有文件。**

**它算什么**：分子 = 当日在 `factor_daily` **已发布版本**下 9 个稠密因子全有行、
**且当日在 `label_daily` 已发布版本下有 `label_valid=1`** 的证券数；
分母 = 当日 `label_valid=1` 行数（**不是代理量**：2023-07-05 = 3,627，与面板当天有效标签行数逐行相同）；
地板 = **运行时从 `gates.py` 源码解析**（`0.90 <= coverage <= 1.0` 那一行），脚本里不重抄，
并有漂移测试盯着——**一台预测某道门的仪器必须去读那道门**。

**验证**：2022–2025 跑一次约 **60 秒**，预测 **24 天**跌破地板；
`m0r_v2` 那次真实运行实测也正是那 24 天，`--compare` 给出 **`identical=True`、零误报、零漏报**。
**逐年覆盖率与 `coverage_attribution` 到浮点末位相同**：2023 年 min **0.8902674386545354**（两边 16 位全同）、
median 0.9093；2022 年 0.8951/0.9254；2024 年 0.9711/0.9924；2025 年 0.9809/0.9859。
969 天也与 G6 = 945/969 的分母相同。**同一个量被两条独立路径算了出来**（签名工件的逐日帧 vs 发布表）。

**一处差点让仪器失效的设计错误（记下来，因为它决定了仪器可不可信）**：
第一版分子数的是"9 个因子全有行的证券"，不管当天有没有有效标签 → **预测 21 天、漏 3 天**
（`2023-04-28`、`2023-05-19`、`2023-06-21`），**全部漏在不安全方向**。
2023-04-28：未限制分子 3,237 → 覆盖率 **0.9070**（预测"存活"）；限制后 3,212 → **0.89997**（预测"丢掉"）；真实结果丢掉。
三天都压在 0.90 上方不到 0.007，所以几十个"不在面板里的证券"就足以翻判定。
修法是把分子与当日有效标签集合做等值连接——而 **DolphinDB 不给子查询或 `loadTable` 起别名**
（`select … from (…) l inner join (…) f on l.x=f.x` 与别名版都报 `Cannot recognize the token`），
只能用**脚本模式**：两侧绑定成变量，再用等值连接函数 ``ej(lab, complete, `signal_date`ts_code)``，按年分块（单月 1.3 秒）。
两版运行的终端记录都留在证据包里作对照。

**另外两条防御**：分子先按 `(日,证券,因子)` 去重再数因子个数（否则一个重复业务键能让 8 读成 9）；
**fail-closed**：窗口内某逻辑分区没有已发布 `data_version` → 抛 `UnpublishedPartitionError`，
而不是把那天悄悄跳过（一台"点名哪些天会缺"的仪器静默省略一天等于自我否定）。

**门禁（worktree 内实跑）**：`ruff` 全清（脚本与测试）；`mypy` 回到基线（只剩既有的 dolphindb stub）；
`pytest tests -q` **2181 passed / 46 skipped（159.91s）**，基线提交 `25c3315` 收集 2214、本分支 2227
= **+13 测试、0 失败**。**4 处变异逐一被对应测试杀死**：地板不再从源码解析、去掉有效标签连接（3 个测试死）、
去掉内层去重、fail-closed 改静默跳过。

**边界（务必照此使用）**：**这是测量工具，不是策略。** 没有任何东西调用它，不改任何门值、不产出判定；
`--fail-on-prediction` 已实现但**默认关闭**，因为"预测到跌破地板之后做什么"
（重发覆盖率完整的基线 / 挪地板 / 只披露）会改动已签名报告，属所有者决定。
建议用法（**未执行**）：① **每次发射一个家族前跑一次**（60 秒），把预测日集写进发射记录——
这次的 24 天是四折跑完、到评价阶段才被看见的；② **换因子集时必跑**（15 因子名单一变，稠密子集与交集都变，
`--factor-source all_v2` 可直接量全 15 个的交集，那会低得多，因为 6 个稀疏因子）。
**不要**擅自把它接进 `preflight_frozen_baseline.py`——要接就是打开那个开关，属所有者决定。

证据：分支 `codex/research-conditional-factor-review` 的
`docs/research/evidence/2026-09-14-model-side-walk-forward/dense_intersection_screen/`
（提交 `3f75cd3`；含仪器与测试副本、24/24 与 21/24 两版运行记录、
`provenance.json` 里记了分支提交号与 `gates.py:35` 原文），叙述见 `problems_found.md` **§O**。

**M2 未受影响**：本轮全部只读查询 + 独立 worktree；驱动 pid 3588838 存活，F1 19/24，三个看守都在，无告警。

## 21. 【2026-09-15 02:30 增补】拿 §20 那台仪器又量了四件事，其中一件**直接关系到"15 个因子最终用哪些"这个待定决定**

全部只读，无代码改动（§20 的分支仍是 `c162073`，未合并）。细节与全部表格在
`codex/research-conditional-factor-review` 的 `docs/research/evidence/…/dense_intersection_screen/finding.md` §9–§13
（提交 `7229fed`、`b2f2657`），叙述版在 `problems_found.md` **§O-续**。

**(1) 交叉验证到三个运行**：仪器的 24 天预测与 `m0r_v2`、`m1_full_v1`、`m1_parallel_v1`
三份实测日集**逐日全同**（各自 `identical=True`、零误报零漏报）。
**这不是三次独立验证**——三者共用同一套冻结基线工件与同一套标签。
它证明的是**"这个日集是基线工件 + 标签的性质、与具体候选无关"**，
而这正是仪器能在候选训练**之前**算出它的原因。
真正的独立性在两条计算路径（签名工件的逐日帧 vs 发布表），两者逐年覆盖率到浮点末位相同（§20）。

**(2) 训练/验证窗口 2014–2021：1,045 / 1,948 天在地板之下。**
2014–2016 三年**全部**跌破，2015 最差（median **0.7631**、min **0.4453**，
最差十天全在 2015-07～10），2017 年 241/243，2018 年 71/243，2019–2021 基本为 0。
**不使任何已发布结果失效**（分数健康地板只在 TEST 日判定），
但它说明**冻结基线工件在 2014–2017 的行集远比 OOS 窗口薄**
（中位覆盖率 0.7631–0.8317 对 OOS 的 0.9845）——
"基线在各折之间可比"这句话在这些年份要打折；候选一侧不受影响（不 `dropna`、每行都打分）。

**归因已测到具体因子，不是推测**：逐日逐因子算 2015 下半年（`probe_dense_per_day_attribution.py`，
复用钉住分母的采集器），`yeslab_206` 在 **146/146 天都是当日最弱的稠密因子**
（median **0.7162**、min 0.4443），其次 `yeslab_443_reg30_max20_v1`（median 0.7665）；
而 2022–2023 的元凶 `yeslab_357` 在这里是 **0.9973**、`yeslab_471` 是 0.9252。
**所以稠密 9 的薄弱是两个不同时代、两个不同原因**：
2014–2017 是**两个修复变体**（206 中位数修复、443 reg30_max20 窗口变体，
都走 `materialize_frozen_factor_daily.py` 的"已批准修复公式"重放），
2022-01..2023-09 是**缺 PIT 行业标签**（§19.1）。
科创板 2019-07 才开板，**两段原因互不重叠**。
（口径提示：本仪器在 2014–2021 量到 1,948 天=有有效标签的天，
逐因子筛查量到 1,950 天=有 universe 成员的天，差 2 天，两个分母不同不可直接相减。）

**(3) 全 15 因子口径：968 / 969 天跌破地板，中位覆盖率仅 0.2194**（逐年 0.2035–0.2361）。
**这条直接关系待定决定**：若把基线工件定义在"15 个因子全齐备"的行上，
OOS 窗口里几乎每一天都不可评，**G6 会变成 ≈0.001**——
因为 6 个稀疏因子中位覆盖率只有 0.46–0.49，且它们的缺失**相关**。
所以稠密/稀疏的分工**不是风格选择，而是评价可行性的前提**：

* **基线（`best_simple`）只能定义在稠密子集上**（它要求行齐备，否则没有可比键集）；
* **候选可以吃全部 15 个**（冻结链不 `dropna`，NaN 交给 `FoldPreprocessor`）。

一句话给所有者：**往冻结集里加一个稀疏因子不会让候选失效，但它若进了稠密子集就会让评价失效。**
**换名单之前该跑的就是这台仪器（60 秒），而不是九天后在评价阶段才发现。**

**(4) 候选真正吃进去的矩阵长什么样**（`profile_candidate_feature_nan.py`，只读已发布 parquet）：
四个测试年里**缺失单元占 20.2%–21.9%**、**至少缺 1 列的行占 73.6%–77.0%**；
缺失**按组到达**而不是均匀散布（2022 年：缺 3 列 **40.6%**、缺 6 列 **26.2%**、缺 0 列 **23.0%**，
四年同一形状），与 6 个稀疏因子互相相关的既有记录一致。
**对账（这一步让上面所有数都可信）**：2022 年"九列全有限"的行数 = **746,521**，
而 §5 阻塞点 3 当时记的是"基线覆盖 747,461 行、其中候选没有的 940 行可打分行数为 0"——
**747,461 − 940 = 746,521，逐行吻合**；2023 年 820,309 对 820,829，差 520，同一口径。
即"面板上九列全有限的行集" = "基线工件里候选能打分的键集"，
这是 §11 那条等式（基线键集 = 九稠密全有限的行集）在**另一个数据源**上的独立复核。

**M2 仍未受影响**：驱动 pid 3588838 存活，F1 19/24，三个看守都在，无 `*.ABORT` / `*.ALARM`。

## 22. 【2026-09-15 02:35 增补】合并窗口已经**先演过一遍**：四个分支按此顺序干净合并、全门禁通过、测试数逐条对账

四个待合并分支里有**两个都改了 M2 驱动本体**
（`codex/deferred-driver-fixes` 与 `codex/serving-thread-contract` 都动
`scripts/run_real_candidate_training.py` 与 `tests/unit/scripts/test_run_real_candidate_training.py`），
所以"到时候再合"是有真实撞车风险的。本轮在**一次性 scratch worktree** 里把整个合并序列真跑了一遍
（`git worktree add --detach`，合完即 `git worktree remove`，**未推送任何合并提交**，主树与 M2 全程未受影响）。

**顺序与结果**（第一次演练基线 = `4fff381`；§23 让 `dense-intersection-screen` 前进到 `4b6aa7d`
之后**又重演了一次**，基线 = `3146871`，两次都干净。下表顺序即建议的合并顺序）：

| 步 | 合并分支 | 结果 | 模拟提交（仅本地，已随 worktree 丢弃） |
|---|---|---|---|
| 1 | `codex/deferred-driver-fixes`（`1dbc36c`） | **干净** | `7fe099b` |
| 2 | `codex/serving-thread-contract`（`b70ad8a`） | **干净** | `0d1f989` |
| 3 | `codex/coverage-screen-denominator`（`3746208`） | **干净** | `21806a3` |
| 4 | `codex/dense-intersection-screen`（`c162073`） | **干净** | `77e189c` |

**文本无冲突不等于语义正确**，所以还查了两件事：

1. **四处改动在合并树里共存**：`write_fold_cache_atomic` 在 `walk_forward_common.py` 与
   `walk_forward_factor_daily.py` 各 2 处、`_requested_folds` 在驱动里 2 处、
   `artifact_loader.py` 的线程契约在、两个筛查脚本（`screen_factor_daily_coverage.py` 修好的版本
   与新增的 `screen_dense_intersection_coverage.py`）都在。
2. **合并树上跑门禁**（第一次演练只跑了 `pytest`，本文当时写成"全量门禁"是**说过头了**；
   已在第二次演练里把 ruff 与 mypy 补齐，下面三项都是同一棵合并树上的实测）：

   * `pytest tests -q` → 第一次演练 **2229 passed / 46 skipped（165.30s）**；
     第二次演练（当前状态）**2232 passed / 46 skipped（163.43s）**，**两次都 0 失败**。
   * `mypy src`（`MYPYPATH=src`）→ **Success: no issues found in 170 source files**，
     两次演练都与各自的基线树逐字相同。
   * `ruff check src tests` → **All checks passed!**；
     对四个分支实际碰到的 **13 个文件**单独跑 → **All checks passed!**。
     **一处必须说清的口径**：`ruff check src tests scripts`（含全部 `scripts/`）在合并树上报
     **589 处**（两次演练都是 589），而**基线树同一命令也是 589 处**——即**零新增**，那 589 处是
     `scripts/` 里与这四个分支无关的既有量。所以"ruff 全清"这句话的正确范围是
     **`src`、`tests` 与被改动的脚本**，不是整个 `scripts/` 目录；
     本文 §5.1 第 4 条与 §18 里那句"ruff 全清（src、tests 与三个被改脚本）"用的就是这个范围。

   **测试数逐条对账**（按当前状态，即第二次演练）：基线 3146871 收集 **2214**，
   `deferred-driver-fixes` +21、`serving-thread-contract` +19、`coverage-screen-denominator` +8、
   `dense-intersection-screen` **+16**（§20 的 13 个 + §23 的 3 个），
   **2214 + 64 = 2278 = 2232 + 46**。
   （第一次演练时该分支还是 +13，故为 2275 = 2229 + 46；两次都对得上各自的分支状态。）
   对得上说明**合并过程没有静默丢掉任何一个测试**——这是"干净合并"最需要复核的一点，
   因为两个分支改同一个测试文件时，最典型的失败模式就是其中一方的测试在合并里消失。

   所有演练都在**一次性 detached worktree** 里做，合完即 `git worktree remove`，
   **没有推送任何合并提交**；合并树顶点分别是 `6235508`（第一次）与 `332ff6f`（第二次），
   都只在本地存在过、已随 worktree 丢弃。**当前 publish HEAD `3146871` 不含任何合并提交。**

**给合并窗口的结论**：按上表顺序合即可；合并后**仍须重跑一次全门禁**（主线到那时会前进），
但已经知道**不会有冲突、也不会有测试丢失**。合并时机不变：**M2 与 M3 都跑完之后**，
因为这四个分支里有两个改的是 M2/M3 正在用的驱动本体。

### 22.1 另外：M2 的评价结果已经被**事前登记**，落地时先对这张表

`dense_intersection_screen` 那台仪器（§20）到目前为止做的都是**事后**对照。
本轮把预测写在了 M2 评价阶段**之前**（登记时刻 **2026-09-14T18:19:37Z = 09-15 02:19 CST**，
当时 M2 仍在 F1 搜索阶段 19/24、`report.json` 不存在）：

| 折 | TEST 年 | 天数 | 跌破地板 | **预测可评天** | 覆盖率 min |
|---|---|---|---|---|---|
| F1 | 2022 | 242 | 1 | **241** | 0.895082 |
| F2 | 2023 | 242 | 23 | **219** | 0.890267 |
| F3 | 2024 | 242 | 0 | **242** | 0.971115 |
| F4 | 2025 | 243 | 0 | **243** | 0.980855 |
| 合计 | — | **969** | **24** | **945** | — |

即 **G6 = 945/969 = 0.9752321981424149**、候选逐日覆盖率恒为 **1.0000**、
`days_dropped_by_the_baseline_side = 24/24`。
前提是 M2 发射时用的基线工件与 `m0r_v2` / `m1_full_v1` 是同一套
（`baseline_v2/baseline_oos` + `baseline_v2/report.json`，见驱动命令行）。

**M2 报告落地时第一件事就是对这张表**：任何一项不符
（945/969、某折可评天数、某年覆盖率 min/median、候选覆盖率不是 1.0000）
都意味着**输入在两次运行之间变了**（基线工件 / 标签 / 因子集 / 地板），
**在解释清楚之前不要接受 M2 的数字**。反过来，一致也**只**说明评价消费了同一套冻结输入，
不说明候选好坏、也不说明那 24 天该被丢掉。

登记文件（含来源测量的 sha256 与推翻条件，生成器直接读仪器 JSON、**没有一个数字是手抄的**）：
分支 `codex/research-conditional-factor-review` 的
`docs/research/evidence/…/dense_intersection_screen/measurements/m2_preregistration.{json,md}`
（提交 `d791b9c`），叙述见同目录 `finding.md` §14。

## 23. 【2026-09-15 02:55 增补】仪器现在能试算**任意因子名单**，于是"换 15 个因子"这个待定决定有了安全的前置步骤

§21(3) 的结论是"换名单之前该先跑一次仪器"。但仪器原本只认驱动里的两个冻结常量
（`FROZEN_FACTOR_SET_V2_DENSE` / `FROZEN_FACTOR_SET_V2`），要试算一个**候选名单**就得改
`research_v2/scripts/walk_forward_factor_daily.py` 里的常量——**而那正是 M2 在飞时绝对不能碰的文件**
（驱动每折从磁盘重新导入它，见 §6 红线）。所以"先算一遍"原先做不到：要么改危险文件，要么等九天。

**已加 `--factors id1,id2,…`**（分支 `codex/dense-intersection-screen`，与 §20 同一个分支）：
显式名单优先于 `--factor-source`，两者同时给直接报错。

**等价性实测**：用 `--factors` 显式列出与 `dense_v2` 完全相同的九个 id 跑 2022–2025，
结果 **predicted 24 vs measured 24、`identical=True`**，与走冻结常量那一路逐日相同——
显式路径不是另一套逻辑，只换了 id 来源。

**四种输入在入口被拒绝，其中第一条是这组校验的真正理由**：

| 输入 | 拒绝理由 | 不拒绝会怎样 |
|---|---|---|
| 重复 id | `contains duplicates …` | **`having count(*) = k` 永不可满足** → 交集查询返回 0 天 → 报告显示"没有任何一天跌破地板"，**与"因子集覆盖完美"同一个形状**。这是本仪器唯一不能犯的错 |
| 空白名单 | `needs at least one factor id` | 同上 |
| 含 `;`／引号等字符 | `ids with unsupported characters` | id 要进带引号的 IN-list，能闭合引号就能改语义 |
| 同时给 `--factor-source` | `cannot both be given` | 两个来源静默取其一，用户不知道到底量了哪个集合 |

**失败模式是"看起来全清"而不是报错**，所以这几条校验不是防御性摆设。

**门禁**：`ruff` 全清；`mypy` 回到基线；新增 3 个测试（共 **16** 个），
**3 处新变异逐一被杀**（去掉重复校验、去掉字符校验、把默认源悄悄改成 `all_v2`），
另有一条测试专门钉住"不加新旗标时默认仍是 `dense_v2`"，防止这次新增改变既有调用的语义。
全量套件见提交信息。

**给所有者的操作顺序（重要）**：15 个因子的名单定了之后，
**先**跑 `--factors <新名单的稠密子集>`（60 秒）确认预测的跌破天数为 0 或可接受，
**再**去动驱动的冻结常量。顺序反了就要重跑整条链。
参考量级：现在的稠密 9 是 **24/969** 天跌破，而全 15 个是 **968/969**（§21(3)）——
**新名单里每一个进入稠密子集的因子都要按这个尺度先过一遍。**
