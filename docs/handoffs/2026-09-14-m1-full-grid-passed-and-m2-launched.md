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
