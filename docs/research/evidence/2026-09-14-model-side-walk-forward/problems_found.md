# 问题记录（模型侧走查，2026-08 至 2026-09-14）

这份文件记的是**过程中发现的问题**，不是结果。结果在
`../../2026-09-14-model-side-walk-forward-results-v0.1.md`。
每条都给出：现象 → 根因 → 处置 → 权威出处（原件已复制到 `vendored/`）。

按对论文的影响排序：A（差一点让所有 torch 候选拿不到 OOS 数）、
D（基线报告的 G3/G5 曾经是 FAIL）、B（工期口径）、其余为工程与治理。

---

## A. 评估阶段的六个阻塞点：两条真实 M1 运行都训练完四折、然后死在评估阶段

**现象**：两次真实 M1 运行（载体与全网格的第一次尝试）都把四折训练跑完了，
然后在**评估阶段**崩溃，写不出 `report.json`，因此没有任何判定。
串行运行**不可恢复**（它早于"搜索记录逐条落盘"这个机制），并行运行可恢复。

**根因（结构性）**：夹具在**复述**写入方的约定，而不是向写入方索取；
而 smoke 路径根本不进评估器。于是"写入方写什么"与"评估方接受什么"之间
没有任何测试层的一致性保证，六个不一致全部只能在真实运行里暴露。

| # | 位置 | 错误信息 | 影响范围 | 修复 |
|---|---|---|---|---|
| 1 | `model_walk_forward._require_verified_model_bundle` | `model bundle training dataset does not match frozen model selection` | 所有 torch 载体（M1 实际中招两次） | `b86cdb0`：card 的 `dataset_id` 有两种合法写法（LightGBM 记 `<dataset>:train`、torch 记裸 `<dataset>`），严格相等改为**成员归属**，是旧接受集的严格超集 |
| 2 | `walk_forward._assemble_oos_frames` | `F1 candidate predictions and labels do not contain identical business keys` | 所有 torch | `6a19ef2`：LightGBM 候选覆盖整个 TEST 截面（F1 804,575 行），torch 只覆盖有可用标签的行（803,572；差的 1,003 行是 `label_valid=False`）。改为**双向覆盖**：不得有 TEST 帧之外的键、且不得漏掉任何**可打分**行 |
| 3 | `walk_forward._require_baseline_covered_by_candidate` | `F1 candidate and baseline predictions: baseline rows are not all covered by the candidate` | 所有 torch | `a950e42`：基线覆盖 747,461 行、候选 803,572 行，基线里候选没有的 940 行中**可打分行数为 0**，指标可用交集两种情况都是 746,521 行。覆盖要求收窄到可打分行 |
| 4 | card `family` | `model bundle family does not match frozen model selection` | **所有 M2** | `9467d63`：M2 的 card family 是 `TEMPORAL_{胜者架构}`，由胜者决定、搜索跑完前不可知，而评价器白名单原本只认载体名 |
| 5 | bundle `config` | `model config has unsupported or missing fields` | **M2（GRU 与 TCN）与 M3** | `e8cfca8`：`torch_bundle_config` 会剥掉架构 dataclass 没声明的字段（M2 剥 `family` 与 `learning_rate`），比对必须只比冻结 config 自己的键 |
| 6 | loader 服务侧 | loader 无 `TEMPORAL_TCN` 条目与工厂分支、builder 无法装配 | **TCN 胜出的 M2** | `9467d63`：即使 TCN 赢了也无法被加载服务 |

**处置与验证**：六处都已修，并且新增了两类"防漂移"测试——
`test_a_bundle_written_by_the_drivers_own_writers_loads_back`
（参数化覆盖 M1 / M3 / M2-GRU / M2-TCN 四个载体，用驱动自己的写入函数产出，不是夹具复述）
与 `test_every_allowlisted_family_has_a_factory_behind_it`（专钉"映射里有、分派里没有"）。

**为什么现在有信心**：blocker 1–3 已在真实 M1 四折上跑通（`m1_full_v1` 出了判定）；
blocker 4–6 打的是 M2/M3，两者都**还没跑到评估阶段**，所以
**逐折预检监视器**照抄评价器的逐折路径，在每折产物一落地就跑一次，
并额外要求 card 的家族同时在白名单里、且 `artifact_loader` 能真的构造出模型。
它已在 M1 的四折上端到端验证过（四折全 PASSED，loader 构造出 `ResidualMLP, 70657 params`）。
**折 1 发现是可修的一天，折 4 之后发现是白跑的八天。**

出处：`vendored/docs/handoffs/2026-09-11-m1-evaluation-crash-lineage-and-footprint.md` §7、§9、§12；
`vendored/docs/handoffs/2026-09-14-m1-full-grid-passed-and-m2-launched.md` §4.1。

---

## B. 成本模型连续三次低估，以及本轮我自己又犯了一次同类错误

| 运行 | 规划值 | 实测 | 低估倍数 | 错在哪 |
|---|---|---|---|---|
| M1 全网格 | 27.8 h | **41.0 h** | 1.47× | 降级因子按旧曲线外推 1.27×，实测 **1.132×**（这一项其实乐观了）；真正的错在别处 |
| M2 四折 | 6.17 天 | **约 9.7–11.1 天**（在跑） | **1.6–1.8×** | 见下三条 |
| M3 四折 | — | 实测 per-fit 2.61 h → **约 1 天** | 已用只读探针在真实折上标定 | 探针先被验证过保真度（对 M1 config0 的 epoch-1 检查点 sha256 逐位相同） |

M2 低估的三个独立原因，每一个都值得单独记：

1. **降级因子不能跨家族搬运。** M1 在 41 路实测 1.132×、worker 载荷 0.381 GiB；
   M2 载荷 **4.16 GiB（11 倍）**、走时序批（`history_length=20`），
   24 个 worker 同时流式读写约 100 GiB 张量，**自变量是访存带宽而不是核数**。
   M2 在 24 路实测降级 **1.437×（中位，24 格全落在 1.332–1.583）**，
   比 M1 在 41 路的还差，尽管并发度只有 M1 的 59%。
2. **一轮 = 3 × 最慢格，不是 3 × 网格均值。** 配置级粒度下 makespan 由最慢的那一格决定。
   M2 最慢格是 TCN/128：21–22 epoch × 约 2400 s ≈ **14.1 h/次拟合**
   （实测 c18 seed 20260801：19:03→09:15 = 14.2 h）。
   单折搜索因此是 **3 × 14.1 ≈ 42.3 h**，而按网格均值算的是 24.4 h（低估 1.74×）。
3. **ms/日 跨折不可比。** 用它把旧折成本折到新折，在 M1 上最小格 −1.3%、最大格 **+9.2%（反方向）**，
   因为两次测量的分母不同（旧基准是 245 训练日 + 60 验证日的探针，新的是整折 967 + 237 日，
   每日行数也不同）。它后来在 M3 上碰巧只差 3.4%，**但那是运气，不是可依赖的规则**。

**本轮（2026-09-14 15:30）我又犯了一次同类错误，记在这里以免重复**：
我用"相邻两个**改进**检查点的时间戳之差"去推每 seed 的墙钟时长，
得到 F1 搜索当晚收口、M2 约 09-20 完成的结论。
错在**改进检查点只覆盖验证分提升的那些 epoch**，而早停 patience 是 10，
所以每次拟合还要在最优 epoch 之后**空跑至多 10 个 epoch**——
TCN/128 的真实单次拟合是 21–22 epoch（14.1 h），我按 11–13 epoch 算了。
正确口径是文档里那条：**取同一 (config, seed) 相邻两个 epoch 检查点之差得到每 epoch 成本，
再乘以"最优 epoch + patience"的总 epoch 数**，而不是用改进检查点的时间跨度。
**M2 的规划口径仍是 09-23 至 09-25。**

**没有免费的加速**：makespan ≈ `ceil(72/W) × 最慢单次拟合 × deg(W)`，
W=24 给 3 × 1.437 = 4.31，W=16 给约 6.5，W=12 给约 7.5 —— **24 路已是最优**，
加核无用。唯一真杠杆是那 4.16 GiB 的批次张量（惰性构造/按需生成/连续内存视图），
它同时降低内存与访存争用，但要改一个已测试的契约、且需重启 M2（损失已跑的小时数），
**属所有者裁决，未动**。注意它**只对 M2 有意义**：M3 是静态族、
走 `build_static_training_batches`、载荷属 M1 量级，所以"赶在 M3 之前改"没有任何收益。

出处：`vendored/docs/handoffs/2026-09-12-m1-full-grid-launch-and-measured-cost-model.md`；
`vendored/docs/handoffs/2026-09-14-m1-full-grid-passed-and-m2-launched.md` §4；
`vendored/…/2026-09-04-real-candidate-model-training-plan.md`（未复制，在主分支）Task 4.2/4.4 的实测段。

---

## C. 基线报告的 G3/G5 曾经是 FAIL：一个 stale 默认值

**现象**：`baseline_v2/report.json` 的 `oos.gate_note` 原文——
"gates re-evaluated with derived fold-contract required_years (F1-F4 -> 2022-2025);
the run-time report used the stale 2020-2025 default, which recorded G3/G5 as FAIL with observed=None"。

**根因**：门控的 `required_years` 有一个 2020–2025 的默认值，
而这套折的测试年其实是 2022–2025；拿默认值去取逐年 RankIC 时，
2020 与 2021 年取不到值（`None`），于是"正年占比"与"最大单年占比"都算不出来 → 记为 FAIL。

**处置**：`gates.py` 现在把 `required_years=None` 解析为**从折契约推导**的
`_ordinary_test_years()`（`research_v2/src/research_v2/evaluation/gates.py` 第 93–114 行），
默认值不再是硬编码年份。基线报告里保留了那条 `gate_note` 作为历史痕迹。

**对论文的含义**：引用 `baseline_v2` 的门控时要带这条注记；
基线的 G4 是 `NOT_APPLICABLE`（它就是被减的那一项），
它的 `all_passed=true` 只表示"作为基线自身是自洽的"，**不表示它是个候选**。

---

## D. 四处"永不触发的守卫"：检查存在，但对任何真实工件从未生效

这一类的共同形状是：**检查以一个字段存在为前提，而没有任何写入方会写那个字段**，
或者**空输入让循环体不执行，于是空契约被判为已校验**。

| # | 守卫 | 为何从不触发 | 修复 |
|---|---|---|---|
| 1 | model card 的 `factor_set_status` 必须为 `WALK_FORWARD_PASSED` | 驱动在 **5 处**把 `"model_status": "WALK_FORWARD_PASSED"` 作为**字面量**写入 card，所以这个字段**不可能为假**；而已核的 `m0_v3` / `m0r_v2` / `m0r_v3` 三个真实 card 里**根本没有** `factor_set_status` 这个字段 | LightGBM 与 torch 两侧都已闭合，并改用**真实工件**（不是夹具）验证 |
| 2 | 另外两处同类守卫 | 用"这个检查到底证明了什么"这一个问题扫晋级→服务路径找到的，**对任何真实工件从未生效过** | `b9eb666` |
| 3 | torch 加载器的 `_allowlisted_factory` | 原写法 `if model_code_version is not None and (…)`，**省略该字段即可跳过整个 allowlist** | `17ece95` |
| 4 | `_validate_inference_contract` | 对**空映射**循环体不执行，于是 `expected_inference_contract={}` 通过校验；而新增的 `unverified_bindings` 判据是 `is None`，空契约反而会把自己**从未校验列表里移除**。实测：剔掉 card 的 8 个契约字段后加载成功，且报告 `unverified_bindings == ('factor_set_status',)`——**加载器主动声称它校验了一个它没校验的绑定**，比静默跳过更糟 | `37b6c44` |

**没有直接改成强制**，而是**让缺口无法被忽略**（显式列出未校验绑定），
理由写在原文 §4.1。

**另一条独立的、更重要的限制**（`04c5acd`）：
签名证据链每一环都闭合（预测分数 bytes 被签名、`_require_candidate_evidence` 把
selection id / model id / fold / dataset 与冻结选择逐一对上、
`_require_verified_model_bundle` 把 bundle 摘要与重新哈希对上），
**但这条链证明的是"这些分数被记录在这个 bundle 名下"，不是"这些分数确由这个 bundle 算出"**。
持有信任根的人可以在正确的 metadata 下签署任意分数 bytes，而现有每一道检查都会通过。
补位的是**独立的复现审计**（从已发布工件重算预测），它已在 M1 的四折上跑完；
写论文时若主张"结果可验证"，应说明**是复现审计而不是签名链**在支撑这个主张。

出处：`vendored/docs/handoffs/2026-09-09-model-status-fail-open-finding.md`；
`vendored/docs/handoffs/2026-09-11-m1-evaluation-crash-lineage-and-footprint.md` §8.1。

---

## E. 两个驱动缺陷：已修好、测好、变异检验过，但**必须等 M2 与 M3 都跑完才能合并**

| 缺陷 | 现象 | 修复 |
|---|---|---|
| fold-cache 临时文件名竞态 | 两个进程（或同进程内同键的两次写入）算出**同一个** `.parquet.tmp` 路径并互相覆盖；失败时还不清理自己的临时文件，那个已有 6 GiB 的目录里每中断一次就多一个永不读取也永不清理的文件 | `4f44e12`：新增 `write_fold_cache_atomic()`，临时名带 **pid + 8 位随机片段**（pid 分隔并发进程、随机片段分隔同进程内的两次写入、并抗 31 天不重启主机上的 pid 复用），失败时删除自己的临时文件；两处重复模式改走同一个助手 |
| `--folds` 是死参数 | 参数被接受但**从未生效**，传子集会静默跑全集 | `1dbc36c`：新增 `_requested_folds(args)`，只接受全集（任意顺序），其余一律 `SystemExit`；**校验放在 `parse_args` 而不是 `main()`**，因为 `parse_args` 是唯一入口，放那里就不可能被人重排到几小时拟合之后 |

* **位置**：worktree `/data/zhangyudong/qrv2_worktrees/deferred-driver-fixes`，
  分支 `codex/deferred-driver-fixes`（从 `9e0b2be` 起两个提交）。**主工作树未被触碰**，
  已验证：`grep -c "_requested_folds\|write_fold_cache_atomic"` 在主树两个驱动文件里都是 **0**。
* **门禁（在 worktree 内实跑）**：`ruff` 全清、`mypy src` 170 文件清、
  `pytest tests -q` **2127 passed / 46 skipped**（基线 2107 / 45，净增 21 个测试、0 失败）。
* **变异检验**：把临时名改回旧写法 → 3 个测试失败，关键那条报
  `two writers computed the same temporary path and would interleave` 并列出两个相同路径；
  删掉 `parse_args` 里的调用 → 2 条失败；**把调用挪到 `main()`（功能等价但可被重排）→ 同样 2 条失败**。
* **为什么现在不能合并**：运行中的驱动会被**下一折的 worker 从磁盘重新导入**，
  合并等于在多日运行中途换代码。合并窗口是 M2 与 M3 都跑完之后（约 09-25 前后），
  届时**必须重跑一次全门禁**，因为主线在此期间可能已前进。

---

## F. 主机层面：GPU 不可用，但**与本管线无关**（一条被我自己误判过的事实）

* `nvidia-smi` → `Failed to initialize NVML: Driver/library version mismatch`（NVML 库版本 580.178）；
  `torch 2.6.0+cu124` 的 `torch.cuda.is_available()` → **False**；
  训练日志里每个 worker 都打一条 CUDA 初始化警告（Error 804）。
* **但这不拖慢任何东西**：驱动在四个训练点都硬编码 `device=torch.device("cpu")`
  （`scripts/run_real_candidate_training.py` 第 597 / 919 / 2630 / 3315 行），
  框架**没有 CUDA 训练路径**。所以"修 GPU 能加速 M2/M3"是**错的**——
  本轮一度这么判断过，特此更正。
* 而且修它需要重载驱动或重启主机，会**直接杀掉在跑的 M2**，所以在 M3 结束前不该碰。
* **若将来真要上 GPU**：`training/runner.py` 第 52 行明确记着
  "Device is numerics-affecting (measured: cpu and cuda digests differ for the same …)"，
  即换设备等于换数值契约，所有已签名的检查点比对（验收 C′）与复现审计都要重做，
  且 `cuda_available` / `cuda_device_names` 会被写进可复现性运行时描述。
  这不是"打开一个开关"，是一次新的谱系。

---

## G. 内存足迹的测量方法有一个坑：只能用"增量"，不能用绝对值

只读探针测 M3 的 SetAttention 激活时，在同一进程里先载入 230 万行原始帧与 prepared 帧、
再 `del` + `gc.collect()`，而 **CPython/PyTorch 的分配器不会立刻把内存还给 OS**，
所以 `VmRSS` 停在 2.61 GiB 的高位；生产 worker 是在**全新进程**里反序列化 0.381 GiB 的批次载荷，
没有这份残留（生产实测常驻 1.26–1.33 GiB）。
**可用的是增量（M3 +0.32 GiB 对 M1 对照 +0.17 GiB），不是绝对值**；
两格的 `VmHWM` 都停在 3.92 GiB 未动，也说明训练期没有超过帧载入期的高水位。

**M2 的申报值已被独立核实为准确**：worker 实测 RSS 均值 12.46 GiB、最大 **17.05 GiB**，
与探针在生产路径上测的 **17.06 GiB** 相差 0.01 GiB；父进程实测 14.74 GiB（低于文档假设的 21.99）。
本轮（09-14 15:30）再次实测：24 个 worker 最大 RSS **16.76 GiB** < 申报 17.1 GiB。
**申报值直接决定 24 路那个台阶能否保住**，所以对 M2 它不是形式字段。

出处：`vendored/docs/handoffs/2026-09-12-m1-full-grid-launch-and-measured-cost-model.md`；
`vendored/docs/handoffs/2026-09-14-m1-full-grid-passed-and-m2-launched.md` §4。

---

## H. 谱系与治理：三个未决项（都归所有者，本轮未代为执行）

1. **M1 从未注册进 registry。** `model_registry/models/` 下**只有 M0R 一条**
   （`sha256:32315c89fb9a65b8…`，状态 `WALK_FORWARD_PASSED`，
   事件链 `None → DRAFT → WALK_FORWARD_PASSED`，理由 `all G1-G6 walk-forward gates passed`）。
   M1 的两次运行都有签名的 `promotion_evidence.json`，但**不在册**。
   要点：`promotion_evidence.json` 是**证据工件**，registry 的状态记录是**在册状态**，两者不是一回事；
   而且 `WALK_FORWARD_PASSED` **不是部署决定**（阶梯是
   `DRAFT → WALK_FORWARD_PASSED → SHADOW → LOCKED_TEST_APPROVED → …`）。
2. **注册哪一份 M1**：载体版（`m1_parallel_v1`，1 格）还是全网格版（`m1_full_v1`，27 格）？
   两者代表不同的搜索规模与不同的谱系证据。全网格已通过，建议以它为准，但这是裁决项。
3. **是否现在对 M0R 启动 SHADOW**：F6.1 的 `WALK_FORWARD_PASSED → SHADOW` 是**人工**动作，
   前置清单已备齐（历史 G1–G6、artifact、推理 manifest contract、V2Test 演练、回滚准备；
   其中 F10.5 影子切换/回滚桌面演练**已完成**），
   见 `vendored/docs/reports/2026-09-12-m0r-shadow-approval-preconditions.md`。
   **Test 环境 smoke run 不计入 60 有效日影子运行。**

---

## I. 运行环境的一个外部事实：本机有约 90 GiB 被另一个用户的常驻进程占着

2026-09-14 15:30 实测：4 个属于用户 `zhangzhenye`（conda 环境 `trellis2`）的
`multiprocessing.spawn` 进程已运行 **27 天 20 小时**，RSS 分别约 22.1 / 23.9 / 21.0 / 23.7 GiB，
CPU 占用 0.3–0.4%（基本空转但**不放内存**）。
主机总内存 629 GiB，当时 `MemAvailable` 约 196 GiB，swap 8 GiB **已用满但 si/so = 0**
（静止，没有在抖）。

**含义**：M2 驱动启动时算的是"511.2 GiB 可用 ≥ 480.5 GiB 门槛"，
那 90 GiB 是被别人占走的，所以**实际余量比申报口径薄**；
M3 的内存门槛是 `MemAvailable ≥ 140 GiB`（48 × 2.5 + 0.6 = 120.6，/0.9 = 134，取 140 留余量），
在 M2 退出后绰绰有余，**不构成阻塞**。
这些进程不属于本项目，**未动**；若将来某次运行报内存不足，这是第一个要查的地方。

---

## J. 一个尚未闭合的度量缺口：逐年增量对基线**不可得**

**现象**：候选报告里有 `yearly_mean_rank_ic`（候选自己的 945 日集）与
`baseline_mean_rank_ic`（**池化**、在候选日集上重算）；基线报告里有它自己的
`yearly_mean_rank_ic`（**969 日集**）。
**没有任何一份签名工件里存着"基线在候选 945 日集上的逐年值"**，
所以"逐年增量"这个量**目前不可从工件直接得出**。

**为什么它要紧**：论文里最有信息量的一张表恰恰是逐年增量
（它能把"2023 年候选可能弱于基线"这件事从推测变成测量），
而现在能写的只有两个不同日集的逐年值并列，加一句量级论证
（日集差异在池化口径下值 −0.0018，而 2023 的表面差是 −0.0051，约 2.8 倍，
所以**符号大概率稳健、幅度不可得**）。

**处置**：本轮**没有**去算它。算它需要读已签名的预测工件
（`<workspace>/predictions/…`，需 HMAC key）并在候选日集上重算基线的逐年 RankIC——
只读、可行、约等于把评价器的指标函数在两个已存在的帧上再跑一次；
逐折预检监视器已经在做同一类读取，所以路径是通的。
**留作所有者决定的可选项**，因为它属于"新增分析"而不是"复述已有证据"，
而且它的结果会进入论文，值得单独一次带验收的执行。
渲染器 `summarize_candidate_comparison.py` 对此的处理是**拒绝发明数字**：
它把基线的逐年值单独列一节，并明写"不要与候选的逐年值相减"。

---

## K. 一句话总结这一批问题的共同形状

六个阻塞点、四处永不触发的守卫、三次成本低估、一次 stale 默认值、一次我自己的 ETA 误算、
一处"工件里根本没有这个量"的缺口，
**没有一个是"算错了数"**——全部是**两处代码/文档各自陈述同一个事实、而没有一方去问另一方**。
因此本轮所有新增测试都改成"向写入方索取真实输出"（驱动自己的写入函数、真实工件、
独立重算的门数值、逐字节比对），而不是在夹具里复述一遍约定。
