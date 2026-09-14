# 设计：让 torch 线程数随工件旅行，并让服务侧无法悄悄用宿主默认值（2026-09-14）

**状态：设计已定，实现被合并窗口挡住。** 本文写给下一个能动手的人（很可能就是本会话的后续），
目的是让他**执行**而不是**重新推导**。发现过程与证据在
`docs/reports/2026-09-14-m1-full-grid-shadow-approval-preconditions.md` §3.1，本文不重复叙述，
只写约束、设计与验收。

---

## 1. 一句话问题

已训练的 torch bundle 的前向**不是线程稳定**的（1/2/4/8 线程逐位相同，16 线程差 4.8e-05、
52 线程差 4.9e-05，是近似并列股票的**秩反转**），而**重算出 0.0 所需的线程数不在已发布的工件里**
（它只在工作区的检查点 manifest 里），**服务侧也一行线程配置都没有**。
于是把 torch 候选放进影子、在默认 52 线程的宿主上打分，会得到与被审计、被签名的分数**不一致**的结果，
而**现有任何一道门都不会发现**。

## 2. 三条硬约束（决定了设计形状，不能绕）

1. **不能让现有 bundle 失效。** `artifact_loader` 里已经写着同一类教训：
   "requiring the field would refuse every existing bundle, including the promoted M0R"。
   已发布的 card 只有四个字段（`dataset_id` / `family` / `model_status` / `model_code_version`），
   M0R 已注册、M1 已归档、**M2 与 M3 的 bundle 也将是这个形状**（因为在它们跑完之前不能改驱动）。
   所以新字段必须是**可选的**，缺失时**不得**当成"通过"。
2. **合并窗口在 M3 之后。** 驱动会被下一折的 worker 从磁盘重新导入，M2 在飞期间不能改；
   M2 一验收完链式看守**立即**发射 M3，中间没有安全窗口。
   所以本改动与 `codex/deferred-driver-fixes` 的两条修复**同一批合并**（约 09-25 之后），
   合并时必须重跑全门禁。
3. **沿用仓库既有的处理方式：说出没校验什么，而不是悄悄跳过。**
   `artifact_loader` 已有 `unverified_bindings` 这个机制（正是 09-09 那次 fail-open 排查引入的）。
   线程数缺失时应当**出现在 `unverified_bindings` 里**，而不是静默用宿主默认值。

## 3. 设计（三部分，每部分都能独立验收）

### 3.1 写入侧：card 带上训练时的线程数

* `torch_model_card(...)` 增加一个**必填**参数 `torch_intraop_threads: int`，
  写进 card。四个调用点（驱动第 577 / 899 / 2588 / 3273 行）都传**该次运行实际使用的值**
  ——不是常量 1，而是 `execution.torch_threads`（并行搜索已强制要求它非空），
  这样"钉 1 线程的谱系"与"没钉住的谱系"在工件里就能区分开。
* card 的字段集因此从 4 个变成 5 个；`artifact_loader` 里那句
  "and nothing else" 的注释与相应校验必须同步更新，
  且**必须同时接受 4 字段（旧）与 5 字段（新）**两种 card。
* **不改 bundle 的其它文件**（`config.json` 是架构 dataclass 的字段集，
  `torch_bundle_config` 会剥掉未声明字段，往里塞运行时信息是错的地方）。

### 3.2 读取侧：服务进程在打分前钉住线程，钉不住就说出来

* 新增一个单一入口（建议放在 `inference/torch_scoring.py`，因为它是驱动与服务共用的打分层）：
  `resolve_serving_threads(card) -> int | None`，
  读 card 的 `torch_intraop_threads`；缺失时返回 `None`。
* 服务侧（`predictor.py` / `daily_pipeline.py` 的进程启动处）：
  * 拿到值 → `configure_torch_threads(n)`（**intra 与 inter 都要设，且必须在任何并行工作之前**，
    这一点 `verify_prediction_reproduction.py` 的 `_pin_scoring_threads` 已经写明并踩过：
    inter-op 一个进程只能配一次）。
  * 拿不到值（旧 bundle）→ **不得**静默用宿主默认：把 `torch_intraop_threads`
    加进 `unverified_bindings`，并要求调用方**显式**传一个线程数才继续；
    没传就拒绝加载。这与 §2 约束 3 一致，也与"空契约不得伪装成已校验"（`37b6c44`）同一原则。
* **部署侧的今天就能做的临时措施**（不需要本设计落地）：
  影子/生产打分进程启动时把 intra 与 inter 都钉到 **1**，并在日志里记录实际线程数。
  这条已写进 M1 前置清单 §3.1，**批准影子前必须落在运行手册里**。

### 3.3 审计侧：把"服务条件"变成可检查的

* `verify_prediction_reproduction.py` 已经在摘要里报
  `scored_under_torch_threads` / `threads_requested`；
  新增一项：当 card 带 `torch_intraop_threads` 时，**校验它与工作区检查点记录的值一致**，
  不一致就非零退出。这样"工件声称的线程数"与"训练实际用的线程数"不会各说各话
  ——正是本仓库反复吃亏的那类形状（两处各自陈述同一事实，谁也不问谁）。

## 4. 测试清单（每条都要做变异检验，即证明它能失败）

1. `torch_model_card` 的输出**含**线程数，且四个调用点传的是运行的实际值而非常量
   （变异：把某个调用点改成硬编码 1 → 测试必须失败）。
2. `artifact_loader` **同时**接受 4 字段与 5 字段的 card
   （变异：删掉兼容分支 → 旧 card 的测试必须失败）。
3. `resolve_serving_threads`：有值返回值；缺字段返回 `None`（不抛、不猜）。
4. 服务侧缺字段且调用方未显式给值 → **拒绝加载**，且 `torch_intraop_threads`
   出现在 `unverified_bindings` 里
   （变异：把拒绝改成静默用宿主默认 → 测试必须失败；
   把该绑定从 `unverified_bindings` 里去掉 → 测试也必须失败）。
5. 审计侧：card 的线程数与检查点记录不一致 → 非零退出
   （变异：把校验删掉 → 测试必须失败）。
6. 既有的 `test_every_card_the_driver_can_write_is_accepted_downstream`
   与 `test_every_allowlisted_family_has_a_factory_behind_it` 必须继续通过
   （它们覆盖 M1 / M3 / M2-GRU / M2-TCN 四家，是 card 与下游接受方之间的一致性锚）。

## 5. 验收标准（合并前逐条核）

* `ruff` 清（`src`、`tests`、被改脚本）；`mypy src` 清；`pytest tests -q` 全绿且**测试数只增不减**。
* **对已发布工件零影响**：`m0r_v2` 与 `m1_full_v1` 的四折复现审计在改动后仍
  `identical=true` / `max_absolute_difference=0.0`（这是最硬的一条：
  改的是 card 的**读取**兼容性，不该动任何既有分数）。
* 与 `codex/deferred-driver-fixes` 一起合并时，`run_real_candidate_training.py` 的三处改动
  （缓存竞态、`--folds`、card 字段）互不冲突；合并后重跑全门禁。
* 合并**必须**在 M2 与 M3 都跑完之后；合并前主树的驱动 sha256 必须仍是
  `33742880f314ca21`（这是"没提前动过"的证据）。

## 6. 本设计**不**做的事

* 不改任何已发布 bundle、card、签名或 registry 状态；不给旧工件补字段（补不了，签名覆盖 card 文件）。
* 不试图让 torch 前向变得线程稳定（那是数值库层面的事，本文只保证"用对的线程数"）。
* 不动 M2/M3 的运行参数；它们的 bundle 会是 4 字段 card，
  因此**它们的线程数只能来自各自工作区的检查点 manifest**（`torch_intraop_threads: 1`），
  这一点要写进它们的判定文档，别让下一个人以为工件里应该有。
