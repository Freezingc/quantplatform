# 2026-09-11：两条 M1 运行都死在评估阶段、四折完整的谱系对照、以及一次被实测否掉的预期

本交接覆盖 2026-09-11 00:31–23:10 的工作，按时间顺序：`898ca4b`（四处「前向线程稳定」的勘误）、
`c016271`（F1 单折的谱系实测）、`2d58a7b`（时序批次构造重写）、`c64cbac`（M2 发射包按实测改写）、
`e1e9c09`（内容哈希流式化）、`b86cdb0`（评估器 card dataset_id 改成员归属，第 1 节的第一个阻塞点）、
`6a19ef2`（OOS 业务键改双向覆盖，第二个）、`554b3d0`（采纳检查上提到构造折输入之前，第 6 节）、
`a950e42`（基线覆盖收窄到可打分行，第三个）、`0c6d7c9` 与 `08297c7`（本交接的前两版）、
`9467d63` 与 `c3afec9`（家族名收敛到一处，第 9 节的第四个阻塞点）、`81377f3`（第 9 节本身）、
`3654b8f`（card 只在一处构造、测试向驱动索取约定，第 10 节的结构性根因）、`f693282`（第 10 节本身）、
`9e8628a`（释放构造期间已死的帧，第 11 节）、`fd3d389`（第 11 节本身与发射算术重算）、
`e8cfca8`（M2/M3 的 bundle config 被 loader 拒收，第 12 节的第五、六个阻塞点）、
`003b5f7`（用驱动自己的写入方走完 loader 全契约，含签名比对）。

**五处以本交接为准，因为它们推翻了先前提交里的说法**：第 3 节的四折对照取代 `c016271` 的单折结果；第 5 节的
足迹实测否掉了 `e1e9c09` 当时预期的 worker 收益；**第 8 节的 OOS 实测否掉了我自己据第 3 节做出的外推**
（「谱系能造成的摆动比它要喂进的那道门还大」——那是把验证阶段的量级搬到 OOS 门槛上，实测差 44 倍）；
第 11 节的逐阶段解剖取代第 5 节「帧是下一个杠杆」的推断，并给出释放后 worker 12.85 GiB、父进程 17.07 GiB
的新基准；同节的发射算术取代第 7 节与 M2 发射包里的 17.54 / 24.67 GiB 与 495.1 GiB 门槛。

## 1. 最要紧的一条：两条真实 M1 运行都训练完了四折，然后在评估阶段崩掉，都没有判定

`m1_v2`（串行、未钉线程、约 44 小时）与 `m1_parallel_v1`（钉 1 线程、约 20 小时）都是同一个结局：F1–F4 全部
`complete: model=m1-fN`、四个 bundle 已冻结并签名、八个签名预测工件已入索引，然后在四折**之后**的评估阶段抛

```
ValueError: model bundle training dataset does not match frozen model selection
  evaluation/model_walk_forward.py:167  _require_verified_model_bundle
```

`report.json` 因此从未写出，两个 watcher 的审计与判定都失败了——**失败原因是「报告不存在」，不是复现失败**，
这两件事在转录里长得很像，不要混。

**根因是 card 里 `dataset_id` 有两种合法写法，而评估器只认其中一种。** 签名选择证据里
`training_dataset_id` 一律是 `<dataset>:train`（所有运行、所有家族都一样，已逐一核对 m0_v3 / m0r_v2 /
m1_v2 / m1_parallel_v1）。LightGBM 的 card 写 `<dataset>:train` → 严格相等成立 → M0/M0R 的评估通过并出了
判定；torch 的 card 写**裸** `<dataset>`（因为复现审计随后要用它去 `load_split(..., "test")`）→ 不相等 →
评估拒绝。**所以这不是某次运行的偶发问题，而是每一个 torch 家族都必然在训练完成之后、最贵的那一刻失败。**

**这件事代码库里早已裁决过，只是评估器没跟上。** 另外两处都按成员归属处理：

- `inference/reproduction.py:412 _checked_card_dataset_id`，文档字符串明写「两个写入方**有意**不一致……
  **两者都合法**，所以检查是成员归属而不是相等」，接受 `{report_dataset_id, report_dataset_id + ":train"}`；
- 驱动自己的 `_fold_adoption`（`run_real_candidate_training.py:1571`），`recorded_dataset not in
  (dataset_id, f"{dataset_id}:train")`。

**修法（`b86cdb0`）**：评估器接受本折签名选择所点名的三个 id —— `training_dataset_id`、`test_dataset_id`、
`test_dataset_id + ":train"`。**这是旧接受集的严格超集**，所以过去能过的一定还能过，没有任何既有测试改变
预期；而集合只由本折自己签名的证据构成，且 bundle 摘要在此之前已由 `load_model_card` 比对过，所以来自别的折
或别的运行的 bundle 仍然报名另一个哈希、仍然被拒。

**测试夹具本身就是这个 bug 能藏住的原因**：它把 `training_dataset_id` 写成 `"train-F1"`——一个没有任何写入方
会产出、且与该折真实 dataset id 无关的字符串，于是整套测试只跑过了一种写法并且全绿。夹具现在写
`f"{test_dataset_id}:train"`（与所有真实运行一致），并加了 `card_form` 参数，使两种约定与「无关数据集」三种
情况都被覆盖。新增两条测试：裸 id 的 bundle 必须通过；报名无关数据集的 bundle 必须**仍然**被同一条检查拒掉
——后一条是前一条的非恒真性守卫，放宽验证器时尤其需要它。

门禁：`ruff check src tests` 干净、`mypy src` 169 文件干净、`pytest tests` **2078 passed / 45 skipped**。

## 2. 恢复路径：并行运行可恢复，串行运行不可（原因是它早于搜索记录逐条落盘）

`_adopt_completed_fold` 的采纳要求里有一条是「每个配置的搜索记录都可采纳」。实测：

- `m1_parallel_v1` 有四条搜索记录（`search/F1..F4/c0-9ef4cccccd3fdf78.json`），`--resume` 重跑后
  **F1 已被采纳**：`F1 config 0: resumed from c0-9ef4cccccd3fdf78.json (ensemble validation RankIC=0.0561)` /
  `F1 adopted from verified artifacts: model=m1-f1`。四折采纳完即进入评估并写 `report.json`。
- `m1_v2` 的 `search/` 目录**为空**——它启动于 `142f8ea`（搜索记录逐条落盘）之前。采纳会失败，而
  `_adopt_completed_fold` 在失败时是**立即 SystemExit、不做任何拟合**（「an operator deserves the reason at
  the start of the run rather than hours into it」），所以试一次不会浪费算力，但也**恢复不出判定**。要它的判定
  只有两条路：新工作区重跑约 44 小时；或者只读取数（见下）。

**只读取数这条路是存在的，且不伪造任何证据。** 评估只需要 `ModelOosFold`（三个 id）加签名工件，而这些都在；
缺的只是 `report.json` 里那份由搜索记录重建的 `search_results` / `best_config`。**不建议事后补造搜索记录**
——那是伪造来源。可行的做法是直接调 `evaluate_model_walk_forward` 取 OOS 指标与六道门结果，只打印、不落盘、
不签发；它回答的是科学问题（M1@52 线程过不过门），产出的不是可归档的判定文档。

## 3. 四折完整的谱系对照（**验证阶段**；它到 OOS 门槛的外推已被第 8 节实测推翻）

`m1_parallel_v1` 跑完 F4 之后，两条运行在**四折上全部可比**：同数据集工件、同配置
（`hidden_dim=64, num_blocks=1, dropout=0.0`）、同三 seed、同 15 因子、同 `residual-mlp-1`、同早停预算，
唯一受控差别是 `torch_intraop_threads`（串行 52 未钉 vs 并行 1 显式钉住）。数值取自两侧**签名的** model
选择证据的 `validation_rank_ic`：

| 折 | 串行 52 线程 | 并行 1 线程 | 差（并行−串行） | 占 G4(5e-3) |
|---|---|---|---|---|
| F1 | 0.05441212880837507 | 0.05605766632279820 | +1.645538e-03 | 32.9% |
| F2 | 0.09434474279284134 | 0.09030626306078236 | **−4.038480e-03** | **80.8%** |
| F3 | 0.08209953348378562 | 0.08220521556268619 | +1.056821e-04 | 2.1% |
| F4 | 0.09934948816614481 | 0.09967804374342573 | +3.285556e-04 | 6.6% |

四条性质，逐条都是实测：

1. **符号 +−++、带符号均值 −4.9e-04 ≈ 0**，所以谱系差不是可事后校正的系统偏移。
2. **平均 |差| = 1.53e-03（G4 的 30.6%），最大 4.04e-03（G4 的 80.8%，F2）**。
3. **四折之间差值的极差是 5.684e-03，已经超过 G4 门槛本身（5e-3）**——即谱系选择能造成的摆动比它要喂进的
   那道门还大。这是本节最该被记住的一个数。
4. 两条谱系**各自内部重拟合都是确定的**：串行与并行的 `c0` 搜索拟合与 `final` 最终拟合，三个 seed 的最佳值
   与最佳 epoch 都逐一相同（串行 6/8/14、并行 6/13/16，值到 16 位小数一致）。所以发散严格是**跨**谱系性质，
   与「同宿主两次全量跑逐字节相同」那条既有性质一致。

**对裁决 2 的含义**：钉线程的一侧不仅身份可移植，而且下面第 4 节证明它在任意宿主上可被精确审计；未钉的一侧
身份依赖宿主核数。但两侧在四折上的验证 RankIC 差别已达到门槛的 80.8%（单折）与超过门槛（极差），所以
**「选哪条谱系作为生产口径」会实质影响 M1 是否过门的结论**，不能当作形式选择。

## 4. 可移植性主张已被实测证明，不再是推论

`m1_parallel_v1` 的 F1（bundle 记录的 `torch_intraop_threads = 1`）在**默认 52 线程**的宿主上审计：

```
scoring under torch intra-op threads=1 (the run recorded 1; the host default was 52)
identical: true   max_absolute_difference: 0.0   row_count: 803572   exit=0
```

即「钉到 1 线程的谱系既身份可移植、又能在任意宿主上被精确审计」现在是证据。反向也仍然成立：`m1_v2`（记录 52）
自动钉到 52 → `exit=0`、`max|diff| 0.0`；钉到 1 → `exit=1`、差 4.923682914836913e-05，**那不是缺陷**。
另：`recorded_run_threads` 对 `m1_v2` 读出 52、对 `m1_parallel_v1` 读出 1、对 LightGBM 基线工作区返回 `None`
（回落宿主默认，因其 `num_threads` 早已钉死）。

## 5. 流式内容哈希：真实收益，以及一个被实测否掉的预期

`e1e9c09` 把 `build_dataset_manifest` 的规范 JSON 改为增量喂给 sha256，不再把整个数据集的每个单元变成 Python
对象、也不再构造出整段 JSON 文本。**在真实工件上逐位不变**：改后 `load_split` 对 `m1_v2` 的 F1 重算出的
四个 content_hash 与改前完全一致（test `28c12cd9a23d32c9`、train `19888596dbec7503`、validation
`8afda36f080df01e`、combined `226a8dfc497c2de6`，最后这个就是已发布的 dataset_id
`226a8dfc497c2de6b07516711f058f26a9270675916c9e67dbf7f087a07d2538`），而 store 只在完全相等时才接受。
改后复现审计重跑 `m1_v2` F1 仍是 `exit=0`、`max|diff| 0.0`、803,572 行。

| 量 | 改前 | 改后 |
|---|---|---|
| `load_split(train)` 峰值 RSS | 8.62 GiB | **4.28 GiB** |
| `load_split(train)` 耗时 | 189.7 s | **168.8 s（−11%）** |
| 其中 combined manifest 构建 | 185.7 s / +5.61 GiB | 165.5 s / +1.26 GiB |
| `temporal_fold_batches` worker 峰值 | 13.61 GiB | **13.62 GiB（无变化）** |
| `temporal_fold_batches` 全程 | 455.9 s | **400.1 s** |
| **父进程**峰值（折批次 + TEST 面板批次） | 19.15 GiB | **17.21 GiB** |
| 父进程 fold / test 批次耗时 | 446.0 s / 621.6 s | 399.6 s / 552.1 s |

**必须纠正一个预期（负面结果）**：改动前我推断它会把时序 worker 足迹从 13.61 GiB 再降到约 9 GiB、从而把
M2 从 24 路解锁到 36 路。**实测否掉了**——worker 峰值仍是 13.62 GiB。原因是 worker 峰值从来不由加载阶段
决定：加载峰值（8.62 GiB）本来就低于后面的 panel + 4.47 GiB 批次张量所形成的峰（13.6 GiB）。削掉的是一个
本来不是瓶颈的峰。父进程降了 1.94 GiB，是因为它要加载三次（TEST 面板路径），加载峰在它这里更接近总峰。

**所以 M2 的 worker 数仍是 24、仍是约 5.8 天，36 路没有解锁**，`c64cbac` 写进发射包的数字不需要改。流式哈希
的真实收益落在**以加载为峰值的那些路径**：静态家族（M1/M3）、复现审计、数据集构建，以及 11% 的加载时间。

**顺带实测否掉的另一个优化**：`_load_verified_snapshot` 里那次在 split 口径下用不到的 `combined` 全量拼接，
用 spy 包住 `pd.concat` 实测只占 **0.27 s / 0.54 GiB**（整个 `load_split(train)` 是 189.7 s / 8.63 GiB，
5 次 concat 合计 0.7 s）。所以「顺手删掉多余 concat」不是杠杆，没有做。

**下一个真杠杆的位置也随之变了**：worker 那 13.6 GiB 里，4.47 GiB 是批次张量（不做惰性批次就无法降），
其余是帧与分配器开销；内容哈希已经不在峰值里了。要继续降足迹，方向是惰性/分批构造批次，或帧的 dtype
（数据集以 `date32[day]` 与 `string` 存储，读进 pandas 后是 object 列，即逐单元 Python 对象），
两者都比本次的改动大，且都要求「输出逐位不变」这条同样的纪律。

## 6. 采纳检查已上提到构造折输入之前（`554b3d0`）

第 1 节那三次恢复重跑把这条从"看起来划算"变成了"实测在付费"：`_run_full_cross_section` 与 `run_full_m2` 都在
调用 `_adopt_completed_fold` **之前**就先 `FoldPreprocessor.fit(train)` 并构造了 train 与 validation 两套批次
（M2 还多一套 TEST 面板），于是一个**已经冻结、马上就要被采纳跳过**的折，仍要先付出这些工作再由 `continue`
全部丢弃。实测 F1：M1 约 6 分钟/折；M2 的三套面板是 400 s + 552 s，约 16 分钟/折。而 `--resume` 是多日运行
**计划内**的跨折手段，不是意外——今天一天就走了三次。

采纳需要的全部输入（`model_id`、三个 dataset id、`factor_order`、网格、预算）在预处理拟合之前就都已知，基线证据
的写入更在前面，M2 绑定的窗口契约来自 `_temporal_search_budget` 读的冻结常量而不是任何面板，所以没有任何东西
强制原来的顺序。LightGBM 那条循环本来就把采纳放在基线证据之后、构造之前——**是两条 torch 循环没对齐**，现在三条
一致了。改的是位置不是内容：同样的调用、同样的顺序、同样的参数；数 3 个采纳点的既有 wiring 测试仍然数到 3。新增
的是一条钉住**顺序本身**的测试（把驱动按顶层函数切开，要求在两条 torch 循环里采纳调用先于 `preprocessor.fit(`，
并断言被检查的恰好是这两条，以免它哪天悄悄什么都不覆盖了）；它在改动前以 3406 对 2378 的偏移失败。

上提之后再跑恢复，每折从约 6 分钟降到约 4.5 分钟：省掉的是预处理拟合与批次构造，剩下的主要是 `extract` 与
`publish_fold_dataset` 的内容哈希校验（约 170 s），而那一部分采纳确实需要——`dataset_id` 是采纳要比对的东西。

## 7. 评估路径上一共是**三个**阻塞点，不是一个；三个都属于同一类

第 1 节写的是第一个。修掉它之后评估往前走了一步，又停在第二个；再修，又停在第三个。**同一个根因重复了三次：
评估路径把 LightGBM 载体的行集/命名约定当成了普适规则，而此前没有任何 torch 家族走到过这么远**，所以这三条
都没有被任何测试或任何真实运行触碰过。

| # | 位置 | 症状 | 真实成因（均为实测） | 修法 | 提交 |
|---|---|---|---|---|---|
| 1 | `model_walk_forward._require_verified_model_bundle` | `model bundle training dataset does not match frozen model selection` | card 的 `dataset_id` 有两种合法写法：LightGBM 记 `<dataset>:train`，torch 记裸 `<dataset>`（复现审计随后要用它加载 TEST 分片）。评估器做严格相等，只认前者 | 改为成员归属：接受本折签名选择点名的 `training_dataset_id` / `test_dataset_id` / `test_dataset_id + ":train"`。**是旧接受集的严格超集** | `b86cdb0` |
| 2 | `walk_forward._assemble_oos_frames` | `F1 candidate predictions and labels do not contain identical business keys` | LightGBM 候选覆盖整个 TEST 截面（F1 为 804,575 行），torch 只覆盖有可用标签的行（**803,572**，差的 1,003 行是 `label_valid=False`）。该检查要求候选的键与**未过滤**的标签帧完全相同 | 改为双向覆盖：候选不得有 TEST 帧之外的键，且不得漏掉任何**可打分**行；`_scoreable_label_keys` 照抄指标自己的有效性表达式 | `6a19ef2` |
| 3 | `walk_forward._require_baseline_covered_by_candidate` | `F1 candidate and baseline predictions: baseline rows are not all covered by the candidate` | 冻结的 best_simple 基线覆盖 747,461 行，torch 候选覆盖 803,572 行，基线里候选没有的那 **940 行中可打分的行数是 0**；指标的可用交集两种情况下都是 **746,521 行** | 覆盖要求收窄到**可打分**行；同时把"基线不得有 TEST 帧之外的键"显式写出（它原先只是恒等检查的副作用，收窄后不再被隐含保证） | `a950e42` |

三条都各自带了非恒真性守卫测试，因为放宽验证器正是最需要这种守卫的改动：第 1 条配"报名无关数据集的 bundle 仍被
拒"，第 2 条配"漏掉一个可打分行仍被拒"与"候选出现 TEST 帧之外的键仍被拒"，第 3 条配"基线出现 TEST 帧之外的键
仍被拒"，并且被接受的两种情形都断言 `delta_rank_ic` 仍为 2.0（即数值没有被动过）。

另外两条值得记下的连带事实：

- **测试夹具本身就是这三个 bug 能藏住的原因。** `test_model_walk_forward.py` 的夹具把 `training_dataset_id` 写成
  `"train-F1"`（没有任何写入方会产出的字符串），并且所有 TEST 行都 `label_valid=True`、候选与基线行集完全相同。
  于是整套测试只跑过一种 card 写法、一种行集约定，并且全绿。夹具现在两者都参数化了。
- **`m1_v2` 的 `--resume` 是安全失败的**：它在**任何拟合之前**就 SystemExit，理由是
  `config 0 has no adoptable search record (no record at c0-9ef4cccccd3fdf78.json)`——该运行早于 `142f8ea`
  （搜索记录逐条落盘）。所以试它不花算力；**它的判定最终由第 8 节的只读路径取回了，没有重训那 44 小时**。

## 8. M1 的判定出来了：两条谱系都过六道门，M0R 仍然最好，而我此前对谱系敏感度的外推是错的

三个阻塞点修完后，`m1_parallel_v1` 用 `--resume` 采纳四折、跑完评估并**正常退出（`resume exit=0`）**，写出
`report.json`、`verdict.md`（**M1 candidate: WALK_FORWARD_PASSED**）与 `promotion_evidence.json`。
**这是链条第一次为 torch 家族产出判定。** `m1_v2`（串行 52 线程）的判定则由只读路径从它已冻结的签名工件里取回
——它的 `search/` 为空、`--resume` 无法采纳、重训要 44 小时，而评估只需要 `ModelOosFold` 的三个 id 加签名工件，
这些都在。**只读路径不签发、不落盘、不伪造搜索记录**：它回答科学问题，产出的不是可归档的判定文档。

四者共用同一份冻结 best_simple 基线（`baseline_mean_rank_ic` 都是 0.0852268662737014）与同一个
`valid_day_count` 945，所以可直接比：

| | M0（`m0_v3`） | **M0R（`m0r_v2`）** | M1 串行 52 线程 | M1 并行 1 线程 |
|---|---|---|---|---|
| mean RankIC | 0.089429123160 | **0.094375961393** | 0.090943058925 | 0.091071468975 |
| ICIR | 0.748719740250 | **0.768172883837** | 0.706794068092 | 0.712568708381 |
| HAC t | 12.383336031330 | **12.905448712183** | 11.729630132912 | 11.851545300157 |
| **delta RankIC（G4）** | 0.004202256887 | **0.009149095119** | 0.005716192652 | 0.005844602702 |
| G4 余量 | **−7.977e-04** | **+4.149e-03** | +7.162e-04 | +8.446e-04 |
| G5 / G6 | 0.2819 / 0.9752 | 0.2791 / 0.9752 | 0.2837 / 0.9752 | 0.2895 / 0.9752 |
| **六道门** | **FAIL（G4）** | **PASS** | **PASS** | **PASS** |

三条结论：

1. **M0R 仍然是最好的候选**，在每一个头条数字上都领先：mean RankIC 0.094376 对 M1 的 0.091071，delta
   0.009149 对 0.005845（约 1.57 倍）。M0 仍是唯一不过门的（G4 差 7.977e-04）。
2. **M1 过门了，但这个判定是"载体验证"而不是"M1 的上限"。** 两次运行都带 `--max-configs 1`，而 M1 的网格是
   `3×3×3 = 27` 格（`hidden_dim ∈ {64,128,256} × num_blocks ∈ {1,2,3} × dropout ∈ {0.0,0.1,0.2}`），实际只搜了
   第一格 `{hidden_dim:64, num_blocks:1, dropout:0.0}`。所以「一个未调参的配置就过了六道门」是正面信号，
   而**「M1 不如 M0R」这个结论不成立**——还有 26 格从未试过。
3. **我此前对谱系敏感度的外推是错的，在此更正。** 第 3 节测到的验证阶段谱系差（最大单折 4.038e-03 = G4 的
   80.8%、四折极差 5.684e-03）是真的，但我据此写下「谱系能造成的摆动比它要喂进的那道门还大」「谱系选择有可能
   决定判定」，**那是把验证阶段的量级直接搬到 OOS 门槛上，实测不支持**。两条谱系的 OOS `delta_rank_ic` 只差
   **1.284100501392e-04**，即 G4 门槛的 **2.57%**、M1 自身余量的约 18%，比验证阶段四折极差小 **44 倍**，
   而且**两条谱系都过 G4**。原因是样本量级不同：验证 RankIC 是单折验证日上的数，OOS 是四折 945 个有效日、
   约 355 万行汇合后的数，逐折约 1e-3 的噪声在汇合后大部分抵消了。
   **对裁决 2 的含义因此变了**：钉线程与否在这个候选上**不改变判定**，所以裁决 2 可以按它本来的理由来定
   （身份可移植、可在任意宿主审计、跨宿主可比），而不必被「怕改变判定」绑架。样本仍是 1 个候选 1 个配置，
   要一般化需要 OOS 层面的多样本，而不是验证阶段的极差。

### 8.1 复现审计也在四折上跑完了（谱系钉 1 线程，宿主默认 52）

`verify_prediction_reproduction.py --workspace m1_parallel_v1 --report …/report.json`：**`exit=0`**、
`reproduced=true`、`scored_under_torch_threads=1`（运行记录 1，宿主默认 52），四折全部 `identical=true`、
`max_absolute_difference=0.0`：F1 803,572 行、F2 881,863 行、F3 931,411 行、F4 936,317 行，
合计 **3,553,163 行逐位相同**。这把第 4 节那条主张从单折证据升级成整轮证据。

### 8.2 晋级证据已签发，但**晋级本身没有发生**

六道门全过，所以驱动执行了 `_maybe_emit_promotion_evidence`，在工作区写出 `promotion_evidence.json`
（129,485 字节、`evidence_sha256` `f934b5c6464a530d8f078957eec723698bc94c1e9f858078308e572dbac6d200`、
`model_id` `sha256:0c438140c0e072e1266701f805de1584fe424a2e3d6be71f0b9433986e771a2a`）。**签发不等于晋级**：
它只是「注册表将来可以消费的一份文件」，写在该运行自己的工作区里，没有触碰注册表。是否晋级、以及晋级哪一条
谱系（两条都过门，但派生 `model_id` 不同、工件不可互换），仍是所有者的裁决。

## 9. 第四个阻塞点是**审出来的**，不是跑出来的：M2 会以和 M1 完全相同的方式死在评估阶段

前三个阻塞点都是「修一个、跑一次、再撞一个」，每次约 40 分钟。所以这次把评估与推理路径上**所有**按家族名分派的
地方静态查了一遍，而不是等 M2 跑五天再重来。查出第四个，而且它不分架构：

`evaluation/model_walk_forward.py` 的 `_MODEL_CARD_FAMILIES["M2"]` 是 `{"M2", "TEMPORAL"}`，而 `run_full_m2` 往
card 里写的是 `f"TEMPORAL_{best_config['family']}"`（网格 `family: [GRU, TCN]`，所以是 `TEMPORAL_GRU` 或
`TEMPORAL_TCN`）。**两个真实拼写都不被接受**，于是每一次 M2 运行都会训练数日、冻结并签名四折，然后在评估阶段死于
family 不匹配。而评估器自己在 `MODEL_FAMILIES` 上方的注释里就记着**同一类漂移曾经已经杀掉过一次跑完的运行**
（M0R 只加进了驱动、没加进这张表）。

后面还藏着两个**只影响 TCN** 的缺口：

| 位置 | 缺口 | 后果 |
|---|---|---|
| `inference/builder.py` `_TEMPORAL_FAMILIES` | 只有 `TEMPORAL_GRU` | TCN bundle 在批次装配处落到「未知家族」拒绝分支 |
| `inference/artifact_loader.py` `_MODEL_CODE_VERSIONS` 与工厂分派 | 无 `TEMPORAL_TCN` 条目、无对应分支 | 驱动写的 `temporal-tcn-1` 无法加载，**TCN 胜出即不可服务** |

**根因是同一件事：四个模块各自把家族名手写成字符串字面量**（评估器的 card 白名单、复现审计的打分器路由、推理
builder 的批次形状分派、artifact loader 的 code version 表），其中三个漂移了。修法是**不再转录**：新增
`research_v2/models/families.py`，从模型类自己的 `model_family` 属性推导静态/时序集合、推导时序的 code version
（`f"temporal-{去掉 TEMPORAL_ 前缀后小写}-1"`，与驱动写的逐字一致）、并给出每个 torch 载体可记录的架构名。四个
消费方全部改为读它，于是改名会**传播**而不是漂移。评估器 M2 条目里的 `"TEMPORAL"` 被去掉：没有任何写入方产出它，
而保留一个没人写的拼写正是白名单开始「什么都不说」的方式。

**这个缺口能藏住，是因为两条既有测试把它钉成了契约**，这一点比缺口本身更值得记：

- `test_unknown_family_is_rejected` 拿 `"TEMPORAL_TCN"` 当「未知家族」的例子——于是它断言的正是缺陷本身。现在换成
  一个没有任何模型类声明的名字，并且分派测试改为对推导出的时序集合做参数化。
- `test_the_torch_code_version_allowlist_is_pinned` 拿一个手写字典去比另一个手写字典。**这种断言冻结遗漏的能力和
  冻结正确值的能力一样强**，所以它把「没有 TCN」钉成了预期。现在家族名取自模型类。

新增 `tests/unit/models/test_families.py` 把四个路由全部钉到推导出的集合上，并要求每个白名单家族**真的能构造**。
它迭代的是模型类声明的家族、不是白名单本身——变异检验暴露了为什么这要紧：把 TCN 从表里删掉后，迭代白名单的版本
只是不再覆盖它，测试照样通过。另加两条端到端测试：M2 选择证据配 `TEMPORAL_GRU` / `TEMPORAL_TCN` 的 card 必须评估
通过，配 `RESIDUAL_MLP` 的必须仍被同一条检查拒掉。

**三次变异检验，每次都恰好杀死该杀死的测试**：评估器 M2 条目改回 `{"M2","TEMPORAL"}` → 两条参数化验收测试 +
一致性测试失败；builder 改回只有 GRU → TCN 分派测试 + 一致性测试失败；TCN 从 loader 表里删掉 → 白名单测试 +
code version 测试 + 工厂测试失败。

**按家族名枚举的地方已全部收敛到一处**（但这句话只覆盖「家族名」这一个轴；bundle 契约还有 `dataset_id`、
`config` 字段集与 `signature` 重建三个轴，其中 `config` 那个轴随后又查出两个阻塞点——见第 12 节与其对本节
这句「完整」的更正）：`registry/promotion_evidence.py` 只记录家族、不做白名单；`inference/torch_scoring.py` 与
`predictor.py` 不按家族分派（接收已装配好的批次）；注册表本身不做家族校验；`lgbm_artifact_loader` 的两个 LightGBM
code version 是完整的（M0/M0R 是仅有的 LightGBM 载体）；`training/temporal_search.py:138/144` 分派的是**配置里的**
`GRU`/`TCN`（另一个命名空间，两支齐全且有 raise）。**按家族名枚举的地方已全部收敛到一处。**

提交：`9467d63`（推导 + 四个消费方 + 测试）、`c3afec9`（loader 分派也改为按类属性，消除同模块内的第二处枚举）。
门禁：`ruff check src tests` 干净、`mypy src` **170** 文件干净、`pytest tests` **2095 passed / 45 skipped**。

## 10. 结构性根因：夹具在**复述**写入方的约定，而不是向写入方索取；smoke 路径根本不进评估器

四个阻塞点是同一个错误在四个位置，但让它们全部活下来的**结构原因只有一个**：每个测试夹具都是把约定
**再写一遍**，而不是从写入方拿。于是"夹具与夹具一致"看起来像"链条一致"，而链条里真正写 card 的那段代码从未
被任何测试问过。前三个阻塞点最终是被**已经训练了数十小时的真实运行**发现的，不是被测试发现的。

所以修法不是再加一条断言，而是把写入方变成唯一来源（`3654b8f`）：

- `torch_model_card(...)` 是**唯一**拼这四个字段的地方，四个写入点（full/smoke × 截面/时序）全部改用它；
- 载体身份改为从模型类的 `model_family` 推导的常量（`M1_FAMILY_NAME` / `M3_FAMILY_NAME`），M2 的两个字段由
  `m2_card_family` / `m2_model_code_version` 从**胜出配置**生成；
- `torch_card_conventions()` 枚举写入方**可能产出的每一个** `(载体, card family, code version)`，其中 M2 的条目
  由 `m2_config_grid(None)` 实际搜索的家族集合推导——所以一个变得可搜索的架构会**自动**变得被检查，而不是
  可训练但不可加载。

两条测试把夹具闭不上的那个环闭上：一条对每个约定调用驱动自己的 card 构造器，把结果交给两个会因不匹配而拒绝的
消费方（评估器的 card family 白名单、artifact loader 的 code version 表）；另一条把约定集合与网格实际搜索的
家族集合相比——**正是这条比较的缺失，让 `{"M2","TEMPORAL"}` 在白名单里安稳待了这么久**。变异检验：把
`m2_card_family` 改回错误拼写，恰好杀死三条测试（两条新的，加上 `test_m2_smoke_pipeline_end_to_end`）。

**另一条结构性事实必须记下，因为它解释了为什么 smoke 全绿而 full 全崩。** smoke 路径直接调
`evaluate_frozen_oos`，**从不进入 `evaluate_model_walk_forward`**——而三个评估阻塞点全部住在后者里
（bundle 校验、OOS 业务键覆盖、基线覆盖）。smoke 不发布数据集工件、不签发选择证据、不冻结预测工件，
所以它结构上不可能碰到那三条检查。它的自我描述是准确的：`"smoke orchestration evidence only; never model
passing evidence"`。它**确实**能管住 card——因为它会用 `load_bundle_for_evaluation` 把 bundle 读回来，
上面的变异检验里它就是因此失败的第三条测试。

**没有做的事，以及为什么**：把 smoke 扩展成跑完整证据链（发布工件 → 签发选择 → 冻结预测 →
`evaluate_model_walk_forward`）。那会让 smoke 真正覆盖评估路径，但也会改变 smoke 的契约（它现在被明确定义为
"只是编排证据"），并且需要为合成帧搭起整条签名链。单元测试侧的覆盖已经补上了：`test_model_walk_forward.py`
的夹具现在能产出无标签行与四种候选覆盖形状，`test_walk_forward_oos.py` 直接在评估器层面钉住双向覆盖。**是否
把 smoke 升级成端到端，属于契约变更，留给所有者决定。**

## 11. worker 峰值的逐阶段解剖，以及据此释放的 1.24 GiB 死帧（`9e8628a`）

第 5 节说过「下一个真杠杆是帧」，但那是推断。这次逐阶段实测了 `temporal_fold_batches`（真实 F1、1 线程、
每步之后读 `/proc/self/statm` 的**当前** RSS，不是只看生命周期峰值）：

| 阶段 | 当前 RSS | 该阶段活着的帧（deep 大小） |
|---|---|---|
| 起始 | 0.45 GiB | — |
| 载入 train 后 | 2.27 GiB | train 0.588 GiB（object 列：`signal_date`、`ts_code`、`exit_date`） |
| 载入 validation 后 | 3.42 GiB | + validation 0.178 GiB |
| prepare train 后 | 4.05 GiB | + prepared_train 0.503 GiB |
| prepare validation 后 | 4.05 GiB | + prepared_validation 0.153 GiB |
| 面板拼接后 | 4.45 GiB | + panel 0.655 GiB（此时 train 仍活着） |
| **train 批次构造后** | **8.28 GiB** | + train 批次张量 3.179 GiB |
| **validation 批次构造后** | **12.77 GiB** | + validation 批次张量 0.984 GiB |

拆出来是三块：**批次张量 4.16 GiB**（不可约，除非改成惰性/分批构造）、**活帧 2.08 GiB**、以及约 **6.5 GiB 的
分配器残留**（载入校验阶段峰值 4.28 GiB，加上 953+237 次独立大张量分配造成的碎片）。

活帧那 2.08 GiB 里有 **1.24 GiB 是死的**：原始 train 分片（0.588，取完 `train_dates` 之后再没人读）、
prepared_train（0.503，面板 `pd.concat` 已复制它、train 批次也已消费完）、prepared_validation（0.153，
面板已复制它）。三者都被局部变量一直握到函数结束，也就是**握过了峰值所在的那两个阶段**。24 路下这是
约 30 GiB 白占，而当时 24 路对 447.3 GiB 的余量只剩 1.7 GiB。

改成用完即 `del`（`prepare_frame` 与 `pd.concat` 都是复制，所以没有任何窗口是透过被删的引用读的）。实测：

| 量 | 改前 | 改后 |
|---|---|---|
| worker 峰值 RSS（`temporal_fold_batches`） | 13.62 GiB | **12.85 GiB** |
| 父进程峰值 RSS（折批次 + TEST 面板批次） | 17.21 GiB | **17.07 GiB**（同日早先另测为 19.15） |
| 批次数（train / validation） | 953 / 237 | 953 / 237 |
| 张量字节数 | 4,470,614,049 | **相同** |
| 单个 train 批次特征张量 | `(1,1610,20,15)` float32 | **相同** |
| 重建耗时 | 400.1 s | 402.2 s（未变，符合预期：这是内存改动） |

**照实说一句不如预期的地方**：释放了 1.24 GiB，worker 峰值只降了 **0.77 GiB**。原因是 `ru_maxrss` 是生命周期
峰值，而 glibc 不会立刻把释放的内存还给操作系统，所以峰值不完全等于「活对象之和」。0.77 GiB 是实测到的、
可写进发射包的那个数，1.24 GiB 是理论释放量，两者都记在这里。

### 11.1 据此重算的发射算术（取代第 7 节与发射包里的旧值）

按 F4 行数等比折算（与既有方法一致），MemAvailable 497 GiB、90% 余量、父进程计入：

| 折 | 单 worker | 父进程 | 安全 worker 数 | 第 7 节旧值 |
|---|---|---|---|---|
| F1 | 12.85 GiB | 17.07 GiB | 33 | 31 |
| F2 | 14.23 GiB | 18.90 GiB | 30 | 28 |
| F3 | 15.51 GiB | 20.60 GiB | 27 | 25 |
| F4 | **16.56 GiB** | **21.99 GiB** | **25** | 24 |

**约束仍取最大的折，但 24 路的前提松了 29.1 GiB**：24 × 16.56 + 21.99 = **419.36 GiB**，要求
MemAvailable ≥ **466.0 GiB**（原为 495.1 GiB）。这不是纸面差别——2026-09-11 02:00 前后另一位租户占着约
86.6 GiB、MemAvailable 只有 488.5 GiB，那时 F4 的安全数是 23、**24 路会被护栏拒**；同样条件下现在安全数是
**25**，24 路有余量地通过。发射包已按此改写（`--worker-memory-gb 16.6`、前置检查阈值 466.0）。

**36 路确定不可达**：需要 686.7 GiB，超过本机 660 GiB 总量。所以 24 路是当前足迹下的天花板，
**要再往上去只能动那 4.16 GiB 的批次张量本身**（惰性构造、按需生成，或把每折的批次改成对一块连续内存做
视图而不是 953+237 次独立分配）——那是架构级改动，且必须继续满足「输出逐位不变」这条同样的纪律，
不在本轮范围内。

### 11.2 一个关于测试的反面记录

第一版释放测试是**恒真的**：它在 `temporal_fold_batches` **返回之后**才采样 weakref，而所有局部变量都随调用
结束而死，所以它对未修改的代码也通过。要观测的性质是**构造过程中**的存活性，因此探针改成从
`temporal_panel_batches` 内部采样（每构造一个面板采一次），断言打在**第二次**采样上——那时原始 train 分片与
两个 prepared 帧除了「还有个局部变量指着它们」之外已经全死。prepared 帧的身份由「它是从哪个原始帧准备好的」
按对象同一性还原，不按行数猜。同时断言 `raw_validation` **必须仍活着**（它作为 `validation_labels` 返回，
是选择指标打分要用的），只断言释放会让这条测试变成「关于删东西」而不是「关于删对东西」。

## 12. 第五、第六个阻塞点：M2 与 M3 写出的 bundle config 是 loader 拒收的（`e8cfca8`、`003b5f7`）

第 9 节说「审计到此是完整的」——那句话对**家族名枚举**这一个轴是准确的，但 bundle 还有另一个轴：**config 的字段
集合**。用「把驱动真实产出交给消费方」这个方法一查，两个都断了。

`artifact_loader._require_fields` 要求 bundle 的 config 与架构自己的 config 字段**完全相等**
（`"model config has unsupported or missing fields"`），而两条 full 运行写入的是
`dict(best_config, raw_factor_count=...)` 原样透传。用驱动自己的网格与 card 构造器实测：

```
M1        LOADED OK   config=['dropout','hidden_dim','num_blocks','raw_factor_count']
M3        REFUSED     config=[…,'learning_rate',…]        7 个字段，架构只要 6 个
M2/GRU    REFUSED     config=[…,'family','learning_rate',…]  5 个字段，架构只要 3 个
M2/TCN    REFUSED     同上
```

即 **M2 与 M3 的运行会训练数日、冻结并签名 bundle，然后发现它无法被评估器、复现审计与服务端任何一方加载**。

**M1 是碰巧干净的**——它的网格恰好只含架构字段。这正是问题隐身的原因：M1 是唯一跑过的 torch 家族，而两条 smoke
路径都到不了这里（截面 smoke 用 `asdict(config_dataclass)`、M2 smoke 手写三个字段，两者写出的字段集恰好正确）。

修法是在**写入方过滤**，不是放松 loader：新增 `torch_bundle_config(config_payload, card_family=…, factor_count=…)`，
向 loader 索取该架构的字段集合（`model_config_fields`），只写这些加 `raw_factor_count`。loader 的严格相等是真契约
（它挡住的是携带陈旧或外来字段的 bundle），保留。过滤不丢信息：完整的胜出配置仍在该折的搜索记录与报告的
`best_config` 里，那才是 `learning_rate` 与架构选择该待的地方；bundle 的 config 只负责重建权重。字段集合两侧都
**从 config dataclass 推导**（`dataclasses.fields`），不再各写一份——这是本条与第 9 节共同的做法。
**目前不存在任何 M2/M3 bundle，所以现在改是零成本，晚改就是白跑一次。**

### 12.1 补上最后一环：签名比对

字段集合的比较只能挡住「因键被拒」的 bundle，挡不住「键对但值重建不出同一个模型」的 bundle。
`load_bundle_for_evaluation` 会从 config 重建架构、再把它的 `artifact_signature()` 与 bundle 里冻结的签名比对——
**这是「运行冻结了什么」与「之后还能加载什么」之间的最后一环**，而此前没有任何测试用驱动自己的写入方走过它
（所有 loader 测试都手搓 card / config / signature，所以写入方与接收方的不一致在构造上就不可见）。

`003b5f7` 补上：对四个 torch 载体（M1、M3、M2/GRU、M2/TCN）各写一个 bundle，全部用 full 运行真正调用的那四个
函数（`torch_model_card`、`torch_bundle_config`、该载体的 model factory、`_preprocessing_artifact`）构造，再用
`load_bundle_for_evaluation` 读回，断言三 seed、家族正确。**变异检验**：给 `torch_bundle_config` 塞进一个多余字段，
四个参数化用例全部失败、外加穷举网格的往返测试——这正是此前缺的灵敏度：在这之前，M1 或 M3 写入方漂移一个字段
没有任何端到端测试会发现。

### 12.2 方法与对第 9 节那句「完整」的更正

第 9 节的静态清扫是按**家族名**枚举点找的，它对那个轴确实穷尽了；但 bundle 契约至少有三个轴——card 的
`family`、card 的 `dataset_id`、以及 `config` 的字段集合——加上 `signature` 的重建比对，一共四个。前三个各出过
阻塞点（1/2/3 是 `dataset_id` 与行集、4 是 `family`、5/6 是 `config`）。

**真正有效的方法不是枚举代码里的常量，而是把写入方的真实产出交给接收方**：拿驱动自己的网格、自己的 card 构造器、
自己的 config 过滤器，去调 loader / 评估器。这一条一次找出两个，而静态清扫一个都没找到。第 12.1 节那条测试就是
这个方法的固化形式，也是今后新增载体或新增网格字段时的默认防线。

六个阻塞点的完整清单（全部已修）：

| # | 轴 | 症状 | 影响的载体 | 提交 |
|---|---|---|---|---|
| 1 | card `dataset_id` | `model bundle training dataset does not match frozen model selection` | 所有 torch（M1 实际中招两次） | `b86cdb0` |
| 2 | OOS 业务键 | `candidate predictions and labels do not contain identical business keys` | 所有 torch | `6a19ef2` |
| 3 | 基线覆盖 | `baseline rows are not all covered by the candidate` | 所有 torch | `a950e42` |
| 4 | card `family` | `model bundle family does not match frozen model selection` | **所有 M2** | `9467d63` |
| 5 | bundle `config` | `model config has unsupported or missing fields` | **M2（GRU 与 TCN）与 M3** | `e8cfca8` |
| 6 | 同上，TCN 服务侧 | loader 无 `TEMPORAL_TCN` 条目与工厂分支、builder 无法装配 | **TCN 胜出的 M2** | `9467d63` |

门禁：`ruff check src tests` 与驱动干净、`mypy src` **170** 文件干净、`pytest tests` **2104 passed / 45 skipped**。
