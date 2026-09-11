# Candidate Search Parallelism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run independent M1/M3 validation configuration searches in spawned processes while preserving exact selection and evidence contracts.

**Architecture:** A reusable process scheduler accepts index-addressed work and returns index-addressed results in canonical order. The candidate driver uses it only for M1/M3 after explicitly pinning Torch threads; serial search, final fits, artifacts, and OOS evaluation remain unchanged.

**Tech Stack:** Python 3.11, `concurrent.futures.ProcessPoolExecutor`, PyTorch CPU, pandas, pytest.

**Spec:** `docs/superpowers/specs/2026-09-10-candidate-search-parallelism-design.md`

## Global Constraints

- Scope is M1/M3 cross-sectional config search only.
- `--parallel-workers` defaults to `1`; this legacy path does not configure Torch threads.
- `--parallel-workers > 1` requires a positive `--torch-threads` value and uses a spawned context.
- `SearchResume.persist` is the sole durable config-result writer.
- Report ordering is increasing config index, never worker completion order.
- Final fits, bundle/evidence/prediction writes, and G1-G6 aggregation stay in the parent.

---

### Task 1: Declare the execution contract

**Files:**

- Modify: `research_v2/src/research_v2/training/seeds.py`
- Modify: `research_v2/scripts/run_real_candidate_training.py`
- Test: `research_v2/tests/unit/training/test_reproducibility_seeds.py`
- Test: `research_v2/tests/unit/scripts/test_run_real_candidate_training.py`

**Interfaces:**

- `configure_torch_threads(thread_count: int) -> None` rejects zero and configures both Torch thread pools.
- `_parallel_execution(args) -> ParallelExecution` parses/validates explicit opt-in settings.

- [x] Write failing tests that a parallel run without `--torch-threads` is rejected and a zero thread count is invalid.
- [x] Run the two targeted test files and confirm the APIs are absent.
- [x] Add the CLI flags, validation helper, and thread setter.
- [x] Re-run the targeted tests (`70 passed`) and commit with `feat(training): declare parallel torch execution settings` (`05d3922`).

### Task 2: Add an ordered spawned scheduler

**Files:**

- Create: `research_v2/src/research_v2/training/parallel_search.py`
- Test: `research_v2/tests/unit/training/test_parallel_search.py`

**Interfaces:**

- `SearchTask(index: int, payload: Mapping[str, object])` binds every result to one grid position.
- `run_parallel_search(tasks, workers, run_task) -> list[tuple[int, dict[str, object]]]` uses `multiprocessing.get_context("spawn")`, caps workers to task count, propagates failures, and returns index order.

- [x] Write failing tests for reverse completion ordering, worker-cap calculation, and worker exception propagation.
- [x] Run `python -m pytest tests/unit/training/test_parallel_search.py -q` and confirm the missing-module failure.
- [x] Implement the scheduler with `ProcessPoolExecutor` and a spawned context.
- [x] Re-run the scheduler tests (`3 passed`) and commit with `feat(training): add ordered spawned config scheduler` (`0f77b95`).

### Task 3: Route M1/M3 search trials through the scheduler

**Files:**

- Modify: `research_v2/scripts/run_real_candidate_training.py`
- Test: `research_v2/tests/unit/scripts/test_run_real_candidate_training.py`

**Interfaces:**

- `CrossSectionConfigTask` contains all three-seed search inputs and a config/seed-unique checkpoint prefix.
- `run_cross_section_config_task(payload) -> dict[str, object]` is package-level and writes no bundle, selection evidence, prediction, or report.
- `_ordered_search_results(configs, completed) -> list[dict[str, Any]]` restores grid order before best-config selection.

- [x] Write a failing test that reverse-order scheduler results become increasing-index search results.
- [x] Run that test and confirm the ordering helper is absent.
- [x] Extract the existing three-seed config body into the worker and call the scheduler only for M1/M3 when workers exceed one. Preserve the serial loop for one worker.
- [x] Run the complete candidate-driver suite (`69 passed`, including a real spawned three-seed worker) and commit with `feat(training): parallelize cross-sectional config search` (`77e8401`).

### Task 4: Verify the supported surface

**Files:**

- Modify: this plan to check completed steps and record exact commands/results.

- [x] Run `python -m pytest tests/unit/training/test_reproducibility_seeds.py tests/unit/training/test_parallel_search.py tests/unit/training/test_search_resume.py tests/unit/scripts/test_run_real_candidate_training.py -q` (`108 passed`).
- [x] Run `python -m pytest tests/unit/evaluation -q` (`72 passed`; two pre-existing pandas FutureWarnings).
- [x] Run `ruff check src tests scripts/run_real_candidate_training.py` (`All checks passed!`).
- [x] Run `git diff --check`, record exact results, and commit with `docs(training): record parallel search verification`.

### Task 5: Route the M2 temporal search through the same scheduler (2026-09-10 increment)

**Why:** Tasks 1-4 left M2 serial because of its temporal-batch contract. M2 is also the family
that cannot afford to stay serial -- measured ~8.6 h per fit and ~161 days for the family, so it is
the one search that will never finish without concurrency. The blocker was never the scheduler
(index-addressed, family-agnostic); it was the payload. A temporal batch holds a 20-day window per
row and the validation panel is train+validation concatenated, so per-task batches would be
gigabytes each.

**Files:**

- Add: `research_v2/src/research_v2/training/search_metrics.py`, `research_v2/src/research_v2/training/temporal_search.py`, `research_v2/tests/unit/training/test_temporal_search.py`.
- Modify: `research_v2/src/research_v2/training/cross_section_search.py`, `research_v2/scripts/run_real_candidate_training.py`, `research_v2/tests/unit/scripts/test_run_real_candidate_training.py`, the design spec.

**Interfaces:**

- `search_metrics.validation_rank_ic(predictions, labels) -> float` -- the number the winning config is selected by, in one place. It had two copies (the driver's serial loops and the M1/M3 worker); three would have been three chances for "the model the search selected" to mean something different from "the model the parent fits".
- `temporal_search.TemporalFoldSpec / TemporalFoldBatches / TemporalConfigTask`, `temporal_fold_batches(spec)` (cached per worker), `temporal_panel_batches`, `temporal_model`, `run_temporal_config_task(payload)`.
- `HISTORY_LENGTH = 20` / `MINIMUM_VALID_HISTORY_LENGTH = 15` -- the window contract, imported by the driver's serial `_temporal_panel_batches` too.
- `_temporal_search_budget(max_epochs)` -- M2's record budget: fit length **plus the window**, because a 20-day and a 10-day window are different models and neither is in the config payload.

**Steps:**

- [x] Write failing tests: the worker rebuilds bitwise the batches the parent would build (against a real published dataset artifact), the validation window reaches back into train (day 21, the split's first day, has a batch only if the panel was concatenated -- and day 15 as the first eligible train date pins that the 15-day minimum counts the signal date itself), one fold is built once per worker, the task payload stays under a 100 KB budget, GRU/TCN follow the config, and a real spawned worker returns a complete three-seed result.
- [x] Run them and confirm the missing-module failure.
- [x] Implement `search_metrics` and `temporal_search`; point `cross_section_search` at the shared metric and delete its private copy.
- [x] Route the M2 search through `run_parallel_search` when `--parallel-workers > 1`, adopting and persisting search records exactly as the M1/M3 parallel path does, and restore grid order through the existing `_ordered_search_results` before the existing `max` selection.
- [x] Bind the window contract into M2's budget and make the driver's serial panel builder read the same constants, so serial loop, spawned worker and records cannot disagree about the window.
- [x] Allow M2 in `_parallel_execution`; keep M0/M0R refused, with the reason in the message.
- [x] Mutation-check: hardcoding the driver's window back to a literal, and dropping `history_length` from the budget, each kill exactly the test that pins them.
- [x] Full gate: `ruff check src tests` clean, `mypy src` **169 files** clean, `pytest tests -q` **1997 passed / 45 skipped**.

**Still open after Task 5:** the FINAL three-seed fit was still serial, and on a fold whose grid
already saturates the workers it was the only serial part left -- three fits out of a 24-config M2
fold's ~52 hours, i.e. half the fold at M2's measured ~8.6 h per fit. Taken by Task 6 below.

### Task 6: Parallelise the final three-seed fit (2026-09-10 increment)

**Why:** config-level parallelism accelerates the search only. The three final fits that produce the
frozen bundle ran serially afterwards in every family, so a fold whose 24-config grid already filled
the workers still paid three full fit-times at the end. They are independent by construction -- same
config, three seeds, `REQUIRED_SEEDS` is a fail-closed contract -- and the parent does not need the
models, only their checkpoint paths and their validation scoring.

**Interfaces:**

- `cross_section_search.CrossSectionSeedTask` / `run_cross_section_seed_task`, `temporal_search.TemporalSeedTask` / `run_temporal_seed_task`, both returning `{seed, checkpoint_path, predictions, validation_rank_ic}`.
- `_fit_one_seed` / `_fit_one_temporal_seed`: one seeded fit, defined ONCE and used by both the config-level worker (which loops the three seeds itself) and the seed-level worker (which the parent loops across processes). Returning the checkpoint path is what lets the parent freeze a bundle from a fit it never ran.

**Steps:**

- [x] Write failing tests: a spawned seed worker hands back a checkpoint that exists on disk under the same `<prefix>-seed-<seed>-epoch-<n>` name the serial path produces, and three spawned seed fits reassemble through `build_seed_ensemble` into a real ensemble with three distinct checkpoints -- for the static families and for M2.
- [x] Refactor both config-level workers onto the shared single-seed fit; their existing spawned-worker tests stay green unchanged, which is the evidence the refactor preserved behaviour.
- [x] Route the final fits of `_run_full_cross_section` (M1/M3) and `run_full_m2` through `run_parallel_search` when `--parallel-workers > 1`, indexed by seed; the serial path stays the default and keeps the identical checkpoint prefixes, so a bundle's provenance cannot depend on the worker count.
- [x] Mutation-check: renaming the parallel branch's checkpoint prefix kills exactly the test that pins serial/parallel prefix equality.
- [x] Full gate: `ruff check src tests` clean, `mypy src` 169 files clean, `pytest tests -q` **2004 passed / 45 skipped**.

**Taken by Task 7 below:** seed-level granularity for the SEARCH.

### Task 7: Seed-level search granularity for narrow grids (2026-09-10 increment)

**Why:** with `--max-configs 1` -- the shape of every zero-tuning comparison run, and what M0R's
+0.009149 was measured at -- there is one config per fold, so `capped_worker_count` returned 1 and
the search stayed serial even with `--parallel-workers 48`. Task 6 parallelised the final fits, which
took such a run from six serial fits per fold to four; splitting the search's unit of work to
(config, seed) takes it to two parallel rounds, i.e. **3x**. At the measured ~2.05 h per fit
(m1_v2: 18 fits in 37 h) that is 8 fit-times per run instead of 24 -- **~16 h instead of ~49 h** --
which is what makes an end-to-end validation of the parallel path one overnight run rather than two
days. Three workers is the ceiling here: a one-config fold holds only three seeds, so the remaining
lever for this run shape is fold-level parallelism, not more workers.

**Rule:** per fold, adopt what the search records cover, then use config-level tasks when the
remaining configs can fill the workers (`len(unadopted) >= workers`) and seed-level tasks otherwise.
Config-level keeps the smaller return payload (scalars only); seed-level returns each seed's
validation predictions so the parent can build the ensemble, which is affordable precisely because
it is chosen only when there are few configs. Both write the same records and the same checkpoint
names, so a workspace does not record which granularity produced it.

**Interfaces:**

- `_ensemble_from_seed_fits(fits, config_payload, validation_labels) -> dict` -- rebuilds one config's
  search result from seed fits that ran in other processes, in the serial loop's exact shape, and
  refuses a seed set that is not exactly `REQUIRED_SEEDS`.
- `_constant_budget(budget)` -- a named factory for the torch families' config-independent budget,
  replacing a lambda whose type mypy could not infer at a binding site.

**Steps:**

- [x] Write failing tests for `_ensemble_from_seed_fits`: the result shape and its four keys, `worst_seed`
      is the lowest-RankIC seed, an incomplete seed set is refused, an unknown seed is refused.
- [x] Implement the seed-level branch in both torch search loops, indexed `config_index * len(REQUIRED_SEEDS) + slot`
      and decoded by integer division, so grid order is restored by the existing `_ordered_search_results`.
- [x] Mutation-check: weakening the seed-set guard to a subset test turns the refusal into a downstream
      `KeyError`, and the incomplete-seed test fails.
- [x] Clean the driver's mypy output while in it: `MYPYPATH=src mypy scripts/run_real_candidate_training.py`
      now reports **zero errors in the driver** (was 13 -- six from today's parallel wiring, six pre-existing
      `ModelWalkForwardResult` name reuse, plus the third-party `dolphindb` stub note which is not ours).
      The gate is `mypy src`, which never covered `scripts/`, so none of these were being caught.
- [x] Full gate: `ruff check src tests` clean, `mypy src` 169 files clean, `pytest tests -q`
      **2008 passed / 45 skipped**.

**End-to-end validation launched 2026-09-10 21:52.** The library level had real spawned workers
(static and temporal, config-level and seed-level, including three-seed reassembly through
`build_seed_ensemble`) and the driver level had source-wiring assertions, but neither is a run. So a
full real-data M1 run was started on the parallel path: `--family M1 --parallel-workers 3
--torch-threads 1 --factor-source frozen_v2 --max-configs 1`, workspace `m1_parallel_v1`, with
`QRV2_MODEL_REGISTRY_ROOT` pointed at a scratch directory under `/tmp` so nothing it produces can
reach the real registry, and the serial `m1_v2` run in flight left untouched. Three spawned workers
were observed training F1's three search seeds concurrently within six minutes of launch.

What that run does and does not settle. It validates the machinery end to end on real data. It also
starts a **new numeric lineage** -- threads pinned to 1 and recorded in every search record's
runtime descriptor, which is the substance of adjudication 2 in the compute brief -- so its numbers
are comparable to themselves and NOT to `m1_v2`'s, and promoting anything from it still needs that
adjudication recorded. The serial run remains the one whose verdict answers the M1-vs-M0R question.

## 实测记录（2026-09-10 22:00–22:30，`m1_parallel_v1` 与串行 `m1_v2` 同时在跑）

**1. 线程数对速度中性，在真实 v2 数据上复现。** 串行运行（52 线程）相邻 epoch 检查点间隔 5m37s
（21:08:28 → 21:14:05 → 21:19:41）；并行运行的三个 worker（各 1 线程、三路并发）epoch-1 检查点落在
22:04:08 / 22:04:38 / 22:06:41，即单 epoch 约 4–6 分钟。两者同量级，所以 3× 加速不是靠降线程换来的，
也没有被降线程吃掉。计划 §3.4 早前的基准是在旧的小折上测的（1/4/8 线程 81.56–81.80 s），这是第一次
在 230 万行的 v2 折上验证同一条结论。

**2. 种子级并行在真实数据上跑通。** 启动 6 分钟内观测到三个 spawned worker 并发训练 F1 的三个搜索 seed，
各自写出自己的检查点（`m1-f1-c0-<seed>-seed-<seed>-epoch-N`），互不覆盖。

**2b. 三路并发下单拟合速度没有下降（2026-09-10 23:05，直接比对检查点时间戳）。** 同一个折 F1、同一个
seed 20260801、同一个配置：并行运行（3 个 worker、各 1 线程）相邻改进检查点的间隔为 5m31s / 5m32s /
5m33s / 5m33s（最后一段跨 epoch 4→6，即两个 epoch）；串行运行（52 线程）在同一折同一 seed 上为
6m / 5m / 6m / 5m（epoch 4→6 同为两个 epoch）。两者同为约 **5.5 分钟/epoch**，即 **3 路并发、每路
1 线程的单拟合降级约 1.00×**。这与计划 §3.4 早前在旧的小折上标定的结果（16 路 1.14×、48 路 1.30×）
一致，也是「M2 可以开到内存允许的上限（20 路）」这个结论的前提——若单拟合速度随并发度显著变慢，20 路的
收益会被降级吃掉一部分。但这一条**不能直接外推到 M2**：M2 的单 worker 面板更大（约 21.6 GiB）、重建更久
（40 分钟），其并发降级尚未实测，应在 M2 首跑前用小规模实测标定，而不是按静态家族的曲线外推。

**3. 钉线程确实生效，并且写进了证据工件。** worker 的 checkpoint manifest 记录
`torch_intraop_threads: 1`、`torch_interop_threads: 1`；串行运行同一位置记录 52。

**4. 两个谱系的差异被量化了——这是裁决 2 的关键输入。** 同一折、同一配置、同一 seed、同一 epoch：

| 谱系 | epoch-1 验证 RankIC | checkpoint sha256（前 12 位） |
|---|---|---|
| 并行，threads=1 | 0.049795006597773174 | `85e9b4a35585` |
| 串行，threads=52 | 0.04979499815492084 | `f1205743b37a` |

指标相差 8.4e-9，而权重摘要完全不同。**但上面这段的结论当晚就被自己的后续测量推翻了，更正如下——
epoch 1 上的 8.4e-9 是起点，不是终点。**

逐 epoch 比对同一折、同一配置、同一 seed 的两条谱系，差异按每轮约 20–45 倍放大：

| epoch | 串行（52 线程） | 并行（1 线程） | 绝对差 |
|---|---|---|---|
| 1 | 0.04979499815492084 | 0.049795006597773174 | 8.4e-09 |
| 2 | 0.05117260526119129 | 0.05117277553763415 | 1.7e-07 |
| 3 | 0.05340960379713739 | 0.05340199099426342 | 7.6e-06 |
| 4 | 0.0570977422920666 | 0.05682709100317665 | 2.7e-04 |
| 6 | 0.0582180901098924 | 0.05771882804489137 | **5.0e-04** |

到 epoch 6 相差 5.0e-4，约为指标本身的 1%。这是**放大**而不是缺陷：训练是混沌系统，浮点求和顺序的微小
差别会被逐轮指数放大。更要紧的是**早停轨迹也会分叉**——同一折的 seed 20260802，串行在 epoch 8 之后再无
改进（patience 10，即停在 18），并行到 epoch 13 仍在改进；seed 20260803 分别是 14 与 16。所以两条谱系
产出的**不是同一个模型的不同字节，而是不同的模型**。

更正后的三条含义：

1. 「科学结论上两者等价」**不成立**。正确说法是：两条谱系是同一程序在不同浮点求和顺序下的两个不同实现，
   其验证 RankIC 相差约 1e-3 量级、早停点可以不同。G4 门槛是 5e-3，所以这个量级约为门槛的 **20%**
   （**2026-09-11 已在完整折上实测：集成口径 1.65e-3，即门槛的 33%，见下**）——
   对一个恰好落在门槛附近的候选，谱系选择有可能影响判定（M0 当初正是以 4.2e-3、差 0.8e-3 而失败）。
2. 工件上不可互换（派生 model_id 不同、跨谱系不存在逐位复现）——这条不变。
3. 搜索记录绑定线程数因此不是过度谨慎而是**必需**：一条在 52 线程下产生的记录若被 1 线程的运行采纳，
   等于把两个不同模型的搜索结果混进同一个 `max` 里选配置。

**整折实测已到位（2026-09-11 00:45）：`m1_parallel_v1` 的 F1 搜索完成，与 `m1_v2` 的 F1 直接可比。** 这是
第一条**端到端、生产数据、同折同配置同三 seed** 的谱系差测量；此前只有逐 epoch 的放大曲线。可比性是核对过
的、不是假定的——两侧同一 F1 数据集工件 `226a8dfc497c2de6b07516711f058f26a9270675916c9e67dbf7f087a07d2538`
（train 2,296,573 / validation 696,906 / test 804,575 行，两侧逐一相同）、同一配置（串行 bundle 的
`config.json` 与并行搜索记录的 `config` 都是 `hidden_dim=64, num_blocks=1, dropout=0.0`）、同三 seed、
同 15 因子 `f00–f14`、同 `model_code_version=residual-mlp-1`、同 torch `2.6.0+cu124`、同早停预算
（patience 10 / max_epochs 100）。受控差别是 `torch_intraop_threads`：**串行 52（未钉，取宿主默认）vs
并行 1（显式钉住）**。严格说还并存着第二个差别——并行侧的拟合在 spawn 出来的 worker 进程里执行——它不改变
数值路径，但不宜当作不存在。

集成口径两侧都由同一句 `F1 config 1/1: ensemble validation RankIC=` 打印，且各自的值都另有一份落盘证据
（串行落在其签名的 model 选择证据 `validation_rank_ic`，并行落在其搜索记录 `ensemble_validation_rank_ic`）：

| 谱系 | F1 集成验证 RankIC |
|---|---|
| 串行 52 线程 | 0.05441212880837507 |
| 并行 1 线程 | 0.056057666322798204 |
| **差（并行 − 串行）** | **+1.6455375144231316e-03** |

**即 G4 门槛（5e-3）的 32.9%**，比上面按逐 epoch 曲线估的「约 1e-3、约 20%」更大。

分 seed 的早停指标（两侧 checkpoint manifest 的 `validation_rank_ic`，也就是早停实际使用的那个数，与上表的
集成口径**不是同一个量**，不可混用）：

| seed | 串行 52 | 最佳 epoch | 并行 1 | 最佳 epoch | 差 |
|---|---|---|---|---|---|
| 20260801 | 0.0582180901098924 | 6 | 0.05771882804489137 | 6 | −4.992621e-04 |
| 20260802 | 0.05796734071944032 | 8 | 0.05700524461384878 | **13** | −9.620961e-04 |
| 20260803 | 0.06535564684647174 | 14 | 0.06562460326049732 | **16** | +2.689564e-04 |

四点必须照实说：

1. **符号不一致**（两负一正），所以谱系差不是可以事后校正的系统偏移，而是量级与门槛同数量级的噪声。
2. **集成把差放大而不是平均掉**：分 seed 最大 |差| 是 9.6e-4（门槛的 19.2%），集成却是 1.65e-3（门槛的
   32.9%）。三 seed 集成**不是**谱系噪声的抑制器——这与「多 seed 求稳」的直觉相反，而且是实测的。
3. **样本量是 1 折 1 配置**，不是一个分布。它能确证的是「量级可达门槛的三分之一」，不能确证「平均而言就是
   三分之一」。要得到分布需要多折多配置，而那会是 M2 正式运行的副产品，不必专门跑一轮。
4. 同谱系内**重拟合是确定的**（正例）：串行侧搜索拟合 `m1-f1-c0-*` 与最终拟合 `m1-f1-final-*` 三个 seed 的
   最佳值与最佳 epoch 逐一相同（epoch 6/8/14，值到 16 位小数一致）。所以发散只发生在**跨**谱系，不发生在
   同谱系的重复拟合之间——这与「同宿主两次全量跑逐字节相同」那条既有性质一致。**并行侧已于 02:36 核对完毕，
   结论相同**：其 `c0` 与 `final` 三个 seed 的最佳值与最佳 epoch 也逐一相同（epoch 6/13/16）。于是两条谱系
   **各自内部确定、彼此不同**，这正是「不同谱系＝不同模型，而不是同一模型的不同字节」的完整证据；也说明
   00:45 那次读到 `final` 只到 epoch 4、值与 `c0` 不符，是拟合尚未跑完的快照，不是同谱系内的不确定。

**上面这一节的单折数字已被四折完整对照取代（2026-09-11 19:00，`m1_parallel_v1` 跑完 F4）。** 完整表与推导见
`docs/handoffs/2026-09-11-m1-evaluation-crash-lineage-and-footprint.md` 第 3 节。要点：F1 +1.646e-03（门槛的
32.9%）、**F2 −4.038e-03（门槛的 80.8%）**、F3 +1.057e-04（2.1%）、F4 +3.286e-04（6.6%）；符号 +−++、带符号
均值 −4.9e-04 ≈ 0；平均 |差| 1.53e-03 = 门槛的 30.6%；**四折差值的极差 5.684e-03 已超过 G4 门槛本身
（5e-3）**。即上面第 1 条说的「谱系选择有可能影响判定」不是可能性而是量级事实：谱系能造成的摆动比它要喂进的
那道门还大。

**更正（2026-09-11 00:31，实测推翻本节原有的「复现审计不受影响」结论）：复现审计**同样**受线程数影响。** 原文是「**但复现审计
不受影响，这是两件不同的事。** 发散只发生在训练；前向推理实测在 1/4/16/52 线程下逐位相同，所以一个线程谱系
下训练出的 bundle，可以在任何线程数的宿主上被精确复现」。那次实测的对象是**未训练**模型（当时钉住该性质的
测试就是这么建的），它在 1/4/16/52 下确实逐位相同；**为什么未训练就干净尚未定论**，能确证的只有「未训练
干净、已训练不干净」这个对照。对**已训练的生产 bundle** 重测：`m1_v2` F1（`RESIDUAL_MLP`，803,572 行）按
1/2/4/8/16/52 线程各跑一次完整前向：

| intra-op 线程 | 与 1 线程逐位相同 | max\|diff\| |
|---|---|---|
| 1 / 2 / 4 / 8 | 是 | **0.0** |
| 16 | **否** | 4.805844e-05 |
| 52 | **否** | 4.923683e-05 |

三点必须照实记下。**(a) 阈值在 8 与 16 之间**：1/2/4/8 四档差值全为 0，直接排除「误差随线程数连续累积」
的解释——分叉是在某一点上出现的，不是逐步长出来的。**(b) 16 与 52 彼此也不相同**（两个差值不等），所以
越过阈值后**不存在唯一的「正确分数」**，只有「该运行当时记录的那个线程数下算出的分数」。**(c) 5e-5 不是
末位舍入**，它是近似并列股票之间的**秩反转**，会改变选股名次。同一 bundle 的 F2 差 4.6777060529579195e-05
（881,863 行）、F3 差 1.2774655084313302e-04（931,411 行）。

机制侧已随之落地：`recorded_run_threads(workspace)` 从 checkpoint manifest 读 `torch_intraop_threads`
（一个工作区里混有多个值就拒绝，不猜），`verify_prediction_reproduction.py` 在进程最早处同时钉住两个线程
池，默认取该记录值（LightGBM 工作区无此记录，回落到宿主默认值——其 `num_threads` 早已钉死），并把
`scored_under_torch_threads` / `threads_requested` 写进 JSON。实测 `m1_v2` F1 自动钉到 52 →
`exit=0`、803,572 行、`max_absolute_difference=0.0`、`identical=true`、`reproduced=true`；同一折钉到 1 →
`exit=1`、差 4.923682914836913e-05。**后者不是缺陷，是审计在错误的线程数下打分的必然结果。**

对本计划裁决的含义因此**加强**而不是减弱：**钉到 1 线程的谱系既身份可移植、又可在任意宿主上被精确审计**
（任何宿主都能把自己钉到 1）；未钉住的谱系只能在与其同核数的宿主上审计通过。审计比对的仍是「存储的分数
是否确由这个 bundle 算出」，与 bundle 出自哪个谱系无关——这一句不变，变的是它成立的前提：必须在同一
线程数下问这个问题。

钉线程的代价仍是**一次性且单向的**：一旦选定某个值，此后所有运行都必须用同一个值，否则「两次全量跑逐
字节相同」这条既有性质就不再成立。注意那条性质目前的 5 份同 md5 报告全部来自 M0R（LightGBM，其
`num_threads` 早已钉死），所以它们不受本条影响；受影响的是从此往后的 torch 运行。

**5. 折级并行不是下一个杠杆（修正 Task 7 末尾的说法）。** 存储层已核实可并发写：选择证据与冻结预测都是
**内容寻址 + `tempfile.mkdtemp` 后 `rename` 原子落盘**，不同折写的是互不相交的 id，所以并发写在存储层是
安全的。但吞吐上不成立：总工作量与 worker 数都不变，配置级并行已经吃满机器时再按折拆分不会更快——24 配置
× 3 seed = 72 次拟合，无论切成 24 个配置任务还是 72 个种子任务交给 24 个 worker，都是 3 轮。折级并行只对
**窄网格**有意义（1 配置时一折只有 3 个 seed，3 个 worker 就是天花板，四折仍一个接一个），而窄网格是载体
对比运行的形状，不是 M2/M3 正式运行的形状。因此 M2 时长的真正杠杆是**能开多少 worker**，即每个时序 worker
的内存占用与重建成本。

**6. 时序 worker 的足迹与重建成本已实测（2026-09-10 22:03–22:45，只读 `m1_v2` 已发布的真实 F1 数据集）。**
`temporal_fold_batches` 在 230 万行 train / 70 万行 validation 的真实折上：**峰值 RSS 16.80 GiB**、
**一次性重建 2409 秒（40.2 分钟）**、产出 953 个训练批次与 237 个验证批次，单个训练批次的特征张量为
`(1, 1610, 20, 15)` float32。两条结论：重建相对 8.6 小时的单次拟合是 2.5%，且**每 worker 只做一次、
跨该折所有配置复用**——这正是当初选择"worker 侧自建"而不是逐任务 pickle 的原因；内存才是 worker 数的约束。

按各折已发布数据集的总行数等比折算（F1 实测 16.8 GiB 为基准），在 497 GiB 可用、90% 余量下：

| 折 | 数据集总行数 | 估算单 worker 足迹 | 安全 worker 数 |
|---|---|---|---|
| F1 | 3,798,054 | 16.8 GiB | 26 |
| F2 | 4,205,787 | 18.6 GiB | 24 |
| F3 | 4,582,910 | 20.3 GiB | 22 |
| F4 | 4,893,642 | 21.6 GiB | **20** |

`--parallel-workers` 是运行级参数、进程池按折创建，所以**约束取最大的折：20 路**。这也说明为什么需要一个
启动前的内存前置条件（Task 8）：48 路 × 21.6 GiB = 1037 GiB，在这台机器上不是启动失败，而是跑到第几小时
被 OOM 挑中。

**本节的两个数字已在 2026-09-11 被实测改写：16.8 GiB → 13.61 GiB、2409 s → 455.9 s，20 路 → 24 路。见第 7 条。**
上表按原样保留，因为它是当时据以做 Task 8 那些决定的证据；新的表在第 7 条，并且**口径不同**（第 7 条把父进程
计入，上表没有）。

### Task 8: 内存前置条件与按 makespan 选择任务粒度（2026-09-10 增量）

**为什么：** 第 6 条的实测把两件事从猜测变成算术。其一，worker 数受内存约束，而超限的失败方式是最贵的
那种——不是启动时被拒，而是数小时后被内核 OOM 挑中。其二，M2 的 24 个配置在 20 路下**不能整除**，于是
Task 7 里"未采纳配置数 ≥ worker 数就用配置级"这条规则是错的：配置级跑 `ceil(24/20)=2` 轮、每轮 3 次拟合
= **6 个拟合时间**；种子级跑 `ceil(72/20)=4` 轮、每轮 1 次 = **4 个拟合时间**。差的这一轮是因为一个配置
任务一旦派发就不能再拆，而多出来的 4 个配置照样占满一整轮。

**接口：**

- `parallel_search.available_memory_gb(path=/proc/meminfo)` —— 读 **MemAvailable 而非 MemFree**：本机 MemFree
  约 125 GiB 而实际可用约 497 GiB，读错那一行会把跑得下的配置判成跑不下，而一个总喊狼来了的护栏会被关掉，
  关掉之后它就什么也不护。
- `parallel_search.worker_memory_budget(...)` —— 返回判定**以及最大的安全 worker 数**，因为被拒之后操作者的
  下一个问题永远是"那到底能开几个"。
- `parallel_search.granularity_makespans / search_granularity` —— 两个候选 makespan（以拟合时间计）与其择优，
  打平时选粗粒度（任务少三分之二、返回标量而非每 seed 的预测帧、每配置一个 worker）。
- 驱动 `--worker-memory-gb`：声明每个 worker 的足迹即成为启动前的硬前置条件；不声明则只把余量打印进转录，
  **不让既有调用方式开始失败**（这是一条新增前置条件，不能顺手把别人的运行弄崩）。

**步骤：**

- [x] 失败测试先行：MemAvailable 与 MemFree 的区分、宿主不提供时为 None、预算的 fit/不 fit/最大安全数、
  90% 余量的边界、退化输入拒绝。
- [x] 实现 `available_memory_gb` / `worker_memory_budget` / `granularity_makespans` / `search_granularity`。
- [x] 驱动接 `--worker-memory-gb` 与 `_check_worker_memory`（`main()` 内、钉线程之前，一次运行只查一次），
  并把两个 torch 搜索循环的粒度判据从"填满 worker"换成 makespan 择优。
- [x] 一条测试预期被实现推翻后**改的是测试而不是实现**：24 配置 / 48 worker 时配置级是 1 轮×3 次拟合，
  种子级是 `ceil(72/48)=2` 轮×1 次——种子级更快，因为配置级会让 24 个 worker 闲着。配置级只在打平时凭
  tie-break 胜出。
- [x] 全量门禁：`ruff check src tests` 干净、`mypy src` 169 文件干净、驱动脚本 mypy 0 错误、
  `pytest tests -q` **2039 passed / 45 skipped**。

**据此重算 M2 的时长**（单次拟合取计划实测的 8.60 h，worker 数取上表的 20）：搜索 4 轮 × 8.6 h ≈ 34.4 h，
加每 worker 一次 40 分钟重建；最终三次拟合 1 轮 ≈ 8.6 h，加新一批 worker 的 40 分钟重建 → **单折约 44 h，
四折约 7.4 天**。若仍按"填满 worker"选粒度则是单折约 61 h、四折约 10 天——**这条规则本身值 2.6 天**。
对照计划里的串行估算（约 161 天），20 路是约 20 倍。

**这段时长算术的输入已在 2026-09-11 变了（见第 7 条）：worker 数 20 → 24、重建 40 分钟 → 7.6 分钟。** 结论
不是等比例变好，而是**台阶式**的：24 配置在 24 路下恰好一轮跑完，搜索从 4 个拟合时降到 3 个（34.4 h → 25.8 h），
而 23 路与 19 路完全相同。单折约 34.7 h、四折约 5.8 天——**前提是发射时 MemAvailable ≥ 495.1 GiB**。

**顺带否掉一个看似划算的优化：** 一折内复用同一个进程池（让最终拟合的 worker 命中搜索阶段的面板缓存）只省
每折约 40 分钟——因为最终拟合只有 3 个任务、只会用到 3 个 worker——占单折 44 小时的 1.5%。已测得重建成本后
就不该做，而不是先做了再解释。

---

## 实测记录（2026-09-11 00:31–02:26）

**7. 第 6 条点名的杠杆已经拉过一次：时序批次构造被重写，worker 足迹与重建成本都实测下降，而批次逐位不变。**

`build_temporal_training_batches` 原来把整折因子装成 `history_by_code`——每个证券一个列表、每一行一个
`(date, dict[15 个 float])`，即**每输入行一个 Python dict 加 15 个 Python float**；同时**逐单元**调用
`pd.to_numeric`（一个为数组设计的函数）、逐单元做 numpy 标量赋值，并且**每个 signal date 都把每个证券的历史
从头重扫一遍**。实测（合成面板 1,200 行 / 4 因子）：追踪到的堆峰值是它自己产出的批次体积的 **5.0 倍**。

改法是把因子放进一个 `(rows, factors)` 的 float64 数组、按索引取窗口。帧本来就按 `(ts_code, signal_date)`
排序，所以一个证券在某 signal date 及之前的观测就是**一段连续行**：`row_of[code_id, date_id]` 加上该证券的行
起点即可直接定位窗口，一个 `codes × dates` 的整数矩阵取代了那些逐行 dict。历史不足 `history_length` 的证券，
其空槽指向第 0 行并被掩码掉——旧实现从不写这些单元，它们保持初值 `0.0/False`，这一点必须照实复刻而不是
"顺手统一"。

**逐位不变是硬要求，不是加分项。** 一个挪动了单个 bit 的构造器等于开出第三条谱系，而今天刚测到谱系差已达 G4
门槛的三分之一（第 4 条）。每个进张量的值仍然是从同一源单元取出的 float64、仍然只向 float32 **转换一次**；
秩仍然在 float32 的目标数组上取；`+inf` 仍然与 NaN 走同一条 missing/0.0 分支。这条等价被钉成测试
（`tests/unit/training/test_temporal_batches_footprint.py`）：把旧算法**逐字冻结**在测试里当参照，对每个批次的
每个张量比**原始字节**（`torch.equal` 会跳过 NaN 位型与带符号零，所以不能用它），面板刻意覆盖每一条分支——
参差的历史长度、无效标签、非有限的目标、NaN 与 `+inf` 因子、并列目标、太薄而不成立的日期、`pad_to`——跨 4 组
参数。另一条测试钉住足迹本身：追踪峰值必须低于自身产出体积的 4 倍，旧实现在 5.0 倍处失败。

**并且在真实折上核对过，不是只在合成面板上。** 只读 `m1_v2` 已发布的 F1 数据集（`226a8dfc…`）、同一份已拟合
预处理状态、1 线程，新旧两个构造器各跑一遍，对每个批次的每个张量字节按日期序做 sha256：

| | 旧构造器 | 新构造器 |
|---|---|---|
| train 批次 sha256 | `dd325a00ab96317bd37965321b54cb8886443704cedff1a8009935e8e1464d0c` | **相同** |
| validation 批次 sha256 | `b9486f115d68861ec88f36624fbdb2ca6ae04c7f84b69af18584e19c4ef8e918` | **相同** |
| train / validation 批次数 | 953 / 237 | 953 / 237 |
| 张量字节数（train / validation） | 3,413,926,736 / 1,056,687,313 | 相同 |
| 单个 train 批次特征张量 | `(1, 1610, 20, 15)` float32 | 相同 |
| **纯构造耗时** | **2027.0 s** | **65.5 s（30.9 倍）** |

同一脚本跑旧构造器时 `加载 + 预处理 + 构造 = 376.9 + 3.3 + 2027.0 = 2407.2 s`，与第 6 条当时记录的
**2409.0 s** 相差 **0.075%**——即这次对照复现了原测量，两组数字可以直接比。

据此更新第 6 条（都走真实入口 `temporal_fold_batches`、同一 F1 数据集、1 线程）：

| 量 | 2026-09-10 实测 | 2026-09-11 实测 | 变化 |
|---|---|---|---|
| `temporal_fold_batches` 全程 | 2409.0 s | **455.9 s** | **5.28×** |
| 其中纯构造 | ~2029 s | **65.5 s** | **30.9×** |
| 其中两次已验证数据集加载 | ~377 s | ~379 s | 未触碰 |
| worker 峰值 RSS | 16.80 GiB | **13.61 GiB** | −19.0% |
| **父进程**峰值 RSS（折批次 + TEST 面板批次） | 从未实测 | **19.15 GiB** | 新增实测 |

端到端只有 5.28× 而不是 30.9×，因为剩下的约 379 s 是两次 `load_split`：每次都要把所有分片字节读一遍、sha256
一遍、再按 `build_split_dataset_manifest` 重算内容哈希。**那一段在证据链上，本次没有动。**

**父进程 19.15 GiB 是这次唯一新增的实测量，它把 worker 数算术里最后一个靠猜的量去掉了。** 护栏
`worker_memory_budget` 的算式本来就是 `safe = (0.9 × MemAvailable − 父进程) // 单 worker`，但父进程此前只有
"可能比任一 worker 大"这句定性描述。现在 F1 上它是 19.15 GiB，比 worker 的 13.61 GiB **大 41%**——因为它还多
一套 242 个 TEST 批次与三段拼接面板，三套批次张量合计 5,689,592,480 字节。

按各折已发布数据集总行数等比折算（F1 实测为基准，与第 6 条同一方法），MemAvailable 497 GiB、90% 余量、
**父进程计入**：

| 折 | 数据集总行数 | 单 worker 足迹 | 父进程足迹 | 安全 worker 数 | 第 6 条原值（不计父进程） |
|---|---|---|---|---|---|
| F1 | 3,798,054 | 13.61 GiB | 19.15 GiB | 31 | 26 |
| F2 | 4,205,787 | 15.07 GiB | 21.21 GiB | 28 | 24 |
| F3 | 4,582,910 | 16.42 GiB | 23.11 GiB | 25 | 22 |
| F4 | 4,893,642 | 17.54 GiB | 24.67 GiB | **24** | 20 |

最后一列是**不计父进程**的口径，所以不要把它当成同一算式的前后对比；不计父进程时新值是 32/29/27/**25**。

**这里有一道必须写明的台阶，否则收益会算错。** M2 是 24 配置 × 3 seed = 72 次拟合，makespan 对 worker 数是
**阶梯函数**：

| worker 数 | 配置级 | 种子级 | 择优 | 搜索时长 |
|---|---|---|---|---|
| 19（旧） | 2 轮 × 3 = 6 拟合时 | 4 轮 × 1 = 4 拟合时 | 种子级 | 34.4 h |
| 23 | 2 轮 × 3 = 6 | 4 轮 × 1 = 4 | 种子级 | **34.4 h（与 19 路完全相同）** |
| **24** | **1 轮 × 3 = 3** | 3 轮 × 1 = 3 | 打平取粗 | **25.8 h** |
| 36 | 1 轮 × 3 = 3 | 2 轮 × 1 = 2 | 种子级 | 17.2 h |

所以**足迹下降只有在把 worker 数抬过 24 这道台阶时才换得到搜索时长**；23 路与 19 路的搜索 makespan 一模一样。
而 24 路 × 17.54 GiB + 父进程 24.67 GiB = **445.63 GiB**，对 447.3 GiB 的余量只剩 **1.7 GiB**，等价于要求
MemAvailable ≥ **495.1 GiB**。36 路需要 656.1 GiB，超过本机 660 GiB 总量，**不可达**——24 路就是这套足迹下的
天花板。

三条可执行结论：

1. **发射 M2 前先看 MemAvailable。** 这台机器是共用的：02:00 前后另一位租户的 4 个进程占着约 86.6 GiB，
   MemAvailable 只有 488.5 GiB，此时 F4 的安全 worker 数是 **23**，24 路会被护栏拒。要么等宿主空下来，要么
   接受 19–23 路——后者搜索时长不变，但仍白拿下面第 2 条。
2. **不依赖台阶的收益也是实的**：每 worker 一次性重建从 2409 s 降到 455.9 s。一折要建两次进程池（搜索一次、
   最终拟合一次），所以每折省约 **65 分钟**墙钟、四折约 **4.3 小时**；重建相对一个配置任务（3 次拟合、25.8 h）
   的占比从 **2.6% 降到 0.5%**。这让"折级并行不值"与"复用进程池不值"两条结论更成立，而不是被推翻。
3. **下一个杠杆在证据链上，因此它是裁决而不是优化。** 单 worker 的 13.61 GiB 里，批次张量只占 4.47 GiB，两次
   `load_split` 的峰值占 8.63 GiB。用 spy 包住 `pd.concat` 实测：一次 `load_split(train)` 里 5 次 concat 合计
   **0.7 s**，其中那次在 split 口径下根本用不到的 `combined` 全量拼接只占 **0.27 s / 0.54 GiB**——所以"顺手删掉
   多余 concat"**不是杠杆，已实测否掉**。真正的大头是 concat 之后把 RSS 从 3.02 GiB 推到 8.63 GiB 的那一段，
   即 `build_split_dataset_manifest` 的规范化与内容哈希重算（外加最后那次 `copy(deep=True)`）。要再降足迹就得动
   它，而动它等于改变"每个工件字节都被读过并校验过"这条性质的实现方式——**这属于所有者的裁决，不该顺手做**。
