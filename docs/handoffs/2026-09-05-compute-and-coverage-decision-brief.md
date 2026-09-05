# 算力口径与数据覆盖决策简报（遗留链 / 冻结集链共用）

> 生成时间：2026-09-05 19:05 CST（Asia/Shanghai）
> 分支 `codex/publish-quant-platform` · 状态：**三项待人工裁决**
> 本简报只记录实测事实与选项；除已注明的两项（指针登记、Task 2.2 测试）外未变更任何运行中的链。

## 1. 一句话结论

两条链（遗留 14 / 冻结 17）当前**都不能按现计划产出正式证据**：冻结集缺 2014–2019 数据导致
F01–F06 全部无法装配；遗留集虽能跑但实测需 **42.6 天/家族**（计划写 6–24 小时）。
同时实测发现真正的提速杠杆**不是 GPU**（GPU 反而慢 2.9 倍），而是**单线程 × 约 100 路并发**
（M1 可从 42.6 天降到约 10 小时），但该杠杆与一个**可复现性缺口**（线程数未钉住、未记录）
绑定，必须先补口径再并发化。

## 2. 实测发现

### 发现 1（阻塞）：`factor_daily` 只覆盖 2020–2025，六折训练窗口全部缺口

活库只读查询：全表 `min(signal_date)=2020-01-02`、`max=2025-12-31`、`68,363,458` 行，
`where signal_date < 2020.01.01` → **0 行**。而已批准折定义
（`datasets/splits.py::annual_folds`，2014–2026 不可变研究口径）需要：

| 折 | train 年 | validation | test | 冻结集可装配性 |
|---|---|---|---|---|
| F01 | 2014–2018 | 2019 | 2020 | **train/validation 全空 → 该折完全不可用** |
| F02 | 2015–2019 | 2020 | 2021 | **train 全空 → 不可用** |
| F03 | 2016–2020 | 2021 | 2022 | train 仅 1/5 年 |
| F04 | 2017–2021 | 2022 | 2023 | train 仅 2/5 |
| F05 | 2018–2022 | 2023 | 2024 | train 仅 3/5 |
| F06 | 2019–2023 | 2024 | 2025 | train 仅 4/5 |

旁证：M1 已发布的 F01 数据集实测分片为 `split=train/year=2014..2018`、
`validation/year=2019`、`test/year=2020`（`manifest.json` 的 `shard_hashes` 逐片列出），
与上表一致。

**危险性**：装配用 `filter_current(...)`，未登记月份 → `expected.isna()` → **整段静默丢弃**，
不报错。因此缺口会产出空/残缺训练集而非显式失败。上一轮「装配源活库端到端验证」通过，
是因为它只验了 **F01/test（2020）**——恰是唯一落在覆盖区内的一段。
`materialization_status.md` 的「可进入训练」结论对 2020–2025 成立，对 walk-forward **不成立**。

**修复（无需新代码）**：`scripts/materialize_frozen_factor_daily.py` 已支持
`--start/--end/--factors/--apply` 且按 (factor_id, 窗口) 幂等。**必须按年分 6 次运行**
（2014…2019），因为 `data_version = sha256(watermark + start + end)` 由窗口决定——
这正是现存 6 个版本恰好一年一个的原因；单窗口跑 6 年会破坏既有粒度与按年幂等。
补齐后再对 2014-01..2019-12 追加 72 条指针登记（复用同一登记脚本）。
成本参照：2020–2025 六年实测约 7 小时 → 2014–2019 同量级 **约 7 小时**；
Tushare token 在仓库根 `.env`，本地 `artifacts/tushare_cache` 已 3.6 GB 且含 2015 年切片。

### 发现 2：CPU 实测 42.6 天/家族，而 GPU 是反向优化，真正杠杆是并发

在跑的 M1（pid 1137022，17:35:17 启动）检查点时间戳直接测量：
epoch-1 @17:48:17 → epoch-8 @18:35:39（validation RankIC 0.1029→0.1057）
→ **稳态 6.77 分钟/epoch**。拟合量 = 每折 27 configs × 3 seeds + 3 次最终 = 84，× 6 折 = **504 次**；
`--max-epochs` 默认 100（启动脚本未覆盖）、patience 10，best@8 → 至少跑到 epoch 18
→ **≈2.03 小时/拟合 → M1 ≈ 42.6 天**（config0 是 27 格中最小格，故为下界；停于 25/30 epoch 则 59/71 天）。

独立进程基准（真实数据 + 驱动自身代码路径，1 epoch，245 train 日 / 60 val 日，10,369 参数）：

| 设备 / 线程 | 单 epoch 秒 | ms/train 日 | checkpoint sha256（前 8） |
|---|---|---|---|
| CPU threads=1 | 81.76 | 333.7 | `1b1f74b7` |
| CPU threads=4 | 81.80 | 333.9 | `d72bc345` |
| CPU threads=8 | 81.56 | 332.9 | `38f78b86` |
| **CUDA:1（RTX 4090）** | **238.62** | **973.9** | `9735f01b` |

交叉校验：M1 实测 406 s / ~1220 train 日 = 333 ms/日，与 CPU 基准吻合（互证）。

1. **GPU 慢 2.9 倍**：小模型 + 逐日小 batch 属派发/启动开销受限，非 FLOP 受限 →
   「启用 4 张 4090」对 M1 是反向优化。
2. **线程数对速度中性**（<0.3% 差异）→ 单次拟合无法靠多线程加速，
   但 **104 核主机上可并发约 100 个拟合、每个 1 线程而单次不减速**
   → M1 ≈ 504 × 2.03 h / 100 ≈ **10 小时**，对比串行 42.6 天 ≈ **100 倍**。
   前提：并发化必须保持并列时的 canonical 选择顺序（E 计划已有
   `sha256(model_family|fold|config_json)` 排序约定可复用），否则选中的 config 可能改变。
3. **M2（GRU/TCN）/M3（set-attention）未基准**：逐日计算量更大，E 计划本为注意力模型设计了
   「不同日期打包只用于 GPU 并行」，故 GPU 结论对这两家族**可能相反**，需另测。

证据全文：`/data/zhangyudong/qrv2_research_artifacts/device_thread_benchmark.txt`。

### 发现 3：线程数未钉住、未记录 → 可复现性契约缺口

同 seed/config/数据在 threads=1/4/8 下产生 **3 个不同 checkpoint sha256**（见上表），
而 torch 训练路径**既不钉住也不记录**线程数：全仓仅 `baselines/lgbm_ranker.py:374`
钉了 `num_threads: 1`（说明项目已知此事要紧）；`training/seeds.py::configure_reproducibility`
记录 seed / torch_version / cuda 可用性 / cuda 设备名，**不含线程数**；checkpoint manifest 亦不含。
故候选模型数值当前依赖未钉住的主机属性（本机 `torch.get_num_threads()` = 核数/2 = 52），
同一命令在不同核数主机上会给出不同结果——与主计划 §4.3、E3.5 的确定性要求相冲突。
**任何并发化改造必须同时显式钉住线程数并写入证据**，否则会把该缺口放大约 100 倍。

### 附带更正：折匹配工件目录名是硬编码的（原路径隔离方案不可达）

`run_real_walk_forward.py:194` 为 `save_baseline_oos_artifacts(result, args.report.parent / "baseline_oos")`
——工件目录 = **报告父目录 + 固定名 `baseline_oos`**。故重建计划原写的
`baseline_oos_report_frozen.json` / `baseline_oos_frozen/` 不可达：若把冻结集报告写在
`qrv2_research_artifacts/` 下，工件会落进 `qrv2_research_artifacts/baseline_oos/`，
**正是遗留链 M2/M3 以 `--baseline-artifacts` 消费的目录**，会静默覆盖 F01–F06.parquet
并把冻结集基线预测喂给遗留候选（污染 G4 增量）。
**正确做法（零代码改动）**：冻结集报告放独立子目录，如
`--report .../qrv2_research_artifacts/frozen/baseline_oos_report.json` → 工件自动落
`frozen/baseline_oos/`；Task 5.1 的 `m{1,2,3}_frozen` 工作区同理置于 `frozen/` 子树下。
同一机制让 Task 2.4 的复现性核验可安全并行（`--report /tmp/legacy_flagcheck/report.json`）。

## 3. 本轮已完成（均已提交推送）

1. **Task 1.7 指针登记已执行并复核**：72 条 `published_version` 指针（2020-01..2025-12，
   6 个按年 data_version）。执行前 72/72 月 dry-run 全 `REGISTER`、0 拒绝、行数合计
   68,363,458 与物化清单逐位相符。执行后复核 72 分区精确齐全；三张遗留表仍各 151 分区、
   版本值未变；M1 进程与 watcher/编排器全程未受影响。
   **对遗留链零影响的机制证明**：两个在跑的驱动只解析
   `("universe_daily","factor_observation_clean","label_daily")`
   （`run_real_candidate_training.py:908-912`、`run_real_walk_forward.py:142-145`），
   且 `current_versions` 以 `where object_name='<表名>'` 过滤；分区键格式与消费端
   `filter_current` 的 `strftime("%Y-%m")` 及三张遗留表既有格式一致。
   执行依据：指针仅断言已核验为真的事实、幂等（重跑 → SKIP_CURRENT）、不写任何因子数据行，
   且为选项 A/B 共同必需、在选项 C 下为惰性未用元数据——不预判 A/B/C 裁决结果。
2. **Task 2.2 已交付**：`test_frozen_set_satisfies_every_model_yaml_factor_bound`
   （遍历 `configs/models/*.yaml` 断言 `min ≤ 17 ≤ max`，并断言 `min > 14` 保留可解释对照）。
   `tests/unit/scripts` **397 passed**；该文件 `ruff check` 0 error（顺带修掉 HEAD 就存在的
   1 个 SIM300）。补充事实核查：`raw_factor_count_min` **无任何代码消费点**
   （模型侧硬约束是 `models/base.py:20` 的 `>= 1`），故候选计划记录的「15 vs 14」张力
   纯属声明式文书层面、不会造成运行期失败，据此正式结案。

## 4. 待裁决（三项）

**裁决 1 — 是否补齐 2014–2019 物化（发现 1 的修复）**
生产库大规模写入（预计新增数千万行），按 6 次年窗口运行 + 追加 72 条指针登记，约 7 小时，
可与 M1 并行。**不补齐则冻结集链（正式证据路径）无法启动**，Task 2.3/2.4/3.x/5.x 全部空转。

**裁决 2 — 算力口径（发现 2/3）**，三个可选方向：
- **(a) 钉住单线程 + 约 100 路并发（实测最优，推荐）**：M1 从 42.6 天 → 约 10 小时。
  需要：显式钉住并记录线程数（补发现 3 的缺口）、搜索循环并发化且保持 canonical 并列顺序、
  为并发重写续跑粒度（现无折/配置级续跑，中断即从 F01/config0/seed1 重来）。
  数值口径变更须记录为已裁决偏差（与「CPU fp32」不冲突——仍是 CPU fp32，只是钉住线程数）。
- **(b) 维持现状串行 CPU**：不改任何口径，接受 M1→M3 全链数月量级。
- **(c) 转向 GPU**：对 M1 实测为**反向优化**（慢 2.9 倍），除非 M2/M3 基准显示相反结论，
  否则不建议；若采用还需撤销「CPU fp32」偏差并在新信任根上重建基线。

**裁决 3 — 遗留链去留（原 A/B/C，新增量化信息）**
M1 至今仅完成 **1 次拟合中的部分 epoch**（504 分之 ~0.03，投入 1.5 小时），
串行跑完需 42.6 天；复现性签发已由基线重跑 #3 的 CONSISTENT 完成。
故「现在停遗留链」的沉没成本几乎为零，而「跑完遗留链」（原选项 A 的前半）在串行口径下
需数月、在并发口径下需先完成裁决 2 的改造。

## 5. 复核命令

```bash
# 发现 1：factor_daily 覆盖范围
cd research_v2 && python - <<'EOF'
import sys; sys.path.insert(0,'scripts')
from walk_forward_common import TARGET_DATABASE, new_holder, query
h=new_holder()
print(query(h, f"select min(signal_date) as mn, max(signal_date) as mx, count(*) as n from loadTable('{TARGET_DATABASE}','factor_daily')")[1])
print(query(h, f"select count(*) as pre2020 from loadTable('{TARGET_DATABASE}','factor_daily') where signal_date < 2020.01.01")[1])
EOF

# 折定义（train 年最早 2014）
grep -n "range(2014, 2020)" research_v2/src/research_v2/datasets/splits.py

# Task 1.7 后置复核：72 分区 + 三张遗留表未变
cd research_v2 && python - <<'EOF'
import sys; sys.path.insert(0,'scripts')
from walk_forward_common import new_holder, current_versions
h=new_holder()
for t in ("factor_daily","universe_daily","factor_observation_clean","label_daily"):
    print(t, len(current_versions(h,t)))
EOF

# 发现 2：M1 实测 epoch 间隔
ls -la --time-style=+%H:%M:%S /data/zhangyudong/qrv2_research_artifacts/m1/checkpoints/

# 发现 3：线程数未钉住 / 未记录
grep -rn "set_num_threads\|OMP_NUM_THREADS" research_v2/src research_v2/scripts
grep -n "thread" research_v2/src/research_v2/training/seeds.py
```

## 6. 关联文书

- `docs/superpowers/plans/2026-09-05-frozen-set-baseline-reconstruction-plan.md`
  （Task 1.7 已勾选含执行证据；新增 Task 3.0 阻塞项；Task 3.1 路径隔离更正；
  Global Constraints 覆盖范围告警）
- `docs/superpowers/plans/2026-09-04-real-candidate-model-training-plan.md`
  （Task 3.4 实测吞吐 + 设备/线程基准 + 可复现性缺口）
- `docs/handoffs/2026-09-05-frozen-factor-set-training-decision.md`（原 A/B/C 备忘）
- `docs/handoffs/2026-09-05-candidate-training-chain-ops.md`（链路运维与标记文件语义）
