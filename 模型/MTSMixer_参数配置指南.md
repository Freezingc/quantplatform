# MTSMixer 参数配置指南

## ❌ 常见错误及解决方案

### 错误1: AssertionError in FactorizedChannelMixing

**错误信息**：
```
File "models/MTSMixer.py", line 50, in __init__
    assert input_dim > factorized_dim
AssertionError
```

**原因**：
- 当 `fac_C=1`（启用因子化通道混合）时
- `d_ff` (channels_hidden_dim) 必须**小于** `enc_in` (channels_dim)
- 你的配置：`d_ff=512 > enc_in=158` ❌

**解决方案**：

#### 方案1：关闭因子化通道混合（推荐）
```bash
--fac_C 0  # 关闭
--d_ff 512  # 可以任意设置
```

#### 方案2：调整 d_ff 参数
```bash
--fac_C 1  # 开启
--d_ff 128  # 必须 < enc_in (158)
```

#### 方案3：增加因子数量
```bash
--enc_in 1000  # 使用更多因子
--d_ff 512     # 现在 512 < 1000
--fac_C 1      # 可以开启
```

---

## 📋 参数配置规则

### 规则1: d_ff 与 enc_in 的关系

| fac_C | 规则 | 示例 |
|-------|------|------|
| 0 (关闭) | `d_ff` 任意 | `enc_in=158, d_ff=512` ✅ |
| 1 (开启) | `d_ff < enc_in` | `enc_in=158, d_ff=128` ✅ |
| 1 (开启) | `d_ff >= enc_in` | `enc_in=158, d_ff=512` ❌ |

### 规则2: sampling 与 seq_len 的关系

`seq_len` 必须能被 `sampling` 整除：

| seq_len | 可用的 sampling | 推荐 |
|---------|----------------|------|
| 30 | 1, 2, 3, 6 | 2 |
| 60 | 1, 2, 3, 4, 6, 12 | 4 |
| 120 | 1, 2, 3, 4, 6, 8, 12 | 6 |

---

## 🎯 推荐配置方案

### 方案1: Alpha158 因子（158个因子）

```bash
python run.py \
    --model MTSMixer \
    --alpha 158 \
    --enc_in 158 \
    --seq_len 30 \
    --e_layers 3 \
    --d_model 256 \
    --d_ff 128 \
    --sampling 2 \
    --fac_T 1 \
    --fac_C 0 \
    --norm 1 \
    --rev 1 \
    --individual 0
```

**说明**：
- ✅ `d_ff=128 < enc_in=158`
- ✅ `fac_C=0`（因子数量较少，不需要因子化）
- ✅ `sampling=2`（30能被2整除）

### 方案2: Alpha360 因子（360个因子）

```bash
python run.py \
    --model MTSMixer \
    --alpha 360 \
    --enc_in 360 \
    --seq_len 60 \
    --e_layers 4 \
    --d_model 512 \
    --d_ff 256 \
    --sampling 4 \
    --fac_T 1 \
    --fac_C 1 \
    --norm 1 \
    --rev 1 \
    --individual 1
```

**说明**：
- ✅ `d_ff=256 < enc_in=360`
- ✅ `fac_C=1`（因子数量多，使用因子化降低参数量）
- ✅ `sampling=4`（60能被4整除）

### 方案3: 自定义因子（47个因子）

```bash
python run.py \
    --model MTSMixer \
    --enc_in 47 \
    --seq_len 30 \
    --e_layers 2 \
    --d_model 128 \
    --d_ff 64 \
    --sampling 2 \
    --fac_T 1 \
    --fac_C 0 \
    --norm 1 \
    --rev 1 \
    --individual 0
```

**说明**：
- ✅ `d_ff=64 < enc_in=47`
- ✅ `fac_C=0`（因子数量很少）
- ✅ 使用较小的模型（因子少）

---

## 🔧 参数调优策略

### 1. 根据因子数量选择配置

```python
if enc_in < 100:
    # 小规模因子
    fac_C = 0
    d_ff = enc_in // 2  # 例如：enc_in=47 -> d_ff=23
    d_model = 128
    e_layers = 2
    
elif enc_in < 300:
    # 中等规模因子
    fac_C = 0 或 1
    d_ff = min(256, enc_in - 10)  # 例如：enc_in=158 -> d_ff=148
    d_model = 256
    e_layers = 3
    
else:
    # 大规模因子
    fac_C = 1
    d_ff = enc_in // 2  # 例如：enc_in=360 -> d_ff=180
    d_model = 512
    e_layers = 4
```

### 2. 根据序列长度选择 sampling

```python
if seq_len <= 30:
    sampling = 2
elif seq_len <= 60:
    sampling = 4
elif seq_len <= 120:
    sampling = 6
else:
    sampling = 8
```

### 3. 根据GPU显存选择模型大小

```python
# GPU < 8GB
d_model = 128
d_ff = 64
e_layers = 2
batch_size = 1

# GPU 8-16GB
d_model = 256
d_ff = 128
e_layers = 3
batch_size = 1

# GPU > 16GB
d_model = 512
d_ff = 256
e_layers = 4
batch_size = 2
```

---

## 📊 参数对照表

### enc_in (因子数量) 推荐配置

| enc_in | d_model | d_ff | fac_C | e_layers | 说明 |
|--------|---------|------|-------|----------|------|
| 47 | 128 | 32 | 0 | 2 | 小规模 |
| 100 | 128 | 64 | 0 | 2 | 中小规模 |
| 158 | 256 | 128 | 0 | 3 | 中等规模 |
| 200 | 256 | 128 | 1 | 3 | 中大规模 |
| 360 | 512 | 256 | 1 | 4 | 大规模 |
| 500+ | 512 | 256 | 1 | 4 | 超大规模 |

### seq_len (序列长度) 推荐配置

| seq_len | sampling | d_model | 训练时间 | 内存占用 |
|---------|----------|---------|----------|----------|
| 20 | 1 或 2 | 128 | 快 | 低 |
| 30 | 2 | 256 | 中等 | 中等 |
| 60 | 4 | 256 | 中等 | 中等 |
| 120 | 6 | 512 | 慢 | 高 |

---

## 🚀 快速配置生成器

### 使用方法

根据你的数据特征，选择对应的配置：

```bash
# 1. 确定因子数量
enc_in=158  # 你的因子数量

# 2. 确定序列长度
seq_len=30  # 你的回看窗口

# 3. 生成配置
if [ $enc_in -lt 100 ]; then
    d_ff=64
    fac_C=0
    d_model=128
elif [ $enc_in -lt 300 ]; then
    d_ff=128
    fac_C=0
    d_model=256
else
    d_ff=256
    fac_C=1
    d_model=512
fi

# 4. 运行训练
python run.py \
    --model MTSMixer \
    --enc_in $enc_in \
    --seq_len $seq_len \
    --d_model $d_model \
    --d_ff $d_ff \
    --fac_C $fac_C \
    ...
```

---

## 🐛 调试技巧

### 1. 验证参数兼容性

```python
# 在训练前检查
enc_in = 158
d_ff = 512
fac_C = 1

if fac_C == 1 and d_ff >= enc_in:
    print(f"❌ 错误：d_ff ({d_ff}) >= enc_in ({enc_in})")
    print(f"✅ 建议：d_ff={enc_in//2} 或 fac_C=0")
else:
    print("✅ 参数配置正确")
```

### 2. 检查 sampling 兼容性

```python
seq_len = 30
sampling = 4

if seq_len % sampling != 0:
    print(f"❌ 错误：seq_len ({seq_len}) 不能被 sampling ({sampling}) 整除")
    print(f"✅ 建议：sampling={[i for i in [1,2,3,4,6,8,12] if seq_len % i == 0]}")
else:
    print("✅ 参数配置正确")
```

### 3. 估算显存占用

```python
# 粗略估算（单位：MB）
batch_size = 1
seq_len = 30
enc_in = 158
d_model = 256
e_layers = 3

memory_mb = (
    batch_size * seq_len * enc_in * 4 +  # 输入数据
    d_model * enc_in * e_layers * 4 +    # 模型参数
    batch_size * seq_len * d_model * e_layers * 4  # 中间激活
) / 1024 / 1024

print(f"预估显存占用: {memory_mb:.2f} MB")
```

---

## 📝 配置检查清单

训练前请检查：

- [ ] `d_ff < enc_in`（当 `fac_C=1` 时）
- [ ] `seq_len % sampling == 0`
- [ ] `enc_in` 与实际因子数量匹配
- [ ] `pred_len = 1`（股票预测任务）
- [ ] GPU 显存足够（至少 4GB）
- [ ] 数据日期范围正确
- [ ] 训练集、验证集、测试集不重叠

---

## 💡 最佳实践

### 1. 首次训练

```bash
# 使用小模型快速验证
--e_layers 2
--d_model 128
--d_ff 64
--train_epochs 10
```

### 2. 正式训练

```bash
# 使用推荐配置
--e_layers 3
--d_model 256
--d_ff 128
--train_epochs 100
```

### 3. 高精度训练

```bash
# 使用大模型
--e_layers 4
--d_model 512
--d_ff 256
--train_epochs 200
--patience 50
```

---

**最后更新**: 2025-01-12  
**版本**: v1.1  
**修复**: AssertionError 问题



