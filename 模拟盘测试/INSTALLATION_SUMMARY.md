# 自动调仓系统安装总结

## ✅ 已创建的文件

### 核心脚本
1. **auto_tiaocang.py** - 主自动化脚本
   - 下载qlib数据
   - 执行模型推理
   - 转换pkl为csv
   - 复制到输出目录
   - 完整的日志记录

2. **setup_cron.sh** - 定时任务安装脚本
   - 自动配置cron定时任务
   - 设置为每周三23:00执行

3. **test_auto_tiaocang.py** - 测试工具
   - 交互式测试界面
   - 可分步测试各个功能
   - 支持快速测试模式

4. **demo.py** - 演示菜单
   - 用户友好的操作界面
   - 集成所有常用功能

### 文档
5. **README_AUTO_TIAOCANG.md** - 详细使用文档
6. **QUICKSTART.md** - 快速开始指南
7. **INSTALLATION_SUMMARY.md** - 本文件

## 🚀 立即开始

### 第一步：安装定时任务

```bash
cd /data/zhangyudong/FDU/Quant-Platform-Prod-new/模拟盘测试
chmod +x *.py *.sh  # 添加执行权限
bash setup_cron.sh
```

### 第二步：验证安装

```bash
crontab -l
```

应该看到：
```
0 23 * * 3 /usr/bin/python3 /data/zhangyudong/FDU/Quant-Platform-Prod-new/模拟盘测试/auto_tiaocang.py >> /data/zhangyudong/FDU/Quant-Platform-Prod-new/模拟盘测试/logs/cron.log 2>&1
```

### 第三步：测试功能（可选）

```bash
# 方式1: 使用演示菜单
python3 demo.py

# 方式2: 快速测试
python3 test_auto_tiaocang.py --quick

# 方式3: 完整测试（会下载数据和运行推理）
python3 auto_tiaocang.py
```

## 📋 执行计划

系统将在**每周三晚上23:00**自动执行以下流程：

```
23:00 - 开始执行
  ├─ 下载当天的qlib数据
  │   └─ https://github.com/chenditc/investment_data/releases/download/YYYY-MM-DD/qlib_bin.tar.gz
  │
  ├─ 执行MTSMatrix模型推理
  │   ├─ 使用GPU 2
  │   ├─ 推理日期：当天
  │   └─ 生成pkl文件
  │
  ├─ 转换pkl为csv
  │   ├─ 转换股票代码格式
  │   ├─ 转换日期格式
  │   └─ 生成宽表格式
  │
  └─ 复制csv到输出目录
      ├─ /data/zhangyudong/FDU/Quant-Platform-Prod-new/回测/input/MTSMatrix_30_5_Test.csv
      └─ /data/xiahongjie/Quant-Platform-2025/输出与展示/调仓输出/MTSMatrix_30_5_Test.csv

完成！
```

## 📊 日志位置

所有执行日志保存在：
```
/data/zhangyudong/FDU/Quant-Platform-Prod-new/模拟盘测试/logs/
```

查看日志：
```bash
# 查看最新日志
ls -lt logs/

# 实时查看日志
tail -f logs/auto_tiaocang_*.log

# 查看cron日志
tail -f logs/cron.log
```

## ⚙️ 配置说明

### 关键参数

| 参数 | 值 | 说明 |
|------|---|------|
| 执行时间 | 周三 23:00 | 可在setup_cron.sh中修改 |
| GPU设备 | GPU 2 | 在MTSMatrixTiaocang.sh中配置 |
| 模型 | MTSMatrix | 序列长度30，预测horizon 5 |
| 训练起始 | 2022-01-01 | 固定起始日期 |
| 训练结束 | 当天日期 | 自动更新 |

### 修改执行时间

编辑 `setup_cron.sh` 中的 `CRON_JOB` 变量：

```bash
# 当前配置：每周三23:00
CRON_JOB="0 23 * * 3 ..."

# 改为每天23:00
CRON_JOB="0 23 * * * ..."

# 改为每周一、三、五 23:00
CRON_JOB="0 23 * * 1,3,5 ..."

# 改为每周五 20:00
CRON_JOB="0 20 * * 5 ..."
```

## 🔧 故障排查

### 1. 定时任务没有执行

**检查cron服务：**
```bash
systemctl status cron    # Ubuntu/Debian
systemctl status crond   # CentOS/RHEL
```

**检查crontab配置：**
```bash
crontab -l
```

### 2. 脚本执行失败

**查看错误日志：**
```bash
tail -100 logs/auto_tiaocang_*.log
```

**常见错误：**
- 权限问题：`chmod +x *.py`
- Python依赖缺失：`pip install pandas`
- GPU不可用：检查CUDA环境
- 网络问题：检查GitHub访问

### 3. 数据下载失败

**可能原因：**
- GitHub访问受限
- 数据还未发布
- 网络连接问题

**解决方案：**
- 配置代理或使用镜像
- 等待数据发布后再执行
- 手动下载后放到指定位置

### 4. 文件复制失败

**检查权限：**
```bash
ls -ld /data/xiahongjie/Quant-Platform-2025/输出与展示/调仓输出
```

**创建目录（如果不存在）：**
```bash
mkdir -p /data/xiahongjie/Quant-Platform-2025/输出与展示/调仓输出
```

## 📝 日常维护

### 清理旧日志

```bash
# 删除30天前的日志
find logs/ -name "*.log" -mtime +30 -delete
```

### 监控磁盘空间

```bash
df -h ~/.qlib/qlib_data/cn_data
```

### 检查最近执行情况

```bash
# 查看最近5次执行日志
ls -lt logs/auto_tiaocang_*.log | head -5
```

## 🔄 更新和升级

如果需要修改脚本：

1. 编辑 `auto_tiaocang.py`
2. 测试修改：`python3 test_auto_tiaocang.py`
3. 无需重新安装cron（除非修改了执行时间）

## 📞 技术支持

遇到问题时：

1. 查看日志文件
2. 检查QUICKSTART.md和README_AUTO_TIAOCANG.md
3. 使用测试工具诊断：`python3 test_auto_tiaocang.py`

---

**安装日期**: 2025-12-22  
**版本**: 1.0  
**状态**: ✅ 就绪

