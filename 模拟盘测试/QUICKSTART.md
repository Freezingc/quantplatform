# 🚀 快速开始指南

## 一键安装

```bash
cd /data/zhangyudong/FDU/Quant-Platform-Prod-new/模拟盘测试
bash setup_cron.sh
```

安装完成后，系统将在**每周三晚上23:00**自动执行调仓。

## 立即测试

### 方法1: 使用交互式菜单

```bash
python3 demo.py
```

### 方法2: 快速测试（跳过耗时步骤）

```bash
python3 test_auto_tiaocang.py --quick
```

### 方法3: 完整执行

```bash
python3 auto_tiaocang.py
```

## 常用命令

| 操作 | 命令 |
|------|------|
| 查看定时任务 | `crontab -l` |
| 查看最新日志 | `ls -lt logs/` |
| 查看日志内容 | `tail -f logs/auto_tiaocang_*.log` |
| 删除定时任务 | `crontab -l \| grep -v auto_tiaocang \| crontab -` |

## 执行流程

```
1. 下载qlib数据
   ↓
2. 执行模型推理 (MTSMatrix)
   ↓
3. 转换pkl → csv
   ↓
4. 复制到输出目录
```

## 输出位置

- `回测/input/MTSMatrix_30_5_Test.csv`
- `/data/xiahongjie/Quant-Platform-2025/输出与展示/调仓输出/MTSMatrix_30_5_Test.csv`

## 文件结构

```
模拟盘测试/
├── auto_tiaocang.py         # 主脚本
├── setup_cron.sh            # 安装脚本
├── test_auto_tiaocang.py    # 测试脚本
├── demo.py                  # 演示菜单
├── README_AUTO_TIAOCANG.md  # 详细文档
├── QUICKSTART.md           # 本文件
└── logs/                    # 日志目录
    ├── auto_tiaocang_*.log
    └── cron.log
```

## 注意事项

⚠️ **首次使用前请确保:**
1. Python环境已安装pandas等依赖
2. GPU资源可用（模型推理需要）
3. 网络可访问GitHub（下载数据）
4. 有权限写入输出目录

## 获取帮助

查看详细文档:
```bash
cat README_AUTO_TIAOCANG.md
```

---

**问题反馈**: 查看日志文件或联系开发者

