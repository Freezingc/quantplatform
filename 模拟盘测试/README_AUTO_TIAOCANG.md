# 自动调仓系统使用说明

## 功能概述

自动调仓系统会在**每周三晚上23:00**自动执行以下流程：

1. 下载当天的qlib数据
2. 执行MTSMatrix模型推理
3. 将推理结果（pkl）转换为csv格式
4. 复制csv文件到指定的输出目录

## 文件说明

- `auto_tiaocang.py` - 主执行脚本
- `setup_cron.sh` - 定时任务安装脚本
- `test_auto_tiaocang.py` - 测试脚本（用于手动测试）
- `logs/` - 日志目录（自动创建）

## 安装步骤

### 1. 设置定时任务

```bash
cd /data/zhangyudong/FDU/Quant-Platform-Prod-new/模拟盘测试
bash setup_cron.sh
```

### 2. 验证定时任务

```bash
crontab -l
```

应该看到类似输出：
```
0 23 * * 3 /usr/bin/python3 /data/zhangyudong/FDU/Quant-Platform-Prod-new/模拟盘测试/auto_tiaocang.py >> /data/zhangyudong/FDU/Quant-Platform-Prod-new/模拟盘测试/logs/cron.log 2>&1
```

## 手动测试

如果想立即测试脚本（不等到周三23:00），可以：

```bash
cd /data/zhangyudong/FDU/Quant-Platform-Prod-new/模拟盘测试
python3 auto_tiaocang.py
```

或使用测试脚本：

```bash
python3 test_auto_tiaocang.py
```

## 日志查看

### 查看最新日志

```bash
ls -lt /data/zhangyudong/FDU/Quant-Platform-Prod-new/模拟盘测试/logs/
tail -f /data/zhangyudong/FDU/Quant-Platform-Prod-new/模拟盘测试/logs/auto_tiaocang_*.log
```

### 查看cron日志

```bash
tail -f /data/zhangyudong/FDU/Quant-Platform-Prod-new/模拟盘测试/logs/cron.log
```

## 输出文件位置

执行完成后，csv文件会被复制到以下位置：

1. `/data/zhangyudong/FDU/Quant-Platform-Prod-new/回测/input/MTSMatrix_30_5_Test.csv`
2. `/data/xiahongjie/Quant-Platform-2025/输出与展示/调仓输出/MTSMatrix_30_5_Test.csv`

## 故障排查

### 1. 定时任务没有执行

检查cron服务状态：
```bash
systemctl status cron    # Ubuntu/Debian
systemctl status crond   # CentOS/RHEL
```

### 2. 脚本执行失败

查看日志文件找到错误信息：
```bash
tail -100 /data/zhangyudong/FDU/Quant-Platform-Prod-new/模拟盘测试/logs/auto_tiaocang_*.log
```

### 3. qlib数据下载失败

可能原因：
- 网络问题（GitHub访问受限）
- 数据还未发布（检查日期）

解决方案：
- 使用代理或镜像
- 手动下载后放到对应位置

### 4. 权限问题

确保脚本有执行权限：
```bash
chmod +x /data/zhangyudong/FDU/Quant-Platform-Prod-new/模拟盘测试/auto_tiaocang.py
```

确保目标目录有写权限：
```bash
ls -ld /data/xiahongjie/Quant-Platform-2025/输出与展示/调仓输出
```

## 修改执行时间

如需修改执行时间，编辑crontab：

```bash
crontab -e
```

Cron时间格式：
```
分 时 日 月 周
0 23 * * 3     # 每周三23:00
0 23 * * 1-5   # 每周一到周五23:00
0 9 * * *      # 每天9:00
```

## 停用定时任务

```bash
crontab -l | grep -v 'auto_tiaocang.py' | crontab -
```

## 注意事项

1. **日期依赖**：脚本依赖GitHub上的qlib数据发布，如果当天数据未发布会失败
2. **计算资源**：模型推理需要GPU资源，确保GPU可用
3. **磁盘空间**：定期清理旧的日志和临时文件
4. **依赖包**：确保Python环境有所需的依赖包（pandas, subprocess等）

