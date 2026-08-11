#!/bin/bash
# 设置定时任务 - 每周二晚上21:00执行

SCRIPT_PATH="/data/zhangyudong/FDU/Quant-Platform-Prod-new/模拟盘测试/auto_tiaocang.py"
PYTHON_PATH="/data/zhangyudong/.conda/envs/quantyd/bin/python"

# 确保脚本有执行权限
chmod +x "$SCRIPT_PATH"

# 定时任务配置：每周三 21:00
# 格式: 分 时 日 月 周
# 周二是2 (0=周日, 1=周一, ..., 6=周六)
CRON_JOB="20 21 * * 2 $PYTHON_PATH $SCRIPT_PATH >> /data/zhangyudong/FDU/Quant-Platform-Prod-new/模拟盘测试/logs/cron.log 2>&1"

# 检查crontab中是否已存在该任务
crontab -l 2>/dev/null | grep -F "$SCRIPT_PATH" > /dev/null
if [ $? -eq 0 ]; then
    echo "定时任务已存在，正在更新..."
    # 删除旧任务
    crontab -l 2>/dev/null | grep -v "$SCRIPT_PATH" | crontab -
fi

# 添加新任务
echo "添加定时任务..."
(crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -

echo "定时任务设置成功！"
echo "当前crontab配置："
crontab -l

echo ""
echo "说明："
echo "- 任务将在每周二晚上21:00自动执行"
echo "- 日志文件保存在: /data/zhangyudong/FDU/Quant-Platform-Prod-new/模拟盘测试/logs/"
echo "- 查看定时任务: crontab -l"
echo "- 编辑定时任务: crontab -e"
echo "- 删除定时任务: crontab -l | grep -v '$SCRIPT_PATH' | crontab -"

