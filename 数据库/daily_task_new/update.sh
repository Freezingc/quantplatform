#!/bin/bash
source /data/share/miniconda3/etc/profile.d/conda.sh
conda activate database

# 获取当天日期（格式：YYYY-MM-DD）
current_date=$(date "+%Y-%m-%d")
current_time=$(date "+%Y-%m-%d %H:%M:%S")
echo "$current_time 开始更新"
# 定义 Python 程序列表

python_scripts=(
    "/home/chenxutian/chenxutian_data/Quant-Platform-Prod/数据库/daily_task_new/update.py"
    "/home/chenxutian/chenxutian_data/Quant-Platform-Prod/数据库/daily_task_new/update_factor.py"
)

# 创建以当天日期命名的文件夹
output_dir="/home/chenxutian/chenxutian_data/database/daily_task_log/$current_date"
mkdir -p "$output_dir"

# 遍历每个 Python 程序
for script in "${python_scripts[@]}"; do
    # 提取程序名称（去掉 .py 后缀）
    script_name=$(basename "$script" .py)

    # 定义输出文件名（包含日期文件夹路径）
    output_file="${output_dir}/${script_name}_output.txt"

    # 使用 nohup 执行 Python 程序，并将输出重定向到文件
    nohup python "$script" > "$output_file" 2>&1 &

    # 等待当前程序执行完成
    wait
done
current_time=$(date "+%Y-%m-%d %H:%M:%S")
echo "$current_time 结束更新"
echo "$current_date 更新已执行完成。输出文件保存在：$output_dir"