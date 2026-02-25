#!/bin/bash

# --- 配置区 ---
TRAIN_SCRIPT="train.py"
SLEEP_TIME=30  # 每轮跑完休息多少秒（给 CPU 降温）

echo "🚀 启动自动化训练守护进程..."

while true; do
    echo "------------------------------------------------"
    echo "📅 当前时间: $(date +%H:%M:%S)"
    echo "🔥 正在启动新一轮训练..."
    
    # 运行 Python 脚本
    # 我们限制它只跑一轮（或者靠代码里的 resume 逻辑自动继续）
    # 修改脚本中的运行行为
    python3 "$TRAIN_SCRIPT" 2>&1 | tee -a training_log.txt
    
    # 获取 Python 的退出状态码
    EXIT_CODE=$?
    
    # 如果代码是因为跑完（或者报错）停止的
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ 这一阶段任务完成。"
    else
        echo "⚠️ 进程意外停止或手动中断 (Exit Code: $EXIT_CODE)。"
        # 如果是你手动按 Ctrl+C，我们就彻底退出脚本
        if [ $EXIT_CODE -eq 130 ]; then
            echo "🛑 检测到手动中断，退出守护程序。"
            break
        fi
    fi

    echo "💤 为了保护硬件，强制休息 $SLEEP_TIME 秒..."
    sleep $SLEEP_TIME
done