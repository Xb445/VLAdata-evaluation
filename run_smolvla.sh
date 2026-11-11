#!/usr/bin/env bash
# =============================================================================
# SmolVLA 真机推理一键启动脚本
# 用法（在项目根目录或任意位置执行）：
#   bash run_smolvla.sh                          # 默认：select_action + 三段任务
#   bash run_smolvla.sh --rtc --preset-pose      # 推荐：RTC 平滑 + 预置工作位姿
#   bash run_smolvla.sh --rtc --dry-run          # 观察模式（不控制机械臂）
#   其余参数与原脚本一致（--task/--fps/--alpha/--log-dir 等）
# =============================================================================
set -e

# 定位脚本目录（项目根）
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="$PROJECT_ROOT/src/lerobot"

# 使用 evo-rl-videos 环境（与项目一致）
PYTHON="/home/ub/anaconda3/envs/evo-rl-videos/bin/python"

if [ ! -x "$PYTHON" ]; then
    echo "❌ 未找到 evo-rl-videos 环境: $PYTHON" >&2
    exit 1
fi

cd "$SCRIPT_DIR"
exec "$PYTHON" -u smolvla_real_inference.py "$@"
