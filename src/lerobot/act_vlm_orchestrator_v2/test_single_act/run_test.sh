#!/bin/bash
# ============================================================================
# 单个 ACT 模型推理效果测试 — 一键启动 (对齐编排器平滑 + MIT 模式)
# ============================================================================
#
# 用法:
#   chmod +x run_test.sh
#   ./run_test.sh                            # 默认 Task_A, 平滑开启, 真机控制
#   ./run_test.sh Task_B                     # 测试 Task_B
#   ./run_test.sh Task_A --dry_run True      # 观察模式 (不控制机械臂)
#   ./run_test.sh Task_A --smooth.mode=none  # 关闭平滑 (模型原始输出)
# ============================================================================

set -e

# 颜色
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
PYTHON="${PYTHON:-/home/ub/anaconda3/envs/evo-rl-videos/bin/python}"
MODEL_BASE="${MODEL_BASE:-/home/ub/Videos/0407/model/ACT_model/model}"

TASK="${1:-Task_A}"
shift || true

# ── 检查模型 ──
MODEL_PATH="${MODEL_BASE}/${TASK}/pretrained_model"
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}❌ 模型路径不存在: ${MODEL_PATH}${NC}"
    echo -e "${YELLOW}可用: MODEL_BASE=... ./run_test.sh${NC}"
    exit 1
fi
echo -e "${GREEN}✅ 模型: ${MODEL_PATH}${NC}"

# ── 相机参数 (可环境变量覆盖) ──
TOP_CAM="${TOP_CAM:-10}"
WRIST_CAM="${WRIST_CAM:-4}"

# ── 启动 ──
cd "$PROJECT_ROOT"
exec "$PYTHON" -m lerobot.act_vlm_orchestrator_v2.test_single_act.test_single_act_inference \
    --robot.type=single_piper \
    --robot.id=single_piper_robot \
    --robot.ctrl_mode="joint" \
    --robot.port=can1 \
    --robot.cameras="{wrist: {type: opencv, index_or_path: ${WRIST_CAM}, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: ${TOP_CAM}, width: 640, height: 480, fps: 30}}" \
    --ckpt_path="$MODEL_PATH" \
    "$@"
