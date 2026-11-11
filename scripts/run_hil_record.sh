#!/bin/bash
# =============================================================================
# 人类在环 (HIL) 数据采集快速启动脚本
#
# 用法: bash scripts/run_hil_record.sh
#
# 功能: 加载策略模型 + 主臂遥操干预 + 数据录制
# =============================================================================

set -e

# ═══════════════════════════════════════════════════════════════════════
# 参数配置区（按需修改）
# ═══════════════════════════════════════════════════════════════════════

# ── Conda 环境 ──
CONDA_ENV="evo-rl-videos"

# ── CAN 接口 ──
CAN_MASTER="can0"      # 主臂 CAN 口
CAN_SLAVE="can1"       # 从臂 CAN 口

# ── 相机配置 ──
CAM_WRIST_INDEX=10      # 腕部相机 /dev/video 索引
CAM_TOP_INDEX=4       # 顶部相机 /dev/video 索引
CAM_WIDTH=640
CAM_HEIGHT=480
CAM_FPS=30

# ── 策略模型路径 ──
# Pi0.5 模型:
POLICY_PATH="//home/ub/Videos/Evo-RL/model/Pimodel/60K/pretrained_model"
# ACT 模型 (取消注释下面一行并注释上面一行):
#POLICY_PATH="/home/ub/Videos/0407/model/ACT_model/Task_B/pretrained_model"
#POLICY_PATH="/home/ub/Videos/0407/model/ACT_model/act_acp_taskB_incremental/pretrained_model"

# ── 数据集参数 ──
DATASET_USER="test"
DATASET_PREFIX="eval_single_piper"   # 策略模式建议使用 eval_ 前缀
NUM_EPISODES=100
EPISODE_TIME_S=60
FPS=30
RESET_TIME_S=30
TASK_DESC="Grab the scanned package and place it into the box."
PUSH_TO_HUB="False"

# ── 机械臂参数 ──
GRIPPER_EFFORT=1000    # 夹爪力矩 (默认 5000 = 5 N·m)

# ── 显示与可视化 ──
DISPLAY_DATA="true"     # 启动 Rerun 可视化
PLAY_SOUNDS="true"      # 语音提示

# ── 干预控制 ──
TOGGLE_KEY="i"          # 干预切换键
SUCCESS_KEY="s"         # 标记成功键
FAILURE_KEY="f"         # 标记失败键

# ═══════════════════════════════════════════════════════════════════════
# 脚本逻辑（无需修改）
# ═══════════════════════════════════════════════════════════════════════

TIMESTAMP=$(date +%m%d_%H%M)
DATASET_ID="${DATASET_USER}/${DATASET_PREFIX}_${TIMESTAMP}"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   🎮 人类在环 (HIL) 数据采集                    ║"
echo "╠══════════════════════════════════════════════╣"
echo "║  策略:  $(basename ${POLICY_PATH})"
echo "║  数据集: ${DATASET_ID}"
echo "║  Episodes: ${NUM_EPISODES} × ${EPISODE_TIME_S}s"
echo "║  相机: wrist=/dev/video${CAM_WRIST_INDEX} top=/dev/video${CAM_TOP_INDEX}"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── 激活 CAN 接口 ──
echo "[1/3] 配置 CAN 接口..."
bash "$(dirname "$0")/setup_can.sh" 2>/dev/null || {
    sudo ip link set ${CAN_MASTER} up type can bitrate 1000000 2>/dev/null || true
    sudo ip link set ${CAN_SLAVE} up type can bitrate 1000000 2>/dev/null || true
}

# ── 激活 conda 环境 ──
echo "[2/3] 激活 conda 环境: ${CONDA_ENV}..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# ── 启动 HIL 录制 ──
echo "[3/3] 启动人类在环录制..."
echo ""

lerobot-human-inloop-record \
    --robot.type=piper_follower \
    --robot.port="${CAN_SLAVE}" \
    --robot.id=slave_arm \
    --robot.require_calibration=False \
    --robot.gripper_effort_default="${GRIPPER_EFFORT}" \
    --robot.cameras="{wrist: {type: opencv, index_or_path: ${CAM_WRIST_INDEX}, width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${CAM_FPS}}, top: {type: opencv, index_or_path: ${CAM_TOP_INDEX}, width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${CAM_FPS}}}" \
    --teleop.type=piper_leader \
    --teleop.port="${CAN_MASTER}" \
    --teleop.id=master_arm \
    --teleop.require_calibration=False \
    --teleop.gripper_effort_default="${GRIPPER_EFFORT}" \
    --policy.path="${POLICY_PATH}" \
    --dataset.repo_id="${DATASET_ID}" \
    --dataset.num_episodes="${NUM_EPISODES}" \
    --dataset.episode_time_s="${EPISODE_TIME_S}" \
    --dataset.fps="${FPS}" \
    --dataset.reset_time_s="${RESET_TIME_S}" \
    --dataset.single_task="${TASK_DESC}" \
    --dataset.push_to_hub="${PUSH_TO_HUB}" \
    --display_data="${DISPLAY_DATA}" \
    --play_sounds="${PLAY_SOUNDS}" \
    --intervention_toggle_key="${TOGGLE_KEY}" \
    --episode_success_key="${SUCCESS_KEY}" \
    --episode_failure_key="${FAILURE_KEY}"
