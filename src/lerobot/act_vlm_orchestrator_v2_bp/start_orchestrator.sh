#!/bin/bash
# ============================================================================
# ACT + VLM 多任务联调编排器 — 一键启动脚本
# ============================================================================
#
# 功能:
#   1. 检查 vLLM 服务是否已启动
#   2. 检查 ACT 模型路径是否存在
#   3. 检查相机是否可用
#   4. 启动编排器主程序
#
# 用法:
#   chmod +x start_orchestrator.sh
#   ./start_orchestrator.sh                     # 默认配置
#   ./start_orchestrator.sh --debug             # 调试模式
#   ./start_orchestrator.sh --device cpu        # CPU 推理
#   ./start_orchestrator.sh --no-preload-all    # 按需加载模型 (节省 GPU)
#
# 前置步骤:
#   1. 启动 vLLM 服务:
#      VLLM_USE_FLASHINFER_SAMPLER=0 python -m vllm.entrypoints.openai.api_server \
#        --model /home/lh/Videos/0407/model/Qwen_model/qwen3_vl_2b_parcel_subtask_merged \
#        --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.7 \
#        --max-model-len 8192 --max-num-seqs 4 --dtype auto \
#        --trust-remote-code --served-model-name qwen3-vl-2b
#
#   2. 确保相机已连接 (top=10, wrist=4)
#   3. 确保 Piper 机械臂已连接 (CAN bus)
# ============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_SRC="$(cd "$SCRIPT_DIR/.." && pwd)"

echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}  ACT + VLM 多任务联调编排器 — 启动检查${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""

# -------------------------------------------------------------------
# 1. 检查 vLLM 服务
# -------------------------------------------------------------------
echo -e "${YELLOW}[1/5] 检查 vLLM 服务...${NC}"
VLLM_URL="${VLLM_URL:-http://localhost:8000/v1}"

if curl -s --connect-timeout 3 "${VLLM_URL}/models" > /dev/null 2>&1; then
    echo -e "  ${GREEN}✅ vLLM 服务运行中: ${VLLM_URL}${NC}"
    MODELS=$(curl -s "${VLLM_URL}/models" | python3 -c "import sys,json; data=json.load(sys.stdin); print(', '.join(m['id'] for m in data.get('data',[])))" 2>/dev/null || echo "N/A")
    echo -e "  可用模型: ${MODELS}"
else
    echo -e "  ${RED}❌ vLLM 服务未运行或不可达: ${VLLM_URL}${NC}"
    echo -e "  ${YELLOW}请先启动 vLLM 服务:${NC}"
    echo -e "  ${YELLOW}  VLLM_USE_FLASHINFER_SAMPLER=0 python -m vllm.entrypoints.openai.api_server \\${NC}"
    echo -e "  ${YELLOW}    --model /home/lh/Videos/0407/model/Qwen_model/qwen3_vl_2b_parcel_subtask_merged \\${NC}"
    echo -e "  ${YELLOW}    --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.7 \\${NC}"
    echo -e "  ${YELLOW}    --max-model-len 8192 --max-num-seqs 4 --dtype auto \\${NC}"
    echo -e "  ${YELLOW}    --trust-remote-code --served-model-name qwen3-vl-2b${NC}"
    echo ""
    read -p "  是否继续? (vLLM 服务不可用会导致 VLM 分类失败) [y/N]: " CONTINUE
    if [[ ! "$CONTINUE" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
echo ""

# -------------------------------------------------------------------
# 2. 检查 ACT 模型
# -------------------------------------------------------------------
echo -e "${YELLOW}[2/5] 检查 ACT 模型路径...${NC}"
MODEL_BASE="${MODEL_BASE:-/home/ub/Videos/0407/model/ACT_model/model}"
TASKS=("Task_A" "Task_B" "Task_C")
ALL_OK=true

for TASK in "${TASKS[@]}"; do
    MODEL_PATH="${MODEL_BASE}/${TASK}/pretrained_model"
    if [ -d "$MODEL_PATH" ]; then
        # 检查关键文件
        if [ -f "$MODEL_PATH/config.json" ] || [ -d "$MODEL_PATH/checkpoint" ]; then
            echo -e "  ${GREEN}✅ ${TASK}: ${MODEL_PATH}${NC}"
        else
            echo -e "  ${YELLOW}⚠️  ${TASK}: ${MODEL_PATH} (目录存在但可能不完整)${NC}"
            ALL_OK=false
        fi
    else
        echo -e "  ${RED}❌ ${TASK}: ${MODEL_PATH} (不存在)${NC}"
        ALL_OK=false
    fi
done

if [ "$ALL_OK" = false ]; then
    echo -e "  ${RED}部分模型路径无效, 请检查!${NC}"
    read -p "  是否继续? [y/N]: " CONTINUE
    if [[ ! "$CONTINUE" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
echo ""

# -------------------------------------------------------------------
# 3. 检查相机
# -------------------------------------------------------------------
echo -e "${YELLOW}[3/5] 检查相机...${NC}"
TOP_CAM=${TOP_CAMERA:-10}
WRIST_CAM=${WRIST_CAMERA:-4}

# 使用 v4l2-ctl 检查相机 (如果可用)
if command -v v4l2-ctl &> /dev/null; then
    CAM_LIST=$(v4l2-ctl --list-devices 2>/dev/null || echo "")
    if [ -n "$CAM_LIST" ]; then
        echo -e "  ${GREEN}✅ 检测到视频设备${NC}"
        echo "$CAM_LIST" | head -20
    else
        echo -e "  ${YELLOW}⚠️  未检测到视频设备${NC}"
    fi
else
    echo -e "  ${YELLOW}⚠️  v4l2-ctl 不可用, 跳过相机检查${NC}"
    echo -e "  将尝试打开相机 ID: top=${TOP_CAM}, wrist=${WRIST_CAM}"
fi
echo ""

# -------------------------------------------------------------------
# 4. 检查 GPIO/CAN 权限
# -------------------------------------------------------------------
echo -e "${YELLOW}[4/5] 检查设备权限...${NC}"
if [ -e /dev/ttyCAN1 ] || [ -e /dev/can1 ]; then
    echo -e "  ${GREEN}✅ CAN 设备存在${NC}"
else
    echo -e "  ${YELLOW}⚠️  CAN 设备未检测到 (可能不影响连接)${NC}"
fi
echo ""

# -------------------------------------------------------------------
# 5. 启动编排器
# -------------------------------------------------------------------
echo -e "${YELLOW}[5/5] 启动编排器...${NC}"
echo ""

cd "$PROJECT_SRC"

# 构建 Python 命令
PYTHON_CMD="python -m lerobot.act_vlm_orchestrator_v2.run_orchestrator"
PYTHON_CMD="$PYTHON_CMD --vlm-url ${VLLM_URL}"
PYTHON_CMD="$PYTHON_CMD --model-a ${MODEL_BASE}/Task_A/pretrained_model"
PYTHON_CMD="$PYTHON_CMD --model-b ${MODEL_BASE}/Task_B/pretrained_model"
PYTHON_CMD="$PYTHON_CMD --model-c ${MODEL_BASE}/Task_C/pretrained_model"
PYTHON_CMD="$PYTHON_CMD --top-camera ${TOP_CAM}"
PYTHON_CMD="$PYTHON_CMD --wrist-camera ${WRIST_CAM}"

# 附加用户参数
PYTHON_CMD="$PYTHON_CMD $@"

echo -e "${GREEN}执行命令:${NC}"
echo -e "  ${BLUE}${PYTHON_CMD}${NC}"
echo ""
echo -e "${GREEN}============================================================================${NC}"
echo -e "${GREEN}  编排器启动中...${NC}"
echo -e "${GREEN}  按 Ctrl+C 停止${NC}"
echo -e "${GREEN}============================================================================${NC}"
echo ""

exec $PYTHON_CMD
