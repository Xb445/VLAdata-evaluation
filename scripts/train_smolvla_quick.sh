#!/usr/bin/env bash
# ==============================================================================
# SmolVLA 快速训练 / 冒烟测试脚本
# ==============================================================================
# 用途:
#   1) 冒烟测试: 验证 conda 环境、数据集、SmolVLA 基座模型能否正常加载并完成训练循环
#   2) 快速试训: 用少量步数观察 loss / 显存 / 数据加载是否正常，再决定正式训练参数
#
# 使用方法:
#   bash scripts/train_smolvla_quick.sh                  # 默认冒烟测试（10 步）
#   bash scripts/train_smolvla_quick.sh --smoke          # 极简冒烟（3 步，batch=1）
#   bash scripts/train_smolvla_quick.sh --dry-run        # 只打印训练命令，不执行
#   bash scripts/train_smolvla_quick.sh --resume         # 从输出目录的最新 checkpoint 续训
#   STEPS=500 BATCH_SIZE=8 bash scripts/train_smolvla_quick.sh   # 快速试训
#
# 常用环境变量覆盖:
#   STEPS=500 BATCH_SIZE=8 OUTPUT_DIR=outputs/train/my_smolvla \
#   bash scripts/train_smolvla_quick.sh
#
# 说明:
#   - 数据集 so101-sim-pickplace-v2 的相机键为 observation.images.front / .wrist，
#     而 smolvla_base 期望 observation.images.camera1/2/3，脚本通过 --rename_map
#     将 front→camera1、wrist→camera2（与官方/社区微调配置一致）。
#   - 数据集视频为 AV1 编码，脚本强制使用 pyav 解码：
#     本机 torchcodec 因缺少 libnppicc.so.12 无法加载，若走默认后端会启动失败。
#   - 基座模型 config.json 中 push_to_hub=true，脚本统一关闭，避免离线环境报错。
# ==============================================================================

set -uo pipefail

# ── 参数解析 ────────────────────────────────────────────────────────────
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --smoke)   SMOKE_MODE=true ;;
        --dry-run) DRY_RUN=true ;;
        --resume)  RESUME=true ;;
        -h|--help)
            sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "未知参数: $arg（可用: --smoke / --dry-run / --resume / --help）" >&2
            exit 2
            ;;
    esac
done

# ── 路径与运行环境 ──────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(dirname "$SCRIPT_DIR")}"

# conda 安装脚本（当前主机为 /home/pd/miniconda3）
CONDA_SH="${CONDA_SH:-/home/pd/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-evo-rl-videos}"

# 数据集（LeRobotDataset v3.0，包含 meta/ data/ videos/）
DATASET_ROOT="${DATASET_ROOT:-/home/pd/datasets/so101-sim-pickplace-v2}"
DATASET_REPO_ID="${DATASET_REPO_ID:-so101-sim-pickplace-v2}"

# SmolVLA 基座模型目录（含 config.json + model.safetensors）
POLICY_PATH="${POLICY_PATH:-/home/pd/zh_videos/tiankung-vla-tutorial-main/pretrained_models/smolvla_base}"

# 输出目录（必须位于项目 outputs/ 下，便于统一管理）
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/train/smolvla_quick_so101_pickplace_v2}"
JOB_NAME="${JOB_NAME:-smolvla_quick_so101_pickplace_v2}"

# ── 训练超参数（冒烟测试默认值）────────────────────────────────────────
RESUME="${RESUME:-false}"
CLEAN="${CLEAN:-true}"            # 非续训时自动清理旧输出目录，避免 FileExistsError
BATCH_SIZE="${BATCH_SIZE:-2}"     # RTX 3090 24GB 冒烟测试足够；显存不足时降为 1
STEPS="${STEPS:-10}"
LOG_FREQ="${LOG_FREQ:-1}"
SAVE_FREQ="${SAVE_FREQ:-10}"      # 冒烟测试结束即保存 1 个 checkpoint
NUM_WORKERS="${NUM_WORKERS:-0}"   # 冒烟测试用主进程加载，避免多进程干扰
EVAL_FREQ="${EVAL_FREQ:-0}"       # 该数据集无仿真评估环境，跳过 eval
SEED="${SEED:-1000}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-true}"

# ── 模型 / 数据加载参数 ─────────────────────────────────────────────────
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
USE_AMP="${USE_AMP:-false}"       # RTX 3090 可用 bf16，如需提速可设为 true
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"   # 见文件头说明：torchcodec 在本机不可用
HF_OFFLINE="${HF_OFFLINE:-true}"  # 基座模型与 tokenizer 均已缓存，默认离线加载
WANDB_ENABLE="${WANDB_ENABLE:-false}"
EMPTY_CAMERAS="${EMPTY_CAMERAS:-}"       # 如需补齐缺失相机可设 1

# 数据集相机键 → 基座模型相机键（SmolVLA 内部固定按 camera1/2/3 排列）
DEFAULT_RENAME_MAP='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'
RENAME_MAP="${RENAME_MAP:-$DEFAULT_RENAME_MAP}"

# 额外透传参数（按空格分隔），例如 EXTRA_ARGS="--policy.optimizer_lr=5e-5"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# --smoke 极简模式：覆盖为最小可运行配置
if [ "${SMOKE_MODE:-false}" = true ]; then
    BATCH_SIZE=1
    STEPS=3
    LOG_FREQ=1
    SAVE_FREQ=3
fi

# ── 基础校验 ────────────────────────────────────────────────────────────
echo "=============================================================="
echo " SmolVLA 快速训练 / 冒烟测试"
echo "=============================================================="
echo "  项目目录:   $PROJECT_DIR"
echo "  数据集:     $DATASET_ROOT"
echo "  基座模型:   $POLICY_PATH"
echo "  输出目录:   $OUTPUT_DIR"
echo "  batch_size: $BATCH_SIZE   steps: $STEPS   workers: $NUM_WORKERS"
echo "  设备:       $POLICY_DEVICE   AMP: $USE_AMP   视频后端: $VIDEO_BACKEND"
echo ""

if [ ! -f "$DATASET_ROOT/meta/info.json" ]; then
    echo "✗ 数据集不完整: 未找到 $DATASET_ROOT/meta/info.json" >&2
    exit 1
fi
if [ ! -f "$POLICY_PATH/config.json" ] || [ ! -f "$POLICY_PATH/model.safetensors" ]; then
    echo "✗ 基座模型不完整: 需要 $POLICY_PATH/config.json 与 model.safetensors" >&2
    exit 1
fi
if [ ! -f "$CONDA_SH" ]; then
    echo "✗ 未找到 conda 初始化脚本: $CONDA_SH（可通过 CONDA_SH 覆盖）" >&2
    exit 1
fi

# ── 激活 conda 环境 ────────────────────────────────────────────────────
cd "$PROJECT_DIR" || { echo "✗ 无法进入项目目录 $PROJECT_DIR" >&2; exit 1; }
# conda 的 activate.d 脚本会引用未定义变量（如 MKL_INTERFACE_LAYER），
# 因此激活期间临时关闭 set -u。
set +u
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV" || { set -u; echo "✗ 无法激活 conda 环境 $CONDA_ENV" >&2; exit 1; }
set -u

echo "[环境] python: $(which python)"
echo "[环境] torch:  $(python -c 'import torch; print(torch.__version__)')"
echo "[环境] cuda:   $(python -c 'import torch; print(torch.cuda.is_available())')"
GPU_NAME=$(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")' 2>/dev/null || echo "N/A")
echo "[环境] gpu:    $GPU_NAME"

# ── 离线模式 ────────────────────────────────────────────────────────────
if [ "$HF_OFFLINE" = true ]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    echo "[环境] 离线加载: 是 (HF_HUB_OFFLINE=1)"
else
    echo "[环境] 离线加载: 否"
fi
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# ── 输出目录处理 ────────────────────────────────────────────────────────
if [ "$RESUME" != true ] && [ "$CLEAN" != false ] && [ -d "$OUTPUT_DIR" ]; then
    echo "[准备] 清理已存在的输出目录: $OUTPUT_DIR"
    rm -rf "$OUTPUT_DIR"
fi
mkdir -p "$(dirname "$OUTPUT_DIR")"

# ── 构建训练命令（数组形式，避免 JSON 中的空格被拆分）──────────────────
CMD=(
    lerobot-train
    --policy.path="$POLICY_PATH"
    --policy.device="$POLICY_DEVICE"
    --policy.use_amp="$USE_AMP"
    --policy.push_to_hub=false
    --dataset.repo_id="$DATASET_REPO_ID"
    --dataset.root="$DATASET_ROOT"
    --dataset.video_backend="$VIDEO_BACKEND"
    --output_dir="$OUTPUT_DIR"
    --job_name="$JOB_NAME"
    --batch_size="$BATCH_SIZE"
    --steps="$STEPS"
    --eval_freq="$EVAL_FREQ"
    --log_freq="$LOG_FREQ"
    --save_freq="$SAVE_FREQ"
    --save_checkpoint="$SAVE_CHECKPOINT"
    --num_workers="$NUM_WORKERS"
    --seed="$SEED"
    --rename_map="$RENAME_MAP"
    --wandb.enable="$WANDB_ENABLE"
)

[ "$RESUME" = true ] && CMD+=(--resume=true)
[ -n "$EMPTY_CAMERAS" ] && CMD+=(--policy.empty_cameras="$EMPTY_CAMERAS")
if [ -n "$EXTRA_ARGS" ]; then
    read -r -a _EXTRA <<< "$EXTRA_ARGS"
    CMD+=("${_EXTRA[@]}")
fi

echo ""
echo "[命令] 训练命令（可复制单独执行）:"
printf '  %q' "${CMD[@]}"
printf '\n\n'

if [ "$DRY_RUN" = true ]; then
    echo "[dry-run] 仅打印命令，未执行训练。"
    exit 0
fi

# ── 开始训练 ────────────────────────────────────────────────────────────
START_TIME=$(date +%s)
"${CMD[@]}"
EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=============================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo " ✓ 训练完成，耗时 ${ELAPSED}s"
else
    echo " ✗ 训练失败（退出码 $EXIT_CODE），耗时 ${ELAPSED}s"
fi
echo "   输出目录: $OUTPUT_DIR"
if [ -d "$OUTPUT_DIR/checkpoints" ]; then
    LATEST=$(ls "$OUTPUT_DIR/checkpoints" 2>/dev/null | grep -vx 'last' | sort -V | tail -1)
    [ -n "$LATEST" ] && echo "   最新 checkpoint: $OUTPUT_DIR/checkpoints/$LATEST"
fi
echo "=============================================================="

exit $EXIT_CODE
