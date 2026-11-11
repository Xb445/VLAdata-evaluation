#!/bin/bash
# =============================================================================
# Evo-RL / LeRobot 项目依赖一键安装脚本
#
# 用法: bash install_deps.sh [--no-cuda] [--no-docs] [--no-dev]
#
# 说明:
#   - 此脚本不包含 conda 环境创建，请在已有的 conda/pyenv 环境中运行
#   - 默认安装 CUDA 版 PyTorch 及全部可选依赖
#   - 可选参数可跳过部分依赖组
# =============================================================================

set -e
set -o pipefail

# ── 颜色输出 ────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()  { echo -e "${CYAN}[STEP]${NC} $*"; }

# ── 参数解析 ────────────────────────────────────────────────────────
NO_CUDA=false
NO_DOCS=false
NO_DEV=false

for arg in "$@"; do
    case $arg in
        --no-cuda) NO_CUDA=true ;;
        --no-docs) NO_DOCS=true ;;
        --no-dev)  NO_DEV=true ;;
        -h|--help)
            echo "用法: bash install_deps.sh [--no-cuda] [--no-docs] [--no-dev]"
            exit 0
            ;;
    esac
done

# ── 检测 Python 可执行文件 ─────────────────────────────────────────
PYTHON_BIN=""
for candidate in python3 python; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON_BIN="$candidate"
        break
    fi
done
if [ -z "$PYTHON_BIN" ]; then
    echo "错误: 未找到 Python，请先激活 conda/pyenv 环境" >&2
    exit 1
fi

# ── 获取脚本/项目根目录 ─────────────────────────────────────────────
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR")
cd "$PROJECT_ROOT"

log_info "项目根目录: $PROJECT_ROOT"
log_info "Python: $($PYTHON_BIN --version)"

# ╔════════════════════════════════════════════════════════════════════╗
# ║  1. 系统级依赖 (apt)                                             ║
# ╚════════════════════════════════════════════════════════════════════╝
log_step "1/5 安装系统级依赖..."

if command -v apt-get &>/dev/null; then
    CAN_SUDO=$(sudo -n true 2>/dev/null && echo true || echo false)
    if $CAN_SUDO; then
        log_info "检测到 apt，安装 CAN/串口/视频 相关系统包..."
        sudo apt-get update -qq
        sudo apt-get install -y -qq \
            can-utils \
            iproute2 \
            usbutils \
            ffmpeg \
            libgl1-mesa-glx \
            libglib2.0-0 \
            libsm6 \
            libxext6 \
            libxrender-dev \
            libgomp1 \
            v4l-utils
        log_info "系统包安装完成"
    else
        log_warn "无 sudo 权限，跳过系统包安装。请手动安装: can-utils, ffmpeg, libgl1 等"
    fi
else
    log_warn "未检测到 apt-get，请手动安装系统依赖: can-utils, ffmpeg, libgl1 等"
fi

# ╔════════════════════════════════════════════════════════════════════╗
# ║  2. 升级 pip / setuptools                                        ║
# ╚════════════════════════════════════════════════════════════════════╝
log_step "2/5 升级 pip 和构建工具..."

$PYTHON_BIN -m pip install --upgrade pip setuptools wheel -q
log_info "pip/setuptools/wheel 已升级"

# ╔════════════════════════════════════════════════════════════════════╗
# ║  3. PyTorch (CUDA / CPU)                                         ║
# ╚════════════════════════════════════════════════════════════════════╝
log_step "3/5 安装 PyTorch..."

if $NO_CUDA; then
    log_info "安装 CPU 版 PyTorch..."
    $PYTHON_BIN -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -q
else
    # 自动检测 CUDA 版本并选择对应的 PyTorch 版本
    CUDA_VER=""
    if command -v nvidia-smi &>/dev/null; then
        CUDA_VER=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" 2>/dev/null || echo "")
    fi
    if [[ -n "$CUDA_VER" ]]; then
        log_info "检测到 CUDA $CUDA_VER"
    else
        log_warn "未检测到 nvidia-smi，默认使用 CUDA 11.8"
        CUDA_VER="11.8"
    fi
    # 根据 CUDA 主版本选择 PyTorch 索引
    CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
    if [[ "$CUDA_MAJOR" -ge 12 ]]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
        log_info "安装 CUDA 12.1 版 PyTorch..."
        $PYTHON_BIN -m pip install torch torchvision torchaudio --index-url "$TORCH_INDEX" -q
    else
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
        log_info "安装 CUDA 11.8 版 PyTorch..."
        $PYTHON_BIN -m pip install torch==2.2.1+cu118 torchvision==0.17.1+cu118 torchaudio==2.2.1 \
            --index-url "$TORCH_INDEX" -q
    fi
fi

log_info "PyTorch 安装完成"

# ╔════════════════════════════════════════════════════════════════════╗
# ║  4. 项目核心 + 全部可选依赖                                      ║
# ╚════════════════════════════════════════════════════════════════════╝
log_step "4/5 安装项目依赖 (lerobot[all] + 额外包)..."

# 安装项目本身 (editable 模式 + all 可选依赖)
$PYTHON_BIN -m pip install -e ".[all]" -q

# 额外依赖（不在 pyproject.toml [all] 中或需要特殊处理）
log_info "安装额外依赖..."

$PYTHON_BIN -m pip install -q \
    rospkg \
    safetensors \
    python-can \
    hidapi \
    pyzmq \
    num2words \
    scipy \
    Pillow \
    matplotlib \
    pandas \
    scikit-image \
    dm-tree \
    timm \
    ninja \
    faker \
    qwen-vl-utils \
    onnxruntime \
    meshcat \
    casadi

# ── piper_sdk (本地包，在 src/ 下) ───────────────────────────────────
PIPER_SDK_DIR="$PROJECT_ROOT/src/piper_sdk"
if [ -d "$PIPER_SDK_DIR" ] && [ -f "$PIPER_SDK_DIR/setup.py" ]; then
    log_info "安装本地 piper_sdk..."
    $PYTHON_BIN -m pip install -e "$PIPER_SDK_DIR" -q
elif [ -d "$PIPER_SDK_DIR" ] && [ -f "$PIPER_SDK_DIR/pyproject.toml" ]; then
    log_info "安装本地 piper_sdk..."
    $PYTHON_BIN -m pip install -e "$PIPER_SDK_DIR" -q
else
    log_warn "piper_sdk 目录未找到或缺少 setup.py/pyproject.toml，跳过"
fi

log_info "项目依赖安装完成"

# ╔════════════════════════════════════════════════════════════════════╗
# ║  5. 文档 & 开发工具 (可选)                                        ║
# ╚════════════════════════════════════════════════════════════════════╝
log_step "5/5 安装文档和开发工具..."

if ! $NO_DOCS; then
    if [ -f "$PROJECT_ROOT/docs-requirements.txt" ]; then
        log_info "安装文档依赖..."
        $PYTHON_BIN -m pip install -r "$PROJECT_ROOT/docs-requirements.txt" -q 2>/dev/null || \
            log_warn "部分文档依赖安装失败（可忽略）"
    fi
fi

if ! $NO_DEV; then
    log_info "安装开发工具..."
    $PYTHON_BIN -m pip install -q \
        pre-commit \
        debugpy \
        mypy \
        pytest \
        pytest-timeout \
        pytest-cov \
        grpcio-tools \
        2>/dev/null || log_warn "部分开发工具安装失败（可忽略）"
fi

# ╔════════════════════════════════════════════════════════════════════╗
# ║  验证                                                            ║
# ╚════════════════════════════════════════════════════════════════════╝
echo ""
log_info "=============================================="
log_info "  依赖安装完成！验证关键包版本:"
log_info "=============================================="
echo ""

$PYTHON_BIN -c "
import sys
# (pip包名, import名) 映射表 — pip包名和import名可能不同
pkg_map = [
    ('torch', 'torch'),
    ('torchvision', 'torchvision'),
    ('numpy', 'numpy'),
    ('opencv-python-headless', 'cv2'),
    ('av', 'av'),
    ('datasets', 'datasets'),
    ('diffusers', 'diffusers'),
    ('huggingface_hub', 'huggingface_hub'),
    ('accelerate', 'accelerate'),
    ('gymnasium', 'gymnasium'),
    ('einops', 'einops'),
    ('rerun-sdk', 'rerun_sdk'),
    ('draccus', 'draccus'),
    ('wandb', 'wandb'),
    ('transformers', 'transformers'),
    ('piper_sdk', 'piper_sdk'),
    ('safetensors', 'safetensors'),
]
for pkg_name, import_name in pkg_map:
    try:
        mod = __import__(import_name)
        ver = getattr(mod, '__version__', '?')
        print(f'  ✅ {pkg_name}: {ver}')
    except Exception:
        print(f'  ❌ {pkg_name}: 未安装')

# CUDA 检测
try:
    import torch
    print(f'  🎯 CUDA 可用: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'  🎯 CUDA 版本: {torch.version.cuda}')
        print(f'  🎯 GPU 数量:  {torch.cuda.device_count()}')
except Exception:
    pass
"

echo ""
log_info "全部完成！可以开始使用 LeRobot 了 🚀"
