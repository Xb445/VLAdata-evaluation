#!/bin/bash
# ============================================================================
# Evo-RL 项目环境 (evo-rl-videos) 一键部署脚本
#
# 适用机器: Ubuntu 22.04 / 24.04 (x86_64) + NVIDIA GPU (CUDA 12.x)
# 目标:     在全新机器上复现与源开发机一致的 evo-rl-videos conda 环境
#
# 用法:
#   bash setup_evo_rl_videos.sh                     # 标准部署 (CUDA GPU)
#   bash setup_evo_rl_videos.sh --no-cuda           # CPU 部署 (无 GPU)
#   bash setup_evo_rl_videos.sh --with-vllm         # 额外部署 vllm-eval (VLM 推理服务)
#   bash setup_evo_rl_videos.sh --offline /mnt/src  # 离线模式: 从 /mnt/src 拷贝项目源码
#   bash setup_evo_rl_videos.sh --skip-system       # 跳过 apt 系统依赖
#   bash setup_evo_rl_videos.sh --skip-conda        # 跳过 conda 创建 (复用已存在的环境)
#   bash setup_evo_rl_videos.sh --env-name xxx      # 自定义环境名 (默认 evo-rl-videos)
#   bash setup_evo_rl_videos.sh --mirror tsinghua    # 清华 pip + 阿里 torch 镜像
#   bash setup_evo_rl_videos.sh --mirror none        # 强制使用官方源
#   bash setup_evo_rl_videos.sh --mirror auto        # 自动检测网络并选择 (默认)
#
# 前置要求:
#   - root 或 sudo 权限 (安装 apt 系统包时)
#   - 网络可访问 pypi / conda-forge / download.pytorch.org (或使用 --mirror)
#   - 项目源码已拷贝到目标机器 (推荐 rsync 整目录), 或使用 --offline 指定来源
# ============================================================================

set -euo pipefail

# ── 默认参数 ────────────────────────────────────────────────────────────
ENV_NAME="evo-rl-videos"
NO_CUDA=false
WITH_VLLM=false
SKIP_SYSTEM=false
SKIP_CONDA=false
OFFLINE_SRC=""
MIRROR="auto"    # none | tsinghua | aliyun | auto (自动检测网络并选择镜像)

# ── 版本锁定 (与源开发机一致) ──────────────────────────────────────────
PYTHON_VERSION="3.10"
TORCH_VERSION="2.7.1"
TORCHVISION_VERSION="0.22.1"
TORCHCODEC_VERSION="0.5"
TORCH_INDEX_CUDA="https://download.pytorch.org/whl/cu126"   # 源机 torch 为 cu126 构建
TORCH_INDEX_CPU="https://download.pytorch.org/whl/cpu"
TRANSFORMERS_GIT="git+https://github.com/huggingface/transformers.git@dcddb970176382c0fcf4521b0c0e6fc15894dfe0#egg=transformers"
EVORL_GIT="https://github.com/MINT-SJTU/Evo-RL.git"
VLLM_VERSION="0.21.0"

# ── 国内镜像源 (网络不通时自动/手动切换) ──────────────────────────────
PIP_TUNA="https://pypi.tuna.tsinghua.edu.cn/simple"
PIP_ALIYUN="https://mirrors.aliyun.com/pypi/simple/"
# PyTorch 国内镜像均为扁平目录结构, 只能配合 --find-links 使用 (不能当 --index-url)
TORCH_ALIYUN="https://mirrors.aliyun.com/pytorch-wheels"
TORCH_SJTU="https://mirror.sjtu.edu.cn/pytorch-wheels"
TORCH_NJU="https://mirror.nju.edu.cn/pytorch-wheels"
TORCH_BFSU="https://mirrors.bfsu.edu.cn/pytorch-wheels"
CONDA_TUNA_MAIN="https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main"
CONDA_TUNA_FREE="https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free"
CONDA_TUNA_CF="https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge"
CONDA_ALIYUN_MAIN="https://mirrors.aliyun.com/anaconda/pkgs/main"
CONDA_ALIYUN_CF="https://mirrors.aliyun.com/anaconda/cloud/conda-forge"

# ── 颜色输出 ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()  { echo -e "${CYAN}[STEP]${NC} $*"; }

# ── conda / 环境路径辅助函数 ──────────────────────────────────────────
# 查找 conda 可执行文件: 优先常见安装路径, 再通过 PATH 查找
# (type -P 只返回真实可执行路径, 不会匹配 conda 的 shell 函数)
find_conda() {
    local c p
    for c in "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" \
             "/opt/conda/bin/conda" "/opt/anaconda3/bin/conda" "/opt/miniconda3/bin/conda" \
             "/data/anaconda3/bin/conda" "/data/miniconda3/bin/conda"; do
        [ -x "$c" ] && { echo "$c"; return 0; }
    done
    p="$(type -P conda 2>/dev/null || true)"
    [ -n "$p" ] && { echo "$p"; return 0; }
    return 1
}

# 查询指定环境的真实前缀路径 (通过 conda 自身查询, 兼容自定义 envs_dirs)
env_prefix_of() {
    "$CONDA_BIN" env list 2>/dev/null | awk -v n="$1" '$1==n && $1!="base" {print $NF; exit}'
}

# 检测 URL 是否可达 (curl 会遵循 http_proxy 等代理环境变量)
net_reachable() {
    timeout 8 curl -fsS --connect-timeout 5 -o /dev/null -I "$1" 2>/dev/null
}

# 根据 MIRROR 选择 pip / torch / conda 源
# TORCH_MIRRORS: 候选 PyTorch 镜像 base 列表 (空格分隔); 为空表示使用官方源
resolve_mirror() {
    TORCH_MIRRORS=""
    case "$MIRROR" in
        tsinghua)
            log_info "使用清华 pip 镜像 + PyTorch 国内镜像"
            export PIP_INDEX_URL="$PIP_TUNA"
            TORCH_MIRRORS="$TORCH_ALIYUN $TORCH_SJTU $TORCH_NJU $TORCH_BFSU"
            CONDA_MAIN="$CONDA_TUNA_MAIN"; CONDA_FREE="$CONDA_TUNA_FREE"; CONDA_CF="$CONDA_TUNA_CF"
            ;;
        aliyun)
            log_info "使用阿里云 pip + PyTorch 国内镜像"
            export PIP_INDEX_URL="$PIP_ALIYUN"
            TORCH_MIRRORS="$TORCH_ALIYUN $TORCH_SJTU $TORCH_NJU $TORCH_BFSU"
            CONDA_MAIN="$CONDA_ALIYUN_MAIN"; CONDA_CF="$CONDA_ALIYUN_CF"; CONDA_FREE=""
            ;;
        auto|*)
            if net_reachable "https://pypi.org/simple/"; then
                log_info "pypi.org 可访问 → pip 使用官方源"
            else
                log_warn "pypi.org 不可达 → pip 自动切换清华镜像"
                export PIP_INDEX_URL="$PIP_TUNA"
            fi
            TORCH_DIR_CK="cu126"; [ $NO_CUDA = true ] && TORCH_DIR_CK="cpu"
            if net_reachable "https://download.pytorch.org/whl/$TORCH_DIR_CK/"; then
                log_info "download.pytorch.org 可访问 → torch 使用官方源"
            else
                log_warn "download.pytorch.org 不可达 → torch 自动切换国内镜像"
                TORCH_MIRRORS="$TORCH_ALIYUN $TORCH_SJTU $TORCH_NJU $TORCH_BFSU"
            fi
            ;;
        none)
            log_info "使用官方源 (--mirror none)"
            ;;
    esac
}

# 检查文件 URL 是否可下载 (200/206 表示存在)
file_ok() {
    local code
    code="$(curl -4 -s -m 10 -o /dev/null -w "%{http_code}" -r 0-0 "$1" 2>/dev/null || echo 000)"
    [ "$code" = "200" ] || [ "$code" = "206" ]
}

# 探测并挑选可用的 PyTorch 镜像 (校验 torch/torchvision/torchcodec 三个 wheel 均存在)
# 结果写入 TORCH_FIND_LINKS: 非空则 torch 安装使用 --find-links, 为空则用官方 --index-url
pick_torch_source() {
    TORCH_FIND_LINKS=""
    [ -z "$TORCH_MIRRORS" ] && return 0
    local dir="cu126"; [ $NO_CUDA = true ] && dir="cpu"
    local suf="%2Bcu126"; [ $NO_CUDA = true ] && suf=""
    local base w need_all ok
    for base in $TORCH_MIRRORS; do
        ok=true
        for w in \
            "torch-$TORCH_VERSION$suf-cp310-cp310-manylinux_2_28_x86_64.whl" \
            "torchvision-$TORCHVISION_VERSION$suf-cp310-cp310-manylinux_2_28_x86_64.whl" \
            "torchcodec-$TORCHCODEC_VERSION$suf-cp310-cp310-manylinux_2_28_x86_64.whl"; do
            if ! file_ok "$base/$dir/$w"; then ok=false; break; fi
        done
        if $ok; then
            TORCH_FIND_LINKS="$base/$dir"
            log_info "PyTorch 镜像可用: $TORCH_FIND_LINKS"
            return 0
        else
            log_warn "镜像缺少所需 wheel, 跳过: $base/$dir"
        fi
    done
    log_error "所有 PyTorch 国内镜像均缺少所需 wheel (torch/torchvision/torchcodec $([ $NO_CUDA = true ] && echo CPU || echo cu126) cp310)"
    log_error "请检查网络或手动指定可用镜像源"
    exit 1
}

# 配置 conda 镜像 channel (仅在选用镜像时生效)
setup_conda_mirror() {
    [ -z "${CONDA_CF:-}" ] && return 0
    log_info "配置 conda 镜像 channel..."
    "$CONDA_BIN" config --add channels "$CONDA_MAIN"
    [ -n "${CONDA_FREE:-}" ] && "$CONDA_BIN" config --add channels "$CONDA_FREE"
    "$CONDA_BIN" config --add channels "$CONDA_CF"   # 最后添加 → 优先级最高
    "$CONDA_BIN" config --set show_channel_urls yes
}

# ── 参数解析 ────────────────────────────────────────────────────────────
usage() {
    sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-cuda)    NO_CUDA=true; shift ;;
        --with-vllm)  WITH_VLLM=true; shift ;;
        --skip-system) SKIP_SYSTEM=true; shift ;;
        --skip-conda)  SKIP_CONDA=true; shift ;;
        --offline)    OFFLINE_SRC="$2"; shift 2 ;;
        --env-name)   ENV_NAME="$2"; shift 2 ;;
        --mirror)     MIRROR="$2"; shift 2 ;;
        -h|--help)    usage ;;
        *) shift ;;
    esac
done

# 解析镜像设置 (在进入安装步骤前完成网络探测)
resolve_mirror
# 挑选可用的 PyTorch 镜像 (探测实际 wheel 文件)
pick_torch_source

# ── 定位脚本目录 / 项目根目录 ───────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/../pyproject.toml" ]; then
    PROJECT_ROOT="$(realpath "$SCRIPT_DIR/..")"
else
    PROJECT_ROOT="$SCRIPT_DIR"
fi
REQ_FILE="$SCRIPT_DIR/requirements-evo-rl-videos.txt"

# ╔════════════════════════════════════════════════════════════════════╗
# ║  1. 前置检查                                                      ║
# ╚════════════════════════════════════════════════════════════════════╝
log_step "1/9 前置检查..."

if [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
    log_error "仅支持 Linux x86_64 机器"
    exit 1
fi

if ! $NO_CUDA; then
    if command -v nvidia-smi &>/dev/null; then
        log_info "检测到 NVIDIA GPU:"
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -5
    else
        log_warn "未检测到 nvidia-smi。若本机有 GPU 请先安装 NVIDIA 驱动; 否则请加 --no-cuda"
    fi
fi

if [ ! -f "$REQ_FILE" ]; then
    log_error "未找到 $REQ_FILE, 请确认与脚本放在同一目录"
    exit 1
fi

# ╔════════════════════════════════════════════════════════════════════╗
# ║  2. 定位 / 安装 conda                                              ║
# ╚════════════════════════════════════════════════════════════════════╝
log_step "2/9 检查 conda..."

CONDA_BIN="$(find_conda || true)"

if [ -z "$CONDA_BIN" ]; then
    log_info "未检测到 conda, 安装 Miniconda3 (Python 3.10) 到 ~/miniconda3 ..."
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    TMP_SH="/tmp/miniconda3.sh"
    curl -fsSL "$MINICONDA_URL" -o "$TMP_SH" || { log_error "下载 Miniconda 失败"; exit 1; }
    bash "$TMP_SH" -b -p "$HOME/miniconda3"
    rm -f "$TMP_SH"
    CONDA_BIN="$HOME/miniconda3/bin/conda"
fi

log_info "conda: $CONDA_BIN ($("$CONDA_BIN" --version))"

# 权威获取 conda 根目录 (兼容符号链接 / condabin / 自定义安装位置)
CONDA_BASE="$("$CONDA_BIN" info --base 2>/dev/null | tail -1)"
[ -z "$CONDA_BASE" ] && CONDA_BASE="$(dirname "$(dirname "$CONDA_BIN")")"
log_info "conda 根目录: $CONDA_BASE"

# 配置 conda 镜像 channel (tsinghua/aliyun 模式)
setup_conda_mirror

# ╔════════════════════════════════════════════════════════════════════╗
# ║  3. 创建 conda 环境                                                ║
# ╚════════════════════════════════════════════════════════════════════╝
log_step "3/9 创建 conda 环境 $ENV_NAME ..."

ENV_PREFIX="$(env_prefix_of "$ENV_NAME")"

if $SKIP_CONDA; then
    log_warn "--skip-conda: 跳过环境创建, 假定 $ENV_NAME 已存在 (请先 conda activate $ENV_NAME)"
else
    if [ -n "$ENV_PREFIX" ]; then
        log_warn "环境 $ENV_NAME 已存在: $ENV_PREFIX (跳过创建)"
    else
        log_info "创建 conda 环境 $ENV_NAME (python=$PYTHON_VERSION)..."
        "$CONDA_BIN" create -n "$ENV_NAME" -y -q "python=$PYTHON_VERSION" -c conda-forge || {
            log_error "conda 创建环境失败! 常见原因与处理:"
            log_error "  1) 无法访问 conda-forge → 配置清华源: conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge"
            log_error "  2) 磁盘空间/权限不足 → 检查安装目录"
            exit 1
        }
        ENV_PREFIX="$(env_prefix_of "$ENV_NAME")"
    fi
    log_info "安装 conda 系统级依赖 (ffmpeg / casadi / pinocchio / coal) ..."
    "$CONDA_BIN" install -n "$ENV_NAME" -y -q \
        ffmpeg casadi pinocchio-python coal-python -c conda-forge \
        || log_warn "conda 依赖部分安装失败 (非核心, 可稍后手动安装)"
fi

# ── 权威定位环境内 python (不硬编码路径, 兼容自定义 envs_dirs) ──────
if [ -z "$ENV_PREFIX" ]; then
    ENV_PREFIX="$(env_prefix_of "$ENV_NAME")"
fi
[ -z "$ENV_PREFIX" ] && ENV_PREFIX="$CONDA_BASE/envs/$ENV_NAME"
ENV_PY="$ENV_PREFIX/bin/python"
if [ ! -x "$ENV_PY" ]; then
    log_error "无法定位环境 $ENV_NAME 的 Python: $ENV_PY 不存在"
    log_error "请检查: 1) conda env list 中是否存在 $ENV_NAME"
    log_error "        2) conda 的 envs_dirs 配置 (cat ~/.condarc | grep envs_dirs)"
    exit 1
fi
log_info "环境 Python: $ENV_PY"
ENV_PIP="$ENV_PY -m pip"

# ╔════════════════════════════════════════════════════════════════════╗
# ║  4. apt 系统依赖                                                   ║
# ╚════════════════════════════════════════════════════════════════════╝
log_step "4/9 安装 apt 系统依赖..."

if $SKIP_SYSTEM; then
    log_warn "--skip-system: 跳过 apt 安装"
elif command -v apt-get &>/dev/null; then
    if sudo -n true 2>/dev/null; then
        sudo apt-get update -qq
        # Ubuntu 24.04 上 libgl1-mesa-glx 为过渡包, 容错处理
        sudo apt-get install -y -qq \
            can-utils iproute2 usbutils ffmpeg \
            libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 v4l-utils \
            || { log_warn "部分 apt 包安装失败, 请手动安装: can-utils ffmpeg libgl1 等"; }
    else
        log_warn "无免密 sudo, 请手动运行: sudo apt-get install -y can-utils iproute2 usbutils ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 v4l-utils"
    fi
else
    log_warn "未检测到 apt-get, 请手动安装系统依赖"
fi

# ╔════════════════════════════════════════════════════════════════════╗
# ║  5. 获取项目源码                                                   ║
# ╚════════════════════════════════════════════════════════════════════╝
log_step "5/9 获取项目源码..."

HAS_LOCAL_CODE=false
if [ -f "$PROJECT_ROOT/pyproject.toml" ] && [ -d "$PROJECT_ROOT/src/lerobot" ]; then
    HAS_LOCAL_CODE=true
    log_info "使用已存在的项目目录: $PROJECT_ROOT"
elif [ -n "$OFFLINE_SRC" ]; then
    if [ -f "$OFFLINE_SRC/pyproject.toml" ]; then
        log_info "从离线目录拷贝项目: $OFFLINE_SRC -> $PWD/Evo-RL ..."
        if command -v rsync &>/dev/null; then
            rsync -a --exclude '.git' --exclude '.conda' --exclude 'outputs' \
                --exclude 'datasets' --exclude 'act_vlm_logs' \
                "$OFFLINE_SRC/" "$PWD/Evo-RL/"
        else
            cp -r "$OFFLINE_SRC" "$PWD/Evo-RL"
        fi
        PROJECT_ROOT="$PWD/Evo-RL"
        HAS_LOCAL_CODE=true
    else
        log_error "--offline 指定的目录中未找到 pyproject.toml"
        exit 1
    fi
else
    log_warn "当前目录没有项目源码, 尝试 git clone 上游仓库 (不含本地自定义代码!)"
    if [ ! -d "$PWD/Evo-RL" ]; then
        git clone --depth 1 "$EVORL_GIT" "$PWD/Evo-RL"
    fi
    PROJECT_ROOT="$PWD/Evo-RL"
    log_warn "⚠ 上游仓库不包含本地代码 (act_vlm_orchestrator_v2 / src/piper_sdk / transformers / robots/single_piper 等)"
    log_warn "⚠ 请手动将源开发机上的这些目录拷贝到 $PROJECT_ROOT 后重新运行本脚本"
    exit 1
fi

# ╔════════════════════════════════════════════════════════════════════╗
# ║  6. 升级 pip + 安装 PyTorch                                        ║
# ╚════════════════════════════════════════════════════════════════════╝
log_step "6/9 安装 PyTorch ($([ $NO_CUDA = true ] && echo CPU || echo CUDA-12.6))..."

$ENV_PIP install --upgrade pip setuptools wheel -q

TORCH_INDEX="$([ $NO_CUDA = true ] && echo "$TORCH_INDEX_CPU" || echo "$TORCH_INDEX_CUDA")"
if [ -n "$TORCH_FIND_LINKS" ]; then
    # 国内镜像为扁平目录, 用 --find-links 下载 torch 三件套 (其余依赖走 pip 索引)
    log_info "PyTorch 下载源 (find-links): $TORCH_FIND_LINKS"
    $ENV_PIP install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" "torchcodec==$TORCHCODEC_VERSION" \
        --find-links "$TORCH_FIND_LINKS" -q
else
    log_info "PyTorch 下载源 (index-url): $TORCH_INDEX"
    $ENV_PIP install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" "torchcodec==$TORCHCODEC_VERSION" \
        --index-url "$TORCH_INDEX" -q
fi

# ╔════════════════════════════════════════════════════════════════════╗
# ║  7. 安装 pip 依赖清单                                               ║
# ╚════════════════════════════════════════════════════════════════════╝
log_step "7/9 安装 pip 依赖清单 (requirements-evo-rl-videos.txt)..."

if $NO_CUDA; then
    # CPU 模式: 过滤掉 nvidia-* CUDA 运行库
    grep -viE '^(nvidia-|triton)' "$REQ_FILE" > /tmp/req-evo-rl-cpu.txt
    $ENV_PIP install -r /tmp/req-evo-rl-cpu.txt -q
else
    $ENV_PIP install -r "$REQ_FILE" -q
fi

# ╔════════════════════════════════════════════════════════════════════╗
# ║  8. 安装源码包 (lerobot / transformers / piper_sdk)               ║
# ╚════════════════════════════════════════════════════════════════════╝
log_step "8/9 安装源码包..."

# 8.1 lerobot (项目本体, editable)
log_info "安装 lerobot (editable)..."
(cd "$PROJECT_ROOT" && $ENV_PIP install -e . -q --no-deps)

# 8.2 transformers (fix/lerobot_openpi 分支, 版本 4.53.3)
if [ -d "$PROJECT_ROOT/transformers" ] && [ -f "$PROJECT_ROOT/transformers/setup.py" ]; then
    log_info "从本地目录安装 transformers: $PROJECT_ROOT/transformers"
    (cd "$PROJECT_ROOT/transformers" && $ENV_PIP install -e . -q --no-deps)
else
    log_info "本地无 transformers 目录, 从 git 安装 (fix/lerobot_openpi)..."
    $ENV_PIP install -e "$TRANSFORMERS_GIT" -q --no-deps
fi

# 8.3 piper_sdk (机械臂 SDK)
if [ -d "$PROJECT_ROOT/src/piper_sdk" ] && [ -f "$PROJECT_ROOT/src/piper_sdk/setup.py" ]; then
    log_info "从本地目录安装 piper_sdk: $PROJECT_ROOT/src/piper_sdk"
    (cd "$PROJECT_ROOT/src/piper_sdk" && $ENV_PIP install -e . -q)
else
    log_info "本地无 piper_sdk 目录, 从 PyPI 安装 piper-sdk==0.6.1..."
    $ENV_PIP install "piper-sdk==0.6.1" -q
fi

# ╔════════════════════════════════════════════════════════════════════╗
# ║  9. 验证 + 可选 vllm-eval                                           ║
# ╚════════════════════════════════════════════════════════════════════╝
log_step "9/9 验证环境..."

set +e
$ENV_PY - <<'PYEOF'
import importlib
pkgs = ["torch", "torchvision", "torchcodec", "numpy", "cv2", "av",
        "datasets", "diffusers", "huggingface_hub", "accelerate",
        "gymnasium", "einops", "rerun_sdk", "draccus", "wandb",
        "transformers", "qwen_vl_utils", "piper_sdk", "safetensors"]
ok = True
for p in pkgs:
    try:
        m = importlib.import_module(p)
        print(f"  ✅ {p}: {getattr(m, '__version__', '?')}")
    except Exception as e:
        ok = False
        print(f"  ❌ {p}: {e}")
try:
    import torch
    print(f"  🎯 CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  🎯 CUDA 版本: {torch.version.cuda}")
        print(f"  🎯 GPU: {torch.cuda.get_device_name(0)}")
except Exception:
    pass
import sys
sys.exit(0 if ok else 1)
PYEOF
VERIFY_RC=$?
set -e

if [ $VERIFY_RC -ne 0 ]; then
    log_error "部分包验证失败, 请检查上方输出"
    exit 1
fi

# ── 可选: vllm-eval 环境 (VLM 推理服务) ────────────────────────────────
if $WITH_VLLM; then
    log_step "额外: 创建 vllm-eval 环境 (vLLM $VLLM_VERSION)..."
    VLLM_PREFIX="$(env_prefix_of vllm-eval)"
    if [ -z "$VLLM_PREFIX" ]; then
        "$CONDA_BIN" create -n vllm-eval -y -q python=3.10 -c conda-forge || {
            log_error "创建 vllm-eval 环境失败"; exit 1; }
        VLLM_PREFIX="$(env_prefix_of vllm-eval)"
    fi
    VLLM_PY="$VLLM_PREFIX/bin/python"
    if [ ! -x "$VLLM_PY" ]; then
        log_error "无法定位 vllm-eval 的 Python: $VLLM_PY"
        exit 1
    fi
    "$VLLM_PY" -m pip install -U pip -q
    "$VLLM_PY" -m pip install "vllm==$VLLM_VERSION" "openai" -q
    log_info "vllm-eval 环境创建完成"
fi

echo ""
log_info "=============================================="
log_info "  环境 $ENV_NAME 部署完成 🚀"
log_info "=============================================="
echo ""
log_info "激活环境:  conda activate $ENV_NAME"
if $WITH_VLLM; then
    log_info "启动 VLM 服务 (vllm-eval):"
    echo ""
    echo "  conda activate vllm-eval"
    echo "  VLLM_USE_FLASHINFER_SAMPLER=0 python -m vllm.entrypoints.openai.api_server \\"
    echo "    --model <Qwen3-VL-2B模型路径> --host 0.0.0.0 --port 8000 \\"
    echo "    --gpu-memory-utilization 0.7 --max-model-len 8192 --max-num-seqs 4 \\"
    echo "    --dtype auto --trust-remote-code --served-model-name qwen3-vl-2b"
    echo ""
fi
log_info "启动编排器:"
log_info "  bash $PROJECT_ROOT/src/lerobot/act_vlm_orchestrator_v2/start_orchestrator.sh"
