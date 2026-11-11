#!/bin/bash
# =============================================================================
# Piper 双臂 CAN 接口快速配置脚本
#
# 用法: bash scripts/setup_can.sh [--diagnose]
#
# can0 → 主臂 (Leader/Master Arm)  → USB 1-5.1:1.0
# can1 → 从臂 (Follower/Slave Arm) → USB 1-5.4:1.0
# =============================================================================

set -e

BITRATE=1000000
CAN0="can0"
CAN1="can1"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ── Check if interface exists ──────────────────────────────────────
check_exists() {
    if ip link show "$1" &>/dev/null; then
        return 0
    else
        return 1
    fi
}

# ── Bring up a CAN interface ───────────────────────────────────────
bring_up() {
    local iface=$1
    local role=$2

    if ! check_exists "$iface"; then
        log_error "接口 $iface 不存在！请确认 USB-CAN 适配器已插入。"
        return 1
    fi

    # Check current state
    local state
    state=$(ip -details link show "$iface" 2>/dev/null | grep -oP 'state \K\w+' | head -1)

    if [ "$state" = "UP" ]; then
        local can_state
        can_state=$(ip -details link show "$iface" 2>/dev/null | grep -oP 'can state \K\S+' | head -1)
        log_info "$iface ($role) 已激活 (state=$state, can_state=$can_state)"
        return 0
    fi

    log_info "正在激活 $iface ($role)..."
    if sudo ip link set "$iface" up type can bitrate "$BITRATE" 2>/dev/null; then
        local new_state new_can_state
        new_state=$(ip -details link show "$iface" 2>/dev/null | grep -oP 'state \K\w+' | head -1)
        new_can_state=$(ip -details link show "$iface" 2>/dev/null | grep -oP 'can state \K\S+' | head -1)
        log_info "$iface ($role) 激活成功 (state=$new_state, can_state=$new_can_state)"
        return 0
    else
        log_error "$iface ($role) 激活失败！请检查："
        log_error "  1. USB-CAN 适配器是否正确插入"
        log_error "  2. 是否需要 sudo 权限"
        log_error "  3. CAN 总线终端电阻 (120Ω)"
        return 1
    fi
}

# ── Diagnostics ─────────────────────────────────────────────────────
run_diagnostics() {
    echo ""
    echo "========================================"
    echo "  CAN 接口诊断"
    echo "========================================"

    for iface in "$CAN0" "$CAN1"; do
        echo ""
        echo "--- $iface ---"
        if ! check_exists "$iface"; then
            log_error "$iface 不存在"
            continue
        fi

        ip -details link show "$iface" 2>/dev/null | grep -E "state|can state|bitrate" || true

        # Check carrier (indicates physical connection to arm)
        if [ -f "/sys/class/net/$iface/carrier" ]; then
            local carrier
            carrier=$(cat "/sys/class/net/$iface/carrier" 2>/dev/null)
            if [ "$carrier" = "1" ]; then
                log_info "Carrier: 已检测到物理连接"
            else
                log_warn "Carrier: 未检测到物理连接 → 臂可能未上电或CAN线断开"
            fi
        fi
    done

    # USB port mapping
    echo ""
    echo "--- USB-CAN 端口映射 ---"
    for iface in "$CAN0" "$CAN1"; do
        if check_exists "$iface"; then
            local usb_path
            usb_path=$(readlink -f "/sys/class/net/$iface/device" 2>/dev/null || echo "未知")
            echo "  $iface → $usb_path"
        fi
    done
    echo ""
    echo "预期映射:"
    echo "  can0 (主臂) → USB 1-5.1:1.0"
    echo "  can1 (从臂) → USB 1-5.4:1.0"
    echo ""
}

# =============================================================================
# Main
# =============================================================================

echo ""
echo "╔══════════════════════════════════════╗"
echo "║   Piper 双臂 CAN 接口配置            ║"
echo "║   主臂: can0 | 从臂: can1 | 1 Mbps  ║"
echo "╚══════════════════════════════════════╝"
echo ""

if [ "$1" = "--diagnose" ] || [ "$1" = "-d" ]; then
    run_diagnostics
    exit 0
fi

# Bring up both interfaces
FAILED=0
bring_up "$CAN0" "主臂" || FAILED=1
bring_up "$CAN1" "从臂" || FAILED=1

echo ""

if [ "$FAILED" -eq 0 ]; then
    log_info "全部 CAN 接口已就绪 ✅"
    echo ""
    echo "现在可以启动 HIL 实验:"
    echo "  conda activate evo-rl-videos"
    echo "  lerobot-human-inloop-record ..."
else
    log_error "部分 CAN 接口配置失败，请运行诊断:"
    echo "  bash scripts/setup_can.sh --diagnose"
fi

echo ""

# Quick arm power check reminder
if [ -f "/sys/class/net/$CAN0/carrier" ] && [ "$(cat /sys/class/net/$CAN0/carrier)" != "1" ]; then
    log_warn "⚠  can0 无物理连接信号 → 请确认主臂已上电 (24V)"
fi
if [ -f "/sys/class/net/$CAN1/carrier" ] && [ "$(cat /sys/class/net/$CAN1/carrier)" != "1" ]; then
    log_warn "⚠  can1 无物理连接信号 → 请确认从臂已上电 (24V)"
fi
