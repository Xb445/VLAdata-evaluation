#!/usr/bin/env bash
# ==============================================================================
# SmolVLA 检查点仿真推理验证脚本
# ==============================================================================
# 作用:
#   使用 Evo-RL 训练产出的 SmolVLA checkpoint，在 pdgs_lab 的 SO101 pick-place
#   仿真环境中做闭环推理验证（Physics + MuJoCo 离屏渲染 + VLA Server）。
#
# 使用方法:
#   bash scripts/eval_smolvla_sim.sh                     # 默认 1 回合 / 150 步（无窗口）
#   bash scripts/eval_smolvla_sim.sh --gui               # MuJoCo 3D GUI 可视化（需 X 显示）
#   bash scripts/eval_smolvla_sim.sh --episodes 3        # 3 个回合
#   bash scripts/eval_smolvla_sim.sh --max-steps 300 --stop-on-success
#   bash scripts/eval_smolvla_sim.sh --dry-run           # 只打印命令
#   bash scripts/eval_smolvla_sim.sh --ckpt /path/to/pretrained_model
#
# 常用环境变量覆盖:
#   CKPT_PATH, PDGS_DIR, OUTPUT_DIR, CONDA_ENV, NUM_EPISODES, MAX_STEPS
#
# 产物 (默认 outputs/eval/smolvla_quick_sim_inference):
#   orchestrator.log      编排器 + Physics + Renderer + VLA 完整日志
#   record/schema.json    物理录制格式说明
#   record/frames.jsonl   逐帧物理状态与执行动作
#   summary.json          推理验证汇总（块/托盘距离、动作统计、是否成功）
#   block_trajectory.csv  红块轨迹与动作序列（便于外部分析/绘图）
#   trajectory.png        关节角与红块 XY 轨迹图
#
# 说明:
#   - pdgs_lab 的 VLA 服务读取 vla_server/vla_config.json；本脚本通过
#     VLA_MODEL_PATH 环境变量覆盖模型目录，不改动其默认模型。
#   - 必须使用 lerobot_eval 环境（含 physical_backend 与 pdgs_lab 的 LeRobot 0.6.2）。
#   - 共享内存同一时刻只允许一个实例；脚本启动前会自动清理残留。
# ==============================================================================

set -uo pipefail

# ── 参数解析 ────────────────────────────────────────────────────────────
DRY_RUN=false
GUI_MODE=false
OUTPUT_DIR_SET=false
while [ $# -gt 0 ]; do
    case "$1" in
        --episodes)        NUM_EPISODES="$2"; shift 2 ;;
        --max-steps)       MAX_STEPS="$2"; shift 2 ;;
        --ckpt)            CKPT_PATH="$2"; shift 2 ;;
        --output-dir)      OUTPUT_DIR="$2"; OUTPUT_DIR_SET=true; shift 2 ;;
        --stop-on-success) STOP_ON_SUCCESS=true; shift ;;
        --gui)             GUI_MODE=true; shift ;;
        --dry-run)         DRY_RUN=true; shift ;;
        -h|--help)
            sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "未知参数: $1" >&2
            echo "可用: --episodes N / --max-steps N / --ckpt PATH / --output-dir PATH / --stop-on-success / --gui / --dry-run" >&2
            exit 2
            ;;
    esac
done

# ── 路径与运行环境 ──────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(dirname "$SCRIPT_DIR")}"

CONDA_SH="${CONDA_SH:-/home/pd/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-lerobot_eval}"

# pdgs_lab：SO101 pick-place 仿真环境（Physics + Renderer + VLA Server）
PDGS_DIR="${PDGS_DIR:-/home/pd/wangsh/WorkSpace/pdgs_lab}"

# 待验证的检查点目录（需包含 config.json + model.safetensors + policy_*processor.json）
CKPT_PATH="${CKPT_PATH:-$PROJECT_DIR/outputs/train/smolvla_quick_so101_pickplace_v2/checkpoints/000010/pretrained_model}"

# 输出目录（默认位于当前项目 outputs/ 下）
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/eval/smolvla_quick_sim_inference}"

# ── 推理参数 ────────────────────────────────────────────────────────────
NUM_EPISODES="${NUM_EPISODES:-1}"
MAX_STEPS="${MAX_STEPS:-150}"          # 150 步 @15Hz ≈ 10 秒
EPISODE_TIMEOUT="${EPISODE_TIMEOUT:-0}" # 0 = 不限制墙钟时间
STOP_ON_SUCCESS="${STOP_ON_SUCCESS:-false}"
PHYSICS_BACKEND="${PHYSICS_BACKEND:-physical_backend}"
RENDERER_BACKEND="${RENDERER_BACKEND:-mujoco}"
CLEAN="${CLEAN:-true}"

# --gui：使用 builtin_mujoco（纯 MuJoCo 物理 + 3D GUI 查看器）
# 注意：GUI 模式需要 X 显示；默认输出目录也会区分，避免覆盖无窗口结果。
if [ "$GUI_MODE" = true ]; then
    PHYSICS_BACKEND="builtin_mujoco"
    RENDERER_BACKEND="mujoco"
    if [ "$OUTPUT_DIR_SET" != true ]; then
        OUTPUT_DIR="$PROJECT_DIR/outputs/eval/smolvla_quick_sim_gui"
    fi
fi

# ── 基础校验 ────────────────────────────────────────────────────────────
echo "=============================================================="
echo " SmolVLA 检查点 × pdgs_lab 仿真闭环推理验证"
echo "=============================================================="
echo "  检查点:     $CKPT_PATH"
echo "  仿真项目:   $PDGS_DIR"
echo "  输出目录:   $OUTPUT_DIR"
echo "  回合/步数:  $NUM_EPISODES 回合 × $MAX_STEPS 步 (15Hz)"
echo "  物理/渲染:  $PHYSICS_BACKEND / $RENDERER_BACKEND"
[ "$GUI_MODE" = true ] && echo "  可视化:     MuJoCo 3D GUI（DISPLAY=${DISPLAY:-未设置}）"
echo ""

for f in config.json model.safetensors policy_preprocessor.json policy_postprocessor.json; do
    if [ ! -f "$CKPT_PATH/$f" ]; then
        echo "✗ 检查点不完整: 缺少 $CKPT_PATH/$f" >&2
        exit 1
    fi
done
if [ ! -f "$PDGS_DIR/orchestrator/orchestrator.py" ]; then
    echo "✗ 未找到 pdgs_lab 编排器: $PDGS_DIR/orchestrator/orchestrator.py" >&2
    exit 1
fi
if ! grep -q "VLA_MODEL_PATH" "$PDGS_DIR/vla_server/vla_server.py"; then
    echo "✗ pdgs_lab 的 VLA 服务不支持 VLA_MODEL_PATH 覆盖，请先应用本仓库提供的补丁。" >&2
    echo "  需要修改: $PDGS_DIR/vla_server/vla_server.py" >&2
    exit 1
fi
if [ ! -f "$CONDA_SH" ]; then
    echo "✗ 未找到 conda 初始化脚本: $CONDA_SH（可通过 CONDA_SH 覆盖）" >&2
    exit 1
fi
if [ "$GUI_MODE" = true ] && [ -z "${DISPLAY:-}" ]; then
    echo "✗ --gui 需要图形显示，但当前 DISPLAY 未设置。请在桌面终端中运行，或改用无窗口模式。" >&2
    exit 1
fi

# ── 激活 conda 环境 ────────────────────────────────────────────────────
# conda 的 activate.d 脚本会引用未定义变量，激活期间临时关闭 set -u。
set +u
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV" || { set -u; echo "✗ 无法激活 conda 环境 $CONDA_ENV" >&2; exit 1; }
set -u

echo "[环境] python: $(which python)"
echo "[环境] torch:  $(python -c 'import torch; print(torch.__version__)')"
echo "[环境] cuda:   $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "[环境] lerobot: $(python -c 'import lerobot; print(lerobot.__file__)')"

# ── 离线加载 ────────────────────────────────────────────────────────────
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

# ── 清理共享内存残留 ────────────────────────────────────────────────────
# 仅在没有相关进程时清理，避免误删正在运行的服务。
if ps -ef | grep -E "orchestrator\.py|mujoco_physics|mujoco_renderer|vla_server" | grep -v grep >/dev/null 2>&1; then
    echo "✗ 检测到 pdgs_lab 推理进程仍在运行，请先停止后再执行本脚本。" >&2
    ps -ef | grep -E "orchestrator\.py|mujoco_physics|mujoco_renderer|vla_server" | grep -v grep >&2
    exit 1
fi
python - <<'PY'
from multiprocessing import shared_memory

for name in ("pose_channel", "image_channel", "vla_channel"):
    try:
        shm = shared_memory.SharedMemory(name=name)
        shm.close()
        shm.unlink()
        print(f"[准备] 已清理共享内存 {name}")
    except FileNotFoundError:
        pass
PY

# ── 准备输出目录 ────────────────────────────────────────────────────────
RECORD_DIR="$OUTPUT_DIR/record"
LOG_FILE="$OUTPUT_DIR/orchestrator.log"

# dry-run 不产生任何文件系统副作用
if [ "$DRY_RUN" != true ]; then
    if [ "$CLEAN" = true ] && [ -d "$OUTPUT_DIR" ]; then
        echo "[准备] 清理已存在的输出目录: $OUTPUT_DIR"
        rm -rf "$OUTPUT_DIR"
    fi
    mkdir -p "$RECORD_DIR"
fi

# ── 构建并展示命令 ──────────────────────────────────────────────────────
CMD=(
    python -u orchestrator/orchestrator.py
    --physics-backend "$PHYSICS_BACKEND"
    --renderer-backend "$RENDERER_BACKEND"
    --num-episodes "$NUM_EPISODES"
    --max-steps "$MAX_STEPS"
    --episode-timeout "$EPISODE_TIMEOUT"
    --physics-record-dir "$RECORD_DIR"
    --physics-record-overwrite
)
[ "$STOP_ON_SUCCESS" = true ] && CMD+=(--stop-on-success)

echo ""
echo "[命令] 仿真推理命令:"
echo "  cd $PDGS_DIR && VLA_MODEL_PATH=$CKPT_PATH ${CMD[*]}"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "[dry-run] 仅打印命令，未执行推理。"
    exit 0
fi

# ── 启动闭环推理 ────────────────────────────────────────────────────────
echo "[运行] 开始闭环推理（模型加载约 5~15 秒）..."
START_TIME=$(date +%s)
(
    cd "$PDGS_DIR" || exit 1
    VLA_MODEL_PATH="$CKPT_PATH" "${CMD[@]}"
) 2>&1 | tee "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}
END_TIME=$(date +%s)

echo ""
echo "=============================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo " ✓ 推理完成，耗时 $((END_TIME - START_TIME))s"
else
    echo " ✗ 推理失败（退出码 $EXIT_CODE），耗时 $((END_TIME - START_TIME))s"
    echo "   请查看日志: $LOG_FILE"
    exit $EXIT_CODE
fi
echo "=============================================================="

# ── 汇总物理录制结果 ────────────────────────────────────────────────────
SUMMARY_SCRIPT="$OUTPUT_DIR/summarize_rollout.py"
cat > "$SUMMARY_SCRIPT" <<'PY'
"""Summarize pdgs_lab physics recording (schema.json + frames.jsonl)."""
import csv
import json
import sys
from pathlib import Path

record_dir = Path(sys.argv[1])
out_dir = Path(sys.argv[2])

schema_path = record_dir / "schema.json"
frames_path = record_dir / "frames.jsonl"
if not frames_path.exists():
    print(f"[汇总] 未找到 {frames_path}，跳过汇总。")
    sys.exit(0)

block_id, tray_id = 8, 9  # SO101 pickplace.xml 默认 body 顺序
if schema_path.exists():
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    for item in schema.get("physics_schema", {}).get("tracked_bodies", []):
        if item.get("name") == "block":
            block_id = item["body_id"]
        elif item.get("name") == "tray":
            tray_id = item["body_id"]

frames = [json.loads(line) for line in frames_path.read_text(encoding="utf-8").splitlines() if line.strip()]
if not frames:
    print("[汇总] frames.jsonl 为空，跳过汇总。")
    sys.exit(0)

def body_xyz(frame, body_id):
    bodies = frame["state"]["bodies"]
    ids = bodies["body_id"]
    idx = ids.index(body_id)
    return bodies["position"][idx]

rows = []
success_any = False
for frame in frames:
    block = body_xyz(frame, block_id)
    tray = body_xyz(frame, tray_id)
    xy = sum((block[i] - tray[i]) ** 2 for i in range(2)) ** 0.5
    z = abs(block[2] - tray[2])
    ok = xy < 0.05 and z < 0.03
    success_any = success_any or ok
    action = list(frame.get("commanded_action", []))
    qpos = list(frame["state"].get("qpos", [])[:6])
    rows.append({
        "frame_index": frame.get("frame_index"),
        "sim_time": frame["state"].get("sim_time"),
        "block_x": block[0], "block_y": block[1], "block_z": block[2],
        "tray_x": tray[0], "tray_y": tray[1], "tray_z": tray[2],
        "xy_distance": xy, "z_distance": z, "success": ok,
        **{f"action_{i}": (action[i] if i < len(action) else None) for i in range(6)},
        **{f"qpos_{i}": (qpos[i] if i < len(qpos) else None) for i in range(6)},
    })

summary = {
    "checkpoint": sys.argv[3] if len(sys.argv) > 3 else None,
    "num_frames": len(rows),
    "sim_duration_s": round(float(rows[-1]["sim_time"]) - float(rows[0]["sim_time"]), 4),
    "block_id": block_id,
    "tray_id": tray_id,
    "block_initial_xyz": [rows[0]["block_x"], rows[0]["block_y"], rows[0]["block_z"]],
    "block_final_xyz": [rows[-1]["block_x"], rows[-1]["block_y"], rows[-1]["block_z"]],
    "tray_xyz": [rows[-1]["tray_x"], rows[-1]["tray_y"], rows[-1]["tray_z"]],
    "final_xy_distance_m": round(rows[-1]["xy_distance"], 6),
    "final_z_distance_m": round(rows[-1]["z_distance"], 6),
    "min_xy_distance_m": round(min(r["xy_distance"] for r in rows), 6),
    "success_final_frame": bool(rows[-1]["success"]),
    "success_any_frame": bool(success_any),
    "success_criterion": "xy_distance < 0.05 m and z_distance < 0.03 m",
}

with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

with (out_dir / "block_trajectory.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print("[汇总] 推理验证结果:")
for key in (
    "num_frames", "sim_duration_s", "block_initial_xyz", "block_final_xyz",
    "tray_xyz", "final_xy_distance_m", "min_xy_distance_m",
    "success_final_frame", "success_any_frame",
):
    print(f"  {key}: {summary[key]}")
print(f"[汇总] 已写入 {out_dir / 'summary.json'}")
print(f"[汇总] 已写入 {out_dir / 'block_trajectory.csv'}")

# 生成轨迹图（matplotlib 不可用时自动跳过）
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = [r["sim_time"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    for i in range(6):
        ax.plot(times, [r[f"qpos_{i}"] for r in rows], label=f"joint{i}")
    ax.set_title("SO101 joint qpos (inference rollout)")
    ax.set_xlabel("sim time (s)")
    ax.set_ylabel("qpos (rad)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1]
    ax.plot([r["block_x"] for r in rows], [r["block_y"] for r in rows], label="block path")
    ax.scatter([rows[0]["block_x"]], [rows[0]["block_y"]], marker="o", label="block start")
    ax.scatter([rows[-1]["block_x"]], [rows[-1]["block_y"]], marker="x", label="block end")
    ax.scatter([rows[-1]["tray_x"]], [rows[-1]["tray_y"]], marker="s", label="tray")
    ax.set_title("Block XY trajectory")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    plot_path = out_dir / "trajectory.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[汇总] 已写入 {plot_path}")
except Exception as exc:
    print(f"[汇总] 绘图跳过: {exc}")
PY

python "$SUMMARY_SCRIPT" "$RECORD_DIR" "$OUTPUT_DIR" "$CKPT_PATH"

echo ""
echo "产物目录: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"
