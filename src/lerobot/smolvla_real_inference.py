#!/usr/bin/env python3
"""
SmolVLA 真机推理控制脚本（顶部相机 + 腕部相机 + Piper 机械臂）
================================================================

功能：
  1. 加载训练好的 SmolVLA 模型（SmolVLM2 backbone 从本机 HF 缓存离线加载）
  2. 读取两路摄像头（BGR->RGB）与 Piper 关节状态
  3. 模型输出动作 -> 平滑 -> 限位/限速 -> 下发 Piper 关节控制
  4. 支持 --dry-run 观察模式：只推理不动作，先验证模型输出

使用方法（在 evo-rl-videos 环境中运行）：
  # 1) 观察模式（不控制机械臂，验证模型输出）
  python smolvla_real_inference.py --dry-run --task "pick up the red block"

  # 2) 真机控制（默认 select_action + exponential 平滑）
  python smolvla_real_inference.py --task "pick up the red block"

  # 3) 自定义参数
  python smolvla_real_inference.py \
      --checkpoint /home/ub/Videos/models/smolVLa_5W_bs64/pretrained_model \
      --top 10 --wrist 4 --can can1 \
      --smoothing moving_average --window_size 5 \
      --fps 20 --task "perform the task"

键盘控制（控制模式）：
  q  -> 平滑退出（失能机械臂）
  Ctrl+C -> 立即退出（失能机械臂）
"""

import argparse
import datetime
import os
import select
import signal
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np

# 将 lerobot 源码路径加入 sys.path
_LEROBOT_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(_LEROBOT_SRC))

from smolvla_inference import ActionSmoother, SmolVLAInference

# Piper SDK
from piper_sdk import C_PiperInterface

# ==============================================================================
#  单位转换常量
# ==============================================================================
RAD2MILLIDEG = 57324.840764  # 弧度 -> 毫度 (1000 * 180 / pi)
GRIPPER_MAX_MM = 70.0        # 夹爪最大行程 70mm（训练数据一致）

# 关节限位（弧度），与 inference_new_joint.py 一致
JOINT_LIMITS = [
    (-2.687, 2.687),
    (0.0, 3.403),
    (-3.0541012, 0.0),
    (-1.5499, 1.5499),
    (-1.22, 1.22),
    (-1.7452, 1.7452),
]


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


# ==============================================================================
#  Piper 机械臂封装
# ==============================================================================

class PiperArm:
    """Piper 机械臂（CAN 控制）封装：连接/使能/读取状态/下发动作。"""

    def __init__(self, can_name: str = "can1"):
        self.can_name = can_name
        self.arm: C_PiperInterface | None = None

    def connect(self):
        """初始化 CAN 并连接机械臂。"""
        from lerobot.utils.piper_sdk import ensure_can_interface_up

        ensure_can_interface_up(self.can_name)

        self.arm = C_PiperInterface(
            can_name=self.can_name,
            judge_flag=False,
            can_auto_init=True,
            logger_level=30,  # WARNING（避免大量 CAN 调试日志）
        )
        self.arm.ConnectPort(start_thread=True)
        time.sleep(0.2)

        # 退出主从联动模式，使用独立控制模式
        self.arm.MasterSlaveConfig(0x00, 0x00, 0x00, 0x00)
        print(f"[Piper] 已连接: {self.can_name}")

    def enable(self, timeout_s: float = 8.0):
        """使能机械臂并等待所有电机使能。"""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            info = self.arm.GetArmLowSpdInfoMsgs()
            enable_flag = all(
                getattr(info, f"motor_{i}").foc_status.driver_enable_status for i in range(1, 7)
            )
            if enable_flag:
                break
            self.arm.EnableArm(7)
            time.sleep(0.5)
        if not enable_flag:
            raise RuntimeError("机械臂使能超时，请检查机械臂供电/急停/连接")
        # 切换关节位置控制模式
        self.arm.MotionCtrl_2(0x01, 0x01, 5, 0x00)
        # 使能夹爪
        self.arm.GripperCtrl(0, 1000, 0x01, 0)
        time.sleep(1.0)
        print("[Piper] 使能完成，已切换关节位置控制模式")

    def get_observation(self) -> np.ndarray:
        """读取关节状态 -> 7 维向量 [j1..j6 弧度, gripper 0-1]。"""
        joint_msg = self.arm.GetArmJointMsgs()
        js = joint_msg.joint_state
        state = np.array(
            [
                getattr(js, f"joint_{i}") * 1e-3 / 57.3 for i in range(1, 7)
            ],
            dtype=np.float32,
        )
        gripper_msg = self.arm.GetArmGripperMsgs()
        gs = gripper_msg.gripper_state
        gripper_norm = abs(gs.grippers_angle * 1e-3 / GRIPPER_MAX_MM)
        return np.concatenate([state, [gripper_norm]]).astype(np.float32)

    def send_action(self, action: np.ndarray):
        """下发动作：action = [j1..j6 弧度, gripper 0-1]。"""
        joints = []
        for i in range(6):
            v = clamp(action[i], *JOINT_LIMITS[i])
            joints.append(round(v * RAD2MILLIDEG))
        gripper = clamp(action[6], 0.0, 1.0)
        gripper_raw = round(gripper * GRIPPER_MAX_MM * 1000)  # 0.001mm

        self.arm.JointCtrl(*joints)
        self.arm.GripperCtrl(gripper_raw, 5000, 0x01, 0)

    def move_to(self, target: np.ndarray, steps: int = 300, delay: float = 0.02):
        """从当前位置慢速插值移动到目标位姿（安全，避免跳变）。

        Args:
            target: 目标 7 维动作 [j1..j6 弧度, gripper 0-1]
            steps: 插值步数（300 步 × 0.02s ≈ 6 秒到位）
            delay: 每步间隔（秒）
        """
        current = self.get_observation()
        print(f"[Piper] 慢速移动机械臂到目标位姿 ({steps} 步) ...")
        for i in range(1, steps + 1):
            t = i / steps
            interp = current + (target - current) * t
            self.send_action(interp)
            time.sleep(delay)
        print(f"[Piper] 到位: {np.round(self.get_observation(), 3)}")

    def stop(self):
        """停止运动（保持当前位置，失能）。"""
        try:
            if self.arm is not None:
                self.arm.DisableArm(7)
                self.arm.DisconnectPort()
        except Exception as e:
            print(f"[Piper] 断开异常: {e}")
        print("[Piper] 已失能并断开")


# ==============================================================================
#  相机封装
# ==============================================================================

class CameraPair:
    """两路相机（顶部 + 腕部），返回 RGB 图像。"""

    def __init__(self, top_id: int = 10, wrist_id: int = 4, width: int = 640, height: int = 480):
        self.top_id = top_id
        self.wrist_id = wrist_id
        self.width = width
        self.height = height
        self._top = None
        self._wrist = None

    def connect(self):
        import cv2

        self._top = cv2.VideoCapture(self.top_id)
        self._wrist = cv2.VideoCapture(self.wrist_id)
        for cam in (self._top, self._wrist):
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cam.set(cv2.CAP_PROP_FPS, 30)
            cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self._top.isOpened() or not self._wrist.isOpened():
            raise RuntimeError(f"相机打开失败: top={self.top_id}, wrist={self.wrist_id}")
        # 预热
        for _ in range(5):
            self._top.read()
            self._wrist.read()
        print(f"[Camera] 已连接: top=/dev/video{self.top_id}, wrist=/dev/video{self.wrist_id}")

    def read(self):
        import cv2

        ret_t, top = self._top.read()
        ret_w, wrist = self._wrist.read()
        if not ret_t or not ret_w:
            raise RuntimeError("相机读取失败")
        # OpenCV BGR -> RGB（模型训练数据为 RGB）
        return cv2.cvtColor(top, cv2.COLOR_BGR2RGB), cv2.cvtColor(wrist, cv2.COLOR_BGR2RGB)

    def close(self):
        if self._top is not None:
            self._top.release()
        if self._wrist is not None:
            self._wrist.release()


# ==============================================================================
#  主循环
# ==============================================================================

DEFAULT_TASKS = [
    "Grab the package and place it on the barcode scanning table.",
    "Flip the package if the barcode is not facing up.",
    "Grab the scanned package and place it into the box.",
]


def _setup_raw_terminal():
    """进入单键 raw 模式（仅当 stdin 是终端）。返回旧的 termios 属性。"""
    try:
        if os.isatty(sys.stdin.fileno()):
            return termios.tcgetattr(sys.stdin.fileno()), True
    except Exception:
        return None, False
    return None, False


def _read_key(timeout=0.0):
    """非阻塞读取单个键盘按键（raw 模式下单键即时生效）。无输入返回 None。"""
    try:
        if select.select([sys.stdin], [], [], timeout)[0]:
            ch = sys.stdin.read(1)
            return ch if ch else None
    except Exception:
        pass
    return None


# ==============================================================================
#  运行日志（终端 + 文件双写）
# ==============================================================================

class _Tee:
    """同时输出到终端与日志文件的流。

    raw 模式下输出用 \r\n，写入日志文件时归一化为 \n（日志更干净）。
    """

    def __init__(self, stream, logfile):
        self.stream = stream
        self.logfile = logfile

    def write(self, data):
        self.stream.write(data)
        self.logfile.write(data.replace("\r\n", "\n"))
        self.logfile.flush()

    def flush(self):
        self.stream.flush()
        self.logfile.flush()


# 原始终端流（关闭日志时需恢复，避免解释器关闭时 flush 已关闭文件）
_ORIG_STDOUT = sys.stdout
_ORIG_STDERR = sys.stderr


def setup_file_logging(log_dir: str):
    """将 stdout/stderr 同时输出到终端和按时间戳命名的日志文件。

    Args:
        log_dir: 日志目录（自动创建）

    Returns:
        日志文件对象（调用方需在退出时调用 close_file_logging）
    """
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(log_dir) / f"smolvla_{ts}.log"
    log_fp = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(_ORIG_STDOUT, log_fp)
    sys.stderr = _Tee(_ORIG_STDERR, log_fp)
    print(f"[Main] 运行日志已保存: {log_path}", flush=True)
    return log_fp


def close_file_logging(log_fp):
    """关闭日志：先恢复原始流，再 flush/close 文件（避免退出时异常）。"""
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    sys.stdout = _ORIG_STDOUT
    sys.stderr = _ORIG_STDERR
    try:
        log_fp.close()
    except Exception:
        pass


def parse_args():
    p = argparse.ArgumentParser(description="SmolVLA 真机推理控制")
    p.add_argument("--checkpoint", type=str,
                   default="/home/ub/Videos/models/smolVLa_5W_bs64/pretrained_model",
                   help="模型 checkpoint 路径")
    p.add_argument("--task", type=str, default=None,
                   help="单个任务描述（与 --tasks 二选一；默认使用训练时的三阶段任务）")
    p.add_argument("--tasks", type=str, default=None,
                   help="多阶段任务列表，用 | 分隔，如 'task1|task2|task3'；运行时按 1/2/3 切换")
    p.add_argument("--top", type=int, default=10, help="顶部相机 /dev/video 编号")
    p.add_argument("--wrist", type=int, default=4, help="腕部相机 /dev/video 编号")
    p.add_argument("--can", type=str, default="can1", help="Piper 机械臂 CAN 接口")
    p.add_argument("--fps", type=float, default=20.0, help="控制频率 (Hz)")
    p.add_argument("--mode", type=str, default="select_action",
                   choices=["select_action", "predict_action_chunk"])
    p.add_argument("--smoothing", type=str, default="exponential",
                   choices=["none", "moving_average", "exponential"])
    p.add_argument("--window_size", type=int, default=5, help="moving_average 窗口")
    p.add_argument("--alpha", type=float, default=0.3, help="exponential 平滑系数(越小越平滑，0.3较稳)")
    p.add_argument("--max_delta", type=float, default=0.1,
                   help="每帧关节最大变化量（弧度），防止模型输出突变/抖动")
    p.add_argument("--gripper_deadband", type=float, default=0.05,
                   help="夹爪死区：目标变化小于该值时不更新，避免夹爪频繁抖动")
    p.add_argument("--preset-pose", action="store_true",
                   help="启动时慢速移动机械臂到训练工作位姿。重要：任务1(抓取到托盘)必须从工作区开始，"
                        "否则home位姿偏离训练分布，模型会输出保持动作")
    p.add_argument("--show-norm", action="store_true",
                   help="日志中显示归一化后的关节状态（诊断是否偏离训练分布）")
    p.add_argument("--save-frame", type=int, default=0, metavar="N",
                   help="每 N 帧保存一次相机图像到 /tmp/smolvla_frames（诊断用，0=不保存）")
    p.add_argument("--log-dir", type=str,
                   default=str(Path(__file__).resolve().parent / "smolvla_logs"),
                   help="运行日志目录（按时间戳自动创建新文件）")
    p.add_argument("--rtc", action="store_true",
                   help="启用 RTC 式 chunk 平滑（后台预推理+重叠融合+滤波，消除 chunk 边界抖动）")
    p.add_argument("--rtc-horizon", type=int, default=15,
                   help="RTC 提前推理触发阈值（帧），需覆盖推理耗时(约0.35s×fps)，且≥blend")
    p.add_argument("--rtc-blend", type=int, default=10,
                   help="RTC 边界融合帧数（新 chunk 前 N 帧与旧 chunk 残留融合）")
    p.add_argument("--dry-run", action="store_true",
                   help="观察模式：只推理不控制机械臂")
    p.add_argument("--no-cam", action="store_true",
                   help="不打开相机（使用黑图，用于纯调试）")
    return p.parse_args()


# 全局标记：收到 SIGTERM 时置位，主循环检测后退出（触发清理）
_terminate = False

# raw 模式下终端不回显也不做换行转换，print 需用 \r\n
_RAW_MODE = False


def rprint(*args):
    """raw 模式下使用 \r\n 打印，保证终端输出对齐。"""
    if _RAW_MODE:
        print(*args, end="\r\n", flush=True)
    else:
        print(*args, flush=True)


def _handle_sigterm(signum, frame):
    global _terminate
    print("\n[Main] 收到终止信号，正在安全退出 ...", flush=True)
    _terminate = True


def main():
    args = parse_args()

    # 日志：按时间保存到 smolvla_logs 目录（后续所有输出同时写入终端与文件）
    _log_fp = setup_file_logging(args.log_dir)

    # 构建任务列表（多阶段任务切换）
    if args.tasks:
        task_list = [t.strip() for t in args.tasks.split("|") if t.strip()]
    elif args.task:
        task_list = [args.task]
    else:
        task_list = list(DEFAULT_TASKS)
    task_idx = 0
    print(f"\n[Main] 共 {len(task_list)} 个阶段任务:")
    for i, t in enumerate(task_list):
        print(f"       [{i+1}] {t}")

    # 平滑器
    smoother = ActionSmoother(
        method=args.smoothing,
        window_size=args.window_size,
        alpha=args.alpha,
        action_dim=7,
    )

    # 加载模型
    print(f"\n{'='*60}\n  加载 SmolVLA 模型\n{'='*60}")
    model = SmolVLAInference(
        checkpoint_path=args.checkpoint,
        device="cuda",
        dtype="float32",
        mode=args.mode,
        smoother=smoother,
    )
    model.load()
    print(f"  chunk_size={model.chunk_size}  n_action_steps={model.n_action_steps}  "
          f"action_dim={model.action_dim}")

    # RTC 式 chunk 平滑器（可选，替代 select_action 队列管理）
    rtc_executor = None
    if args.rtc:
        from smolvla_rtc import RTCInspiredSmoother
        rtc_executor = RTCInspiredSmoother(
            model=model,
            execution_horizon=args.rtc_horizon,
            blend_steps=args.rtc_blend,
            alpha=args.alpha,
            action_dim=model.action_dim,
        )
        print(f"[Main] ✅ 已启用 RTC 式 chunk 平滑 "
              f"(horizon={args.rtc_horizon}, blend={args.rtc_blend}, alpha={args.alpha})")

    # 连接相机
    cams = None
    if not args.no_cam:
        cams = CameraPair(top_id=args.top, wrist_id=args.wrist)
        cams.connect()

    # 连接机械臂（dry-run 也连接以读取真实关节状态，但不使能、不下发动作）
    arm = PiperArm(can_name=args.can)
    arm.connect()
    if not args.dry_run:
        arm.enable()
        # 可选：移动到训练工作位姿（训练 state 均值，接近模型训练分布）
        if args.preset_pose:
            arm.move_to(np.array([0.307, 1.304, -1.277, -0.048, 0.953, 0.116, 0.203]))
            if not args.no_cam:
                # 移动后读两帧丢弃（相机/状态已变化）
                cams.read()

    # 归一化统计量（供 --show-norm 诊断显示）
    _norm_mean = None
    _norm_std = None
    if model._dataset_stats is not None:
        _norm_mean = np.array(model._dataset_stats["observation.state"]["mean"], dtype=np.float32)
        _norm_std = np.array(model._dataset_stats["observation.state"]["std"], dtype=np.float32)

    # 注册 SIGTERM 处理（timeout/kill 也能优雅退出并失能机械臂）
    signal.signal(signal.SIGTERM, _handle_sigterm)

    # 进入单键 raw 模式（供键盘实时切换任务）
    global _RAW_MODE
    _term_attrs, _is_tty = _setup_raw_terminal()
    if _is_tty:
        try:
            tty.setraw(sys.stdin.fileno())
            _RAW_MODE = True
        except Exception:
            pass

    # 主循环
    interval = 1.0 / max(args.fps, 1.0)
    frame_idx = 0
    last_action = None

    _active_mode = "RTC" if args.rtc else args.mode
    print(f"\n{'='*60}\n  开始推理 (mode={_active_mode}, smoothing={args.smoothing}, "
          f"dry_run={args.dry_run})\n"
          f"  键盘: 1..{len(task_list)} 切换任务 | t 显示任务 | q 退出 | Ctrl+C 退出\n{'='*60}")

    try:
        while not _terminate:
            t0 = time.time()

            # 1) 观测
            if args.no_cam:
                top_img = np.zeros((480, 640, 3), dtype=np.uint8)
                wrist_img = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                top_img, wrist_img = cams.read()

            if arm is not None:
                state = arm.get_observation()
            else:
                # dry-run：使用一个固定的合理初始状态
                state = np.array([0.0, 0.0, -1.5, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)

            # 1.4) 诊断：保存相机帧 / 显示归一化状态
            if args.save_frame > 0 and frame_idx % args.save_frame == 0:
                import cv2
                os.makedirs("/tmp/smolvla_frames", exist_ok=True)
                cv2.imwrite(f"/tmp/smolvla_frames/top_{frame_idx:06d}.png",
                            cv2.cvtColor(top_img, cv2.COLOR_RGB2BGR))
                cv2.imwrite(f"/tmp/smolvla_frames/wrist_{frame_idx:06d}.png",
                            cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR))
            if args.show_norm and _norm_mean is not None:
                norm = (state - _norm_mean) / (_norm_std + 1e-8)
                rprint(f"  [norm-state] {np.round(norm, 3)}")

            # 1.5) 键盘控制（任务切换 / 退出）
            key = _read_key()
            if key:
                if key in "123456789":
                    n = int(key)
                    if 1 <= n <= len(task_list):
                        task_idx = n - 1
                        # 清空模型队列/缓存 + RTC 平滑器，确保新任务立即生效
                        model.reset()
                        if rtc_executor is not None:
                            rtc_executor.reset()
                        last_action = None  # 下一帧以当前关节状态为限速基准
                        rprint(f"\n[Main] ▶ 切换任务 -> [{task_idx+1}] {task_list[task_idx]} "
                               f"(已重置模型队列/RTC)\n")
                elif key.lower() == "t":
                    rprint(f"\n[Main] 当前任务 [{task_idx+1}/{len(task_list)}]: {task_list[task_idx]}\n")
                elif key.lower() == "q":
                    rprint("\n[Main] 用户按 q 退出")
                    break

            # 2) 推理（RTC 模式：后台预推理+重叠融合；否则用 select_action 队列）
            if rtc_executor is not None:
                action = rtc_executor.step(
                    wrist_img=wrist_img,
                    top_img=top_img,
                    joint_state=state,
                    task=task_list[task_idx],
                )
            else:
                action = model.step(
                    wrist_img=wrist_img,
                    top_img=top_img,
                    joint_state=state,
                    task=task_list[task_idx],
                )

            # 3) 限速（相对上一帧；首帧以当前关节状态为基准，避免初始跳变）
            if last_action is None:
                last_action = state.copy()
            if args.max_delta > 0:
                delta = action - last_action
                delta = np.clip(delta, -args.max_delta, args.max_delta)
                action = last_action + delta

            # 3.5) 夹爪死区（滞回）：目标变化小于阈值时不更新，
            #      抑制模型对 gripper 输出的高频抖动（夹爪频繁开合）
            if args.gripper_deadband > 0 and abs(action[6] - last_action[6]) < args.gripper_deadband:
                action[6] = last_action[6]
            last_action = action.copy()

            # 4) 下发动作（仅真机控制模式）
            if not args.dry_run and arm is not None:
                arm.send_action(action)

            frame_idx += 1
            elapsed = time.time() - t0
            if frame_idx % 5 == 0:
                rprint(f"[{frame_idx:4d}][T{task_idx+1}] state={np.round(state,3)} "
                       f"action={np.round(action,3)} | 单帧耗时 {elapsed*1000:.0f}ms")

            # 5) 节奏控制
            time.sleep(max(0.0, interval - elapsed))

    except KeyboardInterrupt:
        print("\n[Main] 用户中断，正在退出 ...")
    finally:
        # 恢复终端模式
        if _is_tty and _term_attrs is not None:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _term_attrs)
            except Exception:
                pass
        _RAW_MODE = False
        if rtc_executor is not None:
            rtc_executor.reset()
        if arm is not None:
            arm.stop()
        if cams is not None:
            cams.close()
        print("[Main] 已退出")
        # 关闭日志（先恢复终端流，避免解释器关闭时 flush 已关闭文件）
        close_file_logging(_log_fp)


if __name__ == "__main__":
    main()
