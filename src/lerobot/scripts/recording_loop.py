# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
- Core recording loop used by `lerobot_record.py`.
- 代码作用为记录人类在环实验中的数据，包括策略执行和远程操作设备动作的镜像。
"""

import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np

from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
)
from lerobot.robots import Robot
from lerobot.scripts.recording_hil import (
    INTERVENTION_STATE_ACTIVE,
    INTERVENTION_STATE_POLICY,
    INTERVENTION_STATE_RELEASE,
    ACPInferenceConfig,
    PolicySyncDualArmExecutor,
    _capture_policy_runtime_state,
    _predict_policy_action_with_acp_inference,
)
from lerobot.teleoperators import Teleoperator, koch_leader, omx_leader, so_leader
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.recording_annotations import resolve_collector_policy_id
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device
from lerobot.utils.visualization_utils import log_rerun_data_async

T = TypeVar("T")


""" --------------- record_loop() data flow --------------------------
       [ Robot ]
           V
     [ robot.get_observation() ] ---> raw_obs
           V
     [ robot_observation_processor ] ---> processed_obs
           V
     .-----( ACTION LOGIC )------------------.
     V                                       V
     [ From Teleoperator ]                   [ From Policy ]
     |                                       |
     |  [teleop.get_action] -> raw_action    |   [predict_action]
     |          |                            |          |
     |          V                            |          V
     | [teleop_action_processor]             |          |
     |          |                            |          |
     '---> processed_teleop_action           '---> processed_policy_action
     |                                       |
     '-------------------------.-------------'
                               V
                  [ robot_action_processor ] --> robot_action_to_send
                               V
                    [ robot.send_action() ] -- (Robot Executes)
                               V
                    ( Save to Dataset )
                               V
                  ( Rerun Log / Loop Wait )
"""


@safe_stop_image_writer
def record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs after teleop
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs before robot
    robot_observation_processor: RobotProcessorPipeline[
        RobotObservation, RobotObservation
    ],  # runs after robot
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | list[Teleoperator] | None = None,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    display_compressed_images: bool = False,
    policy_sync_executor: PolicySyncDualArmExecutor | None = None,
    intervention_state_machine_enabled: bool = True,
    collector_policy_id_policy: str = "policy",
    collector_policy_id_human: str = "human",
    acp_inference: ACPInferenceConfig | None = None,
    communication_retry_timeout_s: float = 2.0,
    communication_retry_interval_s: float = 0.1,
):
    if acp_inference is None:
        acp_inference = ACPInferenceConfig()

    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    teleop_arm = teleop_keyboard = None
    if isinstance(teleop, list):
        teleop_keyboard = next((t for t in teleop if isinstance(t, KeyboardTeleop)), None)
        teleop_arm = next(
            (
                t
                for t in teleop
                if isinstance(
                    t,
                    (
                        so_leader.SO100Leader
                        | so_leader.SO101Leader
                        | koch_leader.KochLeader
                        | omx_leader.OmxLeader
                    ),
                )
            ),
            None,
        )

        if not (teleop_arm and teleop_keyboard and len(teleop) == 2 and robot.name == "lekiwi_client"):
            raise ValueError(
                "For multi-teleop, the list must contain exactly one KeyboardTeleop and one arm teleoperator. Currently only supported for LeKiwi robot."
            )

    if dataset is None and policy is not None:
        raise ValueError("Policy-driven recording requires a dataset for feature mapping.")

    action_feature_names = dataset.features[ACTION]["names"] if dataset is not None else None
    if action_feature_names is None:
        if hasattr(robot.action_features, "keys"):
            action_feature_names = list(robot.action_features.keys())
        else:
            action_feature_names = list(robot.action_features)
    zero_policy_action = dict.fromkeys(action_feature_names, 0.0)
    # print(f"srj: zero_policy_action: {zero_policy_action}")
    has_teleop = isinstance(teleop, (Teleoperator, list))
    intervention_enabled = intervention_state_machine_enabled and policy is not None and has_teleop
    intervention_state = INTERVENTION_STATE_POLICY
    last_teleop_action: RobotAction | None = None
    teleop_fallback_warned = False

    # ── PI0.5 策略识别 ──
    # 用于启用 PI0.5 专属的推理动作优化 (夹爪二值滞回)。人类介入(S1) 与
    # ACT 等其它策略不受影响 (保持原逻辑)。
    is_pi05_policy = policy is not None and "pi05" in type(policy).__name__.lower()

    # ── S1 (teaching) intervention: direct teleop→follower mapping ──
    # During S1 the operator physically backdrives the leader arm.
    # The leader's joint positions are sent *directly* to the follower
    # without any EMA smoothing or delta-mapping.  Joint smoothing is
    # reserved exclusively for S0 policy-inference actions (below).
    # ── EMA smoothing for S0 policy actions ──────────────────────────
    # Diffusion-based policy models (like Pi0.5) naturally produce
    # high-frequency noise in the predicted action trajectories.
    # Exponential Moving Average (EMA) smoothing acts as a low-pass filter
    # that removes this noise while preserving the intended motion.
    #
    # Per-joint tuning based on model prediction quality:
    #   J2 & J3 (joint_2.pos, joint_3.pos) — model learns these poorly,
    #       producing large jumps and occasional reverse motion.
    #       → Extra-low alpha (0.12–0.18) + rate limiting.
    #   J1, J4, J5, J6 — moderate smoothing (alpha=0.30).
    #   Gripper — faster response (alpha=0.50), no rate limit.
    _smoothed_policy_action: RobotAction = {}
    _prev_policy_action: RobotAction = {}  # for rate limiting
    _pi05_gripper_state: str = "open"      # PI0.5 夹爪二值滞回状态
    _pi05_gripper_dwell: int = 0           # PI0.5 当前夹爪状态已持续步数

    # ── Teleop (S1) smoothing ────────────────────────────────────────
    # 抖动来源：主臂位置反馈含 0.001° 量化噪声 + 机械微振，S1 遥操直接
    # 透传给从臂（无平滑），且 30Hz 主循环每帧耗时不定（IPC/CAN/相机），
    # 导致从臂 JointCtrl 目标更新时序不均 → 从臂轻微抖动/顿挫。
    # 解法：对 S1 遥操关节动作做轻量 EMA 低通（独立于 S0 策略 EMA）。
    #   alpha=1.0  → 完全不平滑（原始直通）
    #   alpha=0.7  → 轻量平滑（延迟约 1 帧 @30Hz≈16ms，人无感知，滤高频微抖）
    #   alpha=0.4  → 更平滑（延迟约 2 帧，手感偏“黏”）
    # 夹爪不参与 EMA（离散开合动作需要即时响应）。
    TELEOP_EMA_ALPHA: float = 0.7
    _teleop_smoothed_action: RobotAction = {}
    POLICY_EMA_DEFAULT: float = 0.30
    POLICY_EMA_GRIPPER: float = 0.50
    # Per-joint EMA alphas (lower = smoother, less responsive)
    _POLICY_EMA_PER_JOINT: dict[str, float] = {
        "joint_1.pos": 0.30,
        "joint_2.pos": 0.12,  # extra smoothing — model performs poorly on J2
        "joint_3.pos": 0.18,  # extra smoothing — model performs poorly on J3
        "joint_4.pos": 0.30,
        "joint_5.pos": 0.30,
        "joint_6.pos": 0.30,
        "gripper.pos": 0.50,
    }
    # Per-joint rate limits (rad/frame @ 30fps).  Caps the maximum
    # step-to-step change to prevent sudden jumps / reverse motion.
    # J2 & J3 have tighter limits; other joints are unconstrained.
    _POLICY_RATE_LIMIT_PER_JOINT: dict[str, float] = {
        "joint_2.pos": 0.015,  # ~0.86 deg/frame — suppress J2 jumps
        "joint_3.pos": 0.015,  # ~0.86 deg/frame — suppress J3 jumps
    }
    # Gripper dead-zone (normalized [0, 1]).  When the model predicts a
    # gripper position below this threshold, force it to 0 (fully open).
    # This prevents the gripper from twitching near the closed position
    # due to model prediction noise.  Set to 0.0 to disable.
    # Typical values: 0.0 (off) to 0.6 (aggressive).
    # For ACT models: 0.57 (≈ 40mm / 70mm max range) is common.
    #参数含义为：夹爪死区（归一化 [0, 1]）。当模型预测的夹爪位置低于此阈值时，将其强制为 0（完全打开）。这可以防止由于模型预测噪声导致夹爪在接近关闭位置时抖动。设置为 0.0 可禁用。典型值：0.0（关闭）到 0.6（激进）。对于 ACT 模型：0.57（≈ 40mm / 70mm 最大范围）是常见值。
    GRIPPER_DEAD_ZONE: float = 0.0

    # ── PI0.5 夹爪二值滞回控制 (同步 inference_pi05_flowmatch.py 调优) ──
    # PI0.5 输出的夹爪命令常在 chunk 内从全开写到全闭再写回 (夹爪极差>0.8),
    # 导致夹爪快速扇动、抓取时刻无法稳定闭合。用滞回 + 最小驻留时间把意图
    # 转化为稳定的"开/闭"状态: >open_above 判定开, <close_below 判定闭,
    # 中间保持上一状态; 切换后至少驻留 min_dwell 步。闭合时直接发送 0
    # (完全闭合, 跳过平滑造成的缓慢闭合)。
    # 仅作用于 PI0.5 S0 策略动作; 人类介入(S1) 与 ACT 策略不受影响。
    PI05_GRIPPER_BINARY_ENABLED: bool = True
    PI05_GRIPPER_OPEN_ABOVE: float = 0.55     # 夹爪意图 > 此值 → 开
    PI05_GRIPPER_CLOSE_BELOW: float = 0.30    # 夹爪意图 < 此值 → 闭
    PI05_GRIPPER_MIN_DWELL: int = 40          # 状态切换后最小驻留步数 (防扇动)

    teleop_arm_for_mode_switch: Any | None = None
    if isinstance(teleop, Teleoperator):
        teleop_arm_for_mode_switch = teleop
    elif isinstance(teleop, list):
        teleop_arm_for_mode_switch = teleop_arm

    def set_teleop_manual_control(enabled: bool) -> None:
        if teleop_arm_for_mode_switch is None:
            return
        if not hasattr(teleop_arm_for_mode_switch, "set_manual_control"):
            return
        try:
            teleop_arm_for_mode_switch.set_manual_control(enabled)
        except Exception:
            logging.exception("Failed to switch teleop manual-control mode to %s", enabled)

    if policy is None:
        # During reset/teleop-only loops keep leader backdrivable for manual dragging.
        set_teleop_manual_control(True)

    # Reset policy and processor if they are provided
    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    cond_policy_runtime_state: dict[str, Any] | None = None
    uncond_policy_runtime_state: dict[str, Any] | None = None
    if policy is not None and acp_inference.enable and acp_inference.use_cfg:
        cond_policy_runtime_state = _capture_policy_runtime_state(policy)
        uncond_policy_runtime_state = _capture_policy_runtime_state(policy)

    if intervention_enabled:
        # Start in S0.
        # When policy_sync_executor is active the leader receives follower
        # joint targets via send_feedback (position-following mode).
        # When policy_sync is disabled the leader stays in
        # gravity-compensation (backdrivable) mode, ready for S1 takeover.
        if policy_sync_executor is not None:
            set_teleop_manual_control(False)
            # Allow sufficient time for the Piper firmware to complete
            # the teach→position mode transition and begin accepting
            # JointCtrl commands.  Rushing this causes the leader to
            # ignore position targets on the first episode.
            time.sleep(0.5)
        else:
            set_teleop_manual_control(True)
        logging.info("\n🎮 S0 模式：策略驱动从臂，主臂镜像跟随。按 'I' 键进入示教模式。\n")

    def run_with_connection_retry(action_name: str, fn: Callable[[], T]) -> T:
        timeout_s = max(communication_retry_timeout_s, 0.0)
        interval_s = max(communication_retry_interval_s, 0.0)
        deadline_t = time.perf_counter() + timeout_s
        attempts = 0
        first_error: ConnectionError | None = None

        while True:
            attempts += 1
            try:
                result = fn()
                if attempts > 1:
                    elapsed_s = timeout_s - max(deadline_t - time.perf_counter(), 0.0)
                    logging.warning(
                        "%s recovered after %d retries in %.2fs.",
                        action_name,
                        attempts - 1,
                        elapsed_s,
                    )
                return result
            except ConnectionError as error:
                if first_error is None:
                    first_error = error
                    logging.warning(
                        "%s failed with transient communication error; retrying for up to %.2fs (%s)",
                        action_name,
                        timeout_s,
                        error,
                    )

                if timeout_s <= 0.0:
                    raise

                remaining_s = deadline_t - time.perf_counter()
                if remaining_s <= 0.0:
                    raise

                sleep_s = interval_s if interval_s > 0.0 else remaining_s
                time.sleep(min(sleep_s, remaining_s))

    timestamp = 0
    start_episode_t = time.perf_counter()
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()
        # Capture frame timestamp early so all rr.log() calls within this
        # iteration share the same "stable_time" timeline value.  This
        # ensures Rerun displays joint state / action scalars as aligned
        # time-series curves rather than scattered data points.
        # NOTE: use time.monotonic() (NOT perf_counter - start_episode_t)
        # so timestamps increase monotonically *across* episodes.  If every
        # episode resets to t≈0, Rerun overlays all episodes on the same
        # 0–60 s range, causing curves to appear frozen and area-fill
        # rendering artifacts.
        frame_timestamp = time.monotonic()

        if events["exit_early"]:
            events["exit_early"] = False
            break
        # print(f"srj: events: {events}")
        if events.get("toggle_intervention", False):
            events["toggle_intervention"] = False
            if intervention_enabled:
                if intervention_state == INTERVENTION_STATE_POLICY:
                    # ── S0 → S1 transition ──
                    intervention_state = INTERVENTION_STATE_ACTIVE
                    set_teleop_manual_control(True)
                    # Reset S0 policy smoothing state so S0→S1→S0
                    # round-trips start with fresh EMA history.
                    _smoothed_policy_action = {}
                    _prev_policy_action = {}
                    _teleop_smoothed_action = {}
                    # PI0.5 夹爪二值滞回复位 (人类示教期间可能已打开夹爪)
                    _pi05_gripper_state = "open"
                    _pi05_gripper_dwell = 0

                    # ── Verify S1 teach mode activation ──
                    # The Piper firmware may require the physical teaching
                    # pendant button to be pressed before gravity compensation
                    # engages, even after the software sends MotionCtrl_2(0x01,0x02).
                    # Check the actual control mode and guide the operator.
                    _s1_mode_ok = False
                    if teleop_arm_for_mode_switch is not None and hasattr(teleop_arm_for_mode_switch, "arm"):
                        try:
                            from lerobot.utils.piper_sdk import read_piper_ctrl_mode
                            from lerobot.utils.piper_sdk import (
                                PIPER_CTRL_MODE_TEACH as _M_TEACH,
                            )
                            from lerobot.utils.piper_sdk import (
                                PIPER_CTRL_MODE_LINKAGE_TEACH_INPUT as _M_LINK,
                            )
                            _mode = read_piper_ctrl_mode(teleop_arm_for_mode_switch.arm, timeout_s=0.3)
                            if _mode in {_M_TEACH, _M_LINK}:
                                _s1_mode_ok = True
                            else:
                                logging.warning(
                                    "⚠️  S1 activated but leader ctrl_mode=0x%02X (expected 0x02 teach). "
                                    "Gravity compensation may not be engaged. "
                                    "Try pressing the physical teaching button on the leader arm.",
                                    _mode if _mode is not None else -1,
                                )
                        except Exception:
                            pass  # Non-Piper teleop or mode read not supported

                    if _s1_mode_ok:
                        logging.info(
                            "🟢 S1 示教模式已激活：主臂重力补偿已启用，拖动主臂控制从臂。按 'I' 返回策略模式。"
                        )
                    else:
                        logging.info(
                            "🟡 S1 待命中：请按下主臂上的示教按键以启用重力补偿，"
                            "然后拖动主臂控制从臂。按 'I' 返回策略模式。"
                        )
                else:
                    # ── S1/S2 → S2/S0 transition ──
                    intervention_state = INTERVENTION_STATE_RELEASE
                    if policy_sync_executor is not None:
                        set_teleop_manual_control(False)
                    else:
                        set_teleop_manual_control(True)
                    if policy is not None and preprocessor is not None and postprocessor is not None:
                        policy.reset()
                        preprocessor.reset()
                        postprocessor.reset()
                        if acp_inference.enable and acp_inference.use_cfg:
                            cond_policy_runtime_state = _capture_policy_runtime_state(policy)
                            uncond_policy_runtime_state = _capture_policy_runtime_state(policy)
                    if policy is not None and preprocessor is not None and postprocessor is not None:
                        logging.info("Policy cache reset on release: next policy action is recomputed.")
                    logging.info(
                        "🔵 控制权已释放，返回策略模式 S0。策略将重新推理。"
                    )
            else:
                logging.info("Intervention toggle ignored because policy+teleop are not both active.")

        # Get robot observation
        obs = robot.get_observation()
        # print(f"srj: obs: {obs}")

        # Applies a pipeline to the raw robot observation, default is IdentityProcessor
        obs_processed = robot_observation_processor(obs)
        # print(f"srj: obs_processed: {obs_processed}")

        if dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

        # Get action from policy and/or teleop
        act_processed_policy: RobotAction | None = None
        act_processed_teleop: RobotAction | None = None
        if (
            policy is not None
            and preprocessor is not None
            and postprocessor is not None
            and not (intervention_enabled and intervention_state == INTERVENTION_STATE_ACTIVE)
        ):
            # print(f"srj: intervention_enabled: {intervention_enabled}; intervention_state: {intervention_state}")
            policy_action = _predict_policy_action_with_acp_inference(
                observation_frame=observation_frame,
                policy=policy,
                device=get_safe_torch_device(policy.config.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=single_task,
                robot_type=robot.robot_type,
                acp_inference=acp_inference,
                cond_runtime_state=cond_policy_runtime_state,
                uncond_runtime_state=uncond_policy_runtime_state,
            )
            # print(f"srj: policy_action: {policy_action}")
            act_processed_policy = make_robot_action(policy_action, dataset.features)
            # print(f"srj: act_processed_policy: {act_processed_policy}")

        if isinstance(teleop, Teleoperator):
            act = run_with_connection_retry("teleop.get_action", teleop.get_action)
            # print(f"srj: teleop_act: {act}")

            # Applies a pipeline to the raw teleop action, default is IdentityProcessor
            act_processed_teleop = teleop_action_processor((act, obs))

        elif isinstance(teleop, list):
            arm_action = run_with_connection_retry("teleop_arm.get_action", teleop_arm.get_action)
            arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
            keyboard_action = teleop_keyboard.get_action()
            base_action = robot._from_keyboard_to_base_action(keyboard_action)
            act = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
            act_processed_teleop = teleop_action_processor((act, obs))

        if act_processed_policy is None and act_processed_teleop is None:
            logging.info(
                "No policy or teleoperator provided, skipping action generation."
                "This is likely to happen when resetting the environment without a teleop device."
                "The robot won't be at its rest position at the start of the next episode."
            )
            continue

        if act_processed_teleop is not None:
            last_teleop_action = act_processed_teleop
            # print(f"srj: last_teleop_action: {last_teleop_action}")
            teleop_fallback_warned = False
            # print(f"srj: teleop_fallback_warned: {teleop_fallback_warned}")

        policy_action_for_storage = (
            act_processed_policy if act_processed_policy is not None else zero_policy_action
        )
        # print(f"srj: policy_action_for_storage: {policy_action_for_storage}")

        # ── Track whether the action came from the policy (vs teleop) ──
        # This flag must be captured BEFORE EMA smoothing creates a new
        # dict (which breaks the `is` identity check below).
        _action_is_from_policy = act_processed_policy is not None

        is_intervention = 0.0
        if intervention_enabled and intervention_state == INTERVENTION_STATE_ACTIVE:
            is_intervention = 1.0
            if act_processed_teleop is not None:
                # ---- Direct leader→follower mapping (no smoothing) ----
                # During S1 (human-in-the-loop teaching), the operator
                # physically backdrives the leader arm.  We send the
                # leader's joint positions *directly* to the follower
                # without any EMA smoothing, delta-mapping, or rate
                # limiting.  This gives the operator instant, 1:1 control
                # of the follower with zero perceptible lag.
                #
                # Joint smoothing and rate limiting are reserved ONLY for
                # S0 policy-inference actions where the diffusion model's
                # high-frequency noise warrants low-pass filtering.
                action_values = dict(act_processed_teleop)

                # ── 轻量 EMA（S1 遥操防抖）──────────────────────────
                # 主臂反馈噪声直接透传会让从臂轻微抖动。加轻量低通（默认
                # alpha=0.7，延迟≈1 帧），仅滤高频微抖；夹爪不做平滑以保
                # 证即时开合。TELEOP_EMA_ALPHA=1.0 可完全关闭。
                if TELEOP_EMA_ALPHA < 1.0:
                    if not _teleop_smoothed_action:
                        _teleop_smoothed_action = dict(action_values)
                    else:
                        _prev_tel = _teleop_smoothed_action
                        for _k, _v in action_values.items():
                            if _k == "gripper.pos":
                                _prev_tel[_k] = float(_v)
                                continue
                            _pv = _prev_tel.get(_k, float(_v))
                            _prev_tel[_k] = (
                                TELEOP_EMA_ALPHA * float(_v)
                                + (1.0 - TELEOP_EMA_ALPHA) * _pv
                            )
                        action_values = dict(_prev_tel)
            elif last_teleop_action is not None:
                action_values = last_teleop_action
                if not teleop_fallback_warned:
                    logging.warning(
                        "Intervention is active but no fresh teleop action is available; reusing last teleop action."
                    )
                    teleop_fallback_warned = True
            elif act_processed_policy is not None:
                action_values = act_processed_policy
                if not teleop_fallback_warned:
                    logging.warning(
                        "Intervention is active but teleop action is unavailable; falling back to policy action."
                    )
                    teleop_fallback_warned = True
            else:
                action_values = zero_policy_action
                if not teleop_fallback_warned:
                    logging.warning(
                        "Intervention is active but no teleop/policy action is available; sending zero action."
                    )
                    teleop_fallback_warned = True
        else:
            action_values = act_processed_policy if act_processed_policy is not None else act_processed_teleop

            # ── EMA smoothing for S0 policy actions ──────────────────
            # The diffusion model produces high-frequency noise in its
            # action predictions.  Apply per-joint EMA low-pass filtering
            # + rate limiting to remove jitter while preserving intentional
            # motion.  J2/J3 get extra-strong smoothing due to poor model
            # prediction quality on these joints.
            if act_processed_policy is not None and action_values is act_processed_policy:
                _is_first_policy_frame = (len(_smoothed_policy_action) == 0)
                for k, v in action_values.items():
                    # ── Per-joint EMA alpha ──────────────────────────
                    alpha = _POLICY_EMA_PER_JOINT.get(k, POLICY_EMA_DEFAULT)
                    v_float = float(v)

                    if _is_first_policy_frame:
                        _smoothed_policy_action[k] = v_float
                    else:
                        prev = _smoothed_policy_action.get(k, v_float)
                        ema_val = alpha * v_float + (1.0 - alpha) * prev

                        # ── Rate limiting for problematic joints ─────
                        # Cap the per-frame change to prevent sudden jumps
                        # and reverse motion (especially J2, J3).
                        rate_limit = _POLICY_RATE_LIMIT_PER_JOINT.get(k)
                        if rate_limit is not None and k in _prev_policy_action:
                            prev_sent = _prev_policy_action[k]
                            delta = ema_val - prev_sent
                            if abs(delta) > rate_limit:
                                ema_val = prev_sent + (
                                    rate_limit if delta > 0 else -rate_limit
                                )

                        _smoothed_policy_action[k] = ema_val

                action_values = dict(_smoothed_policy_action)

                _prev_policy_action = dict(action_values)

        # ── Gripper dead-zone (S0 policy actions only) ──
        # When the model predicts a gripper position below the configured
        # threshold (normalized [0,1]), force it to 0 (fully open).  This
        # suppresses low-amplitude gripper twitch from policy noise.
        #
        # NOT applied during S1 teleop — the human operator's direct grip
        # commands must pass through without modification.
        _gz_key = "gripper.pos"
        if GRIPPER_DEAD_ZONE > 0.0 and _gz_key in action_values and not is_intervention:
            if float(action_values[_gz_key]) < GRIPPER_DEAD_ZONE:
                action_values[_gz_key] = 0.0

        # ── PI0.5 夹爪二值滞回 (仅 PI0.5 S0 策略动作, 同步 flowmatch 调优) ──
        # 将 PI0.5 振荡的夹爪意图转化为稳定"开/闭"状态, 解决"夹不到"/"闭合
        # 缓慢"。人类介入(S1) 与 ACT 策略不经过此逻辑 (保持原行为)。
        if is_pi05_policy and PI05_GRIPPER_BINARY_ENABLED and _gz_key in action_values and not is_intervention:
            _ratio = float(action_values[_gz_key])
            _pi05_gripper_dwell += 1
            if _ratio > PI05_GRIPPER_OPEN_ABOVE and _pi05_gripper_state != "open":
                if _pi05_gripper_dwell >= PI05_GRIPPER_MIN_DWELL:
                    _pi05_gripper_state = "open"
                    _pi05_gripper_dwell = 0
            elif _ratio < PI05_GRIPPER_CLOSE_BELOW and _pi05_gripper_state != "closed":
                if _pi05_gripper_dwell >= PI05_GRIPPER_MIN_DWELL:
                    _pi05_gripper_state = "closed"
                    _pi05_gripper_dwell = 0
            action_values[_gz_key] = 0.0 if _pi05_gripper_state == "closed" else 1.0

        # Applies a pipeline to the action, default is IdentityProcessor
        robot_action_to_send = robot_action_processor((action_values, obs))
        # print(f"srj: robot_action_to_send: {robot_action_to_send}")

        # Send action to robot
        # Action can eventually be clipped using `max_relative_target`,
        # so action actually sent is saved in the dataset. action = postprocessor.process(action)
        # TODO(steven, pepijn, adil): we should use a pipeline step to clip the action, so the sent action is the action that we input to the robot.
        selected_from_policy = _action_is_from_policy
        if policy_sync_executor is not None and selected_from_policy:
            _sent_action = run_with_connection_retry(
                "policy_sync_executor.send_action",
                lambda robot_action_to_send=robot_action_to_send: policy_sync_executor.send_action(
                    robot_action_to_send
                ),
            )
            # print(f"srj: _sent_action: {_sent_action}")
        else:
            _sent_action = run_with_connection_retry(
                "robot.send_action",
                lambda robot_action_to_send=robot_action_to_send: robot.send_action(robot_action_to_send),
            )

        # Write to dataset
        if dataset is not None:
            action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
            policy_action_frame = build_dataset_frame(
                dataset.features, policy_action_for_storage, prefix="complementary_info.policy_action"
            )
            frame = {**observation_frame, **action_frame, **policy_action_frame, "task": single_task}

            if "complementary_info.is_intervention" in dataset.features:
                frame["complementary_info.is_intervention"] = np.array([is_intervention], dtype=np.float32)
            if "complementary_info.state" in dataset.features:
                frame["complementary_info.state"] = np.array([intervention_state], dtype=np.float32)
            if "complementary_info.collector_policy_id" in dataset.features:
                frame["complementary_info.collector_policy_id"] = resolve_collector_policy_id(
                    intervention_enabled=intervention_enabled,
                    is_intervention=bool(is_intervention),
                    selected_from_policy=selected_from_policy,
                    policy_id=collector_policy_id_policy,
                    human_id=collector_policy_id_human,
                )
            dataset.add_frame(frame)

        if display_data:
            log_rerun_data_async(
                observation=obs_processed,
                action=action_values,
                compress_images=display_compressed_images,
                timestamp=frame_timestamp,
            )

        if intervention_state == INTERVENTION_STATE_RELEASE:
            intervention_state = INTERVENTION_STATE_POLICY
            # Reset S0 policy smoothing state so fresh EMA history is
            # used when policy inference resumes after S1→S0 return.
            _smoothed_policy_action = {}
            _prev_policy_action = {}
            _teleop_smoothed_action = {}
            # PI0.5 夹爪二值滞回复位 (策略重新推理, 从打开状态开始)
            _pi05_gripper_state = "open"
            _pi05_gripper_dwell = 0

        dt_s = time.perf_counter() - start_loop_t
        precise_sleep(max(1 / fps - dt_s, 0.0))

        timestamp = time.perf_counter() - start_episode_t
