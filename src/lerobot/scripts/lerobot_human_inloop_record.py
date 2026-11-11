#!/usr/bin/env python

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
Records with policy execution and teleop-device action mirroring enabled.

Phase A/B goals:
- Same policy action is executed on the follower robot and mirrored to the teleop arm.
- Keyboard-toggled intervention lets teleop temporarily take over execution.
- 代码中断后，teleop可以将机器人手臂移动到一个安全的重置位置，然后按下键盘上的成功/失败键来结束当前episode。
- 代码作用为记录人类在环实验中的数据，包括策略执行和远程操作设备动作的镜像。
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

from lerobot.configs import parser
from lerobot.scripts.lerobot_record import RecordConfig, record
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.recording_annotations import (
    EPISODE_FAILURE,
    EPISODE_SUCCESS,
    infer_collector_policy_version,
)


def _default_failure_reset_pose_path(cfg: RecordConfig) -> Path:
    robot_id = cfg.robot.id if cfg.robot.id else "default"
    robot_type = cfg.robot.type if hasattr(cfg.robot, "type") else type(cfg.robot).__name__
    return HF_LEROBOT_HOME / "failure_reset_pose" / f"{robot_type}_{robot_id}.json"


def _extract_joint_pos_from_observation(observation: dict[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in observation.items() if key.endswith(".pos")}


def _save_failure_reset_pose(robot: Any, pose_path: Path) -> dict[str, float]:
    observation = robot.get_observation()
    joint_pos = _extract_joint_pos_from_observation(observation)
    if not joint_pos:
        raise ValueError("Could not capture failure reset pose: no '.pos' joints found in observation.")

    pose_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"robot_type": robot.robot_type, "joint_pos": joint_pos}
    with open(pose_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    logging.info("Saved failure_reset_pose to %s", pose_path)
    return joint_pos


def _load_failure_reset_pose(pose_path: Path) -> dict[str, float]:
    with open(pose_path) as f:
        payload = json.load(f)
    joint_pos_raw = payload["joint_pos"] if isinstance(payload, dict) and "joint_pos" in payload else payload
    if not isinstance(joint_pos_raw, dict):
        raise ValueError(
            f"Invalid failure reset pose payload in {pose_path}: expected dict, got {type(joint_pos_raw)}"
        )
    joint_pos = {str(key): float(value) for key, value in joint_pos_raw.items() if str(key).endswith(".pos")}
    if not joint_pos:
        raise ValueError(f"Invalid failure reset pose payload in {pose_path}: no '.pos' joints found.")
    logging.info("Loaded failure_reset_pose from %s", pose_path)
    return joint_pos


def _slow_reset_all_arms_to_pose(
    robot: Any,
    teleop: Any,
    target_pose: dict[str, float],
    duration_s: float = 3.0,
) -> None:
    joint_keys = [key for key in robot.action_features if key.endswith(".pos") and key in target_pose]
    if not joint_keys:
        logging.warning("No matching '.pos' joints found for the stored reset pose.")
        return

    current_pose = _extract_joint_pos_from_observation(robot.get_observation())
    start_pose = {key: current_pose.get(key, float(target_pose[key])) for key in joint_keys}
    goal_pose = {key: float(target_pose[key]) for key in joint_keys}

    # ── Move both arms to reset pose ──
    # Do NOT force-set manual_control here.  send_feedback() internally
    # ensures the leader is in position-command mode when needed.  Forcing
    # the mode here would cause unnecessary teach→position→teach→position
    # ping-pong across the episode lifecycle.
    step_dt_s = 0.05
    steps = max(int(duration_s / step_dt_s), 1)
    for idx in range(1, steps + 1):
        alpha = idx / steps
        action = {key: start_pose[key] + (goal_pose[key] - start_pose[key]) * alpha for key in joint_keys}
        robot.send_action(action)
        if teleop is not None and not isinstance(teleop, list):
            teleop.send_feedback(action)
        time.sleep(step_dt_s)

    logging.info("Episode ended. Arms returned to the stored reset pose in %.1fs.", duration_s)


class _HumanInloopFailureResetController:
    def __init__(self, cfg: RecordConfig):
        self.pose_path = _default_failure_reset_pose_path(cfg)
        self.failure_reset_pose: dict[str, float] | None = None

    def on_record_connected(self, robot: Any, teleop: Any) -> None:
        if self.pose_path.is_file():
            self.failure_reset_pose = _load_failure_reset_pose(self.pose_path)
            return

        input(
            "Human-inloop with policy detected.\n"
            "Please ensure ALL robot arms are at reset position, then press ENTER to capture:\n"
            f"{self.pose_path}\n"
        )
        self.failure_reset_pose = _save_failure_reset_pose(robot=robot, pose_path=self.pose_path)

    def on_episode_outcome(self, robot: Any, teleop: Any, episode_success: str | None) -> None:
        if episode_success in {EPISODE_FAILURE, EPISODE_SUCCESS} and self.failure_reset_pose is not None:
            _slow_reset_all_arms_to_pose(robot=robot, teleop=teleop, target_pose=self.failure_reset_pose)


@parser.wrap()
def human_inloop_record(cfg: RecordConfig):
    if cfg.teleop is None:
        raise ValueError("`lerobot-human-inloop-record` requires `teleop` config.")

    cfg.policy_sync_to_teleop = cfg.policy is not None
    cfg.intervention_state_machine_enabled = cfg.policy is not None
    cfg.enable_episode_outcome_labeling = True
    cfg.default_episode_success = "failure"
    cfg.enable_collector_policy_id = True
    if cfg.collector_policy_id_policy is None:
        cfg.collector_policy_id_policy = infer_collector_policy_version(cfg.policy)
    if cfg.policy is not None:
        failure_reset_controller = _HumanInloopFailureResetController(cfg)
        cfg._on_record_connected = failure_reset_controller.on_record_connected
        cfg._on_record_episode_outcome = failure_reset_controller.on_episode_outcome

    logging.info(
        "\n╔══════════════════════════════════════════════════════════╗\n"
        "║         🎮 人类在环 (Human-in-the-Loop) 录制模式          ║\n"
        "╠══════════════════════════════════════════════════════════╣\n"
        "║  按 '%s' 键 → 切换 策略模式 / 示教模式                   ║\n"
        "║  按 '%s' 键 → 标记成功并结束当前 episode                 ║\n"
        "║  按 '%s' 键 → 标记失败并结束当前 episode                 ║\n"
        "║  按 '→' 键 → 跳过当前 episode                           ║\n"
        "║  按 'ESC'  → 停止录制并退出                              ║\n"
        "╠══════════════════════════════════════════════════════════╣\n"
        "║  S0 策略模式：AI 驱动从臂，主臂镜像跟随                   ║\n"
        "║  S1 示教模式：拖动主臂遥控从臂，人工辅助完成任务          ║\n"
        "╚══════════════════════════════════════════════════════════╝\n",
        cfg.intervention_toggle_key,
        cfg.episode_success_key,
        cfg.episode_failure_key,
    )
    return record(cfg)


def main():
    register_third_party_plugins()
    human_inloop_record()


if __name__ == "__main__":
    main()
