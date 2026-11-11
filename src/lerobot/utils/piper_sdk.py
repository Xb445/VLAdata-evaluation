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

from __future__ import annotations

import os
import subprocess
import time
from functools import lru_cache
from typing import Any

PIPER_JOINT_NAMES = (
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
)
PIPER_JOINT_ACTION_KEYS = tuple(f"{joint}.pos" for joint in PIPER_JOINT_NAMES)
PIPER_ACTION_KEYS = PIPER_JOINT_ACTION_KEYS + ("gripper.pos",)
PIPER_CTRL_MODE_TEACH = 0x02
PIPER_CTRL_MODE_LINKAGE_TEACH_INPUT = 0x06


def milli_to_unit(value: float | int) -> float:
    return float(value) * 1e-3


def unit_to_milli(value: float | int) -> int:
    return int(round(float(value) * 1e3))


@lru_cache(maxsize=1)
def get_piper_sdk() -> tuple[type[Any], Any]:
    try:
        from piper_sdk import C_PiperInterface_V2, LogLevel

        return C_PiperInterface_V2, LogLevel
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import `piper_sdk`. Install Evo-RL dependencies first (for example: `pip install -e .`)."
        ) from exc


def parse_piper_log_level(level_name: str) -> Any:
    _, log_level_enum = get_piper_sdk()
    normalized = level_name.upper()
    try:
        return getattr(log_level_enum, normalized)
    except AttributeError as exc:
        raise ValueError(
            f"Invalid Piper log level '{level_name}'. "
            "Expected one of: DEBUG, INFO, WARNING, ERROR, CRITICAL, SILENT."
        ) from exc


# ── CAN interface helpers ──────────────────────────────────────────

def _read_sysfs_operstate(interface_name: str) -> str | None:
    path = f"/sys/class/net/{interface_name}/operstate"
    try:
        with open(path) as f:
            return f.read().strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None

def _read_sysfs_carrier(interface_name: str) -> str | None:
    path = f"/sys/class/net/{interface_name}/carrier"
    try:
        with open(path) as f:
            return f.read().strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None

def _get_can_state_details(interface_name: str) -> dict[str, str] | None:
    try:
        result = subprocess.run(
            ["ip", "-details", "link", "show", interface_name],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    info: dict[str, str] = {}
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith(f"{interface_name}:"):
            if "state UP" in line: info["state"] = "UP"
            elif "state DOWN" in line: info["state"] = "DOWN"
            elif "state UNKNOWN" in line: info["state"] = "UNKNOWN"
        if "can state " in line:
            info["can_state"] = line.split("can state ")[1].split()[0]
        if "bitrate " in line:
            info["bitrate"] = line.split("bitrate ")[1].split()[0]
    return info if info else None

def can_interface_is_up(interface_name: str) -> bool:
    operstate = _read_sysfs_operstate(interface_name)
    if operstate is None:
        return False
    if operstate != "up":
        details = _get_can_state_details(interface_name)
        if details is not None and details.get("state") == "UP":
            return True
        return False
    return True

def _try_bring_up_can(interface_name: str, bitrate: int, *, use_sudo: bool) -> bool:
    cmd = ["ip", "link", "set", interface_name, "up", "type", "can", "bitrate", str(bitrate)]
    if use_sudo:
        cmd = ["sudo", "-n"] + cmd
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False
    if result.returncode == 0:
        time.sleep(0.3)
        return can_interface_is_up(interface_name)
    return False

def _raise_can_down_help(interface_name: str, bitrate: int) -> None:
    details = _get_can_state_details(interface_name)
    can_state = details.get("can_state", "unknown") if details else "unknown"
    carrier = _read_sysfs_carrier(interface_name)
    extra = ""
    if carrier == "0":
        extra = ("  ⚠  No carrier detected. Check: arm powered ON, CAN cable connected, "
                 "120Ω termination at both ends.\n")
    raise RuntimeError(
        f"[{interface_name}] CAN interface is DOWN (can_state={can_state}).\n\n"
        f"Please run:  sudo ip link set {interface_name} up type can bitrate {bitrate}\n"
        f"{extra}"
    )

def ensure_can_interface_up(interface_name: str, *, bitrate: int = 1_000_000, auto_bring_up: bool = True) -> None:
    operstate = _read_sysfs_operstate(interface_name)
    if operstate is None:
        raise RuntimeError(
            f"CAN interface '{interface_name}' does not exist.\n"
            f"Create: sudo ip link add dev {interface_name} type can bitrate {bitrate}\n"
            f"Then:   sudo ip link set {interface_name} up type can bitrate {bitrate}"
        )
    if can_interface_is_up(interface_name):
        details = _get_can_state_details(interface_name)
        can_state = details.get("can_state", "?") if details else "?"
        print(f"[{interface_name}] CAN interface is UP (can_state={can_state}).")
        if can_state == "BUS-OFF":
            raise RuntimeError(
                f"[{interface_name}] CAN is BUS-OFF. Run:\n"
                f"  sudo ip link set {interface_name} down\n"
                f"  sudo ip link set {interface_name} up type can bitrate {bitrate}"
            )
        # Allow the CAN controller to stabilize after any external
        # reconfiguration (e.g. setup_can.sh) or previous session
        # cleanup.  Without this delay the socket opened by
        # ConnectPort() may fail to transmit (SEND_MESSAGE_FAILED).
        time.sleep(0.5)
        return
    print(f"[{interface_name}] CAN interface is DOWN. Attempting to bring UP...")
    if not auto_bring_up:
        _raise_can_down_help(interface_name, bitrate)
    if _try_bring_up_can(interface_name, bitrate, use_sudo=False):
        print(f"[{interface_name}] Brought UP (without sudo).")
        return
    if _try_bring_up_can(interface_name, bitrate, use_sudo=True):
        print(f"[{interface_name}] Brought UP (via sudo).")
        return
    _raise_can_down_help(interface_name, bitrate)

# ── End CAN helpers ─────────────────────────────────────────────────


def wait_enable_piper(arm: Any, timeout_s: float, retry_interval_s: float = 0.2) -> bool:
    deadline = time.monotonic() + max(0.0, timeout_s)
    interval_s = max(0.01, retry_interval_s)
    while time.monotonic() < deadline:
        if bool(arm.EnablePiper()):
            return True
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            break
        time.sleep(min(interval_s, remaining_s))
    return False


def _piper_mode_to_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    if hasattr(value, "value") and isinstance(value.value, int):
        return int(value.value)
    if hasattr(value, "__int__"):
        return int(value)
    return None


def read_piper_ctrl_mode(arm: Any, timeout_s: float = 1.0, poll_s: float = 0.02) -> int | None:
    deadline = time.monotonic() + max(0.0, timeout_s)
    while time.monotonic() < deadline:
        status_msg = arm.GetArmStatus()
        if getattr(status_msg, "time_stamp", 0.0) > 0.0:
            arm_status = getattr(status_msg, "arm_status", None)
            if arm_status is not None:
                mode = _piper_mode_to_int(getattr(arm_status, "ctrl_mode", None))
                if mode is not None:
                    return mode
        time.sleep(max(0.005, poll_s))
    return None


def _try_exit_teach_mode_via_motion_ctrl(
    arm: Any, *, interface_name: str, speed_ratio: int = 100, high_follow: bool = True,
    max_retries: int = 5, read_timeout_s: float = 0.5, poll_s: float = 0.02,
) -> bool:
    mit_mode = 0xAD if high_follow else 0x00
    for attempt in range(1, max_retries + 1):
        arm.MotionCtrl_2(0x01, 0x01, speed_ratio, mit_mode)
        settle_s = 0.2 + 0.1 * attempt
        time.sleep(settle_s)
        verify_mode = read_piper_ctrl_mode(arm, timeout_s=read_timeout_s, poll_s=poll_s)
        if verify_mode is None:
            print(f"[{interface_name}] MotionCtrl_2 attempt {attempt}/{max_retries}: cannot read mode.")
            continue
        if verify_mode not in {PIPER_CTRL_MODE_TEACH, PIPER_CTRL_MODE_LINKAGE_TEACH_INPUT}:
            print(f"[{interface_name}] Mode switch OK (0x{verify_mode:02X}) after {attempt} attempt(s).")
            return True
        print(f"[{interface_name}] Attempt {attempt}/{max_retries}: still teach (0x{verify_mode:02X}).")
    return False

def _try_exit_teach_mode_via_reset(
    arm: Any, *, interface_name: str, speed_ratio: int = 100, high_follow: bool = True,
    read_timeout_s: float = 0.5, poll_s: float = 0.02, disable_param: int = 7,
) -> bool:
    mit_mode = 0xAD if high_follow else 0x00
    print(f"[{interface_name}] Performing full disable→enable reset...")
    try:
        arm.DisableArm(disable_param)
        time.sleep(0.5)
        arm.EnablePiper()
        time.sleep(0.5)
        arm.MotionCtrl_2(0x01, 0x01, speed_ratio, mit_mode)
        time.sleep(0.3)
        verify_mode = read_piper_ctrl_mode(arm, timeout_s=read_timeout_s, poll_s=poll_s)
        if verify_mode is not None and verify_mode not in {PIPER_CTRL_MODE_TEACH, PIPER_CTRL_MODE_LINKAGE_TEACH_INPUT}:
            print(f"[{interface_name}] Mode OK after reset (0x{verify_mode:02X}).")
            return True
    except Exception as exc:
        print(f"[{interface_name}] Reset failed: {exc}")
    return False

def guard_piper_ctrl_mode_on_connect(
    arm: Any, *, interface_name: str,
    timeout_s: float = 0.5, poll_s: float = 0.02, settle_s: float = 0.05,
) -> None:
    _ = settle_s
    # Step 0: pre-cleanup (Disable + Enable to clear stale state)
    preclean_failures = 0
    try:
        arm.DisableArm(7)
        time.sleep(0.3)
    except Exception:
        preclean_failures += 1
    try:
        arm.EnablePiper()
        time.sleep(0.3)
    except Exception:
        preclean_failures += 1
    can_appears_dead = (preclean_failures >= 2)

    # Step 1: read mode with retries
    mode = None
    max_read_retries = 1 if can_appears_dead else 3
    for read_attempt in range(1, max_read_retries + 1):
        mode = read_piper_ctrl_mode(arm, timeout_s=timeout_s, poll_s=poll_s)
        print(f"srj: mode: {mode} ")
        if mode is not None:
            break
        if read_attempt < max_read_retries:
            wait_s = 0.5 * read_attempt
            print(f"[{interface_name}] Mode read {read_attempt}/{max_read_retries} returned None. Waiting {wait_s:.1f}s...")
            time.sleep(wait_s)

    # Step 2: mode=None → blind recovery or hardware error
    if mode is None:
        if can_appears_dead:
            raise RuntimeError(
                f"[{interface_name}] CAN unreachable. Check power, wiring, termination."
            )
        print(f"[{interface_name}] Arm unresponsive (stuck in teach mode?). Attempting blind recovery...")
        if _try_exit_teach_mode_via_motion_ctrl(arm, interface_name=interface_name, max_retries=3,
                                                 read_timeout_s=timeout_s, poll_s=poll_s):
            mode = read_piper_ctrl_mode(arm, timeout_s=timeout_s, poll_s=poll_s)
            if mode is not None and mode not in {PIPER_CTRL_MODE_TEACH, PIPER_CTRL_MODE_LINKAGE_TEACH_INPUT}:
                return
        if _try_exit_teach_mode_via_reset(arm, interface_name=interface_name,
                                           read_timeout_s=timeout_s, poll_s=poll_s):
            mode = read_piper_ctrl_mode(arm, timeout_s=timeout_s, poll_s=poll_s)
            if mode is not None and mode not in {PIPER_CTRL_MODE_TEACH, PIPER_CTRL_MODE_LINKAGE_TEACH_INPUT}:
                return
        raise RuntimeError(
            f"[{interface_name}] Arm unresponsive (CAN UP but no status data).\n\n"
            "The CAN interface is working but the arm's firmware is not responding.\n"
            "Most likely causes:\n"
            "  1. Arm is NOT powered ON → check the 24V power supply LED.\n"
            "  2. Arm needs a power-cycle → disconnect & reconnect 24V, wait 5s.\n"
            "  3. CAN wiring issue → check cable and 120Ω termination.\n\n"
            "Quick fix:  bash scripts/setup_can.sh --diagnose"
        )

    # Step 3: valid mode → done
    if mode not in {PIPER_CTRL_MODE_TEACH, PIPER_CTRL_MODE_LINKAGE_TEACH_INPUT}:
        return

    # Step 4: teach mode → recover
    print(f"[{interface_name}] Arm in teach mode (0x{mode:02X}). Recovering...")
    if _try_exit_teach_mode_via_motion_ctrl(arm, interface_name=interface_name,
                                             read_timeout_s=timeout_s, poll_s=poll_s):
        return
    print(f"[{interface_name}] Trying full reset...")
    if _try_exit_teach_mode_via_reset(arm, interface_name=interface_name,
                                       read_timeout_s=timeout_s, poll_s=poll_s):
        return
    raise RuntimeError(
        f"[{interface_name}] Stuck in teach mode (0x{mode:02X}). Power-cycle the arm and retry."
    )
