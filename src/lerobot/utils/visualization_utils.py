# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import numbers
import os

import numpy as np
import rerun as rr

from lerobot.processor import RobotAction, RobotObservation

from .constants import ACTION, ACTION_PREFIX, OBS_PREFIX, OBS_STR


def init_rerun(
    session_name: str = "lerobot_control_loop",
    ip: str | None = None,
    port: int | None = None,
    observation_keys: list[str] | None = None,
    action_keys: list[str] | None = None,
    camera_keys: list[str] | None = None,
) -> None:
    """
    Initializes the Rerun SDK for visualizing the control loop.

    Args:
        session_name: Name of the Rerun session.
        ip: Optional IP for connecting to a Rerun server.
        port: Optional port for connecting to a Rerun server.
        observation_keys: Optional list of scalar observation key names
            (e.g. ``["joint_1.pos", "joint_2.pos"]``) used to build
            TimeSeries views.  If omitted, the default blueprint will
            rely on Rerun heuristics.
        action_keys: Optional list of scalar action key names used to
            build TimeSeries views.
        camera_keys: Optional list of camera entity names (e.g.
            ``["wrist", "top"]``) used to build SpatialView panels.
            These must be simple names without ``.pos`` suffixes.
    """
    # Use a larger flush batch (1 MiB) so the SDK sends fewer, bigger
    # messages instead of flooding the wire with tiny per-scalar packets.
    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", str(1024 * 1024))
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
    rr.init(session_name)

    # ── Build and send a blueprint so that joint state / action
    #     time-series are visible alongside camera feeds ──
    if observation_keys or action_keys or camera_keys:
        _send_hil_blueprint(
            observation_keys=observation_keys or [],
            action_keys=action_keys or [],
            camera_keys=camera_keys or [],
        )

    # Default to 25 % of system RAM so multiple episodes fit without
    # hitting "Reached memory limit … dropping oldest data".
    memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "25%")
    if ip and port:
        rr.connect_grpc(url=f"rerun+http://{ip}:{port}/proxy")
    else:
        rr.spawn(memory_limit=memory_limit)


def _send_hil_blueprint(
    observation_keys: list[str],
    action_keys: list[str],
    camera_keys: list[str],
) -> None:
    """Send a Rerun blueprint with TimeSeries views for HIL recording.

    Creates tabbed panels:
    1. **Camera feeds** — one SpatialView per camera key.
    2. **Observation state** — TimeSeriesView for every ``observation.*`` scalar.
    3. **Action** — TimeSeriesView for every ``action.*`` scalar.
    """
    import rerun.blueprint as rrb

    obs_state_paths: list[str] = []
    for k in observation_keys:
        path = k if str(k).startswith(OBS_PREFIX) else f"{OBS_STR}.{k}"
        obs_state_paths.append(path)

    act_paths: list[str] = []
    for k in action_keys:
        path = k if str(k).startswith(ACTION_PREFIX) else f"{ACTION}.{k}"
        act_paths.append(path)

    # ── Camera views (one per unique camera key) ──
    camera_views: list[rrb.Spatial2DView] = []
    for cam_name in sorted(set(camera_keys)):
        cam_path = f"{OBS_STR}.{cam_name}"
        camera_views.append(
            rrb.Spatial2DView(
                name=f"Camera: {cam_name}",
                origin=cam_path,
                contents=[cam_path],
            )
        )

    # ── TimeSeries views ──
    obs_ts_view: rrb.TimeSeriesView | None = None
    if obs_state_paths:
        obs_ts_view = rrb.TimeSeriesView(
            name="Observation State",
            origin=obs_state_paths[0],
            contents=obs_state_paths,
        )

    act_ts_view: rrb.TimeSeriesView | None = None
    if act_paths:
        act_ts_view = rrb.TimeSeriesView(
            name="Action",
            origin=act_paths[0],
            contents=act_paths,
        )

    # ── Tabs layout: Cameras tab + State curves tab ──
    camera_tab = rrb.Vertical(
        *camera_views,
        name="📷 Cameras",
    ) if camera_views else None

    curves_column_contents: list[rrb.TimeSeriesView] = []
    if obs_ts_view is not None:
        curves_column_contents.append(obs_ts_view)
    if act_ts_view is not None:
        curves_column_contents.append(act_ts_view)

    curves_tab = rrb.Vertical(
        *curves_column_contents,
        name="📈 State Curves",
    ) if curves_column_contents else None

    tabs: list[rrb.Container] = []
    if camera_tab is not None:
        tabs.append(camera_tab)
    if curves_tab is not None:
        tabs.append(curves_tab)

    if tabs:
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                *tabs,
            ),
            collapse_panels=True,
        )
        rr.send_blueprint(blueprint)


def _is_scalar(x):
    return isinstance(x, (float | numbers.Real | np.integer | np.floating)) or (
        isinstance(x, np.ndarray) and x.ndim == 0
    )


def log_rerun_data(
    observation: RobotObservation | None = None,
    action: RobotAction | None = None,
    compress_images: bool = False,
    timestamp: float | None = None,
) -> None:
    """
    Logs observation and action data to Rerun for real-time visualization.

    This function iterates through the provided observation and action
    dictionaries and sends their contents to the Rerun viewer.  It handles
    different data types appropriately:
    - Scalars values (floats, ints) are logged as ``rr.Scalars``.
    - 3D NumPy arrays that resemble images (e.g., with 1, 3, or 4 channels
      first) are transposed from CHW to HWC format, (optionally) compressed
      to JPEG and logged as ``rr.Image`` or ``rr.EncodedImage``.
    - 1D NumPy arrays are logged as a series of individual scalars, with
      each element indexed.
    - Other multi-dimensional arrays are flattened and logged as individual
      scalars.

    Keys are automatically namespaced with ``"observation."`` or ``"action."``
    if not already present.

    .. important::
       All scalars logged within a single call share the same
       ``stable_time`` timeline value so that Rerun displays them as
       aligned time-series curves.  If *timestamp* is ``None`` the
       call-order log-time is used (still shared within this call).

    Args:
        observation: An optional dictionary containing observation data to log.
        action: An optional dictionary containing action data to log.
        compress_images: Whether to compress images before logging to save
            bandwidth & memory in exchange for cpu and quality.
        timestamp: Optional frame timestamp (seconds) used for the
            ``stable_time`` timeline.  When omitted each call still shares
            a single log-time point.
    """
    # ── Set shared timeline for this frame ──
    if timestamp is not None:
        rr.set_time_seconds("stable_time", timestamp)

    if observation:
        for k, v in observation.items():
            if v is None:
                continue
            key = k if str(k).startswith(OBS_PREFIX) else f"{OBS_STR}.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
            elif isinstance(v, np.ndarray):
                arr = v
                # Convert CHW -> HWC when needed
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.ndim == 1:
                    for i, vi in enumerate(arr):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
                else:
                    img_entity = rr.Image(arr).compress() if compress_images else rr.Image(arr)
                    # IMPORTANT: do NOT use static=True for images – it marks
                    # the data as timeless and prevents frame-by-frame updates
                    # in the Rerun viewer (camera feed appears frozen).
                    rr.log(key, entity=img_entity)

    if action:
        for k, v in action.items():
            if v is None:
                continue
            key = k if str(k).startswith(ACTION_PREFIX) else f"{ACTION}.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
            elif isinstance(v, np.ndarray):
                if v.ndim == 1:
                    for i, vi in enumerate(v):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
                else:
                    # Fall back to flattening higher-dimensional arrays
                    flat = v.flatten()
                    for i, vi in enumerate(flat):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))


# ── Non-blocking async Rerun logger ─────────────────────────────────
# `rr.log()` is synchronous and can block the main control loop when
# the Rerun viewer is slow to consume data (especially after video
# encoding starts in background processes for previous episodes).
# This offloads ALL Rerun SDK calls (including set_time_seconds) to
# a single background thread to avoid race conditions.

import queue as _queue
import threading as _threading

class AsyncRerunLogger:
    """Non-blocking Rerun logger.  ALL rr.* calls happen on one daemon thread."""

    def __init__(self, max_queue_size=96):
        self._q = _queue.Queue(maxsize=max_queue_size)
        self._t = _threading.Thread(target=self._run, daemon=True, name="rerun-logger")
        self._t.start()

    def _run(self):
        while True:
            entry = self._q.get()
            if entry is None:
                break
            try:
                entry()  # entry is a zero-arg callable
            except Exception:
                pass

    def _enqueue(self, fn):
        try:
            self._q.put_nowait(fn)
        except _queue.Full:
            pass  # drop silently when viewer can't keep up

    def set_time(self, timeline, seconds):
        """Set the shared timeline.  Executed on the logger thread to avoid races."""
        self._enqueue(lambda: rr.set_time_seconds(timeline, seconds))

    def log(self, entity_path, entity):
        self._enqueue(lambda: rr.log(entity_path, entity))

    def stop(self):
        try:
            self._q.put_nowait(None)
        except _queue.Full:
            pass
        self._t.join(timeout=2.0)


_async_logger: AsyncRerunLogger | None = None

def get_async_logger() -> AsyncRerunLogger:
    global _async_logger
    if _async_logger is None:
        _async_logger = AsyncRerunLogger()
    return _async_logger


def log_rerun_data_async(
    observation=None, action=None, compress_images=False, timestamp=None,
):
    """Non-blocking version of log_rerun_data. All rr.* calls are offloaded."""
    logger = get_async_logger()
    if timestamp is not None:
        logger.set_time("stable_time", timestamp)

    if observation:
        for k, v in observation.items():
            if v is None:
                continue
            key = k if str(k).startswith(OBS_PREFIX) else f"{OBS_STR}.{k}"
            if _is_scalar(v):
                logger.log(key, rr.Scalars(float(v)))
            elif isinstance(v, np.ndarray):
                arr = v
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.ndim == 1:
                    for i, vi in enumerate(arr):
                        logger.log(f"{key}_{i}", rr.Scalars(float(vi)))
                else:
                    img = rr.Image(arr).compress() if compress_images else rr.Image(arr)
                    logger.log(key, img)

    if action:
        for k, v in action.items():
            if v is None:
                continue
            key = k if str(k).startswith(ACTION_PREFIX) else f"{ACTION}.{k}"
            if _is_scalar(v):
                logger.log(key, rr.Scalars(float(v)))
            elif isinstance(v, np.ndarray):
                if v.ndim == 1:
                    for i, vi in enumerate(v):
                        logger.log(f"{key}_{i}", rr.Scalars(float(vi)))
