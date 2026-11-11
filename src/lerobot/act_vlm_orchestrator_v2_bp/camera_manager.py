"""
相机管理器 — 统一管理顶部和腕部相机, 为 VLM 和 ACT 提供场景图像

【v2.0 架构】
  统一帧缓存模式: 后台线程是唯一的相机读取者, 持续更新帧缓存。
  ACT 推理和 VLM 分类都从缓存中获取最新帧, 不再各自独立 read(),
  从而消除多方竞争同一 VideoCapture 导致的帧错位/抖动问题。

支持两路相机:
    - top (顶部视角): 观察桌面/托盘/盒子区域的全局场景
    - wrist (腕部视角): 观察机械臂末端执行器附近的局部场景
"""

import time
import logging
import threading
from typing import Optional, Dict, Tuple

import numpy as np

from lerobot.act_vlm_orchestrator_v2.config import OrchestratorConfig

logger = logging.getLogger(__name__)


class CameraManager:
    """
    相机管理器 (v2.0 统一帧缓存)

    核心设计:
      - 后台线程以 camera_fps 频率持续读取两路相机
      - 最新帧存入线程安全的缓存 (带时间戳)
      - ACT 和 VLM 统一通过 get_latest_frames() 获取缓存帧
      - 帧缓存带有时间戳, 可检测帧是否过期

    使用示例:
        cam_mgr = CameraManager(config)
        cam_mgr.connect()
        cam_mgr.start_async()  # 启动后台拉流

        # 获取最新帧 (ACT 和 VLM 通用)
        top_img, wrist_img, ts = cam_mgr.get_latest_frames()

        # 检查帧是否足够新鲜
        if cam_mgr.is_frame_fresh(max_age=0.1):
            ...
    """

    # 帧过期阈值: 超过此时间的缓存帧视为过期
    FRAME_STALE_AGE = 0.5  # 秒

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self._top_camera_id = config.top_camera_id
        self._wrist_camera_id = config.wrist_camera_id
        self._width = config.camera_width
        self._height = config.camera_height
        self._fps = config.camera_fps

        # 相机实例
        self._top_cam = None
        self._wrist_cam = None

        # 最新帧缓存 (带时间戳)
        self._top_frame: Optional[np.ndarray] = None
        self._wrist_frame: Optional[np.ndarray] = None
        self._top_frame_ts: float = 0.0
        self._wrist_frame_ts: float = 0.0
        self._frame_lock = threading.Lock()

        # 异步读取线程 (唯一相机消费者)
        self._async_thread: Optional[threading.Thread] = None
        self._async_running: bool = False
        self._async_interval: float = 1.0 / max(self._fps, 1)

        # 状态
        self._connected: bool = False
        self._frame_count: int = 0
        self._error_count: int = 0
        self._consecutive_errors: int = 0
        self._max_consecutive_errors: int = 10

    # ========================================================================
    # 连接管理
    # ========================================================================

    def connect(self) -> bool:
        """连接两路相机"""
        logger.info(f"[Camera] 连接相机: top={self._top_camera_id}, wrist={self._wrist_camera_id}")

        try:
            import cv2

            # Top 相机
            self._top_cam = cv2.VideoCapture(self._top_camera_id)
            if not self._top_cam.isOpened():
                logger.error(f"[Camera] ❌ 无法打开 top 相机 (ID={self._top_camera_id})")
                return False
            self._top_cam.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            self._top_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            self._top_cam.set(cv2.CAP_PROP_FPS, self._fps)
            self._top_cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Wrist 相机
            self._wrist_cam = cv2.VideoCapture(self._wrist_camera_id)
            if not self._wrist_cam.isOpened():
                logger.error(f"[Camera] ❌ 无法打开 wrist 相机 (ID={self._wrist_camera_id})")
                self._top_cam.release()
                self._top_cam = None
                return False
            self._wrist_cam.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            self._wrist_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            self._wrist_cam.set(cv2.CAP_PROP_FPS, self._fps)
            self._wrist_cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self._connected = True
            logger.info(f"[Camera] ✅ 相机连接成功 ({self._width}x{self._height}@{self._fps}fps)")

            # 预热: 读取几帧丢弃 (相机刚打开时可能曝光不稳定)
            for _ in range(5):
                self._top_cam.read()
                self._wrist_cam.read()
                time.sleep(0.05)

            return True

        except Exception as e:
            logger.error(f"[Camera] ❌ 相机连接失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def disconnect(self):
        """断开相机连接"""
        self.stop_async()

        if self._top_cam is not None:
            self._top_cam.release()
            self._top_cam = None
        if self._wrist_cam is not None:
            self._wrist_cam.release()
            self._wrist_cam = None

        self._connected = False
        logger.info("[Camera] 相机已断开")

    # ========================================================================
    # 图像采集
    # ========================================================================

    def capture_frame(self, camera_name: str) -> Optional[np.ndarray]:
        """
        同步读取单路相机的一帧。

        Args:
            camera_name: "top" 或 "wrist"

        Returns:
            RGB 图像 (H, W, 3) numpy array, 或 None

        注意: OpenCV 的 cv2.VideoCapture.read() 返回 BGR,
        但 lerobot 相机管线 (OpenCVCamera) 会转成 RGB 后喂给模型,
        ACT 训练数据为 RGB。因此这里必须做 BGR→RGB 转换,
        否则模型输入的通道顺序颠倒会导致输出异常 (如"保持姿态")。
        """
        cam = self._top_cam if camera_name == "top" else self._wrist_cam
        if cam is None:
            logger.warning(f"[Camera] {camera_name} 相机未连接")
            return None

        try:
            import cv2
            ret, frame = cam.read()
            if ret and frame is not None:
                self._frame_count += 1
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                self._error_count += 1
                logger.warning(f"[Camera] {camera_name} 读取失败 (总错误: {self._error_count})")
                return None
        except Exception as e:
            self._error_count += 1
            logger.error(f"[Camera] {camera_name} 读取异常: {e}")
            return None

    def capture_vlm_images(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        同步读取两路相机, 返回 VLM 用图像。

        Returns:
            (top_image, wrist_image): RGB 格式 numpy array (与 lerobot 相机管线一致)
        """
        top_img = self.capture_frame("top")
        wrist_img = self.capture_frame("wrist")
        return top_img, wrist_img

    def capture_act_observation(self) -> Dict[str, np.ndarray]:
        """
        获取 ACT 模型观测用的图像。

        Returns:
            {"top": image, "wrist": image} 字典
            注意: ACT 模型期望的 key 格式可能不同,
            实际使用时需要在 orchestrator 中做 key 映射。
        """
        top_img = self.capture_frame("top")
        wrist_img = self.capture_frame("wrist")

        obs = {}
        if top_img is not None:
            obs["top"] = top_img
        if wrist_img is not None:
            obs["wrist"] = wrist_img

        return obs

    # ========================================================================
    # 异步读取 (后台线程持续拉流, 降低延迟)
    # ========================================================================

    def start_async(self):
        """启动后台异步读取线程"""
        if self._async_running:
            return

        self._async_running = True
        self._async_thread = threading.Thread(
            target=self._async_loop,
            daemon=True,
            name="camera-async",
        )
        self._async_thread.start()
        logger.info("[Camera] 异步读取线程已启动")

    def stop_async(self):
        """停止后台异步读取线程"""
        self._async_running = False
        if self._async_thread is not None:
            self._async_thread.join(timeout=2.0)
            self._async_thread = None
        logger.info("[Camera] 异步读取线程已停止")

    def _async_loop(self):
        """后台线程: 唯一相机读取者, 持续更新帧缓存"""
        import cv2
        while self._async_running:
            try:
                now = time.time()

                # 读取 top (OpenCV 返回 BGR, 统一转 RGB 与 lerobot 相机管线/训练数据一致)
                if self._top_cam is not None and self._top_cam.isOpened():
                    ret, frame = self._top_cam.read()
                    if ret and frame is not None:
                        with self._frame_lock:
                            self._top_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            self._top_frame_ts = now
                        self._consecutive_errors = 0
                    else:
                        self._consecutive_errors += 1

                # 读取 wrist
                if self._wrist_cam is not None and self._wrist_cam.isOpened():
                    ret, frame = self._wrist_cam.read()
                    if ret and frame is not None:
                        with self._frame_lock:
                            self._wrist_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            self._wrist_frame_ts = now
                        self._consecutive_errors = 0
                    else:
                        self._consecutive_errors += 1

                self._frame_count += 2

                # 连续错误过多时告警
                if self._consecutive_errors >= self._max_consecutive_errors:
                    logger.error(
                        f"[Camera] ⚠️ 连续 {self._consecutive_errors} 次读取失败, "
                        f"相机可能已断开!"
                    )

            except Exception as e:
                logger.error(f"[Camera] 异步读取异常: {e}")
                self._error_count += 1

            time.sleep(self._async_interval)

    def get_latest_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        获取最新缓存帧 (线程安全, 非阻塞)。

        Returns:
            (top_image, wrist_image, timestamp):
                - top_image: 顶部相机最新帧 (RGB numpy array) 或 None
                - wrist_image: 腕部相机最新帧 或 None
                - timestamp: 帧的时间戳 (取两帧中较新的), 用于判断新鲜度
        """
        with self._frame_lock:
            top = self._top_frame.copy() if self._top_frame is not None else None
            wrist = self._wrist_frame.copy() if self._wrist_frame is not None else None
            ts = max(self._top_frame_ts, self._wrist_frame_ts)
        return top, wrist, ts

    def is_frame_fresh(self, max_age: float = None) -> bool:
        """
        检查缓存帧是否足够新鲜。

        Args:
            max_age: 最大允许年龄 (秒), 默认使用 FRAME_STALE_AGE

        Returns:
            True 如果两路相机的最新帧都在 max_age 以内
        """
        if max_age is None:
            max_age = self.FRAME_STALE_AGE
        now = time.time()
        with self._frame_lock:
            top_fresh = (now - self._top_frame_ts) < max_age if self._top_frame is not None else False
            wrist_fresh = (now - self._wrist_frame_ts) < max_age if self._wrist_frame is not None else False
        return top_fresh and wrist_fresh

    # ========================================================================
    # 兼容旧接口 (capture_vlm_images / capture_act_observation)
    # 现在统一从帧缓存获取, 不再直接 read()
    # ========================================================================

    def capture_vlm_images(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        VLM 用图像 (从缓存获取)。

        Returns:
            (top_image, wrist_image): RGB 格式 numpy array
        """
        top_img, wrist_img, _ = self.get_latest_frames()
        return top_img, wrist_img

    def capture_act_observation(self) -> Dict[str, np.ndarray]:
        """
        ACT 模型观测用图像 (从缓存获取)。

        Returns:
            {"top": image, "wrist": image} 字典
        """
        top_img, wrist_img, _ = self.get_latest_frames()

        obs = {}
        if top_img is not None:
            obs["top"] = top_img
        if wrist_img is not None:
            obs["wrist"] = wrist_img

        return obs

    # ========================================================================
    # 状态查询
    # ========================================================================

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def stats(self) -> dict:
        return {
            "connected": self._connected,
            "frame_count": self._frame_count,
            "error_count": self._error_count,
            "async_running": self._async_running,
        }
