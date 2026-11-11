"""
VLM 场景分类客户端 — 调用 vLLM 服务进行包裹场景识别

对齐微调数据格式 (train_data_ABC_v3.json):
    输入:  "<image><image>你是一个包裹抓取任务的视觉调度器。...完整任务描述..."
    输出:  {"parcel_location": "桌面|托盘|盒子", "label_visible": true|false, "subtask": "subtask1|subtask2|subtask3|subtask4"}
"""

import json
import time
import base64
import logging
import re
from io import BytesIO
from typing import Optional, Dict
from dataclasses import dataclass, field

import numpy as np
import requests
from PIL import Image

from lerobot.pi05_vlm_orchestrator.config import OrchestratorConfig, SUBTASK_TO_MODEL, SUBTASK_DESCRIPTIONS

logger = logging.getLogger(__name__)

# 微调对齐的 prompt — 与 train_data_ABC_v3.json 训练数据逐字一致。
# 推理时必须使用与训练完全相同的指令, 否则模型学到的
# "两步推理 (位置+标签可见性) + JSON 输出" 行为不会被正确触发。
VLM_PROMPT = (
    "<image><image>你是一个包裹抓取任务的视觉调度器。\n\n"
    "【相机信息】\n"
    "- 第一张图：顶部全局相机，固定在工作台正上方，提供俯视全局视图，用于判断包裹位置（桌面/托盘/盒子）和快递单标签是否朝上可见\n"
    "- 第二张图：腕部近景相机，安装在机械臂夹爪末端，提供包裹近距离特写，用于辅助确认包裹位置与快递单标签可见性\n\n"
    "【场景物体】\n"
    "- 包裹：灰白色包装袋/棕色纸箱，表面可能贴有白色快递单（含条码、运单文字），可能位于桌面、托盘或盒子内\n"
    "- 托盘：深色圆形，位置固定在工作台上\n"
    "- 盒子：矩形纸箱，位置固定在工作台上\n\n"
    "【判断规则】\n"
    "1. 先用第一张图（顶部俯视）判断包裹位置：桌面 / 托盘 / 盒子\n"
    "2. 再结合第一张图（顶部俯视）与第二张图（腕部近景）判断快递单标签是否可见\n"
    "3. 根据 包裹位置 + 标签可见性 输出当前任务阶段：\n"
    "   - subtask1：包裹在桌面上 → 抓取包裹放入托盘\n"
    "   - subtask2：包裹在托盘中，且标签不可见（朝下）→ 翻转包裹使标签朝上\n"
    "   - subtask3：包裹在托盘中，且标签可见（朝上）→ 夹取包裹放入盒子\n"
    "   - subtask4：包裹已在盒子中（托盘和桌面均无包裹）→ 任务完成（done）\n\n"
    "请严格按JSON格式输出，不要添加Markdown标记，不要解释：\n"
    '{"parcel_location": "桌面|托盘|盒子", "label_visible": true|false, "subtask": "subtask1|subtask2|subtask3|subtask4"}'
)

# vLLM OpenAI-compatible chat endpoint
CHAT_ENDPOINT = "/chat/completions"


@dataclass
class VLMDecision:
    """VLM 单次场景分类结果"""
    subtask: str = ""               # 原始输出: subtask1/2/3/4
    parcel_location: str = "未知"    # 包裹位置: 桌面/托盘/盒子
    label_visible: bool = False     # 面单是否可见
    target_model: str = ""          # 对应的 ACT 模型: Task_A/B/C/standby
    raw_response: str = ""          # 原始 API 返回文本
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)
    success: bool = False           # API 调用是否成功
    error_message: str = ""

    def __post_init__(self):
        if self.subtask and not self.target_model:
            self.target_model = SUBTASK_TO_MODEL.get(self.subtask, "standby")

    def __repr__(self) -> str:
        if self.success:
            return (f"VLMDecision(subtask={self.subtask}, "
                    f"location={self.parcel_location}, "
                    f"label_visible={self.label_visible}, "
                    f"model={self.target_model})")
        return f"VLMDecision(FAILED: {self.error_message})"


class VLMClient:
    """
    VLM 场景分类客户端

    通过 vLLM OpenAI-compatible API 调用微调后的 Qwen3-VL-2B 模型，
    输入顶部+腕部两张相机图像，输出场景分类结果。
    """

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self._base_url = config.vlm_base_url.rstrip("/")
        self._chat_url = f"{self._base_url}{CHAT_ENDPOINT}"
        self._model = config.vlm_model_name
        self._timeout = config.vlm_timeout
        self._max_tokens = config.vlm_max_tokens
        self._temperature = config.vlm_temperature
        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
        })

        # 统计
        self._total_calls: int = 0
        self._success_calls: int = 0
        self._last_call_time: float = 0.0

        # 验证连接
        self._check_connection()

    # ========================================================================
    # 连接检查
    # ========================================================================

    def _check_connection(self):
        """检查 vLLM 服务是否可达"""
        try:
            resp = self._session.get(
                f"{self._base_url}/models",
                timeout=5.0,
            )
            if resp.status_code == 200:
                models_data = resp.json()
                model_ids = [m.get("id", "") for m in models_data.get("data", [])]
                logger.info(f"[VLM] vLLM 服务连接成功, 可用模型: {model_ids}")
            else:
                logger.warning(f"[VLM] vLLM 服务响应异常: HTTP {resp.status_code}")
        except requests.exceptions.ConnectionError:
            logger.warning(f"[VLM] 无法连接到 vLLM 服务 ({self._base_url}), "
                           f"将使用回退策略")
        except Exception as e:
            logger.warning(f"[VLM] 连接检查失败: {e}")

    # ========================================================================
    # 图像编码
    # ========================================================================

    @staticmethod
    def _encode_image_b64(image: np.ndarray) -> str:
        """将 numpy 图像编码为 base64 data URL"""
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        # 确保是 RGB 格式
        if image.ndim == 2:
            pil_img = Image.fromarray(image, mode="L").convert("RGB")
        elif image.shape[2] == 1:
            pil_img = Image.fromarray(image.squeeze(), mode="L").convert("RGB")
        elif image.shape[2] == 3:
            pil_img = Image.fromarray(image)
        else:
            pil_img = Image.fromarray(image[..., :3])

        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG", quality=90)
        b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_str}"

    # ========================================================================
    # 核心接口: 场景分类
    # ========================================================================

    def classify(
        self,
        top_image: Optional[np.ndarray] = None,
        wrist_image: Optional[np.ndarray] = None,
    ) -> VLMDecision:
        """
        对当前场景进行分类, 返回 subask 决策。

        Args:
            top_image: 顶部相机图像 (H, W, 3) numpy array
            wrist_image: 腕部相机图像 (H, W, 3) numpy array

        Returns:
            VLMDecision 包含场景分类结果
        """
        self._total_calls += 1

        # 构建多模态消息 (对齐微调格式: 两张图 + 文本 prompt)
        content_parts = []

        # 图像1: top view
        if top_image is not None:
            img_b64 = self._encode_image_b64(top_image)
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": img_b64},
            })

        # 图像2: wrist view
        if wrist_image is not None:
            img_b64 = self._encode_image_b64(wrist_image)
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": img_b64},
            })

        # 文本 prompt (对齐微调格式)
        content_parts.append({
            "type": "text",
            "text": VLM_PROMPT,
        })

        messages = [
            {"role": "user", "content": content_parts},
        ]

        payload = {
            "model": self._model,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }

        try:
            self._last_call_time = time.time()
            resp = self._session.post(
                self._chat_url,
                json=payload,
                timeout=self._timeout,
            )

            if resp.status_code == 200:
                data = resp.json()
                raw_text = data["choices"][0]["message"]["content"].strip()
                decision = self._parse_response(raw_text)
                decision.success = True
                self._success_calls += 1
                return decision
            else:
                error_msg = f"HTTP {resp.status_code}: {resp.text[:200]}"
                logger.error(f"[VLM] API 调用失败: {error_msg}")
                return VLMDecision(
                    success=False,
                    error_message=error_msg,
                )

        except requests.exceptions.Timeout:
            logger.error(f"[VLM] API 调用超时 ({self._timeout}s)")
            return VLMDecision(success=False, error_message="timeout")
        except requests.exceptions.ConnectionError:
            logger.error("[VLM] vLLM 服务连接失败")
            return VLMDecision(success=False, error_message="connection_error")
        except Exception as e:
            logger.error(f"[VLM] API 调用异常: {e}")
            return VLMDecision(success=False, error_message=str(e))

    # ========================================================================
    # 响应解析
    # ========================================================================

    def _parse_response(self, raw_text: str) -> VLMDecision:
        """
        解析 VLM 输出的 JSON 响应。

        期望格式: {"parcel_location": "桌面", "label_visible": false, "subtask": "subtask1"}
        兼容非严格 JSON 格式 (如模型输出中包含额外文本)。

        重要: 由于 VLM 输出的 subtask 字段可能与 parcel_location/label_visible
        不一致 (模型并非联合推理这三个字段), 解析后必须做一致性校验,
        以 parcel_location + label_visible 推导的 subtask 为准。
        """
        decision = VLMDecision(raw_response=raw_text)

        # 尝试多种方式提取 JSON
        json_str = self._extract_json(raw_text)

        if json_str:
            try:
                data = json.loads(json_str)
                decision.subtask = data.get("subtask", "")
                decision.parcel_location = data.get("parcel_location", "未知")
                # label_visible 必须归一化为 bool: 模型可能输出字符串 "true"/"false",
                # 直接 bool("false") 会错误地得到 True
                decision.label_visible = self._normalize_bool(data.get("label_visible", False))
                decision.confidence = 0.9
            except json.JSONDecodeError as e:
                logger.warning(f"[VLM] JSON 解析失败: {e}, raw='{raw_text[:200]}'")
        else:
            # JSON 提取失败 → 关键词回退
            decision = self._keyword_fallback(raw_text, decision)
            decision.confidence = 0.3

        # ── 一致性校验: 基于 parcel_location + label_visible 推导正确 subtask ──
        derived_subtask = self._derive_subtask(decision.parcel_location, decision.label_visible)
        if derived_subtask and derived_subtask != decision.subtask:
            logger.warning(
                f"[VLM] ⚠️ subtask 不一致! "
                f"VLM输出={decision.subtask}, "
                f"推导值={derived_subtask} "
                f"(基于 location={decision.parcel_location}, label_visible={decision.label_visible}), "
                f"以推导值为准"
            )
            decision.subtask = derived_subtask

        # 如果 subtask 仍为空 (推导也失败), 最后一次尝试
        if not decision.subtask:
            decision.subtask = self._derive_subtask(decision.parcel_location, decision.label_visible)
            if decision.subtask:
                logger.warning(f"[VLM] subtask 缺失, 从位置推导: {decision.subtask}")

        decision.target_model = SUBTASK_TO_MODEL.get(decision.subtask, "standby")
        return decision

    @staticmethod
    def _normalize_bool(value) -> bool:
        """
        将 VLM 输出的 label_visible 统一归一化为 bool。

        兼容类型:
            bool: 原样返回
            int/float: 非 0 即 True
            str: "true"/"1"/"yes"/"是"/"可见" → True
                 "false"/"0"/"no"/"否"/"不可见" → False
            None/其他: 默认 False (与训练标签默认一致)
        """
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            v = value.strip().lower()
            if v in ("true", "1", "yes", "是", "可见"):
                return True
            if v in ("false", "0", "no", "否", "不可见"):
                return False
        return False

    @staticmethod
    def _derive_subtask(parcel_location: str, label_visible: bool) -> str:
        """
        根据 parcel_location 和 label_visible 推导正确的 subtask。

        映射关系 (由任务定义决定, 是确定性的):
          - 桌面 → subtask1  (抓取到托盘)
          - 托盘 + 标签不可见 → subtask2 (翻转包裹)
          - 托盘 + 标签可见   → subtask3 (放入盒子)
          - 盒子 → subtask4  (待机)
        """
        loc = parcel_location.strip()
        if "桌面" in loc or "桌" in loc or "table" in loc.lower() or "desk" in loc.lower():
            return "subtask1"
        elif "托盘" in loc or "盘" in loc or "tray" in loc.lower():
            return "subtask2" if not label_visible else "subtask3"
        elif "盒子" in loc or "盒" in loc or "box" in loc.lower():
            return "subtask4"
        return ""

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        """从文本中提取 JSON 对象字符串"""
        # 方式0: 去除可能的 markdown 代码块围栏 (```json ... ```)
        cleaned = re.sub(r"```(?:json|JSON)?\s*", "", text)
        cleaned = cleaned.replace("```", "")

        # 方式1: 直接匹配 {...}
        json_match = re.search(r'\{[^{}]*\}', cleaned)
        if json_match:
            return json_match.group(0)

        # 方式2: 尝试匹配嵌套的 {...} (简单处理)
        start = cleaned.find('{')
        end = cleaned.rfind('}')
        if start >= 0 and end > start:
            return cleaned[start:end + 1]

        return None

    @staticmethod
    def _keyword_fallback(raw_text: str, decision: VLMDecision) -> VLMDecision:
        """JSON 解析失败时, 通过关键词推断分类"""
        text_lower = raw_text.lower()

        # 包裹位置关键词
        if "桌面" in raw_text or "table" in text_lower or "桌" in raw_text:
            decision.parcel_location = "桌面"
        elif "托盘" in raw_text or "tray" in text_lower or "盘" in raw_text:
            decision.parcel_location = "托盘"
        elif "盒子" in raw_text or "box" in text_lower or "盒" in raw_text:
            decision.parcel_location = "盒子"

        # 标签可见性 — 注意: "不可见" 是 "可见" 的子串, 必须先排除 "不可见"/"false"
        if "false" in text_lower or "不可见" in raw_text or "朝下" in raw_text:
            decision.label_visible = False
        elif "true" in text_lower or "可见" in raw_text or "朝上" in raw_text:
            decision.label_visible = True

        # subtask 关键词
        if "subtask1" in text_lower:
            decision.subtask = "subtask1"
        elif "subtask2" in text_lower:
            decision.subtask = "subtask2"
        elif "subtask3" in text_lower:
            decision.subtask = "subtask3"
        elif ("subtask4" in text_lower or "done" in text_lower
              or "任务完成" in raw_text or "完成" in raw_text):
            decision.subtask = "subtask4"
        # 注意: 不在此处基于位置推导 subtask,
        # 而是留给 _parse_response 中统一的一致性校验处理
        return decision

    # ========================================================================
    # 统计信息
    # ========================================================================

    @property
    def stats(self) -> dict:
        return {
            "total_calls": self._total_calls,
            "success_calls": self._success_calls,
            "success_rate": self._success_calls / max(self._total_calls, 1),
            "last_call_time": self._last_call_time,
        }

    def reset_stats(self):
        self._total_calls = 0
        self._success_calls = 0
