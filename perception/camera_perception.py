#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
camera_perception.py  (v3 — VLM 升级版)
=========================================
核心升级：用视觉语言模型（VLM）替代 HSV 颜色阈值检测。

【原版问题】
  - HSV 检测需要对每种颜色手动标定阈值
  - 实验室有新颜色方块时必须重新标定
  - 光线变化会导致阈值失效

【升级后】
  - 把摄像头图像发给 VLM（GPT-4o / DeepSeek-VL / Qwen-VL 等）
  - 用自然语言描述问 VLM："图中有哪些颜色的方块和垃圾桶，像素坐标在哪里？"
  - VLM 返回结构化 JSON，包含颜色名称和像素坐标
  - 再通过标定矩阵把像素坐标转换成机械臂坐标系
  - 无论什么颜色，只要 VLM 认识这个颜色词就能检测，无需预先标定

【相机位置约定】
  本项目默认摄像头安装在机械臂末端（eye-in-hand，Lab2 方式）。
  摄像头朝下俯拍桌面。标定矩阵把像素坐标 (u,v) 映射到机械臂基坐标系 (x,y)。
  如果摄像头安装位置不同，只需更新 calib 字典。

【精度保证措施】
  1. Prompt 明确要求返回像素坐标的中心点，不是边界框
  2. 要求 VLM 区分方块（小正方体）和桶（大矩形容器）
  3. 多次重试取均值，降低单次抖动
  4. 保留 HSV 作为 fallback，VLM 失败时自动切换
  5. 提供 scan_scene_with_retry() 高精度模式，多帧平均

【真机部署时的图像质量建议】
  - 摄像头分辨率 >= 640x480
  - 桌面铺白色/浅色背景布，增加对比度
  - 避免强侧光，均匀照明
  - 每次 episode 开始前静止 0.5 秒再扫描
"""

import os
import re
import json
import time
import base64
import textwrap
import numpy as np
from typing import Dict, List, Optional, Tuple

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import rospy
    from sensor_msgs.msg import Image
    HAS_ROS = True
except ImportError:
    HAS_ROS = False

try:
    from cv_bridge import CvBridge
    HAS_CV_BRIDGE = True
except ImportError:
    HAS_CV_BRIDGE = False

# ── OpenAI 兼容层（与 llm_policy.py 相同的双版本支持） ────────────────────
_FORCE_LEGACY = os.environ.get("USE_OPENAI_LEGACY", "").strip() in {
    "1", "true", "True", "YES", "yes"}
try:
    if _FORCE_LEGACY:
        raise ImportError("forced legacy")
    from openai import OpenAI as _OpenAIClient
    _OPENAI_IS_V1 = True
except Exception:
    try:
        import openai as _openai_legacy
        _OpenAIClient = None
        _OPENAI_IS_V1 = False
    except ImportError:
        _OpenAIClient = None
        _OPENAI_IS_V1 = None


def _create_client(api_key: str, base_url: Optional[str]):
    """创建 OpenAI 兼容客户端（支持新旧两版 SDK）。"""
    if _OPENAI_IS_V1 and _OpenAIClient:
        kw = {"api_key": api_key, "timeout": 45.0}
        if base_url:
            kw["base_url"] = base_url
        return _OpenAIClient(**kw)
    elif _OPENAI_IS_V1 is False:
        _openai_legacy.api_key = api_key
        if base_url:
            _openai_legacy.api_base = base_url
        # 返回一个鸭子类型对象，兼容 client.chat.completions.create(...)
        class _Legacy:
            class chat:
                class completions:
                    @staticmethod
                    def create(model, messages, temperature, max_tokens, **kw):
                        resp = _openai_legacy.ChatCompletion.create(
                            model=model, messages=messages,
                            temperature=temperature, max_tokens=max_tokens)
                        content = resp["choices"][0]["message"]["content"]
                        class _R:
                            class choices:
                                pass
                        class _C:
                            class message:
                                pass
                        _C.message.content = content
                        _R.choices = [_C()]
                        return _R()
        return _Legacy()
    else:
        raise RuntimeError(
            "openai 库未安装，请运行：pip install openai")


# ── VLM 模型 preset（与 llm_policy.py 一致） ──────────────────────────────
VLM_PRESETS = {
    # DeepSeek-VL（有视觉能力）
    "deepseek-vl": {
        "base_url": "https://api.deepseek.com/v1",
        "model":    "deepseek-chat",        # 或 deepseek-vl2 如果开通了
    },
    # Qwen-VL（阿里，推荐，视觉能力强）
    "qwen-vl": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model":    "qwen-vl-max",
    },
    # OpenAI GPT-4o（视觉最强，但费用高）
    "gpt-4o": {
        "base_url": None,
        "model":    "gpt-4o",
    },
    "gpt-4o-mini": {
        "base_url": None,
        "model":    "gpt-4o-mini",
    },
    # Kimi（月之暗面，有视觉版本）
    "kimi-vl": {
        "base_url": "https://api.moonshot.cn/v1",
        "model":    "moonshot-v1-8k-vision-preview",
    },
}

# ── 场景区域参数（与 pick_place_env.py 保持一致） ─────────────────────────
# 摄像头图像宽度中点（像素），左侧为方块区，右侧为桶区
# 这个值需要根据你的实际摄像头分辨率和安装位置调整
# 默认 640x480 摄像头，分割线在水平中点 x=320
DEFAULT_IMAGE_WIDTH  = 640
DEFAULT_IMAGE_HEIGHT = 480
DEFAULT_SPLIT_X      = 320   # 左半区=方块，右半区=桶

# ── 标定参数默认值（运行 Lab2 标定后替换） ────────────────────────────────
DEFAULT_CALIB = {
    "kx": -0.00029,   # x_robot = kx * v_pixel + bx
    "bx":  0.31084,
    "ky":  0.00030,   # y_robot = ky * u_pixel + by
    "by":  0.09080,
}

# ── Prompt 模板 ────────────────────────────────────────────────────────────

# 完整扫描 prompt：识别所有物体（方块 + 桶）
SCAN_PROMPT = textwrap.dedent("""
You are analyzing a top-down camera image of a robot workspace.
The image shows colored blocks (small cubes ~5cm) and colored bins (larger
rectangular containers ~7cm×7cm×9cm with an open top) on a table surface.

TASK: Identify ALL visible colored objects and return their pixel coordinates.

RULES:
1. Blocks are SMALL (~30-50 pixels wide), cube-shaped, solid colored.
2. Bins are LARGER (~50-80 pixels wide), rectangular container shape, open top.
3. Return the CENTER pixel of each object.
4. Color names must be lowercase English (e.g. "red", "blue", "green",
   "yellow", "pink", "orange", "purple", "cyan", "brown", "white", "black").
5. If the same color appears as both a block AND a bin, report both separately.
6. Image size is {W}x{H} pixels. Coordinates must be within this range.
7. Only report objects you are confident about. Skip unclear/occluded objects.

Return ONLY valid JSON, no other text:
{{
  "blocks": [
    {{"color": "red",   "u": 145, "v": 230}},
    {{"color": "blue",  "u": 310, "v": 195}}
  ],
  "bins": [
    {{"color": "green", "u": 480, "v": 280}},
    {{"color": "red",   "u": 520, "v": 310}}
  ]
}}

If no objects are visible, return: {{"blocks": [], "bins": []}}
""").strip()


def _resolve_vlm_model_name(explicit: Optional[str]) -> str:
    """显式 vlm_model 非空则用之，否则读 env_config.vlm_model()。"""
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    try:
        from env_config import vlm_model
        return vlm_model()
    except ImportError:
        return os.environ.get("VLM_MODEL", "").strip() or "qwen-vl"


# 单目标查询 prompt：查找特定颜色特定类型的物体
FIND_PROMPT = textwrap.dedent("""
You are analyzing a top-down camera image of a robot workspace.
Find the {obj_type} that is {color} colored.

A block is a small cube (~5cm, ~30-50px wide).
A bin is a larger rectangular container (~7cm×7cm×9cm, ~50-80px wide, open top).

Return ONLY valid JSON with the CENTER pixel coordinate:
{{"found": true, "u": 145, "v": 230, "confidence": 0.95}}

If not found:
{{"found": false, "u": null, "v": null, "confidence": 0.0}}

Image size: {W}x{H} pixels.
""").strip()


class VLMPerception:
    """
    基于视觉语言模型（VLM）的场景感知模块。

    真机部署时用这个类替代原来的 HSV 检测，
    可以识别任意颜色，无需预先标定 HSV 阈值。
    """

    def __init__(self,
                 api_key: str,
                 vlm_model: Optional[str] = None,
                 base_url: str = None,
                 calib: Optional[Dict] = None,
                 image_width: int = DEFAULT_IMAGE_WIDTH,
                 image_height: int = DEFAULT_IMAGE_HEIGHT,
                 split_x: int = DEFAULT_SPLIT_X,
                 use_zone_constraint: bool = True):
        """
        Args:
            api_key            : VLM API key
            vlm_model          : 模型名称，见 VLM_PRESETS；None/空则读环境变量 VLM_MODEL
            base_url           : 自定义 API base URL（None=使用preset默认）
            calib              : 标定参数 {kx,bx,ky,by}，None=使用默认值
            image_width        : 摄像头图像宽度（像素）
            image_height       : 摄像头图像高度（像素）
            split_x            : 左右分区分割线（像素），左=方块区，右=桶区
            use_zone_constraint: 是否用左右分区约束来消歧同色物体
        """
        self.calib = calib or dict(DEFAULT_CALIB)
        self.img_w = image_width
        self.img_h = image_height
        self.split_x = split_x
        self.use_zone = use_zone_constraint

        vlm_model = _resolve_vlm_model_name(vlm_model)

        # 解析模型preset
        if vlm_model in VLM_PRESETS:
            preset   = VLM_PRESETS[vlm_model]
            base_url = base_url or preset["base_url"]
            self.model = preset["model"]
        else:
            self.model = vlm_model
        self._client = _create_client(api_key, base_url)

        # ROS 图像订阅
        self._latest_image: Optional[np.ndarray] = None
        self._bridge = CvBridge() if HAS_CV_BRIDGE else None
        if HAS_ROS and HAS_CV_BRIDGE:
            rospy.Subscriber("/usb_cam/image_raw", Image,
                             self._image_cb, queue_size=1)
            if HAS_ROS:
                rospy.loginfo(
                    f"[VLMPerception] 初始化完成，模型={self.model}")
        else:
            print(f"[VLMPerception] 初始化完成（离线模式），模型={self.model}")

    def _image_cb(self, msg):
        if self._bridge:
            try:
                self._latest_image = self._bridge.imgmsg_to_cv2(msg, "bgr8")
            except Exception:
                pass

    # ── 图像处理工具 ──────────────────────────────────────────────────────

    def _encode_image(self, img: np.ndarray) -> str:
        """
        把 OpenCV BGR 图像编码为 base64 JPEG 字符串，用于 VLM API。

        精度优化：
          - 发送前裁剪到桌面工作区域（去掉机械臂本体部分）
          - 降低 JPEG 压缩率（quality=90），减少颜色失真
        """
        if not HAS_CV2:
            raise RuntimeError("cv2 未安装")
        # 裁剪到桌面区域（可根据实际摄像头视野调整）
        h, w = img.shape[:2]
        # 去掉图像顶部 10%（通常是机械臂底座）和左侧 5%
        y0 = int(h * 0.10)
        x0 = int(w * 0.05)
        cropped = img[y0:, x0:]
        _, buf = cv2.imencode(".jpg", cropped,
                              [cv2.IMWRITE_JPEG_QUALITY, 90])
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    def _draw_debug(self, img: np.ndarray,
                    scene: Dict) -> np.ndarray:
        """在图像上标注检测结果，用于调试。"""
        if not HAS_CV2:
            return img
        out = img.copy()
        # 画分割线
        cv2.line(out, (self.split_x, 0), (self.split_x, self.img_h),
                 (0, 255, 255), 1)
        for item in scene.get("blocks", []):
            if item is None:
                continue
            u, v = int(item.get("u_orig", 0)), int(item.get("v_orig", 0))
            cv2.circle(out, (u, v), 8, (0, 255, 0), 2)
            cv2.putText(out, f"{item['color']}_block",
                        (u+10, v), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 255, 0), 1)
        for item in scene.get("bins", []):
            if item is None:
                continue
            u, v = int(item.get("u_orig", 0)), int(item.get("v_orig", 0))
            cv2.circle(out, (u, v), 12, (255, 0, 0), 2)
            cv2.putText(out, f"{item['color']}_bin",
                        (u+10, v), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 0, 0), 1)
        return out

    # ── 坐标转换 ──────────────────────────────────────────────────────────

    def pixel_to_robot(self, u: float, v: float,
                       crop_offset_x: int = 0,
                       crop_offset_y: int = 0) -> Tuple[float, float]:
        """
        像素坐标 (u, v) → 机械臂基坐标系 (x, y)。

        注意：VLM 返回的坐标是裁剪后图像的坐标，
        需要加回裁剪偏移量换算到原始图像坐标，再做标定转换。

        标定公式（来自 Lab2）：
          x_robot = kx * v_original + bx
          y_robot = ky * u_original + by
        """
        u_orig = u + crop_offset_x
        v_orig = v + crop_offset_y
        x = self.calib["kx"] * v_orig + self.calib["bx"]
        y = self.calib["ky"] * u_orig + self.calib["by"]
        return float(x), float(y)

    def update_calibration(self, kx: float, bx: float,
                           ky: float, by: float):
        """更新标定系数（运行 Lab2 标定程序后调用）。"""
        self.calib = {"kx": kx, "bx": bx, "ky": ky, "by": by}
        print(f"[VLMPerception] 标定已更新: {self.calib}")

    def load_calibration_from_yaml(self, yaml_path: str):
        """从 Lab2 vision_config.yaml 加载标定系数。"""
        import yaml
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        if "calibration" in data:
            c = data["calibration"]
        else:
            c = data
        self.calib = {
            "kx": float(c.get("kx", self.calib["kx"])),
            "bx": float(c.get("bx", self.calib["bx"])),
            "ky": float(c.get("ky", self.calib["ky"])),
            "by": float(c.get("by", self.calib["by"])),
        }
        print(f"[VLMPerception] 从 {yaml_path} 加载标定: {self.calib}")

    # ── VLM 调用核心 ─────────────────────────────────────────────────────

    def _call_vlm_scan(self, img: np.ndarray) -> Dict:
        """
        调用 VLM 扫描整个场景，返回所有物体的原始 VLM 响应（已解析 JSON）。

        返回格式：
          {
            "blocks": [{"color": "red",  "u": 145, "v": 230}, ...],
            "bins":   [{"color": "blue", "u": 480, "v": 280}, ...]
          }
        """
        img_b64 = self._encode_image(img)
        prompt  = SCAN_PROMPT.format(W=self.img_w, H=self.img_h)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}",
                            "detail": "high",   # 高分辨率模式，坐标更准确
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,    # 低温度 → 更稳定的坐标输出
            max_tokens=800,
        )
        content = resp.choices[0].message.content.strip()

        # 清理可能的 markdown 代码块
        content = re.sub(r"```[a-z]*\n?", "", content)
        content = re.sub(r"```",           "", content)
        content = content.strip()

        data = json.loads(content)

        # 验证格式
        if "blocks" not in data or "bins" not in data:
            raise ValueError(f"VLM 返回格式错误: {content[:200]}")

        return data

    def _call_vlm_find(self, img: np.ndarray,
                       color: str, obj_type: str) -> Optional[Dict]:
        """
        调用 VLM 查找特定颜色特定类型的物体（更精确的单目标查询）。

        用于在 scan_scene 失败后的精确补查。
        """
        img_b64 = self._encode_image(img)
        prompt  = FIND_PROMPT.format(
            color=color, obj_type=obj_type,
            W=self.img_w, H=self.img_h)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{img_b64}",
                                   "detail": "high"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        resp = self._client.chat.completions.create(
            model=self.model, messages=messages,
            temperature=0.05, max_tokens=100)
        content = resp.choices[0].message.content.strip()
        content = re.sub(r"```[a-z]*\n?|```", "", content).strip()

        data = json.loads(content)
        if data.get("found") and data.get("u") is not None:
            return {"u": float(data["u"]), "v": float(data["v"]),
                    "confidence": float(data.get("confidence", 0.8))}
        return None

    # ── 区域约束消歧 ──────────────────────────────────────────────────────

    def _apply_zone_constraint(self, raw: Dict) -> Dict:
        """
        用左右分区约束对 VLM 结果做后处理。

        规则：
          - 左半区 (u < split_x)：认为是方块
          - 右半区 (u >= split_x)：认为是桶
          - 如果 VLM 把方块放到了右半区，或把桶放到了左半区，
            且没有更好的同色候选，则接受（不强制过滤，只做警告）
          - 如果同颜色在同一区域出现多次，取置信度最高的（面积最大/位置最中心）

        这个规则不是强制性的，而是在有歧义时优先选择符合区域的那个。
        """
        if not self.use_zone:
            return raw

        # 按颜色去重：同颜色同类型保留最可信的（靠近区域中心）
        def _deduplicate(items: List[Dict], expected_side: str) -> List[Dict]:
            by_color = {}
            for item in items:
                c = item["color"]
                u = item["u"]
                # 计算区域分数：越符合预期区域得分越高
                if expected_side == "left":
                    zone_score = (self.split_x - u) / self.split_x
                else:
                    zone_score = (u - self.split_x) / (self.img_w - self.split_x)

                item["_zone_score"] = zone_score
                if c not in by_color or zone_score > by_color[c]["_zone_score"]:
                    by_color[c] = item

            return list(by_color.values())

        raw["blocks"] = _deduplicate(raw.get("blocks", []), "left")
        raw["bins"]   = _deduplicate(raw.get("bins",   []), "right")
        return raw

    # ── 主扫描接口 ────────────────────────────────────────────────────────

    def scan_scene(self,
                   wait_sec: float = 0.5,
                   n_retry: int = 2
                   ) -> Dict[str, Dict[str, Optional[np.ndarray]]]:
        """
        扫描当前场景，返回所有检测到的方块和垃圾桶位置。

        Args:
            wait_sec : 等待图像稳定的时间（秒）
            n_retry  : VLM 调用失败时的重试次数

        Returns:
            {
              "blocks": {color_name: np.array([x,y]) or None},
              "bins":   {color_name: np.array([x,y]) or None},
            }
            坐标都是机械臂基坐标系下的 (x,y)，单位米。
        """
        time.sleep(wait_sec)

        # 初始化返回结构（颜色列表从 color_config 动态获取）
        try:
            from config.color_config import get_color_config
            known_colors = get_color_config().colors
        except Exception:
            known_colors = []

        result = {
            "blocks": {c: None for c in known_colors},
            "bins":   {c: None for c in known_colors},
        }
        # 也保留动态颜色槽（VLM 可能返回 known_colors 之外的颜色）
        result["_raw_blocks"] = []
        result["_raw_bins"]   = []

        if self._latest_image is None:
            msg = "[VLMPerception] 没有收到图像！请检查摄像头连接。"
            if HAS_ROS:
                rospy.logwarn(msg)
            else:
                print(msg)
            return result

        img = self._latest_image.copy()

        # 计算裁剪偏移（与 _encode_image 一致）
        h, w   = img.shape[:2]
        crop_x = int(w * 0.05)
        crop_y = int(h * 0.10)

        raw = None
        last_err = None
        for attempt in range(n_retry + 1):
            try:
                raw = self._call_vlm_scan(img)
                break
            except Exception as e:
                last_err = e
                print(f"[VLMPerception] VLM 调用失败 (尝试{attempt+1}/{n_retry+1}): {e}")
                if attempt < n_retry:
                    time.sleep(1.0)

        if raw is None:
            print(f"[VLMPerception] 所有重试失败，最后错误: {last_err}")
            return result

        # 应用区域约束
        raw = self._apply_zone_constraint(raw)

        # 保存原始结果（供调试用）
        result["_raw_blocks"] = raw.get("blocks", [])
        result["_raw_bins"]   = raw.get("bins",   [])

        # 转换坐标并填入结果
        for item in raw.get("blocks", []):
            color = item["color"].lower().strip()
            u, v  = float(item["u"]), float(item["v"])
            rx, ry = self.pixel_to_robot(u, v, crop_x, crop_y)
            pos = np.array([rx, ry], dtype=np.float32)
            result["blocks"][color] = pos
            # 也填入 known_colors 之外的颜色（动态扩展）
            if color not in known_colors:
                print(f"[VLMPerception] 发现新颜色方块: {color} @ ({rx:.3f},{ry:.3f})")

        for item in raw.get("bins", []):
            color = item["color"].lower().strip()
            u, v  = float(item["u"]), float(item["v"])
            rx, ry = self.pixel_to_robot(u, v, crop_x, crop_y)
            pos = np.array([rx, ry], dtype=np.float32)
            result["bins"][color] = pos
            if color not in known_colors:
                print(f"[VLMPerception] 发现新颜色桶: {color} @ ({rx:.3f},{ry:.3f})")

        self._log_result(result)
        return result

    def scan_scene_with_retry(self,
                              target_block: str,
                              target_bin: str,
                              n_frames: int = 3,
                              wait_sec: float = 0.3
                              ) -> Dict[str, Dict[str, Optional[np.ndarray]]]:
        """
        高精度扫描：对目标颜色的方块和桶各扫描 n_frames 帧，取均值。

        在 episode 开始时调用一次，比 scan_scene 更准确。
        """
        block_positions = []
        bin_positions   = []

        for i in range(n_frames):
            scene = self.scan_scene(wait_sec=wait_sec, n_retry=1)
            bp = scene["blocks"].get(target_block)
            binp = scene["bins"].get(target_bin)
            if bp   is not None: block_positions.append(bp)
            if binp is not None: bin_positions.append(binp)
            if i < n_frames - 1:
                time.sleep(0.1)

        # 均值计算
        scene_final = self.scan_scene(wait_sec=0.1, n_retry=1)
        if block_positions:
            scene_final["blocks"][target_block] = np.mean(
                block_positions, axis=0).astype(np.float32)
        if bin_positions:
            scene_final["bins"][target_bin] = np.mean(
                bin_positions, axis=0).astype(np.float32)

        return scene_final

    def get_block_position(self, color: str) -> Optional[np.ndarray]:
        """获取指定颜色方块的位置（单目标精确查询）。"""
        if self._latest_image is None:
            return None
        try:
            result = self._call_vlm_find(
                self._latest_image, color, "block")
            if result:
                h, w = self._latest_image.shape[:2]
                crop_x = int(w * 0.05)
                crop_y = int(h * 0.10)
                rx, ry = self.pixel_to_robot(
                    result["u"], result["v"], crop_x, crop_y)
                return np.array([rx, ry], dtype=np.float32)
        except Exception as e:
            print(f"[VLMPerception] get_block_position({color}) 失败: {e}")
        return None

    def get_bin_position(self, color: str) -> Optional[np.ndarray]:
        """获取指定颜色桶的位置（单目标精确查询）。"""
        if self._latest_image is None:
            return None
        try:
            result = self._call_vlm_find(
                self._latest_image, color, "bin")
            if result:
                h, w = self._latest_image.shape[:2]
                crop_x = int(w * 0.05)
                crop_y = int(h * 0.10)
                rx, ry = self.pixel_to_robot(
                    result["u"], result["v"], crop_x, crop_y)
                return np.array([rx, ry], dtype=np.float32)
        except Exception as e:
            print(f"[VLMPerception] get_bin_position({color}) 失败: {e}")
        return None

    def get_debug_image(self) -> Optional[np.ndarray]:
        """返回带检测标注的调试图像。"""
        if self._latest_image is None:
            return None
        try:
            scene = self.scan_scene(wait_sec=0.1)
            return self._draw_debug(self._latest_image, scene)
        except Exception:
            return self._latest_image.copy()

    def print_scene_summary(self):
        """打印当前场景的简洁摘要（调试用）。"""
        scene = self.scan_scene(wait_sec=0.5)
        print("\n[VLMPerception] 场景扫描结果：")
        print("  方块：")
        for c, pos in scene["blocks"].items():
            if pos is not None:
                print(f"    {c:10s}: x={pos[0]:.3f}, y={pos[1]:.3f}")
            else:
                print(f"    {c:10s}: 未检测到")
        print("  垃圾桶：")
        for c, pos in scene["bins"].items():
            if pos is not None:
                print(f"    {c:10s}: x={pos[0]:.3f}, y={pos[1]:.3f}")
            else:
                print(f"    {c:10s}: 未检测到")

    def _log_result(self, result: Dict):
        n_blocks = sum(1 for v in result["blocks"].values() if v is not None)
        n_bins   = sum(1 for v in result["bins"].values()   if v is not None)
        msg = (f"[VLMPerception] 检测到 {n_blocks} 个方块，"
               f"{n_bins} 个桶")
        if HAS_ROS:
            rospy.loginfo(msg)
        else:
            print(msg)


# ── HSV Fallback（仅在 VLM 不可用时使用） ────────────────────────────────

class HSVPerceptionFallback:
    """
    HSV 颜色检测后备方案。

    仅在 VLM API 不可用时使用（如网络断线、API 配额耗尽）。
    功能和精度均弱于 VLMPerception，仅支持预先标定的颜色。
    """

    BLOCK_MAX_AREA = 2500
    BIN_MIN_AREA   = 2500

    def __init__(self, color_config=None, calib=None):
        from config.color_config import get_color_config
        self.color_cfg = color_config or get_color_config()
        self.calib = calib or dict(DEFAULT_CALIB)
        self._latest_image = None
        self._bridge = CvBridge() if HAS_CV_BRIDGE else None
        if HAS_ROS:
            rospy.Subscriber("/usb_cam/image_raw", Image,
                             self._image_cb, queue_size=1)

    def _image_cb(self, msg):
        if self._bridge:
            try:
                self._latest_image = self._bridge.imgmsg_to_cv2(msg, "bgr8")
            except Exception:
                pass

    def pixel_to_robot(self, u, v):
        x = self.calib["kx"] * v + self.calib["bx"]
        y = self.calib["ky"] * u + self.calib["by"]
        return float(x), float(y)

    def scan_scene(self, wait_sec=0.5):
        import time
        time.sleep(wait_sec)
        result = {
            "blocks": {c: None for c in self.color_cfg.colors},
            "bins":   {c: None for c in self.color_cfg.colors},
        }
        if self._latest_image is None or not HAS_CV2:
            return result
        img = self._latest_image.copy()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for color in self.color_cfg.colors:
            lo, hi = self.color_cfg.get_hsv_range(color)
            if self.color_cfg.needs_wrap(color):
                lo2 = np.array([lo[0], lo[1], lo[2]], dtype=np.uint8)
                hi2 = np.array([180,   hi[1], hi[2]], dtype=np.uint8)
                lo1 = np.array([0,     lo[1], lo[2]], dtype=np.uint8)
                hi1 = np.array([hi[0], hi[1], hi[2]], dtype=np.uint8)
                mask = cv2.inRange(hsv, lo1, hi1) | cv2.inRange(hsv, lo2, hi2)
            else:
                mask = cv2.inRange(hsv, lo, hi)
            cnts, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in sorted(cnts, key=cv2.contourArea, reverse=True):
                area = cv2.contourArea(cnt)
                if area < 200:
                    continue
                M = cv2.moments(cnt)
                if M["m00"] < 1e-6:
                    continue
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                rx, ry = self.pixel_to_robot(cx, cy)
                pos = np.array([rx, ry], dtype=np.float32)
                if area >= self.BIN_MIN_AREA:
                    if result["bins"][color] is None:
                        result["bins"][color] = pos
                else:
                    if result["blocks"][color] is None:
                        result["blocks"][color] = pos
        return result


# ── 自适应感知：优先 VLM，失败时回落 HSV ────────────────────────────────

class AdaptivePerception:
    """
    自适应感知模块。

    正常情况下使用 VLMPerception（支持任意颜色）。
    VLM 调用失败时自动回落到 HSVPerceptionFallback（仅支持预标定颜色）。

    这是真机部署时推荐使用的入口类。
    """

    def __init__(self,
                 api_key: str,
                 vlm_model: Optional[str] = None,
                 base_url: str = None,
                 calib: Optional[Dict] = None,
                 color_config=None,
                 **kwargs):
        self._vlm = VLMPerception(
            api_key=api_key,
            vlm_model=vlm_model,
            base_url=base_url,
            calib=calib,
            **kwargs,
        )
        from config.color_config import get_color_config
        self._hsv = HSVPerceptionFallback(
            color_config=color_config or get_color_config(),
            calib=calib,
        )
        self._vlm_ok = True   # 是否 VLM 可用（失败后自动切换到 HSV）

    @property
    def _latest_image(self):
        return self._vlm._latest_image

    def pixel_to_robot(self, u, v):
        return self._vlm.pixel_to_robot(u, v)

    def update_calibration(self, kx, bx, ky, by):
        self._vlm.update_calibration(kx, bx, ky, by)
        self._hsv.calib = {"kx": kx, "bx": bx, "ky": ky, "by": by}

    def load_calibration_from_yaml(self, path):
        self._vlm.load_calibration_from_yaml(path)
        self._hsv.calib = dict(self._vlm.calib)

    def scan_scene(self, wait_sec=0.5, n_retry=2):
        if self._vlm_ok:
            try:
                result = self._vlm.scan_scene(
                    wait_sec=wait_sec, n_retry=n_retry)
                # 如果 VLM 什么都没检测到，尝试 HSV 补充
                n_detected = sum(
                    1 for v in list(result["blocks"].values()) +
                    list(result["bins"].values()) if v is not None)
                if n_detected == 0:
                    print("[AdaptivePerception] VLM 无检测结果，尝试 HSV 补充...")
                    hsv_result = self._hsv.scan_scene(wait_sec=0.1)
                    # 合并：VLM 优先，HSV 补充空缺
                    for c in hsv_result["blocks"]:
                        if result["blocks"].get(c) is None:
                            result["blocks"][c] = hsv_result["blocks"][c]
                    for c in hsv_result["bins"]:
                        if result["bins"].get(c) is None:
                            result["bins"][c] = hsv_result["bins"][c]
                return result
            except Exception as e:
                print(f"[AdaptivePerception] VLM 完全失败: {e}")
                print("[AdaptivePerception] 切换到 HSV 模式（功能受限）")
                self._vlm_ok = False

        return self._hsv.scan_scene(wait_sec=wait_sec)

    def scan_scene_with_retry(self, target_block, target_bin,
                               n_frames=3, wait_sec=0.3):
        if self._vlm_ok:
            try:
                return self._vlm.scan_scene_with_retry(
                    target_block, target_bin, n_frames, wait_sec)
            except Exception as e:
                print(f"[AdaptivePerception] VLM 重试模式失败: {e}")
        return self.scan_scene(wait_sec=wait_sec)

    def get_debug_image(self):
        if self._vlm_ok:
            return self._vlm.get_debug_image()
        return None

    def print_scene_summary(self):
        scene = self.scan_scene()
        mode = "VLM" if self._vlm_ok else "HSV(fallback)"
        print(f"\n[AdaptivePerception | {mode}] 场景摘要：")
        for c, pos in scene["blocks"].items():
            if pos is not None:
                print(f"  block/{c:10s}: ({pos[0]:.3f}, {pos[1]:.3f})")
        for c, pos in scene["bins"].items():
            if pos is not None:
                print(f"  bin/{c:10s}:   ({pos[0]:.3f}, {pos[1]:.3f})")

    # 兼容旧代码的别名
    def get_block_position(self, color):
        return self._vlm.get_block_position(color) if self._vlm_ok else None

    def get_bin_position(self, color):
        return self._vlm.get_bin_position(color) if self._vlm_ok else None


# ── 向后兼容别名 ──────────────────────────────────────────────────────────
# 旧代码 from perception.camera_perception import CameraPerception 仍然有效
# 但现在 CameraPerception 需要 api_key 参数
CameraPerception = AdaptivePerception
