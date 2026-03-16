#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llm_policy.py
=============
基于 LLM 的探索策略（πEXP），实现 ExploRLLM 的 Algorithm 1。

支持：
  - OpenAI 兼容 API（GPT-4o-mini、DeepSeek、Kimi/Moonshot 等）
  - 高层策略 πH：根据场景状态选择 (primitive, object_index)
  - 低层策略 πL：生成 Python 可操作图代码 → 采样位置

用法：
    from llm.llm_policy import LLMExplorationPolicy
    policy = LLMExplorationPolicy(api_key=..., base_url=..., model=...)
    action = policy.get_exploration_action(obs_dict, crops)
"""

import os
import re
import ast
import time
import textwrap
import traceback
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Please install openai: pip install openai")


# ── 模型配置预设 ──────────────────────────────────────────────────────────────

MODEL_PRESETS = {
    # OpenAI
    "gpt-4o-mini": {
        "base_url": "https://api.openai.com/v1",
        "model":    "gpt-4o-mini",
    },
    # DeepSeek（性价比高、代码生成能力强）
    "deepseek-v3": {
        "base_url": "https://api.deepseek.com/v1",
        "model":    "deepseek-chat",
    },
    # Kimi / Moonshot
    "kimi": {
        "base_url": "https://api.moonshot.cn/v1",
        "model":    "moonshot-v1-8k",
    },
    # Qwen（阿里）
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model":    "qwen-turbo",
    },
}


# ── 状态描述辅助 ──────────────────────────────────────────────────────────────

def _describe_state(obs_dict: Dict[str, Any]) -> str:
    """
    将解析后的观测字典转成自然语言场景描述，供 LLM 提示使用。

    obs_dict 键：
        positions   : 每个跟踪物体 {name: [x,y]}
        gripper     : "open" | "closed"
        pick_color  : str
        place_color : str
        held_object : str | None（夹爪中拿着的块，若有）
    """
    lines = []
    lines.append(f"任务：将 {obs_dict['pick_color']} 块放入 {obs_dict['place_color']} 碗中。")
    lines.append(f"夹爪：{obs_dict['gripper']}")
    if obs_dict.get("held_object"):
        lines.append(f"当前手持：{obs_dict['held_object']}")
    else:
        lines.append("当前手持：无")
    lines.append("物体位置（相对机器人基座的 x,y，单位米）：")
    for name, pos in obs_dict["positions"].items():
        lines.append(f"  {name}: x={pos[0]:.3f}, y={pos[1]:.3f}")
    return "\n".join(lines)


# ── 高层 few-shot 示例 ───────────────────────────────────────────────────────

HIGH_LEVEL_EXAMPLES = [
    {
        "scene": textwrap.dedent("""
            任务：将红块放入蓝碗中。
            夹爪：open
            当前手持：无
            物体位置（相对机器人基座的 x,y，单位米）：
              red_block:   x=0.25, y=-0.05
              green_block: x=0.28, y= 0.08
              blue_block:  x=0.22, y= 0.12
              red_bowl:    x=0.35, y=-0.15
              green_bowl:  x=0.35, y= 0.00
              blue_bowl:   x=0.35, y= 0.15
        """).strip(),
        "response": '{"primitive": 0, "object_name": "red_block", "object_index": 0, "reason": "夹爪张开且未持物，应先抓取红块。"}'
    },
    {
        "scene": textwrap.dedent("""
            任务：将红块放入蓝碗中。
            夹爪：closed
            当前手持：red_block
            物体位置（相对机器人基座的 x,y，单位米）：
              red_block:   x=0.25, y=-0.05
              blue_bowl:   x=0.35, y= 0.15
        """).strip(),
        "response": '{"primitive": 1, "object_name": "blue_bowl", "object_index": 4, "reason": "已持红块，下一步应放入蓝碗。"}'
    },
]

HIGH_LEVEL_SYSTEM = textwrap.dedent("""
    你是一个六自由度机械臂的操控规划器。
    根据当前场景状态，决定下一步的单一原语动作：
      - primitive 0 = 抓取（PICK，抓取一个物体）
      - primitive 1 = 放置（PLACE，放入容器）

    物体索引对应关系：
      0=red_block, 1=green_block, 2=blue_block,
      3=red_bowl,  4=green_bowl,  5=blue_bowl

    规则：
    1. 若夹爪张开且未持物 → 选择 PICK。
    2. 若夹爪闭合且持物 → 选择 PLACE。
    3. 抓取时选择与任务 pick_color 匹配的块。
    4. 放置时选择与任务 place_color 匹配的碗。

    仅回复一个 JSON 对象，不要其他文字。
    格式：{"primitive": <int>, "object_name": <str>, "object_index": <int>, "reason": <str>}
""").strip()


LOW_LEVEL_SYSTEM = textwrap.dedent("""
    你是为机器人拾放系统生成 Python 代码的视觉专家。
    给定一个物体的 28×28 RGB 图像块，请写一个 Python 函数，返回
    28×28 的 numpy 概率图，表示适合抓取或放置的位置。

    规则：
    1. 仅使用 numpy 和 opencv (cv2)，不要其他 import。
    2. 函数签名：generate_probability_map(img) -> np.ndarray，形状 (28,28)，float32。
    3. 数值非负，后续会归一化为和为 1。
    4. 对矩形/立方体物体：优先中心区域。
    5. 对碗/圆形容器：优先内缘区域。
    6. 妥善处理边界情况（如全零图像）。
    7. 只返回 Python 函数代码，不要解释、不要 markdown、不要反引号。
""").strip()

LOW_LEVEL_EXAMPLE_BLOCK = textwrap.dedent("""
    def generate_probability_map(img):
        import cv2
        import numpy as np
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)
        prob = mask.astype(np.float32)
        s = prob.sum()
        if s > 0:
            prob /= s
        return prob
""").strip()

LOW_LEVEL_EXAMPLE_BOWL = textwrap.dedent("""
    def generate_probability_map(img):
        import cv2
        import numpy as np
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        prob = dist.astype(np.float32)
        s = prob.sum()
        if s > 0:
            prob /= s
        return prob
""").strip()


# ── 可操作图代码评估 ─────────────────────────────────────────────────────────

def _evaluate_candidate(code: str, crop: np.ndarray, n_samples: int = 5) -> float:
    """
    在给定 crop 上运行生成的代码并返回得分。
    得分为采样点落在概率非零区域的比例。
    若代码执行失败或输出无效则返回 0.0。
    """
    try:
        namespace = {}
        exec(code, namespace)
        fn = namespace.get("generate_probability_map")
        if fn is None:
            return 0.0

        prob = fn(crop)
        if not isinstance(prob, np.ndarray):
            return 0.0
        if prob.shape != (28, 28):
            return 0.0
        if np.isnan(prob).any() or np.isinf(prob).any():
            return 0.0

        # 得分：类似分布质量的度量，概率分布在合理区域（非退化）时得分更高
        prob = np.clip(prob, 0, None)
        s = prob.sum()
        if s < 1e-9:
            return 0.0
        prob = prob / s

        # 采样 n_samples 个位置
        flat = prob.flatten()
        indices = np.random.choice(len(flat), size=n_samples, p=flat)
        ys, xs = np.unravel_index(indices, (28, 28))

        # 得分：有多少采样点落在图像中心 50% 区域（惩罚退化到角落的情况）
        in_center = np.sum((xs >= 7) & (xs <= 21) & (ys >= 7) & (ys <= 21))
        return float(in_center) / n_samples

    except Exception:
        return 0.0


# ── 主策略类 ─────────────────────────────────────────────────────────────────

class LLMExplorationPolicy:
    """
    实现 ExploRLLM Algorithm 1 中的 πEXP = ε-greedy LLM 探索。

    每个时间步以概率 ε：
        1. 调用 πH_LLM → (primitive, obj_index)
        2. 调用 πL_LLM → 从可操作图采样 x_r
        3. 返回 ã_t = (primitive, obj_index, x_r)
    否则由外部 RL 策略处理。
    """

    def __init__(self,
                 api_key:    str,
                 base_url:   str = None,
                 model:      str = "deepseek-v3",
                 epsilon:    float = 0.2,
                 n_candidates: int = 3,
                 api_timeout: float = 30.0):
        """
        Args:
            api_key     : LLM 服务商的 API key
            base_url    : API 根地址（None 则用 OpenAI 默认）
            model       : 模型名字符串（用 MODEL_PRESETS 的 key 或完整名）
            epsilon     : 探索概率
            n_candidates: 低层生成的代码候选数量
            api_timeout : 请求超时（秒）
        """
        self.epsilon      = epsilon
        self.n_candidates = n_candidates
        self.api_timeout  = api_timeout

        # 解析预设
        if model in MODEL_PRESETS:
            preset   = MODEL_PRESETS[model]
            base_url = base_url or preset["base_url"]
            model    = preset["model"]
        self.model = model

        # 构建 OpenAI 兼容客户端
        kwargs = {"api_key": api_key, "timeout": api_timeout}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)

        # 低层代码候选缓存（按物体名索引）
        self._code_cache: Dict[str, str] = {}

        print(f"[LLM] Initialised with model={self.model}, ε={self.epsilon}, "
              f"n_candidates={self.n_candidates}")

    # ── 高层策略 πH ──────────────────────────────────────────────────────────

    def call_high_level(self, obs_dict: Dict[str, Any]) -> Tuple[int, int]:
        """
        πH_LLM：根据场景状态返回 (primitive, object_index)。

        失败时返回 (0, 0) 作为兜底。
        """
        scene_str = _describe_state(obs_dict)

        # 构建 few-shot 消息
        messages = [{"role": "system", "content": HIGH_LEVEL_SYSTEM}]
        for ex in HIGH_LEVEL_EXAMPLES:
            messages.append({"role": "user",      "content": ex["scene"]})
            messages.append({"role": "assistant", "content": ex["response"]})
        messages.append({"role": "user", "content": scene_str})

        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=200,
            )
            content = resp.choices[0].message.content.strip()

            # 解析 JSON 回复
            import json
            # 去掉可能的 markdown 代码块包裹
            content = re.sub(r"```[a-z]*\n?", "", content).strip()
            data = json.loads(content)
            primitive  = int(data["primitive"])
            obj_index  = int(data["object_index"])
            reason     = data.get("reason", "")
            print(f"[LLM-H] primitive={primitive}, obj={data['object_name']}, "
                  f"reason: {reason}")
            return primitive, obj_index

        except Exception as e:
            print(f"[LLM-H] Failed: {e}. Using fallback (0,0).")
            return 0, 0

    # ── 低层策略 πL ──────────────────────────────────────────────────────────

    def call_low_level(self, obj_name: str, crop: np.ndarray,
                       force_regenerate: bool = False) -> np.ndarray:
        """
        πL_LLM：为给定物体 crop 生成可操作图代码，返回采样的 (x_r, y_r) 残差偏移（米）。

        策略：
          1. 查代码缓存；未命中则用 LLM 生成 n_candidates 个候选。
          2. 在 crop 上评估候选，保留最佳。
          3. 从最佳可操作图采样位置。
          4. 将像素偏移 (0..27) 转为米。
        """
        PIXEL_TO_METER = 0.003  # 典型 Sagittarius 工作空间下约 3mm/像素

        if obj_name not in self._code_cache or force_regenerate:
            self._code_cache[obj_name] = self._generate_best_code(
                obj_name, crop)

        code = self._code_cache[obj_name]

        try:
            namespace = {}
            exec(code, namespace)
            fn = namespace["generate_probability_map"]
            prob = fn(crop)
            prob = np.clip(prob, 0, None)
            s = prob.sum()
            if s < 1e-9:
                return np.zeros(2, dtype=np.float32)
            prob = prob / s

            # 从可操作图按概率采样一个像素位置
            flat_idx = np.random.choice(len(prob.flatten()), p=prob.flatten())
            py, px = np.unravel_index(flat_idx, (28, 28))

            # 将相对 crop 中心的像素偏移转为米
            center = 14.0
            dx_pix = float(px) - center
            dy_pix = float(py) - center
            x_r = dx_pix * PIXEL_TO_METER
            y_r = dy_pix * PIXEL_TO_METER
            return np.array([x_r, y_r], dtype=np.float32)

        except Exception as e:
            print(f"[LLM-L] Execution failed for {obj_name}: {e}")
            return np.zeros(2, dtype=np.float32)

    def _generate_best_code(self, obj_name: str,
                            crop: np.ndarray) -> str:
        """
        通过 LLM 生成 n_candidates 个可操作图函数，在 crop 上评估后返回最佳。
        LLM 失败时退回简单默认代码。
        """
        is_bowl = "bowl" in obj_name
        example = LOW_LEVEL_EXAMPLE_BOWL if is_bowl else LOW_LEVEL_EXAMPLE_BLOCK

        # 构建提示
        prompt = textwrap.dedent(f"""
            请为名为 '{obj_name}' 的物体的 28×28 RGB 图像块，生成一个 Python 函数 `generate_probability_map(img)`。
            该物体是**{'碗/容器' if is_bowl else '矩形块'}**。

            正确函数示例：
            {example}

            请按相同规则写一个**新的、不同的**函数。
            只返回函数代码，不要 markdown、不要解释。
        """).strip()

        candidates = []
        messages   = [{"role": "system",  "content": LOW_LEVEL_SYSTEM},
                      {"role": "user",    "content": prompt}]

        for i in range(self.n_candidates):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,  # 较高温度以得到更多样化的代码
                    max_tokens=400,
                )
                code_raw = resp.choices[0].message.content.strip()
                # 去掉可能的 markdown 代码围栏
                code_raw = re.sub(r"```[a-z]*\n?", "", code_raw).strip()
                code_raw = re.sub(r"```", "", code_raw).strip()
                candidates.append(code_raw)
            except Exception as e:
                print(f"[LLM-L] Candidate {i} generation failed: {e}")

        if not candidates:
            print(f"[LLM-L] All candidates failed, using default code.")
            return example  # 退回硬编码示例

        # 评估所有候选，保留得分最高者
        scores = [_evaluate_candidate(c, crop) for c in candidates]
        best_idx = int(np.argmax(scores))
        print(f"[LLM-L] {len(candidates)} candidates scored: {scores} → "
              f"selecting #{best_idx} (score={scores[best_idx]:.3f})")
        return candidates[best_idx]

    # ── 主探索接口 ────────────────────────────────────────────────────────────

    def should_explore(self) -> bool:
        """按概率采样本步是否使用 LLM 探索。"""
        return np.random.random() < self.epsilon

    def get_exploration_action(self,
                               obs_dict: Dict[str, Any],
                               crops: np.ndarray) -> np.ndarray:
        """
        完整 LLM 探索：先 πH 再 πL → ã_t。

        Args:
            obs_dict  : 解析后的观测字典（见 _describe_state）
            crops     : (N_total, 3, 28, 28) 图像块

        Returns:
            action: [primitive, obj_index, res_x, res_y]
        """
        # 步骤 1：高层决策
        primitive, obj_idx = self.call_high_level(obs_dict)

        # 步骤 2：从可操作图得到低层残差
        from envs.pick_place_env import ALL_OBJECTS
        obj_name = (ALL_OBJECTS[obj_idx]
                    if 0 <= obj_idx < len(ALL_OBJECTS) else "unknown")

        # 取对应 crop， (C,H,W) → (H,W,C) 供 OpenCV 使用
        if 0 <= obj_idx < len(crops):
            crop_chw = crops[obj_idx]  # (3, 28, 28)
            crop_hwc = (crop_chw.transpose(1, 2, 0) * 255).astype(np.uint8)
        else:
            crop_hwc = np.zeros((28, 28, 3), dtype=np.uint8)

        res_xy = self.call_low_level(obj_name, crop_hwc)

        action = np.array([float(primitive), float(obj_idx),
                           float(res_xy[0]),  float(res_xy[1])],
                          dtype=np.float32)
        return action

    def clear_cache(self):
        """清空低层代码缓存（场景变化较大时调用）。"""
        self._code_cache.clear()
        print("[LLM] Code cache cleared.")
