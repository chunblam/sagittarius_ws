#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llm_policy.py
================
升级版LLM探索策略，支持任意多种颜色和随机桶位置。

关键变化：
  - prompt里描述的物体从3种颜色扩展到N种颜色
  - prompt里现在包含桶的位置（因为桶是随机的，LLM需要知道桶在哪）
  - object_index映射更新：前N个是方块，后N个是桶

凭证、网关与模型名（勿写入仓库，用环境变量或 .env）：
  - api_key / base_url：调用方优先；为空则读 LLM_API_KEY、LLM_BASE_URL
  - model：调用方可传 preset 名；为空则读 LLM_MODEL（默认 deepseek-v3）
  - base_url 最终仍可由 MODEL_PRESETS 补全

本地示例：  export LLM_API_KEY='...'  export LLM_MODEL='deepseek-v3'
"""

import os
import re
import json
import time
import textwrap
import numpy as np
from typing import Dict, Tuple, Optional, Any

"""
OpenAI SDK 版本兼容说明
----------------------
本项目日常开发默认使用 openai>=1.x（推荐），调用方式为：
  from openai import OpenAI
  client = OpenAI(...)
  client.chat.completions.create(...)

但实验室机器可能只能安装到 openai==0.9.1（旧版 SDK），其调用方式为：
  import openai
  openai.api_key = ...
  openai.api_base = ...
  openai.ChatCompletion.create(...)

为避免在两台机器来回改代码，这里提供“双写法兼容层”：
- 默认优先走 1.x 新版写法（保持现有调用路径：client.chat.completions.create）
- 如检测不到 OpenAI 类，或设置环境变量 USE_OPENAI_LEGACY=1，则自动走 0.9.x 旧版写法
"""

# ── OpenAI SDK 兼容层：优先 1.x，必要时回落到 0.9.x ───────────────────────
_FORCE_LEGACY = os.environ.get("USE_OPENAI_LEGACY", "").strip() in {"1", "true", "True", "YES", "yes"}
try:
    if _FORCE_LEGACY:
        raise ImportError("Forced legacy OpenAI SDK")
    # openai>=1.x
    from openai import OpenAI as _OpenAIClient  # type: ignore
    _OPENAI_IS_V1 = True
except Exception:
    # openai==0.9.x
    import openai as _openai_legacy  # type: ignore
    _OpenAIClient = None
    _OPENAI_IS_V1 = False


class _CompatMsg:
    def __init__(self, content: str):
        self.content = content


class _CompatChoice:
    def __init__(self, content: str):
        self.message = _CompatMsg(content)


class _CompatResp:
    def __init__(self, content: str):
        self.choices = [_CompatChoice(content)]


class _LegacyChatCompletions:
    """把 openai==0.9.x 的 ChatCompletion.create 适配到 1.x 风格接口。"""

    @staticmethod
    def create(model: str, messages, temperature: float, max_tokens: int):
        resp = _openai_legacy.ChatCompletion.create(  # type: ignore[attr-defined]
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # 旧版返回通常是 dict-like：choices[0]['message']['content']
        try:
            content = resp["choices"][0]["message"]["content"]
        except Exception:
            content = str(resp)
        return _CompatResp(str(content).strip())


class _LegacyChat:
    completions = _LegacyChatCompletions()


class _LegacyClient:
    class chat:  # noqa: N801（保持与新版接口一致的属性名）
        completions = _LegacyChatCompletions()


def _create_openai_client(api_key: str, base_url: Optional[str]):
    """
    返回一个具备 `client.chat.completions.create(...)` 方法的对象。
    - openai>=1.x: 返回 OpenAI(...) 实例
    - openai==0.9.x: 配置模块级 api_key/api_base 并返回 _LegacyClient()
    """
    if _OPENAI_IS_V1:
        kwargs = {"api_key": api_key, "timeout": 30.0}
        if base_url:
            kwargs["base_url"] = base_url
        return _OpenAIClient(**kwargs)  # type: ignore[misc]

    # legacy 0.9.x
    _openai_legacy.api_key = api_key  # type: ignore[attr-defined]
    if base_url:
        _openai_legacy.api_base = base_url  # type: ignore[attr-defined]
    return _LegacyClient()

from config.color_config import ColorConfig, get_color_config

# 与 train.py 中 argparse 默认值保持一致，便于只配环境变量、不把密钥写进命令行
LLM_API_KEY_ENV = "LLM_API_KEY"
LLM_BASE_URL_ENV = "LLM_BASE_URL"


def _resolve_llm_api_key(explicit: Optional[str]) -> str:
    """显式 api_key 非空则用之，否则读环境变量 LLM_API_KEY。"""
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    return os.environ.get(LLM_API_KEY_ENV, "").strip()


def _resolve_llm_model_name(explicit: Optional[str]) -> str:
    """显式 model 非空则用之，否则读 env_config.llm_model()。"""
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    try:
        from env_config import llm_model
        return llm_model()
    except ImportError:
        return os.environ.get("LLM_MODEL", "").strip() or "deepseek-v3"


def _resolve_llm_base_url(
    explicit: Optional[str],
    preset_base_url: Optional[str],
) -> Optional[str]:
    """
    显式 base_url 非空则用之；
    否则读 LLM_BASE_URL；
    再否则用 preset（如 deepseek 官方地址）；gpt-4o-mini 等可为 None（走 OpenAI 默认）。
    """
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    env_u = os.environ.get(LLM_BASE_URL_ENV, "").strip()
    if env_u:
        return env_u
    return preset_base_url


MODEL_PRESETS = {
    "deepseek-v3": {"base_url": "https://api.deepseek.com/v1",
                    "model":    "deepseek-chat"},
    "kimi":        {"base_url": "https://api.moonshot.cn/v1",
                    "model":    "moonshot-v1-8k"},
    "qwen":        {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "model":    "qwen-turbo"},
    "gpt-4o-mini": {"base_url": None,
                    "model":    "gpt-4o-mini"},
}

HIGH_LEVEL_SYSTEM = textwrap.dedent("""
    You are a robot arm planner for a pick-and-place task.

    The robot has a parallel gripper. It can execute two primitives:
      primitive 0 = PICK  (grasp a block from the table)
      primitive 1 = PLACE (release into a bin)

    Rules:
    1. If gripper is open and holding nothing → choose PICK.
       Pick the block that matches the task's pick_color.
    2. If gripper is closed → choose PLACE.
       Place into the bin that matches the task's place_color.
    3. Always choose the object index that matches the target color.

    Respond ONLY with a JSON object:
    {"primitive": <0 or 1>, "object_name": <str>, "object_index": <int>, "reason": <str>}
    No other text.
""").strip()


def _build_scene_description(obs_dict: Dict[str, Any],
                              color_cfg: ColorConfig) -> str:
    """构建场景描述字符串供LLM理解。"""
    lines = []
    lines.append(f"Task: Pick the {obs_dict['pick_color']} block, "
                 f"place it in the {obs_dict['place_color']} bin.")
    lines.append(f"Gripper: {obs_dict['gripper']}")
    held = obs_dict.get("held_object")
    lines.append(f"Holding: {held if held else 'nothing'}")
    lines.append("")
    lines.append("Object positions (x,y in meters):")
    lines.append("  Blocks:")

    N = color_cfg.n_colors
    for i, color in enumerate(color_cfg.colors):
        key = f"{color}_block"
        pos = obs_dict["positions"].get(key, [0, 0])
        lines.append(f"    [{i}] {color}_block: x={pos[0]:.3f}, y={pos[1]:.3f}")

    lines.append("  Bins:")
    for i, color in enumerate(color_cfg.colors):
        key = f"{color}_bin"
        pos = obs_dict["positions"].get(key, [0, 0])
        lines.append(f"    [{N+i}] {color}_bin: x={pos[0]:.3f}, y={pos[1]:.3f}")

    return "\n".join(lines)


def _default_affordance_code() -> str:
    """默认的affordance map代码（LLM不可用时的后备）。"""
    return textwrap.dedent("""
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


class LLMExplorationPolicy:
    """
    升级版LLM探索策略。
    支持任意多种颜色，桶的位置是动态的。

    api_key / base_url 可传 None 或空字符串，此时从环境变量
    LLM_API_KEY、LLM_BASE_URL 读取。
    model 可传 None 或空字符串，此时从 LLM_MODEL 读取（见 env_config / .env）。
    """

    def __init__(self,
                 api_key:      Optional[str] = None,
                 base_url:     Optional[str] = None,
                 model:        Optional[str] = None,
                 epsilon:      float = 0.2,
                 n_candidates: int = 3,
                 color_config: ColorConfig = None):

        self.epsilon      = epsilon
        self.n_candidates = n_candidates
        self.color_cfg    = color_config or get_color_config()

        model = _resolve_llm_model_name(model)

        # 解析 model preset（得到官方默认 base_url，可被环境变量覆盖）
        preset_base: Optional[str] = None
        if model in MODEL_PRESETS:
            preset        = MODEL_PRESETS[model]
            preset_base   = preset["base_url"]
            model         = preset["model"]

        api_key_resolved = _resolve_llm_api_key(api_key)
        base_url_resolved = _resolve_llm_base_url(base_url, preset_base)
        self.model = model

        # client 统一接口：self._client.chat.completions.create(...)
        # - 日常：openai>=1.x（默认）
        # - 实验室若只能装 openai==0.9.1：设置 USE_OPENAI_LEGACY=1 或自动回落
        self._client = _create_openai_client(
            api_key=api_key_resolved, base_url=base_url_resolved)

        self._code_cache: Dict[str, str] = {}
        print(f"[LLM] model={self.model}, ε={self.epsilon}, "
              f"colors={self.color_cfg.colors}")

    def should_explore(self) -> bool:
        return np.random.random() < self.epsilon

    def call_high_level(self, obs_dict: Dict) -> Tuple[int, int]:
        """
        πH: 返回 (primitive, object_index)
        object_index: 0..N-1=方块, N..2N-1=桶
        """
        scene_str = _build_scene_description(obs_dict, self.color_cfg)
        N = self.color_cfg.n_colors

        messages = [
            {"role": "system",  "content": HIGH_LEVEL_SYSTEM},
            {"role": "user",    "content": scene_str},
        ]

        try:
            resp = self._client.chat.completions.create(
                model=self.model, messages=messages,
                temperature=0.2, max_tokens=250)
            content = resp.choices[0].message.content.strip()
            content = re.sub(r"```[a-z]*\n?|```", "", content).strip()
            data    = json.loads(content)
            prim    = int(data["primitive"])
            obj_idx = int(data["object_index"])
            print(f"[LLM-H] prim={prim}, obj={data['object_name']}: "
                  f"{data.get('reason','')}")
            return prim, obj_idx
        except Exception as e:
            print(f"[LLM-H] 失败: {e}，使用fallback(0,0)")
            return 0, 0

    def call_low_level(self, obj_name: str,
                       crop: np.ndarray) -> np.ndarray:
        """πL: 从affordance map采样残差偏移(x_r, y_r)（米）。"""
        PIXEL_TO_METER = 0.003

        if obj_name not in self._code_cache:
            self._code_cache[obj_name] = self._generate_best_code(
                obj_name, crop)

        code = self._code_cache[obj_name]
        try:
            ns = {}
            exec(code, ns)
            fn   = ns["generate_probability_map"]
            prob = fn(crop)
            prob = np.clip(prob, 0, None)
            s    = prob.sum()
            if s < 1e-9:
                return np.zeros(2, dtype=np.float32)
            prob = prob / s
            idx  = np.random.choice(len(prob.flatten()), p=prob.flatten())
            py, px = np.unravel_index(idx, (28, 28))
            return np.array(
                [(px - 14) * PIXEL_TO_METER,
                 (py - 14) * PIXEL_TO_METER], dtype=np.float32)
        except Exception as e:
            print(f"[LLM-L] 执行失败 {obj_name}: {e}")
            return np.zeros(2, dtype=np.float32)

    def _generate_best_code(self, obj_name: str,
                             crop: np.ndarray) -> str:
        is_bin = "bin" in obj_name
        prompt = textwrap.dedent(f"""
            Write a Python function `generate_probability_map(img)` for a
            28x28 RGB image of a **{'cylindrical bin/container' if is_bin else 'small rectangular block'}**
            named '{obj_name}'.

            Rules:
            - Only numpy and cv2. Return np.ndarray shape (28,28) float32.
            - Values non-negative, will be normalised to sum 1.
            - {'For bins: prefer the center interior area.' if is_bin else 'For blocks: prefer the top center region.'}
            - Return ONLY the function, no markdown.
        """).strip()

        candidates = []
        for _ in range(self.n_candidates):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7, max_tokens=400)
                code = resp.choices[0].message.content.strip()
                code = re.sub(r"```[a-z]*\n?|```", "", code).strip()
                candidates.append(code)
            except Exception:
                pass

        if not candidates:
            return _default_affordance_code()

        scores = [self._score_code(c, crop) for c in candidates]
        best   = candidates[int(np.argmax(scores))]
        print(f"[LLM-L] {len(candidates)}个候选, scores={scores}")
        return best

    def _score_code(self, code: str, crop: np.ndarray) -> float:
        try:
            ns = {}
            exec(code, ns)
            fn   = ns.get("generate_probability_map")
            if fn is None: return 0.0
            prob = fn(crop)
            if not isinstance(prob, np.ndarray): return 0.0
            if prob.shape != (28, 28): return 0.0
            prob = np.clip(prob, 0, None)
            if prob.sum() < 1e-9: return 0.0
            prob = prob / prob.sum()
            idx  = np.random.choice(784, size=10, p=prob.flatten())
            ys, xs = np.unravel_index(idx, (28, 28))
            in_center = np.sum((xs >= 7) & (xs <= 21) & (ys >= 7) & (ys <= 21))
            return float(in_center) / 10
        except Exception:
            return 0.0

    def get_exploration_action(self, obs_dict: Dict,
                               crops: np.ndarray) -> np.ndarray:
        """完整LLM探索：πH → πL → ã_t"""
        primitive, obj_idx = self.call_high_level(obs_dict)

        N = self.color_cfg.n_colors
        if obj_idx < N:
            obj_name = f"{self.color_cfg.idx_to_color(obj_idx)}_block"
        else:
            obj_name = f"{self.color_cfg.idx_to_color(obj_idx - N)}_bin"

        # 取对应的图像crop（crops是方块的crops，桶没有单独crop）
        crop_idx = min(obj_idx % N, len(crops) - 1)
        if 0 <= crop_idx < len(crops):
            chw  = crops[crop_idx]
            hwc  = (chw.transpose(1, 2, 0) * 255).astype(np.uint8)
        else:
            hwc = np.zeros((28, 28, 3), dtype=np.uint8)

        res_xy = self.call_low_level(obj_name, hwc)

        return np.array([float(primitive), float(obj_idx),
                         float(res_xy[0]), float(res_xy[1])],
                        dtype=np.float32)

    def clear_cache(self):
        self._code_cache.clear()
