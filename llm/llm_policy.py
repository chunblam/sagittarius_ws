#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llm_policy.py
=============
LLM-based exploration strategy (πEXP) implementing Algorithm 1 from ExploRLLM.

Supports:
  - OpenAI-compatible API (GPT-4o-mini, DeepSeek, Kimi/Moonshot, etc.)
  - High-level policy πH: selects (primitive, object_index) from scene state
  - Low-level policy πL: generates Python affordance map code → sample position

Usage:
    from llm.llm_policy import LLMExplorationPolicy
    policy = LLMExplorationPolicy(api_key=..., base_url=..., model=...)
    action = policy.explore(state)
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


# ── Model config presets ──────────────────────────────────────────────────────

MODEL_PRESETS = {
    # OpenAI
    "gpt-4o-mini": {
        "base_url": "https://api.openai.com/v1",
        "model":    "gpt-4o-mini",
    },
    # DeepSeek  (cost-effective, strong code generation)
    "deepseek-v3": {
        "base_url": "https://api.deepseek.com/v1",
        "model":    "deepseek-chat",
    },
    # Kimi / Moonshot
    "kimi": {
        "base_url": "https://api.moonshot.cn/v1",
        "model":    "moonshot-v1-8k",
    },
    # Qwen (Alibaba)
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model":    "qwen-turbo",
    },
}


# ── State description helpers ─────────────────────────────────────────────────

def _describe_state(obs_dict: Dict[str, Any]) -> str:
    """
    Convert parsed observation dict to a natural-language scene description
    for the LLM prompt.

    obs_dict keys:
        positions   : {name: [x,y]} for each tracked object
        gripper     : "open" | "closed"
        pick_color  : str
        place_color : str
        held_object : str | None   (which block is in gripper, if any)
    """
    lines = []
    lines.append(f"Task: Pick the {obs_dict['pick_color']} block and place it "
                 f"in the {obs_dict['place_color']} bowl.")
    lines.append(f"Gripper: {obs_dict['gripper']}")
    if obs_dict.get("held_object"):
        lines.append(f"Currently holding: {obs_dict['held_object']}")
    else:
        lines.append("Currently holding: nothing")
    lines.append("Object positions (x,y in meters from robot base):")
    for name, pos in obs_dict["positions"].items():
        lines.append(f"  {name}: x={pos[0]:.3f}, y={pos[1]:.3f}")
    return "\n".join(lines)


# ── Few-shot examples ─────────────────────────────────────────────────────────

HIGH_LEVEL_EXAMPLES = [
    {
        "scene": textwrap.dedent("""
            Task: Pick the red block and place it in the blue bowl.
            Gripper: open
            Currently holding: nothing
            Object positions:
              red_block:   x=0.25, y=-0.05
              green_block: x=0.28, y= 0.08
              blue_block:  x=0.22, y= 0.12
              red_bowl:    x=0.35, y=-0.15
              green_bowl:  x=0.35, y= 0.00
              blue_bowl:   x=0.35, y= 0.15
        """).strip(),
        "response": '{"primitive": 0, "object_name": "red_block", "object_index": 0, "reason": "Gripper is open and holding nothing, so we should pick the red block first."}'
    },
    {
        "scene": textwrap.dedent("""
            Task: Pick the red block and place it in the blue bowl.
            Gripper: closed
            Currently holding: red_block
            Object positions:
              red_block:   x=0.25, y=-0.05
              blue_bowl:   x=0.35, y= 0.15
        """).strip(),
        "response": '{"primitive": 1, "object_name": "blue_bowl", "object_index": 4, "reason": "Holding the red block, so next we place it in the blue bowl."}'
    },
]

HIGH_LEVEL_SYSTEM = textwrap.dedent("""
    You are a robot manipulation planner for a 6-DOF robotic arm.
    Given the current scene state, decide the NEXT single primitive action:
      - primitive 0 = PICK (grasp an object)
      - primitive 1 = PLACE (release into a container)

    Object index mapping:
      0=red_block, 1=green_block, 2=blue_block,
      3=red_bowl,  4=green_bowl,  5=blue_bowl

    Rules:
    1. If gripper is open and holding nothing → choose PICK.
    2. If gripper is closed and holding something → choose PLACE.
    3. Pick the block matching the task's pick_color.
    4. Place into the bowl matching the task's place_color.

    Respond ONLY with a JSON object. No other text.
    Format: {"primitive": <int>, "object_name": <str>, "object_index": <int>, "reason": <str>}
""").strip()


LOW_LEVEL_SYSTEM = textwrap.dedent("""
    You are a robot vision expert generating Python code for a robotic pick-and-place system.
    Given a 28x28 RGB image crop of an object, write a Python function that returns
    a 28x28 numpy probability map indicating good grasp or place positions.

    RULES:
    1. Only use numpy and opencv (cv2). No other imports.
    2. Function signature: generate_probability_map(img) -> np.ndarray of shape (28,28), float32
    3. Values should be non-negative; they will be normalised to sum to 1.
    4. For rectangular/cubic objects: prefer the center region.
    5. For bowls/round containers: prefer the interior rim area.
    6. Handle edge cases (all-zero image) gracefully.
    7. Return ONLY the Python function. No explanation, no markdown, no backticks.
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


# ── Affordance map evaluation ─────────────────────────────────────────────────

def _evaluate_candidate(code: str, crop: np.ndarray, n_samples: int = 5) -> float:
    """
    Run the generated code on a crop and return a score.
    Score = fraction of samples that land in the non-zero region of the mask.
    Returns 0.0 if code fails to execute or produces invalid output.
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

        # Score: entropy-like measure of the distribution quality
        # Higher score if probability is spread over plausible region (not degenerate)
        prob = np.clip(prob, 0, None)
        s = prob.sum()
        if s < 1e-9:
            return 0.0
        prob = prob / s

        # Sample n_samples positions
        flat = prob.flatten()
        indices = np.random.choice(len(flat), size=n_samples, p=flat)
        ys, xs = np.unravel_index(indices, (28, 28))

        # Score: how many samples are in the center 50% of the image
        # (penalises degenerate corners)
        in_center = np.sum((xs >= 7) & (xs <= 21) & (ys >= 7) & (ys <= 21))
        return float(in_center) / n_samples

    except Exception:
        return 0.0


# ── Main policy class ─────────────────────────────────────────────────────────

class LLMExplorationPolicy:
    """
    Implements πEXP = ε-greedy LLM exploration from Algorithm 1 of ExploRLLM.

    At each timestep, with probability ε:
        1. Call πH_LLM → (primitive, obj_index)
        2. Call πL_LLM → sample x_r from affordance map
        3. Return ã_t = (primitive, obj_index, x_r)
    Otherwise, defer to the RL policy (handled externally).
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
            api_key     : API key for your LLM provider
            base_url    : API base URL (None = OpenAI default)
            model       : model name string (use MODEL_PRESETS key or full name)
            epsilon     : exploration probability
            n_candidates: number of low-level code candidates to generate
            api_timeout : request timeout in seconds
        """
        self.epsilon      = epsilon
        self.n_candidates = n_candidates
        self.api_timeout  = api_timeout

        # Resolve preset
        if model in MODEL_PRESETS:
            preset   = MODEL_PRESETS[model]
            base_url = base_url or preset["base_url"]
            model    = preset["model"]
        self.model = model

        # Build OpenAI-compatible client
        kwargs = {"api_key": api_key, "timeout": api_timeout}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)

        # Cache for low-level code candidates (keyed by object name)
        self._code_cache: Dict[str, str] = {}

        print(f"[LLM] Initialised with model={self.model}, ε={self.epsilon}, "
              f"n_candidates={self.n_candidates}")

    # ── High-level policy πH ──────────────────────────────────────────────────

    def call_high_level(self, obs_dict: Dict[str, Any]) -> Tuple[int, int]:
        """
        πH_LLM: Given scene state, return (primitive, object_index).

        Returns (0, 0) as fallback on failure.
        """
        scene_str = _describe_state(obs_dict)

        # Build few-shot messages
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

            # Parse JSON response
            import json
            # Handle potential markdown wrapping
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

    # ── Low-level policy πL ───────────────────────────────────────────────────

    def call_low_level(self, obj_name: str, crop: np.ndarray,
                       force_regenerate: bool = False) -> np.ndarray:
        """
        πL_LLM: Generate affordance map code for the given object crop.
        Returns a sampled (x_r, y_r) residual offset in meters.

        Strategy:
          1. Check code cache; if miss, generate n_candidates via LLM.
          2. Evaluate candidates on the crop, keep best.
          3. Sample position from the best affordance map.
          4. Convert pixel offset (0..27) → meter offset.
        """
        PIXEL_TO_METER = 0.003  # ~3mm per pixel at typical Sagittarius workspace

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

            # Sample pixel position from affordance map
            flat_idx = np.random.choice(len(prob.flatten()), p=prob.flatten())
            py, px = np.unravel_index(flat_idx, (28, 28))

            # Convert pixel offset from crop center to meters
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
        Generate n_candidates affordance map functions via LLM,
        evaluate on the crop, return the best one.
        Falls back to a simple default if LLM fails.
        """
        is_bowl = "bowl" in obj_name
        example = LOW_LEVEL_EXAMPLE_BOWL if is_bowl else LOW_LEVEL_EXAMPLE_BLOCK

        # Build prompt
        prompt = textwrap.dedent(f"""
            Generate a Python function `generate_probability_map(img)` for
            a 28x28 RGB image crop of a **{'bowl/container' if is_bowl else 'rectangular block'}**
            called '{obj_name}'.

            Example of a correct function:
            {example}

            Now write a NEW, DIFFERENT function following the same rules.
            Return only the function code, no markdown, no explanation.
        """).strip()

        candidates = []
        messages   = [{"role": "system",  "content": LOW_LEVEL_SYSTEM},
                      {"role": "user",    "content": prompt}]

        for i in range(self.n_candidates):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,  # higher temperature → more diverse code
                    max_tokens=400,
                )
                code_raw = resp.choices[0].message.content.strip()
                # Strip markdown code fences if present
                code_raw = re.sub(r"```[a-z]*\n?", "", code_raw).strip()
                code_raw = re.sub(r"```", "", code_raw).strip()
                candidates.append(code_raw)
            except Exception as e:
                print(f"[LLM-L] Candidate {i} generation failed: {e}")

        if not candidates:
            print(f"[LLM-L] All candidates failed, using default code.")
            return example  # fallback to hardcoded example

        # Evaluate all candidates, keep the best
        scores = [_evaluate_candidate(c, crop) for c in candidates]
        best_idx = int(np.argmax(scores))
        print(f"[LLM-L] {len(candidates)} candidates scored: {scores} → "
              f"selecting #{best_idx} (score={scores[best_idx]:.3f})")
        return candidates[best_idx]

    # ── Main exploration interface ────────────────────────────────────────────

    def should_explore(self) -> bool:
        """Sample whether this step should use LLM exploration."""
        return np.random.random() < self.epsilon

    def get_exploration_action(self,
                               obs_dict: Dict[str, Any],
                               crops: np.ndarray) -> np.ndarray:
        """
        Full LLM exploration: πH then πL → ã_t.

        Args:
            obs_dict  : parsed observation dict (see _describe_state)
            crops     : (N_total, 3, 28, 28) image crops

        Returns:
            action: [primitive, obj_index, res_x, res_y]
        """
        # Step 1: High-level decision
        primitive, obj_idx = self.call_high_level(obs_dict)

        # Step 2: Low-level residual from affordance map
        from envs.pick_place_env import ALL_OBJECTS
        obj_name = (ALL_OBJECTS[obj_idx]
                    if 0 <= obj_idx < len(ALL_OBJECTS) else "unknown")

        # Get the corresponding crop (C,H,W) → (H,W,C) for OpenCV
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
        """Clear low-level code cache (call when scene changes significantly)."""
        self._code_cache.clear()
        print("[LLM] Code cache cleared.")
