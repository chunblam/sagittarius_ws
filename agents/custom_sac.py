#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
custom_sac.py
=============
自定义 SAC 策略：物体中心观测编码器
    CNN（逐物体图像块）→ 拼接位置与语言 → 自注意力 → MLP

并实现 ExploRLLMSAC：在 rollout 收集阶段注入 LLM 探索的 SAC 子类（对应 Algorithm 1）。

结构（来自 ExploRLLM 论文）：
    对每个物体 i：
        crop_i (3,28,28) ──► 2 层 CNN ──► feature_i (d_feat)
        feature_i + pos_i + gripper + lang ──► φ'_i (d')
    [φ'_0, ..., φ'_{N-1}] ──► Self-Attention ──► global_feat
    global_feat ──► 2 层 MLP ──► Q 或 π
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Type, Union, Any

from stable_baselines3 import SAC
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import GymEnv, Schedule
import gymnasium as gym


# ── 维度常量（须与 pick_place_env.py 一致）────────────────────────────────────

N_TOTAL    = 6       # 块+碗
CROP_SIZE  = 28
IMG_DIM    = N_TOTAL * 3 * CROP_SIZE * CROP_SIZE   # 14112
POS_DIM    = N_TOTAL * 2                            # 12
GRIP_DIM   = 1
LANG_DIM   = 6
OBS_DIM    = IMG_DIM + POS_DIM + GRIP_DIM + LANG_DIM   # 14131


# ── 逐物体 CNN 特征提取器 ────────────────────────────────────────────────────

class ObjectCropEncoder(nn.Module):
    """
    共享 2 层 CNN，将单个 28×28 裁剪编码为特征向量。
    对每个物体的图像块独立应用。
    """

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.cnn = nn.Sequential(
            # 输入: (3, 28, 28)
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=0),  # → (16,12,12)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0), # → (32,5,5)
            nn.ReLU(),
            nn.Flatten(),                                            # → 800
            nn.Linear(800, out_dim),
            nn.ReLU(),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 3, 28, 28) → (batch, out_dim)"""
        return self.cnn(x)


# ── 物体特征上的自注意力 ──────────────────────────────────────────────────────

class ObjectSelfAttention(nn.Module):
    """
    对 N 个物体特征向量做多头自注意力。
    通过对注意力输出做 mean-pooling 聚合成一个全局特征。
    """

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,   # (batch, seq, dim)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, N_obj, d_model)
        returns: (batch, d_model) – mean-pooled 后的注意力特征
        """
        attn_out, _ = self.attn(x, x, x)
        attn_out = self.norm(attn_out + x)      # 残差
        return attn_out.mean(dim=1)             # (batch, d_model)


# ── 完整物体中心特征提取器 ────────────────────────────────────────────────────

class ObjectCentricExtractor(BaseFeaturesExtractor):
    """
    与 stable-baselines3 兼容的特征提取器。

    输入：形状 (obs_dim,) 的扁平观测向量
    输出：(features_dim,) 的全局特征向量

    内部流程：un-flatten → 逐 crop CNN → 拼接 pose+lang+gripper → self-attention → 输出
    """

    def __init__(self,
                 observation_space: gym.Space,
                 cnn_out_dim:    int = 64,
                 attn_dim:       int = 80,  # cnn_out + pos(2) + grip(1) + lang(2)=80 ≈ next multiple-of-heads
                 n_attn_heads:   int = 4,
                 features_dim:   int = 128):
        super().__init__(observation_space, features_dim=features_dim)

        self.n_objects   = N_TOTAL
        self.cnn_out_dim = cnn_out_dim

        # 拼接后每个物体的维度：cnn_out(64)+pos(2)+gripper(1)+lang(6) → 投影到 attn_dim
        per_obj_raw = cnn_out_dim + 2 + 1 + (LANG_DIM // N_TOTAL + LANG_DIM % N_TOTAL)
        self.project = nn.Linear(per_obj_raw, attn_dim)

        self.crop_encoder = ObjectCropEncoder(out_dim=cnn_out_dim)
        self.attention    = ObjectSelfAttention(d_model=attn_dim, n_heads=n_attn_heads)

        self.mlp = nn.Sequential(
            nn.Linear(attn_dim, features_dim),
            nn.ReLU(),
        )

        self._per_obj_raw = per_obj_raw

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (batch, OBS_DIM)
        returns: (batch, features_dim)
        """
        batch = obs.shape[0]

        # ── 将观测拆开（反扁平化）─────────────────────────────────────────────
        img_flat  = obs[:, :IMG_DIM]                         # (B, 14112)
        pos_flat  = obs[:, IMG_DIM : IMG_DIM+POS_DIM]       # (B, 12)
        grip      = obs[:, IMG_DIM+POS_DIM : IMG_DIM+POS_DIM+GRIP_DIM]   # (B, 1)
        lang      = obs[:, IMG_DIM+POS_DIM+GRIP_DIM :]     # (B, 6)

        # 将图像重排为 (B, N_obj, 3, 28, 28)
        crops = img_flat.view(batch, self.n_objects, 3, CROP_SIZE, CROP_SIZE)

        # 位置 (B, N_obj, 2)
        positions = pos_flat.view(batch, self.n_objects, 2)

        # ── 对每个物体独立编码 ────────────────────────────────────────────────
        # crops: (B, N_obj, 3, 28, 28) → (B*N_obj, 3, 28, 28)
        crops_flat = crops.view(batch * self.n_objects, 3, CROP_SIZE, CROP_SIZE)
        cnn_feats  = self.crop_encoder(crops_flat)           # (B*N_obj, cnn_out)
        cnn_feats  = cnn_feats.view(batch, self.n_objects, self.cnn_out_dim)

        # ── 构建每个物体的特征向量 ────────────────────────────────────────────
        # 将 gripper、lang 广播到每个物体
        grip_exp  = grip.unsqueeze(1).expand(-1, self.n_objects, -1)   # (B, N, 1)
        lang_exp  = lang.unsqueeze(1).expand(-1, self.n_objects, -1)   # (B, N, 6)

        per_obj = torch.cat([cnn_feats, positions, grip_exp, lang_exp], dim=-1)
        # per_obj: (B, N_obj, cnn_out+2+1+6)

        # 若不足 _per_obj_raw 则 padding
        if per_obj.shape[-1] < self._per_obj_raw:
            pad = self._per_obj_raw - per_obj.shape[-1]
            per_obj = F.pad(per_obj, (0, pad))

        per_obj = self.project(per_obj)  # (B, N_obj, attn_dim)

        # ── 自注意力聚合 ──────────────────────────────────────────────────────
        global_feat = self.attention(per_obj)   # (B, attn_dim)

        return self.mlp(global_feat)            # (B, features_dim)


# ── 注入 LLM 探索的 ExploRLLM SAC ────────────────────────────────────────────

class ExploRLLMSAC(SAC):
    """
    SAC 子类：在收集 rollout 时注入基于 LLM 的探索。

    通过重写 _sample_action 实现 Algorithm 1：
        j ~ U[0,1)
        if j <= ε:  使用 LLM 策略
        else:       使用 SAC 策略
    """

    def __init__(self,
                 policy,
                 env,
                 llm_policy=None,      # LLMExplorationPolicy 实例
                 warmup_steps: int = 20_000,
                 obs_parser_fn=None,   # fn(obs_array) → obs_dict，供 LLM 使用
                 **kwargs):
        super().__init__(policy, env, **kwargs)
        self.llm_policy    = llm_policy
        self.warmup_steps  = warmup_steps
        self.obs_parser_fn = obs_parser_fn
        self._llm_steps    = 0   # LLM 引导的步数
        self._rl_steps     = 0   # RL 引导的步数

    def _sample_action(
        self,
        learning_starts: int,
        action_noise=None,
        n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        重写 stable-baselines3 的 _sample_action，在其中注入 LLM 探索。

        warmup 之后：
            - 以概率 ε：用 LLM 生成动作
            - 否则：按原样使用 SAC 策略
        """
        # warmup 期间或未提供 LLM 策略时：保持标准 SAC 行为
        if (self.num_timesteps < self.warmup_steps
                or self.llm_policy is None):
            return super()._sample_action(learning_starts, action_noise, n_envs)

        # warmup 之后：ε-greedy 探索
        if self.llm_policy.should_explore():
            # 使用 LLM 探索
            obs = self._last_obs  # (n_envs, obs_dim)
            actions = []
            for i in range(n_envs):
                obs_i = obs[i]
                if self.obs_parser_fn is not None:
                    obs_dict = self.obs_parser_fn(obs_i)
                else:
                    obs_dict = self._default_obs_parser(obs_i)

                # 从观测中取出图像块
                crops = obs_i[:IMG_DIM].reshape(N_TOTAL, 3, CROP_SIZE, CROP_SIZE)

                try:
                    action = self.llm_policy.get_exploration_action(
                        obs_dict, crops)
                except Exception as e:
                    print(f"[ExploRLLM] LLM action failed: {e}. Falling back to SAC.")
                    action, _ = super()._sample_action(
                        learning_starts, action_noise, 1)
                    action = action[0]

                actions.append(action)

            self._llm_steps += n_envs
            buffer_actions = np.array(actions)
            actions_array  = buffer_actions   # 已是正确尺度
            return actions_array, buffer_actions

        else:
            # 使用 SAC 策略
            self._rl_steps += 1
            return super()._sample_action(learning_starts, action_noise, n_envs)

    def _default_obs_parser(self, obs: np.ndarray) -> dict:
        """
        默认观测解析器：从扁平观测中解析出结构化字典。
        与 SagittariusPickPlaceEnv._build_observation 的布局一致。
        """
        from envs.pick_place_env import ALL_OBJECTS, BLOCK_NAMES, BOWL_NAMES

        pos_start = IMG_DIM
        pos_flat  = obs[pos_start : pos_start + POS_DIM]
        positions = pos_flat.reshape(N_TOTAL, 2)

        grip_val = float(obs[pos_start + POS_DIM])
        gripper  = "open" if grip_val < 0.5 else "closed"

        lang = obs[pos_start + POS_DIM + GRIP_DIM :]
        # 解码 one-hot 语言目标
        pick_idx  = int(np.argmax(lang[:3]))
        place_idx = int(np.argmax(lang[3:]))
        color_map = {0: "red", 1: "green", 2: "blue"}
        pick_color  = color_map.get(pick_idx,  "red")
        place_color = color_map.get(place_idx, "blue")

        pos_dict = {}
        for i, name in enumerate(ALL_OBJECTS):
            pos_dict[name] = positions[i].tolist()

        held = None
        if gripper == "closed":
            held = f"{pick_color}_block"

        return {
            "positions":   pos_dict,
            "gripper":     gripper,
            "pick_color":  pick_color,
            "place_color": place_color,
            "held_object": held,
        }

    def get_exploration_stats(self) -> dict:
        """返回 LLM 与 RL 步数相关的统计。"""
        total = self._llm_steps + self._rl_steps
        return {
            "llm_steps":  self._llm_steps,
            "rl_steps":   self._rl_steps,
            "total_steps": total,
            "llm_fraction": self._llm_steps / max(total, 1),
        }


# ── 策略注册辅助函数 ──────────────────────────────────────────────────────────

def make_sac_kwargs(features_dim: int = 128) -> dict:
    """
    返回供 stable-baselines3 SAC 使用的 policy_kwargs，
    以使用 ObjectCentricExtractor。
    """
    return {
        "features_extractor_class": ObjectCentricExtractor,
        "features_extractor_kwargs": {
            "cnn_out_dim":  64,
            "attn_dim":     80,
            "n_attn_heads": 4,
            "features_dim": features_dim,
        },
        "net_arch": [128, 128],
        "activation_fn": nn.ReLU,
    }
