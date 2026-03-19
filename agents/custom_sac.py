#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
custom_sac.py
================
升级版SAC，相比原版的核心变化：

  变化1：颜色用Embedding层编码（而不是one-hot）
         支持任意数量颜色，颜色数量变化时网络结构不需要改
  变化2：Observation里现在包含桶的位置，encoder要处理更多位置数据
  变化3：ObsDecoder正确拆解新的observation布局
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Type
import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


CROP_SIZE = 28


class CropEncoder(nn.Module):
    """对单个28×28 RGB图像crop编码。"""
    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2),   # → (16,12,12)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2),  # → (32,5,5)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(800, out_dim),
            nn.ReLU(),
        )
        self.out_dim = out_dim

    def forward(self, x):
        return self.net(x)


class ObjectCentricExtractor(BaseFeaturesExtractor):
    """
    升级版特征提取器，支持：
      - 任意数量颜色（N_colors动态传入）
      - 颜色任务用Embedding编码
      - 同时处理方块位置和桶位置
    """

    def __init__(self,
                 observation_space: gym.Space,
                 n_colors:      int = 3,
                 cnn_out_dim:   int = 64,
                 embed_dim:     int = 16,
                 attn_dim:      int = 96,
                 n_attn_heads:  int = 4,
                 features_dim:  int = 128):
        super().__init__(observation_space, features_dim=features_dim)

        self.n_colors    = n_colors
        self.cnn_out_dim = cnn_out_dim

        # ── 计算observation各部分的偏移量 ──────────────────────────────
        self.img_dim  = n_colors * 3 * CROP_SIZE * CROP_SIZE
        self.pos_dim  = n_colors * 2   # 方块位置
        self.bin_dim  = n_colors * 2   # 桶位置
        self.grip_dim = 1
        self.task_dim = 2

        # ── 模块定义 ──────────────────────────────────────────────────
        # 图像编码（每个颜色一个crop）
        self.crop_encoder = CropEncoder(out_dim=cnn_out_dim)

        # 颜色任务embedding（把颜色index映射到向量）
        self.color_embed = nn.Embedding(n_colors, embed_dim)

        # 每个对象的特征维度：
        #   cnn_out(64) + block_pos(2) + bin_pos(2) + gripper(1) = 69
        #   → project to attn_dim
        per_obj_raw = cnn_out_dim + 2 + 2 + 1
        self.proj = nn.Linear(per_obj_raw, attn_dim)

        # 自注意力聚合
        self.attn = nn.MultiheadAttention(
            attn_dim, n_attn_heads, batch_first=True)
        self.norm = nn.LayerNorm(attn_dim)

        # 任务编码融合（颜色embedding）
        task_encoded_dim = embed_dim * 2   # pick + place
        self.task_proj = nn.Linear(task_encoded_dim, attn_dim)

        # 最终MLP
        self.mlp = nn.Sequential(
            nn.Linear(attn_dim * 2, features_dim),   # attn + task
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        N = self.n_colors

        # ── 拆解observation ────────────────────────────────────────
        ptr = 0

        # 图像crops
        crops_flat = obs[:, ptr : ptr + self.img_dim]
        ptr += self.img_dim
        crops = crops_flat.view(B, N, 3, CROP_SIZE, CROP_SIZE)

        # 方块位置
        block_pos = obs[:, ptr : ptr + self.pos_dim].view(B, N, 2)
        ptr += self.pos_dim

        # 桶位置（新增）
        bin_pos = obs[:, ptr : ptr + self.bin_dim].view(B, N, 2)
        ptr += self.bin_dim

        # 夹爪状态
        gripper = obs[:, ptr : ptr + self.grip_dim]   # (B, 1)
        ptr += self.grip_dim

        # 任务编码（两个颜色index）
        task_raw = obs[:, ptr : ptr + self.task_dim]  # (B, 2) float
        pick_idx  = task_raw[:, 0].long().clamp(0, N-1)
        place_idx = task_raw[:, 1].long().clamp(0, N-1)

        # ── 编码图像crops ─────────────────────────────────────────
        crops_2d = crops.view(B * N, 3, CROP_SIZE, CROP_SIZE)
        cnn_feats = self.crop_encoder(crops_2d).view(B, N, self.cnn_out_dim)

        # ── 构建每对象特征 ────────────────────────────────────────
        # 对每个颜色：[cnn | block_pos | bin_pos | gripper]
        grip_exp = gripper.unsqueeze(1).expand(-1, N, -1)   # (B,N,1)

        per_obj = torch.cat([
            cnn_feats,          # (B,N,64)
            block_pos,          # (B,N,2)
            bin_pos,            # (B,N,2)
            grip_exp,           # (B,N,1)
        ], dim=-1)              # (B,N,69)

        per_obj = self.proj(per_obj)   # (B,N,attn_dim)

        # ── 自注意力聚合 ─────────────────────────────────────────
        attn_out, _ = self.attn(per_obj, per_obj, per_obj)
        attn_out = self.norm(attn_out + per_obj)
        global_feat = attn_out.mean(dim=1)   # (B, attn_dim)

        # ── 任务编码（颜色embedding）────────────────────────────
        pick_emb  = self.color_embed(pick_idx)    # (B, embed_dim)
        place_emb = self.color_embed(place_idx)   # (B, embed_dim)
        task_feat = self.task_proj(
            torch.cat([pick_emb, place_emb], dim=-1))   # (B, attn_dim)

        # ── 融合并输出 ───────────────────────────────────────────
        combined = torch.cat([global_feat, task_feat], dim=-1)
        return self.mlp(combined)   # (B, features_dim)


class ExploRLLMSAC(SAC):
    """
    升级版SAC，注入LLM探索，兼容新的observation布局。
    """

    def __init__(self, policy, env,
                 llm_policy=None,
                 warmup_steps: int = 20_000,
                 n_colors: int = 3,
                 **kwargs):
        super().__init__(policy, env, **kwargs)
        self.llm_policy   = llm_policy
        self.warmup_steps = warmup_steps
        self.n_colors     = n_colors
        self._llm_steps   = 0
        self._rl_steps    = 0

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        if (self.num_timesteps < self.warmup_steps
                or self.llm_policy is None):
            return super()._sample_action(
                learning_starts, action_noise, n_envs)

        if self.llm_policy.should_explore():
            obs = self._last_obs
            actions = []
            for i in range(n_envs):
                obs_dict = self._parse_obs(obs[i])
                crops    = self._extract_crops(obs[i])
                try:
                    action = self.llm_policy.get_exploration_action(
                        obs_dict, crops)
                except Exception as e:
                    print(f"[SAC] LLM失败: {e}")
                    action, _ = super()._sample_action(
                        learning_starts, action_noise, 1)
                    action = action[0]
                actions.append(action)
            self._llm_steps += n_envs
            arr = np.array(actions)
            return arr, arr
        else:
            self._rl_steps += 1
            return super()._sample_action(
                learning_starts, action_noise, n_envs)

    def _parse_obs(self, obs: np.ndarray) -> dict:
        """从flat observation重建结构化字典，供LLM使用。"""
        from config.color_config import get_color_config
        cfg = get_color_config()
        N   = cfg.n_colors

        img_dim  = N * 3 * CROP_SIZE * CROP_SIZE
        pos_dim  = N * 2
        bin_dim  = N * 2

        block_pos = obs[img_dim : img_dim+pos_dim].reshape(N, 2)
        bin_pos   = obs[img_dim+pos_dim : img_dim+pos_dim+bin_dim].reshape(N, 2)
        gripper   = float(obs[img_dim+pos_dim+bin_dim])
        task      = obs[img_dim+pos_dim+bin_dim+1 :]

        pick_idx  = int(round(float(task[0])))
        place_idx = int(round(float(task[1])))
        pick_color  = cfg.idx_to_color(pick_idx)
        place_color = cfg.idx_to_color(place_idx)

        positions = {}
        for i, c in enumerate(cfg.colors):
            positions[f"{c}_block"] = block_pos[i].tolist()
            positions[f"{c}_bin"]   = bin_pos[i].tolist()

        return {
            "positions":   positions,
            "gripper":     "open" if gripper < 0.5 else "closed",
            "pick_color":  pick_color,
            "place_color": place_color,
            "held_object": f"{pick_color}_block" if gripper >= 0.5 else None,
        }

    def _extract_crops(self, obs: np.ndarray) -> np.ndarray:
        N = self.n_colors
        img_dim = N * 3 * CROP_SIZE * CROP_SIZE
        return obs[:img_dim].reshape(N, 3, CROP_SIZE, CROP_SIZE)

    def get_exploration_stats(self) -> dict:
        total = self._llm_steps + self._rl_steps
        return {
            "llm_steps":    self._llm_steps,
            "rl_steps":     self._rl_steps,
            "llm_fraction": self._llm_steps / max(total, 1),
        }


def make_sac_kwargs(n_colors: int = 3,
                       features_dim: int = 128) -> dict:
    """给stable-baselines3的SAC构造policy_kwargs。"""
    return {
        "features_extractor_class":  ObjectCentricExtractor,
        "features_extractor_kwargs": {
            "n_colors":    n_colors,
            "cnn_out_dim": 64,
            "embed_dim":   16,
            "attn_dim":    96,
            "n_attn_heads": 4,
            "features_dim": features_dim,
        },
        "net_arch": [128, 128],
        "activation_fn": nn.ReLU,
    }
