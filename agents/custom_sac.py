#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
custom_sac.py
=============
Custom SAC policy with object-centric observation encoder:
    CNN (per-object image patches) → concat positions + lang → self-attention → MLP

Also implements ExploRLLMSAC: a subclass of SAC that injects LLM exploration
into the rollout collection phase (Algorithm 1).

Architecture (from ExploRLLM paper):
    For each object i:
        crop_i (3,28,28) ──► 2-layer CNN ──► feature_i (d_feat)
        feature_i + pos_i + gripper + lang ──► φ'_i (d')
    [φ'_0, ..., φ'_{N-1}] ──► Self-Attention ──► global_feat
    global_feat ──► 2-layer MLP ──► Q or π
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


# ── Dimension constants (must match pick_place_env.py) ───────────────────────

N_TOTAL    = 6       # blocks + bowls
CROP_SIZE  = 28
IMG_DIM    = N_TOTAL * 3 * CROP_SIZE * CROP_SIZE   # 14112
POS_DIM    = N_TOTAL * 2                            # 12
GRIP_DIM   = 1
LANG_DIM   = 6
OBS_DIM    = IMG_DIM + POS_DIM + GRIP_DIM + LANG_DIM   # 14131


# ── CNN feature extractor (per object) ───────────────────────────────────────

class ObjectCropEncoder(nn.Module):
    """
    Shared 2-layer CNN that encodes a single 28×28 crop into a feature vector.
    Applied independently to each object's image crop.
    """

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.cnn = nn.Sequential(
            # Input: (3, 28, 28)
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


# ── Self-attention over object features ──────────────────────────────────────

class ObjectSelfAttention(nn.Module):
    """
    Single-head self-attention over N object feature vectors.
    Aggregates into a single global feature via mean-pooling of attended outputs.
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
        returns: (batch, d_model) – mean-pooled attended features
        """
        attn_out, _ = self.attn(x, x, x)
        attn_out = self.norm(attn_out + x)      # residual
        return attn_out.mean(dim=1)             # (batch, d_model)


# ── Full object-centric feature extractor ────────────────────────────────────

class ObjectCentricExtractor(BaseFeaturesExtractor):
    """
    stable-baselines3 compatible feature extractor.

    Input:  flat observation vector of shape (obs_dim,)
    Output: (features_dim,) global feature vector

    Internal pipeline:
        un-flatten → CNN per crop → concat pose+lang+gripper → self-attention → output
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

        # Per-object dimension after concat:
        # cnn_out(64) + pos(2) + gripper(1) + lang(6/N_obj≈1) = 67 → project to attn_dim
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

        # ── Un-flatten observation ─────────────────────────────────────────
        img_flat  = obs[:, :IMG_DIM]                         # (B, 14112)
        pos_flat  = obs[:, IMG_DIM : IMG_DIM+POS_DIM]       # (B, 12)
        grip      = obs[:, IMG_DIM+POS_DIM : IMG_DIM+POS_DIM+GRIP_DIM]   # (B, 1)
        lang      = obs[:, IMG_DIM+POS_DIM+GRIP_DIM :]     # (B, 6)

        # Reshape images to (B, N_obj, 3, 28, 28)
        crops = img_flat.view(batch, self.n_objects, 3, CROP_SIZE, CROP_SIZE)

        # Positions (B, N_obj, 2)
        positions = pos_flat.view(batch, self.n_objects, 2)

        # ── Encode each object independently ─────────────────────────────
        # crops: (B, N_obj, 3, 28, 28) → (B*N_obj, 3, 28, 28)
        crops_flat = crops.view(batch * self.n_objects, 3, CROP_SIZE, CROP_SIZE)
        cnn_feats  = self.crop_encoder(crops_flat)           # (B*N_obj, cnn_out)
        cnn_feats  = cnn_feats.view(batch, self.n_objects, self.cnn_out_dim)

        # ── Build per-object feature vector ──────────────────────────────
        # Broadcast gripper and lang to per-object
        grip_exp  = grip.unsqueeze(1).expand(-1, self.n_objects, -1)   # (B, N, 1)
        lang_exp  = lang.unsqueeze(1).expand(-1, self.n_objects, -1)   # (B, N, 6)

        per_obj = torch.cat([cnn_feats, positions, grip_exp, lang_exp], dim=-1)
        # per_obj: (B, N_obj, cnn_out+2+1+6)

        # Pad if needed to match self._per_obj_raw
        if per_obj.shape[-1] < self._per_obj_raw:
            pad = self._per_obj_raw - per_obj.shape[-1]
            per_obj = F.pad(per_obj, (0, pad))

        per_obj = self.project(per_obj)  # (B, N_obj, attn_dim)

        # ── Self-attention aggregation ─────────────────────────────────
        global_feat = self.attention(per_obj)   # (B, attn_dim)

        return self.mlp(global_feat)            # (B, features_dim)


# ── ExploRLLM SAC with LLM exploration injection ─────────────────────────────

class ExploRLLMSAC(SAC):
    """
    SAC subclass that injects LLM-based exploration during rollout collection.

    Overrides collect_rollouts() to implement Algorithm 1:
        j ~ U[0,1)
        if j <= ε:  use LLM policy
        else:       use SAC policy
    """

    def __init__(self,
                 policy,
                 env,
                 llm_policy=None,      # LLMExplorationPolicy instance
                 warmup_steps: int = 20_000,
                 obs_parser_fn=None,   # fn(obs_array) → obs_dict for LLM
                 **kwargs):
        super().__init__(policy, env, **kwargs)
        self.llm_policy    = llm_policy
        self.warmup_steps  = warmup_steps
        self.obs_parser_fn = obs_parser_fn
        self._llm_steps    = 0   # count of LLM-guided steps
        self._rl_steps     = 0   # count of RL-guided steps

    def _sample_action(
        self,
        learning_starts: int,
        action_noise=None,
        n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Override stable-baselines3's _sample_action to inject LLM exploration.

        After warmup:
            - with probability ε: use LLM to generate action
            - otherwise:          use SAC policy as normal
        """
        # During warmup or if no LLM policy: standard SAC behaviour
        if (self.num_timesteps < self.warmup_steps
                or self.llm_policy is None):
            return super()._sample_action(learning_starts, action_noise, n_envs)

        # After warmup: ε-greedy exploration
        if self.llm_policy.should_explore():
            # Use LLM exploration
            obs = self._last_obs  # (n_envs, obs_dim)
            actions = []
            for i in range(n_envs):
                obs_i = obs[i]
                if self.obs_parser_fn is not None:
                    obs_dict = self.obs_parser_fn(obs_i)
                else:
                    obs_dict = self._default_obs_parser(obs_i)

                # Extract crops from observation
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
            actions_array  = buffer_actions   # already in correct scale
            return actions_array, buffer_actions

        else:
            # Use SAC policy
            self._rl_steps += 1
            return super()._sample_action(learning_starts, action_noise, n_envs)

    def _default_obs_parser(self, obs: np.ndarray) -> dict:
        """
        Default observation parser: extracts structured dict from flat obs.
        Matches the layout defined in SagittariusPickPlaceEnv._build_observation.
        """
        from envs.pick_place_env import ALL_OBJECTS, BLOCK_NAMES, BOWL_NAMES

        pos_start = IMG_DIM
        pos_flat  = obs[pos_start : pos_start + POS_DIM]
        positions = pos_flat.reshape(N_TOTAL, 2)

        grip_val = float(obs[pos_start + POS_DIM])
        gripper  = "open" if grip_val < 0.5 else "closed"

        lang = obs[pos_start + POS_DIM + GRIP_DIM :]
        # Decode one-hot lang
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
        """Return stats about LLM vs RL step counts."""
        total = self._llm_steps + self._rl_steps
        return {
            "llm_steps":  self._llm_steps,
            "rl_steps":   self._rl_steps,
            "total_steps": total,
            "llm_fraction": self._llm_steps / max(total, 1),
        }


# ── Policy registration helper ────────────────────────────────────────────────

def make_sac_kwargs(features_dim: int = 128) -> dict:
    """
    Returns policy_kwargs for stable-baselines3 SAC to use
    the ObjectCentricExtractor.
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
