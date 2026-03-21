#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py  (v3 — VLM 感知升级版)
================================
变化：
  - 新增 --vlm-api-key / --vlm-model 参数（真机评估时用）
  - 训练阶段仍用 Gazebo GT 坐标（不调 VLM，节省 API 费用）
  - 其余训练逻辑与上一版本保持一致
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import torch
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import env_config  # noqa: F401 — 加载 .env
from env_config import (
    llm_api_key,
    llm_base_url,
    llm_model,
    vlm_api_key,
    vlm_base_url,
    vlm_model,
)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epsilon",            type=float, default=0.2)
    p.add_argument("--total-steps",        type=int,   default=50_000)
    p.add_argument("--warmup-steps",       type=int,   default=20_000)
    p.add_argument("--seed",               type=int,   default=0)
    p.add_argument("--task",               type=str,   default="short_horizon")
    p.add_argument("--max-episode-steps",  type=int,   default=10)
    p.add_argument("--pose-id-count",      type=int,   default=1,
                   help="动作中的姿态候选数；两阶段建议 1 -> 2~4")

    # 颜色配置
    p.add_argument("--yaml-path",          type=str,   default=None)
    p.add_argument("--colors",             type=str,   nargs="+", default=None)

    # SAC
    p.add_argument("--learning-rate",      type=float, default=3e-4)
    p.add_argument("--buffer-size",        type=int,   default=50_000)
    p.add_argument("--batch-size",         type=int,   default=256)
    p.add_argument("--learning-starts",    type=int,   default=1000)

    # LLM 探索策略（训练时引导 SAC 探索）
    p.add_argument("--model",              type=str,   default=llm_model(),
                   help="LLM 探索模型；默认 LLM_MODEL")
    p.add_argument("--api-key",            type=str,
                   default=llm_api_key(),
                   help="LLM API Key（ε>0 时必填；默认 LLM_API_KEY）")
    p.add_argument("--base-url",           type=str,
                   default=llm_base_url(),
                   help="LLM 网关；默认 LLM_BASE_URL")
    p.add_argument("--n-candidates",       type=int,   default=3)

    # VLM 感知（真机评估时用，训练时不调用；与 LLM 密钥/网关完全独立）
    p.add_argument("--vlm-api-key",        type=str,
                   default=vlm_api_key(),
                   help="VLM API Key（默认 VLM_API_KEY，与 LLM 分开）")
    p.add_argument("--vlm-model",          type=str,   default=vlm_model(),
                   help="VLM 模型；默认 VLM_MODEL")
    p.add_argument("--vlm-base-url",       type=str,   default=vlm_base_url(),
                   help="VLM 网关；默认 VLM_BASE_URL（不设则由各 VLM preset 决定）")
    p.add_argument("--calib-yaml",         type=str,   default=None,
                   help="Lab2 标定 yaml（真机评估用）")

    # 训练课程：2+2（先训） / 3+2（方块多于桶，含干扰色）
    p.add_argument(
        "--curriculum", type=str, default="2+2", choices=["2+2", "3+2"],
        help="2+2=2方块2桶；3+2=3方块2桶（少一个桶，含干扰块）")

    # 实验
    p.add_argument("--ablation",           action="store_true")
    p.add_argument("--seeds",              type=int, nargs="+", default=[0,1,2])
    p.add_argument("--epsilons",           type=float, nargs="+",
                   default=[0.0, 0.2, 0.5])

    # 输出
    p.add_argument("--log-dir",            type=str,   default="./logs")
    p.add_argument("--save-freq",          type=int,   default=5000)
    p.add_argument("--eval-freq",          type=int,   default=5000)
    p.add_argument("--eval-episodes",      type=int,   default=10)
    p.add_argument("--device",             type=str,   default="auto")
    p.add_argument("--verbose",            type=int,   default=1)
    return p.parse_args()


def train_single(args, epsilon: float, seed: int, run_name: str):
    import rospy
    import torch.nn as nn
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

    from config.color_config import ColorConfig
    from envs.pick_place_env import SagittariusPickPlaceEnv
    from agents.custom_sac import ExploRLLMSAC, make_sac_kwargs

    print(f"\n{'='*58}")
    print(f"  Training : {run_name}  (ε={epsilon}, seed={seed})")
    print(f"  [感知升级] 训练阶段用 Gazebo GT，真机评估用 VLM")
    print(f"{'='*58}\n")

    color_cfg = ColorConfig(yaml_path=args.yaml_path)
    if args.colors:
        color_cfg.colors = [c for c in args.colors if c in color_cfg.colors or True]
        color_cfg._build_index()
    log_dir  = Path(args.log_dir) / run_name
    ckpt_dir = log_dir / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    torch.manual_seed(seed)

    if not rospy.core.is_initialized():
        rospy.init_node("explorllm_train", anonymous=True)

    train_env = Monitor(
        SagittariusPickPlaceEnv(
            task=args.task, max_steps=args.max_episode_steps,
            color_config=color_cfg, pose_id_count=args.pose_id_count,
            curriculum_mode=args.curriculum),
        filename=str(log_dir / "train_monitor.csv"))

    eval_env = Monitor(
        SagittariusPickPlaceEnv(
            task=args.task, max_steps=args.max_episode_steps,
            color_config=color_cfg, noise_sigma=0.0,
            pose_id_count=args.pose_id_count,
            curriculum_mode=args.curriculum),
        filename=str(log_dir / "eval_monitor.csv"))

    # 网络与动作空间维度 = 固定槽位 n_active=SLOT_COUNT（3），与课程 2+2 / 3+2 无关
    na = train_env.env.n_active
    print(f"[Train] 颜色配置: {color_cfg.colors}  (ColorConfig N={color_cfg.n_colors})")
    print(f"[Train] 课程 curriculum={args.curriculum}  "
          f"观测/策略槽位 n_active={na}  obs_dim={train_env.env.obs_dim}")

    # LLM 探索策略（文字接口，训练阶段使用）
    # 密钥可从环境变量 LLM_API_KEY 读取，无需在命令行传 --api-key
    llm_policy = None
    _llm_key = (args.api_key or llm_api_key()).strip()
    if epsilon > 0.0 and _llm_key:
        from llm.llm_policy import LLMExplorationPolicy
        llm_policy = LLMExplorationPolicy(
            api_key=args.api_key or None,
            base_url=args.base_url,
            model=args.model, epsilon=epsilon,
            n_candidates=args.n_candidates, color_config=color_cfg,
            n_active=na)
        print(f"[Train] LLM 探索策略：model={args.model}, ε={epsilon}")
    elif epsilon > 0.0:
        print("[Train] 警告：ε>0 但无 API Key（请设 LLM_API_KEY 或 --api-key），以 ε=0 运行")

    policy_kwargs = make_sac_kwargs(n_colors=na, features_dim=128)
    model = ExploRLLMSAC(
        policy="MlpPolicy", env=train_env,
        llm_policy=llm_policy, warmup_steps=args.warmup_steps,
        n_colors=na, learning_rate=args.learning_rate,
        buffer_size=args.buffer_size, batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        policy_kwargs=policy_kwargs, device=args.device,
        verbose=args.verbose, tensorboard_log=str(log_dir/"tb"), seed=seed)

    params = sum(p.numel() for p in model.policy.parameters())
    print(f"[Train] 模型参数量：{params:,}")

    class EvalCB(BaseCallback):
        def __init__(self):
            super().__init__()
            self._results = []
            self._last = 0

        def _on_step(self):
            if self.num_timesteps - self._last >= args.eval_freq:
                self._last = self.num_timesteps
                s, r = 0, []
                for _ in range(args.eval_episodes):
                    o, _ = eval_env.reset()
                    done = tr = False; ep = 0
                    while not (done or tr):
                        a, _ = self.model.predict(o, deterministic=True)
                        o, rr, done, tr, _ = eval_env.step(a); ep += rr
                    r.append(ep)
                    if done: s += 1
                sr = s / args.eval_episodes
                mr = float(np.mean(r))
                print(f"\n[Eval@{self.num_timesteps}] success={sr:.2%}  "
                      f"reward={mr:.3f}\n")
                self._results.append({"step": self.num_timesteps,
                                      "success_rate": sr, "mean_reward": mr})
                with open(log_dir/"eval_results.json","w") as f:
                    json.dump(self._results, f, indent=2)
            return True

    callbacks = [
        EvalCB(),
        CheckpointCallback(save_freq=args.save_freq,
                           save_path=str(ckpt_dir), name_prefix="sac"),
    ]

    print(f"[Train] 开始训练 {args.total_steps} 步...")
    t0 = time.time()
    try:
        model.learn(total_timesteps=args.total_steps, callback=callbacks)
    except KeyboardInterrupt:
        print("\n[Train] 中断")

    elapsed = time.time() - t0
    print(f"[Train] 完成，耗时 {elapsed/3600:.1f}h")
    model.save(str(log_dir / "final_model"))
    print(f"[Train] 模型保存到 {log_dir}/final_model.zip")

    # 保存 VLM 相关配置（便于 eval.py 直接读取）
    vlm_config = {
        "vlm_model":   args.vlm_model,
        "vlm_base_url":args.vlm_base_url,
        "calib_yaml":  args.calib_yaml,
        "colors":      color_cfg.colors,
        "n_active":    na,
        "curriculum":  args.curriculum,
    }
    with open(log_dir/"vlm_config.json","w") as f:
        json.dump(vlm_config, f, indent=2)
    print(f"[Train] VLM 配置保存到 {log_dir}/vlm_config.json")

    train_env.close()
    eval_env.close()


def main():
    args = get_args()
    if args.ablation:
        for eps in args.epsilons:
            for seed in args.seeds:
                train_single(args, eps, seed, f"eps{eps:.1f}_seed{seed}")
    else:
        train_single(args, args.epsilon, args.seed,
                     f"eps{args.epsilon:.1f}_seed{args.seed}")


if __name__ == "__main__":
    main()
