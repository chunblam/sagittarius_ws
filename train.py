#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py
===========
升级版训练脚本，支持多颜色 + 随机桶位置。

用法：
    # 纯SAC（不需要API Key，测试环境是否能跑通）
    python train.py --epsilon 0.0

    # 完整ExploRLLM，DeepSeek
    python train.py --epsilon 0.2 --model deepseek-v3 --api-key sk-xxx

    # 指定要使用的颜色（如果只想用3种颜色测试）
    python train.py --epsilon 0.2 --colors red green blue

    # 消融实验
    python train.py --ablation --seeds 0 1 2

    # 指定vision_config.yaml路径（从Lab2标定结果加载颜色）
    python train.py --epsilon 0.2 --yaml-path ~/sagittarius_ws/.../vision_config.yaml
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


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epsilon",       type=float, default=0.2)
    p.add_argument("--total-steps",   type=int,   default=50_000)
    p.add_argument("--warmup-steps",  type=int,   default=20_000)
    p.add_argument("--seed",          type=int,   default=0)
    p.add_argument("--task",          type=str,   default="short_horizon")
    p.add_argument("--max-episode-steps", type=int, default=10)

    # 颜色配置
    p.add_argument("--yaml-path",     type=str,   default=None,
                   help="Lab2 vision_config.yaml路径（自动加载颜色）")
    p.add_argument("--colors",        type=str,   nargs="+", default=None,
                   help="手动指定颜色列表（覆盖yaml），例如 --colors red green blue")

    # SAC
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--buffer-size",   type=int,   default=50_000)
    p.add_argument("--batch-size",    type=int,   default=256)
    p.add_argument("--learning-starts",type=int,  default=1000)

    # LLM
    p.add_argument("--model",         type=str,   default="deepseek-v3")
    p.add_argument("--api-key",       type=str,
                   default=os.environ.get("LLM_API_KEY", ""))
    p.add_argument("--base-url",      type=str,
                   default=os.environ.get("LLM_BASE_URL", None))
    p.add_argument("--n-candidates",  type=int,   default=3)

    # 实验
    p.add_argument("--ablation",      action="store_true")
    p.add_argument("--seeds",         type=int, nargs="+", default=[0,1,2])
    p.add_argument("--epsilons",      type=float, nargs="+",
                   default=[0.0, 0.2, 0.5])

    # 输出
    p.add_argument("--log-dir",       type=str,   default="./logs")
    p.add_argument("--save-freq",     type=int,   default=5000)
    p.add_argument("--eval-freq",     type=int,   default=5000)
    p.add_argument("--eval-episodes", type=int,   default=10)
    p.add_argument("--device",        type=str,   default="auto")
    p.add_argument("--verbose",       type=int,   default=1)
    return p.parse_args()


def train_single(args, epsilon: float, seed: int, run_name: str):
    import rospy
    import torch.nn as nn
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

    from configs.color_config import ColorConfig
    from envs.pick_place_env import SagittariusPickPlaceEnv
    from agents.custom_sac import ExploRLLMSAC, make_sac_kwargs

    print(f"\n{'='*55}")
    print(f"  Training: {run_name}  (ε={epsilon}, seed={seed})")
    print(f"{'='*55}\n")

    # ── 颜色配置 ──────────────────────────────────────────────────
    color_cfg = ColorConfig(yaml_path=args.yaml_path)
    if args.colors:
        # 如果手动指定了颜色，过滤只保留这些颜色
        color_cfg.colors = [c for c in args.colors
                            if c in color_cfg.colors or True]
        color_cfg._build_index()
        print(f"[Train] 使用指定颜色：{color_cfg.colors}")
    else:
        print(f"[Train] 从yaml加载颜色：{color_cfg.colors}")

    N = color_cfg.n_colors
    print(f"[Train] 颜色总数 N={N}")

    # ── 目录 ──────────────────────────────────────────────────────
    log_dir  = Path(args.log_dir) / run_name
    ckpt_dir = log_dir / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    torch.manual_seed(seed)

    if not rospy.core.is_initialized():
        rospy.init_node("explorllm_train", anonymous=True)

    # ── 环境 ──────────────────────────────────────────────────────
    train_env = Monitor(
        SagittariusPickPlaceEnv(
            task=args.task,
            max_steps=args.max_episode_steps,
            color_config=color_cfg,
        ),
        filename=str(log_dir / "train_monitor.csv"),
    )

    eval_env = Monitor(
        SagittariusPickPlaceEnv(
            task=args.task,
            max_steps=args.max_episode_steps,
            color_config=color_cfg,
            noise_sigma=0.0,
        ),
        filename=str(log_dir / "eval_monitor.csv"),
    )

    # ── LLM策略 ──────────────────────────────────────────────────
    llm_policy = None
    if epsilon > 0.0 and args.api_key:
        from llm.llm_policy import LLMExplorationPolicy
        llm_policy = LLMExplorationPolicy(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            epsilon=epsilon,
            n_candidates=args.n_candidates,
            color_config=color_cfg,
        )
    elif epsilon > 0.0:
        print("[Train] 警告：ε>0 但没有API Key，以ε=0运行。")

    # ── SAC模型 ──────────────────────────────────────────────────
    policy_kwargs = make_sac_kwargs(n_colors=N, features_dim=128)

    model = ExploRLLMSAC(
        policy="MlpPolicy",
        env=train_env,
        llm_policy=llm_policy,
        warmup_steps=args.warmup_steps,
        n_colors=N,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        policy_kwargs=policy_kwargs,
        device=args.device,
        verbose=args.verbose,
        tensorboard_log=str(log_dir / "tb"),
        seed=seed,
    )

    params = sum(p.numel() for p in model.policy.parameters())
    print(f"[Train] 模型参数量：{params:,}")

    # ── 回调 ──────────────────────────────────────────────────────
    class EvalCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self._results = []
            self._last_eval = 0

        def _on_step(self):
            if self.num_timesteps - self._last_eval >= args.eval_freq:
                self._last_eval = self.num_timesteps
                successes, rewards = 0, []
                for _ in range(args.eval_episodes):
                    obs, _ = eval_env.reset()
                    done = trunc = False
                    ep_r = 0
                    while not (done or trunc):
                        a, _ = self.model.predict(obs, deterministic=True)
                        obs, r, done, trunc, _ = eval_env.step(a)
                        ep_r += r
                    rewards.append(ep_r)
                    if done: successes += 1
                sr = successes / args.eval_episodes
                mr = float(np.mean(rewards))
                print(f"\n[Eval@{self.num_timesteps}] "
                      f"success={sr:.2%}  reward={mr:.3f}\n")
                self._results.append({"step": self.num_timesteps,
                                      "success_rate": sr,
                                      "mean_reward": mr})
                with open(log_dir / "eval_results.json", "w") as f:
                    json.dump(self._results, f, indent=2)
            return True

    callbacks = [
        EvalCallback(),
        CheckpointCallback(save_freq=args.save_freq,
                           save_path=str(ckpt_dir),
                           name_prefix="sac"),
    ]

    # ── 训练 ──────────────────────────────────────────────────────
    print(f"[Train] 开始训练 {args.total_steps} 步...")
    t0 = time.time()
    try:
        model.learn(total_timesteps=args.total_steps, callback=callbacks)
    except KeyboardInterrupt:
        print("\n[Train] 用户中断。")

    elapsed = time.time() - t0
    print(f"[Train] 完成，耗时 {elapsed/3600:.1f}h")

    model.save(str(log_dir / "final_model"))
    print(f"[Train] 模型已保存到 {log_dir}/final_model.zip")

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
