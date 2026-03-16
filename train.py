#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py
========
Main training script for ExploRLLM on Sagittarius SGR532.

Usage examples:
    # Single run with ε=0.2
    python train.py --epsilon 0.2 --seed 0

    # Ablation: sweep ε ∈ {0, 0.2, 0.5}
    python train.py --ablation --seeds 0 1 2

    # With DeepSeek
    python train.py --epsilon 0.2 --model deepseek-v3 --api-key sk-xxx

    # Without LLM (pure SAC baseline)
    python train.py --epsilon 0.0

Environment variables (alternative to CLI args):
    LLM_API_KEY    : your API key
    LLM_BASE_URL   : custom API endpoint
    LLM_MODEL      : model name
"""

import os
import sys
import argparse
import time
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

# Add project root to path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


# ── Argument parsing ──────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="ExploRLLM training on Sagittarius")

    # Training config
    p.add_argument("--epsilon",      type=float, default=0.2,
                   help="LLM exploration probability (0=pure SAC)")
    p.add_argument("--total-steps",  type=int, default=50_000,
                   help="Total environment steps")
    p.add_argument("--warmup-steps", type=int, default=20_000,
                   help="Steps before LLM exploration starts")
    p.add_argument("--seed",         type=int, default=0)
    p.add_argument("--task",         type=str, default="short_horizon",
                   choices=["short_horizon", "long_horizon"])
    p.add_argument("--max-episode-steps", type=int, default=10)

    # SAC hyperparameters
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--buffer-size",   type=int,   default=50_000)
    p.add_argument("--batch-size",    type=int,   default=256)
    p.add_argument("--tau",           type=float, default=0.005)
    p.add_argument("--gamma",         type=float, default=0.99)
    p.add_argument("--train-freq",    type=int,   default=1)
    p.add_argument("--gradient-steps",type=int,   default=1)
    p.add_argument("--learning-starts",type=int,  default=1000)

    # LLM config
    p.add_argument("--model",        type=str, default="deepseek-v3",
                   help="LLM model name or preset key")
    p.add_argument("--api-key",      type=str,
                   default=os.environ.get("LLM_API_KEY", ""),
                   help="LLM API key (or set LLM_API_KEY env var)")
    p.add_argument("--base-url",     type=str,
                   default=os.environ.get("LLM_BASE_URL", None))
    p.add_argument("--n-candidates", type=int, default=3,
                   help="Number of low-level code candidates per object")

    # Ablation
    p.add_argument("--ablation",     action="store_true",
                   help="Run ablation sweep over ε ∈ {0, 0.2, 0.5}")
    p.add_argument("--seeds",        type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epsilons",     type=float, nargs="+",
                   default=[0.0, 0.2, 0.5])

    # Output
    p.add_argument("--log-dir",      type=str, default="./logs")
    p.add_argument("--save-freq",    type=int, default=5000,
                   help="Save checkpoint every N steps")
    p.add_argument("--eval-freq",    type=int, default=5000,
                   help="Evaluate every N steps")
    p.add_argument("--eval-episodes",type=int, default=10,
                   help="Episodes per evaluation")

    # Misc
    p.add_argument("--device",       type=str, default="auto",
                   help="'auto', 'cuda', or 'cpu'")
    p.add_argument("--verbose",      type=int, default=1)

    return p.parse_args()


# ── Logging & callbacks ───────────────────────────────────────────────────────

class TrainingLogger:
    """Minimal training logger that saves episode rewards to JSON."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.data = {
            "config": {},
            "episode_rewards": [],   # list of (step, reward)
            "eval_results":    [],   # list of eval dicts
        }
        log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_episode(self, step: int, reward: float, info: dict = None):
        entry = {"step": step, "reward": reward}
        if info:
            entry.update(info)
        self.data["episode_rewards"].append(entry)

    def log_eval(self, step: int, eval_dict: dict):
        eval_dict["step"] = step
        self.data["eval_results"].append(eval_dict)

    def save(self):
        with open(self.log_path, "w") as f:
            json.dump(self.data, f, indent=2)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_policy(model, env, n_episodes: int = 10) -> dict:
    """
    Run n_episodes with the current policy (no LLM exploration).
    Returns success rate, mean reward, low-level error rate.
    """
    successes      = 0
    total_rewards  = []
    motion_errors  = 0
    total_steps    = 0

    for ep in range(n_episodes):
        obs, info = env.reset()
        ep_reward = 0.0
        done = False
        truncated = False

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            total_steps += 1
            if not info.get("success", True):
                motion_errors += 1

        total_rewards.append(ep_reward)
        if done:  # terminated = task success
            successes += 1

    return {
        "success_rate":   successes / n_episodes,
        "mean_reward":    float(np.mean(total_rewards)),
        "std_reward":     float(np.std(total_rewards)),
        "low_level_error_rate": motion_errors / max(total_steps, 1),
        "n_episodes":     n_episodes,
    }


# ── Single training run ───────────────────────────────────────────────────────

def train_single(args, epsilon: float, seed: int, run_name: str):
    """Run one training experiment."""
    import rospy
    from envs.pick_place_env import SagittariusPickPlaceEnv
    from agents.custom_sac import ExploRLLMSAC, make_sac_kwargs
    from stable_baselines3.common.callbacks import (
        CheckpointCallback, EvalCallback, BaseCallback)
    from stable_baselines3.common.monitor import Monitor

    print(f"\n{'='*60}")
    print(f"  Training: {run_name}  (ε={epsilon}, seed={seed})")
    print(f"{'='*60}\n")

    # ── Setup directories ──────────────────────────────────────────────────
    log_dir  = Path(args.log_dir) / run_name
    ckpt_dir = log_dir / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = TrainingLogger(log_dir / "training_log.json")
    logger.data["config"] = vars(args)
    logger.data["config"]["epsilon"] = epsilon
    logger.data["config"]["seed"]    = seed

    # ── Random seeds ──────────────────────────────────────────────────────
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ── ROS init (done once per process) ──────────────────────────────────
    if not rospy.core.is_initialized():
        rospy.init_node("explorllm_train", anonymous=True)

    # ── Create environments ────────────────────────────────────────────────
    train_env = Monitor(
        SagittariusPickPlaceEnv(
            task=args.task,
            max_steps=args.max_episode_steps,
        ),
        filename=str(log_dir / "train_monitor.csv"),
    )
    eval_env = Monitor(
        SagittariusPickPlaceEnv(
            task=args.task,
            max_steps=args.max_episode_steps,
            noise_sigma=0.0,   # eval with less noise for cleaner metrics
        ),
        filename=str(log_dir / "eval_monitor.csv"),
    )

    # ── LLM policy ────────────────────────────────────────────────────────
    llm_policy = None
    if epsilon > 0.0 and args.api_key:
        from llm.llm_policy import LLMExplorationPolicy
        llm_policy = LLMExplorationPolicy(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            epsilon=epsilon,
            n_candidates=args.n_candidates,
        )
        print(f"[Train] LLM policy initialised (ε={epsilon})")
    elif epsilon > 0.0 and not args.api_key:
        print(f"[Train] WARNING: ε={epsilon} but no API key provided. "
              f"Running as pure SAC (ε=0).")

    # ── Device ────────────────────────────────────────────────────────────
    device = "auto"
    if args.device == "cpu":
        device = "cpu"
    elif args.device == "cuda" and torch.cuda.is_available():
        device = "cuda"

    # ── Build SAC model ────────────────────────────────────────────────────
    policy_kwargs = make_sac_kwargs(features_dim=128)

    model = ExploRLLMSAC(
        policy="MlpPolicy",
        env=train_env,
        llm_policy=llm_policy,
        warmup_steps=args.warmup_steps,
        # SAC hyperparameters
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        learning_starts=args.learning_starts,
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=args.verbose,
        tensorboard_log=str(log_dir / "tb"),
        seed=seed,
    )

    print(f"[Train] Model parameters: "
          f"{sum(p.numel() for p in model.policy.parameters()):,}")

    # ── Callbacks ─────────────────────────────────────────────────────────

    class EpisodeLogCallback(BaseCallback):
        """Log episode reward at end of each episode."""
        def __init__(self, _logger: TrainingLogger):
            super().__init__()
            self._ep_logger = _logger

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                if "episode" in info:
                    ep_r = info["episode"]["r"]
                    self._ep_logger.log_episode(
                        self.num_timesteps, ep_r, info)
                    if self.verbose and self.num_timesteps % 1000 == 0:
                        stats = self.model.get_exploration_stats()
                        print(f"  step={self.num_timesteps:6d}  "
                              f"ep_reward={ep_r:.3f}  "
                              f"llm_frac={stats['llm_fraction']:.2f}")
            return True

        def _on_training_end(self) -> None:
            self._ep_logger.save()

    class EvalLogCallback(BaseCallback):
        """Periodic evaluation + logging."""
        def __init__(self, _logger, eval_env, freq, n_ep):
            super().__init__()
            self._ep_logger  = _logger
            self._eval_env   = eval_env
            self._freq       = freq
            self._n_episodes = n_ep
            self._last_eval  = 0

        def _on_step(self) -> bool:
            if self.num_timesteps - self._last_eval >= self._freq:
                self._last_eval = self.num_timesteps
                results = evaluate_policy(
                    self.model, self._eval_env, self._n_episodes)
                self._ep_logger.log_eval(self.num_timesteps, results)
                self._ep_logger.save()
                print(f"\n[Eval @ {self.num_timesteps}] "
                      f"success={results['success_rate']:.2%}  "
                      f"reward={results['mean_reward']:.3f}  "
                      f"error_rate={results['low_level_error_rate']:.2%}\n")
            return True

    checkpoint_cb = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=str(ckpt_dir),
        name_prefix="sac",
    )

    callbacks = [
        EpisodeLogCallback(logger),
        EvalLogCallback(logger, eval_env, args.eval_freq, args.eval_episodes),
        checkpoint_cb,
    ]

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"[Train] Starting training for {args.total_steps} steps...")
    t0 = time.time()

    try:
        model.learn(
            total_timesteps=args.total_steps,
            callback=callbacks,
            reset_num_timesteps=True,
        )
    except KeyboardInterrupt:
        print("\n[Train] Interrupted by user.")

    elapsed = time.time() - t0
    print(f"\n[Train] Done. Elapsed: {elapsed/3600:.1f}h")

    # ── Final save ────────────────────────────────────────────────────────
    final_path = str(log_dir / "final_model")
    model.save(final_path)
    print(f"[Train] Model saved to {final_path}.zip")

    # Final eval
    print("[Train] Running final evaluation...")
    final_results = evaluate_policy(model, eval_env, args.eval_episodes * 2)
    logger.log_eval(args.total_steps, {**final_results, "final": True})
    logger.save()
    print(f"[Train] Final: success={final_results['success_rate']:.2%}, "
          f"reward={final_results['mean_reward']:.3f}")

    # Cleanup
    train_env.close()
    eval_env.close()

    return final_results


# ── Ablation sweep ────────────────────────────────────────────────────────────

def run_ablation(args):
    """Run full ablation: ε × seeds grid."""
    all_results = {}

    for epsilon in args.epsilons:
        for seed in args.seeds:
            run_name = f"eps{epsilon:.1f}_seed{seed}"
            results  = train_single(args, epsilon, seed, run_name)
            key = f"eps={epsilon}"
            if key not in all_results:
                all_results[key] = []
            all_results[key].append(results)

    # Print summary table
    print("\n" + "="*60)
    print("  ABLATION SUMMARY")
    print("="*60)
    print(f"{'ε':>8}  {'Success(mean)':>14}  {'Success(std)':>13}  {'Reward(mean)':>13}")
    print("-"*60)
    for key, results_list in sorted(all_results.items()):
        success_rates = [r["success_rate"] for r in results_list]
        rewards       = [r["mean_reward"]  for r in results_list]
        print(f"{key:>8}  {np.mean(success_rates):>14.2%}  "
              f"{np.std(success_rates):>13.2%}  {np.mean(rewards):>13.3f}")

    # Save summary
    summary_path = Path(args.log_dir) / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = get_args()

    # Validate
    if args.epsilon > 0 and not args.api_key:
        print("WARNING: --epsilon > 0 requires --api-key or LLM_API_KEY env var.")
        print("Continuing with ε=0 (pure SAC).")
        args.epsilon = 0.0

    if args.ablation:
        print(f"Running ablation over ε={args.epsilons}, seeds={args.seeds}")
        run_ablation(args)
    else:
        run_name = f"eps{args.epsilon:.1f}_seed{args.seed}"
        train_single(args, args.epsilon, args.seed, run_name)


if __name__ == "__main__":
    main()
