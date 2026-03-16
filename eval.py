#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval.py
=======
已训练 ExploRLLM 策略的评估脚本。

模式：
  1. 在 Gazebo 中评估已保存模型（无需相机）
  2. 在真实机械臂上用相机评估（sim-to-real 演示）
  3. 根据日志绘制训练曲线

用法：
    # 评估已保存模型
    python eval.py --model-path logs/eps0.2_seed0/final_model.zip --n-episodes 30

    # 真实机械臂 + 相机演示
    python eval.py --model-path logs/eps0.2_seed0/final_model.zip --real-robot

    # 绘制训练曲线（ablation）
    python eval.py --plot --log-dir logs/
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path",   type=str, default=None)
    p.add_argument("--n-episodes",   type=int, default=30)
    p.add_argument("--task",         type=str, default="short_horizon")
    p.add_argument("--real-robot",   action="store_true",
                   help="Use real camera for perception (sim-to-real)")
    p.add_argument("--plot",         action="store_true",
                   help="Plot training curves from log-dir")
    p.add_argument("--log-dir",      type=str, default="./logs")
    p.add_argument("--output-dir",   type=str, default="./results")
    return p.parse_args()


# ── 仿真评估 ─────────────────────────────────────────────────────────────────

def eval_sim(model_path: str, n_episodes: int, task: str):
    """在 Gazebo 仿真中评估已保存的模型。"""
    import rospy
    from envs.pick_place_env import SagittariusPickPlaceEnv
    from agents.custom_sac import ExploRLLMSAC

    if not rospy.core.is_initialized():
        rospy.init_node("explorllm_eval", anonymous=True)

    print(f"[Eval] Loading model: {model_path}")
    model = ExploRLLMSAC.load(model_path)

    env = SagittariusPickPlaceEnv(task=task, max_steps=10)

    results = {
        "episodes": [],
        "success_rate": 0.0,
        "mean_reward":  0.0,
    }

    successes = 0
    rewards   = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        ep_reward = 0.0
        done = truncated = False
        steps = 0

        print(f"\n[Eval] Episode {ep+1}/{n_episodes}: "
              f"pick={info.get('pick_color')}, place={info.get('place_color')}")

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1
            print(f"  step {steps}: prim={int(round(action[0]))}, "
                  f"obj={int(round(action[1]))}, "
                  f"res=({action[2]:.3f},{action[3]:.3f}), "
                  f"r={reward:.3f}, success={info.get('success')}")

        rewards.append(ep_reward)
        if done:
            successes += 1
            print(f"  ✓ Task completed! reward={ep_reward:.3f}")
        else:
            print(f"  ✗ Truncated. reward={ep_reward:.3f}")

        results["episodes"].append({
            "ep": ep,
            "reward": ep_reward,
            "success": done,
            "steps": steps,
            "pick_color":  info.get("pick_color"),
            "place_color": info.get("place_color"),
        })

    results["success_rate"] = successes / n_episodes
    results["mean_reward"]  = float(np.mean(rewards))
    results["std_reward"]   = float(np.std(rewards))

    print(f"\n{'='*50}")
    print(f"  Evaluation Results ({n_episodes} episodes)")
    print(f"{'='*50}")
    print(f"  Success rate : {results['success_rate']:.2%}")
    print(f"  Mean reward  : {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
    print(f"{'='*50}\n")

    env.close()
    return results


# ── 真实机械臂评估（sim-to-real）─────────────────────────────────────────────

def eval_real_robot(model_path: str, n_episodes: int, task: str):
    """
    在真实 Sagittarius 机械臂上用相机感知进行评估。

    用 HSV 颜色检测（Lab2 sagittarius_object_color_detector）的相机检测
    替代 Gazebo 真值位置。

    需先启动：
        roslaunch sagittarius_moveit demo_true.launch
        roslaunch sagittarius_object_color_detector color_classification_fixed.launch
    """
    import rospy
    from envs.pick_place_env import SagittariusPickPlaceEnv, ALL_OBJECTS
    from agents.custom_sac import ExploRLLMSAC
    from perception.camera_perception import CameraPerception

    if not rospy.core.is_initialized():
        rospy.init_node("explorllm_real_eval", anonymous=True)

    print("[RealEval] Loading model and camera perception...")
    model      = ExploRLLMSAC.load(model_path)
    camera     = CameraPerception()

    # 使用仿真 env，但可用相机感知覆盖（当前为占位）
    env = SagittariusPickPlaceEnv(task=task, max_steps=10)

    def camera_perception_override(obs_arr):
        """用真实相机检测结果替换 Gazebo 真值位置。"""
        detected = camera.get_object_positions()
        # 将检测位置写回 obs 数组中的位置区段
        # 具体实现依赖相机标定与观测布局
        pos_start = env.observation_space.shape[0]  # 图像区段结束位置
        # 占位：目前直接返回未修改的 obs
        return obs_arr

    print("[RealEval] Starting real robot evaluation.")
    print("[RealEval] Make sure arm is powered on and demo_true.launch is running.")
    input("[RealEval] Press Enter to start...")

    successes = 0
    rewards   = []

    for ep in range(n_episodes):
        print(f"\n[RealEval] Episode {ep+1}/{n_episodes}")
        obs, info = env.reset()

        print(f"  Task: pick {info.get('pick_color')} → {info.get('place_color')} bowl")
        print("  Place objects on table. Press Enter when ready...")
        input()

        ep_reward = 0.0
        done = truncated = False

        while not (done or truncated):
            # 此处可选用相机感知覆盖 obs 后再 predict
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward

        rewards.append(ep_reward)
        if done:
            successes += 1
            print(f"  ✓ Success! Reward={ep_reward:.3f}")
        else:
            print(f"  ✗ Failed. Reward={ep_reward:.3f}")

    success_rate = successes / n_episodes
    print(f"\n[RealEval] Success rate: {success_rate:.2%}")
    env.close()
    return {"success_rate": success_rate, "mean_reward": float(np.mean(rewards))}


# ── 训练曲线绘制 ──────────────────────────────────────────────────────────────

def plot_training_curves(log_dir: str, output_dir: str):
    """
    绘制 log_dir 下所有运行的训练奖励曲线。
    按 epsilon 分组，对多种子求 mean ± std。

    输出：training_curves.png
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("[Plot] matplotlib not installed. Run: pip install matplotlib")
        return

    log_dir    = Path(log_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有训练日志
    runs = {}  # epsilon_str → [(steps, reward)] 列表
    for log_file in sorted(log_dir.rglob("training_log.json")):
        try:
            with open(log_file) as f:
                data = json.load(f)
            eps = data["config"].get("epsilon", 0.0)
            key = f"ε={eps:.1f}"
            episodes = data.get("episode_rewards", [])
            if not episodes:
                continue
            steps   = [e["step"]   for e in episodes]
            rewards = [e["reward"] for e in episodes]

            # 滑动窗口平滑
            window = 50
            rewards_smooth = np.convolve(
                rewards, np.ones(window)/window, mode="valid")
            steps_smooth   = steps[window-1:]

            if key not in runs:
                runs[key] = []
            runs[key].append((np.array(steps_smooth), np.array(rewards_smooth)))
        except Exception as e:
            print(f"[Plot] Failed to load {log_file}: {e}")

    if not runs:
        print("[Plot] No training logs found.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"ε=0.0": "#888787", "ε=0.2": "#1D9E75", "ε=0.5": "#534AB7"}

    for key, seed_runs in sorted(runs.items()):
        color = colors.get(key, "#333333")
        # 将多种子对齐到同一步数轴（插值）
        max_steps = max(r[0][-1] for r in seed_runs)
        common_steps = np.linspace(0, max_steps, 200)
        interp_rewards = []
        for steps, rewards in seed_runs:
            interp = np.interp(common_steps, steps, rewards)
            interp_rewards.append(interp)
        arr = np.array(interp_rewards)
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)

        ax.plot(common_steps / 1000, mean,
                label=f"{key} (n={len(seed_runs)})",
                color=color, linewidth=2)
        ax.fill_between(common_steps / 1000,
                        mean - std, mean + std,
                        alpha=0.2, color=color)

    ax.set_xlabel("环境步数 (×10³)", fontsize=12)
    ax.set_ylabel("Episode 奖励（平滑）", fontsize=12)
    ax.set_title("ExploRLLM 训练曲线 — Sagittarius SGR532", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = output_dir / "training_curves.png"
    plt.savefig(str(out_path), dpi=150)
    print(f"[Plot] Saved training curves to {out_path}")
    plt.close()


# ── 入口 ───────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    if args.plot:
        plot_training_curves(args.log_dir, args.output_dir)
        return

    if args.model_path is None:
        print("ERROR: --model-path required for evaluation.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.real_robot:
        results = eval_real_robot(
            args.model_path, args.n_episodes, args.task)
    else:
        results = eval_sim(
            args.model_path, args.n_episodes, args.task)

    # 保存评估结果
    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Eval] Results saved to {results_path}")


if __name__ == "__main__":
    main()
