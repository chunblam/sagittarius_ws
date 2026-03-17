#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval.py  
==================
评估脚本：测试训练好的policy或绘制训练曲线。

用法：
    # 在Gazebo里评估
    python eval.py --model-path logs/eps0.2_seed0/final_model.zip --n-episodes 30

    # 真机部署（使用摄像头感知）
    python eval.py --model-path logs/eps0.2_seed0/final_model.zip --real-robot

    # 绘制消融实验曲线
    python eval.py --plot --log-dir logs/

    # 指定颜色子集评估
    python eval.py --model-path ... --colors red green blue
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path",    type=str, default=None)
    p.add_argument("--n-episodes",    type=int, default=30)
    p.add_argument("--task",          type=str, default="short_horizon")
    p.add_argument("--real-robot",    action="store_true")
    p.add_argument("--plot",          action="store_true")
    p.add_argument("--log-dir",       type=str, default="./logs")
    p.add_argument("--output-dir",    type=str, default="./results")
    p.add_argument("--colors",        type=str, nargs="+", default=None,
                   help="指定评估时使用的颜色子集")
    p.add_argument("--yaml-path",     type=str, default=None)
    return p.parse_args()


# ── 仿真评估 ──────────────────────────────────────────────────────────────────

def eval_sim(model_path: str, n_episodes: int,
             task: str, colors=None, yaml_path=None):
    import rospy
    from configs.color_config import ColorConfig
    from envs.pick_place_env import SagittariusPickPlaceEnv
    from agents.custom_sac import ExploRLLMSAC

    if not rospy.core.is_initialized():
        rospy.init_node("explorllm_eval", anonymous=True)

    cfg = ColorConfig(yaml_path=yaml_path)
    if colors:
        cfg.colors = [c for c in colors if c in cfg.colors]
        cfg._build_index()
        print(f"[Eval] 使用颜色子集：{cfg.colors}")

    print(f"[Eval] 加载模型：{model_path}")
    model = ExploRLLMSAC.load(model_path)
    env   = SagittariusPickPlaceEnv(task=task, max_steps=10,
                                      color_config=cfg)

    results  = {"episodes": []}
    successes = 0
    rewards   = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        ep_r = 0.0
        done = trunc = False
        steps = 0

        print(f"\n[Eval] Episode {ep+1}/{n_episodes}  "
              f"pick={info.get('pick_color')} → "
              f"place={info.get('place_color')}")

        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, step_info = env.step(action)
            ep_r += r
            steps += 1
            prim  = int(round(float(action[0])))
            obj_i = int(round(float(action[1])))
            c     = cfg.idx_to_color(obj_i % cfg.n_colors)
            t_str = "block" if obj_i < cfg.n_colors else "bin"
            print(f"  step {steps}: {['pick','place'][prim]} {c}_{t_str}  "
                  f"res=({action[2]:.3f},{action[3]:.3f})  "
                  f"r={r:.3f}  ok={step_info.get('success')}")

        rewards.append(ep_r)
        if done:
            successes += 1
            print(f"  ✓ 成功！reward={ep_r:.3f}")
        else:
            print(f"  ✗ 超时。reward={ep_r:.3f}")

        results["episodes"].append({
            "ep":          ep,
            "reward":      ep_r,
            "success":     done,
            "steps":       steps,
            "pick_color":  info.get("pick_color"),
            "place_color": info.get("place_color"),
        })

    results["success_rate"] = successes / n_episodes
    results["mean_reward"]  = float(np.mean(rewards))
    results["std_reward"]   = float(np.std(rewards))

    print(f"\n{'='*52}")
    print(f"  评估结果（{n_episodes} episodes）")
    print(f"{'='*52}")
    print(f"  成功率  : {results['success_rate']:.2%}")
    print(f"  平均奖励: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
    print(f"{'='*52}\n")

    env.close()
    return results


# ── 真机评估 ──────────────────────────────────────────────────────────────────

def eval_real_robot(model_path: str, n_episodes: int,
                    task: str, colors=None, yaml_path=None):
    """
    真机部署评估。

    使用摄像头的 CameraPerception 同时检测方块和垃圾桶位置，
    替换Gazebo GT坐标注入observation。

    前置条件：
        roslaunch sagittarius_moveit demo_true.launch
        roslaunch sagittarius_object_color_detector hsv_params.launch（确保颜色阈值正确）
    """
    import rospy
    from configs.color_config import ColorConfig
    from envs.pick_place_env import SagittariusPickPlaceEnv
    from agents.custom_sac import ExploRLLMSAC
    from perception.camera_perception import CameraPerception

    if not rospy.core.is_initialized():
        rospy.init_node("explorllm_real", anonymous=True)

    cfg  = ColorConfig(yaml_path=yaml_path)
    if colors:
        cfg.colors = [c for c in colors if c in cfg.colors]
        cfg._build_index()

    print(f"[RealEval] 加载模型和摄像头感知...")
    model  = ExploRLLMSAC.load(model_path)
    camera = CameraPerception(color_config=cfg)
    env    = SagittariusPickPlaceEnv(task=task, max_steps=10,
                                       color_config=cfg)

    print(f"[RealEval] 使用颜色：{cfg.colors}")
    print(f"[RealEval] 摆放说明：")
    print(f"  - 方块：随意摆在桌面左侧区域（x=0.15~0.30）")
    print(f"  - 垃圾桶：随意摆在桌面右侧区域（x=0.28~0.40）")
    print(f"  - 左右半区无需精确，只需大致分开即可")
    print(f"  - 方块和桶靠面积自动区分，同颜色不会混淆")
    input("\n[RealEval] 摆好物品后按 Enter 开始评估...")

    successes = 0
    rewards   = []

    for ep in range(n_episodes):
        print(f"\n[RealEval] Episode {ep+1}/{n_episodes}")

        # 用摄像头扫描获取当前物体位置
        print("  扫描桌面场景...")
        scene = camera.scan_scene(wait_sec=1.5)

        print("  检测到的物体：")
        for c in cfg.colors:
            bp   = scene["blocks"].get(c)
            binp = scene["bins"].get(c)
            bs   = f"({bp[0]:.3f},{bp[1]:.3f})"   if bp   is not None else "未检测"
            bns  = f"({binp[0]:.3f},{binp[1]:.3f})" if binp is not None else "未检测"
            print(f"    {c:8s}: block={bs}  bin={bns}")

        obs, info = env.reset()

        # 用摄像头坐标覆盖observation里的位置部分
        obs = _inject_camera_positions(obs, scene, cfg)

        pick_c  = info.get("pick_color")
        place_c = info.get("place_color")
        print(f"  任务: pick {pick_c} block → place in {place_c} bin")

        ep_r = 0.0
        done = trunc = False
        while not (done or trunc):
            # 每步刷新一次摄像头（方块被抓后位置会变）
            scene = camera.scan_scene(wait_sec=0.3)
            obs   = _inject_camera_positions(obs, scene, cfg)

            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, _ = env.step(action)
            obs = _inject_camera_positions(obs, scene, cfg)
            ep_r += r

        rewards.append(ep_r)
        if done:
            successes += 1
            print(f"  ✓ 成功！")
        else:
            print(f"  ✗ 超时。")

    sr = successes / n_episodes
    print(f"\n[RealEval] 成功率: {sr:.2%}  "
          f"平均奖励: {float(np.mean(rewards)):.3f}")
    env.close()
    return {"success_rate": sr, "mean_reward": float(np.mean(rewards))}


def _inject_camera_positions(obs: np.ndarray, scene: dict,
                              cfg) -> np.ndarray:
    """
    将摄像头检测到的位置坐标注入observation向量，
    替换Gazebo GT位置（用于真机部署）。
    """
    obs = obs.copy()
    N       = cfg.n_colors
    img_dim = N * 3 * 28 * 28

    # 方块位置
    for i, c in enumerate(cfg.colors):
        pos = scene["blocks"].get(c)
        if pos is not None:
            start = img_dim + i * 2
            obs[start]   = pos[0]
            obs[start+1] = pos[1]

    # 桶位置
    pos_dim = N * 2
    for i, c in enumerate(cfg.colors):
        pos = scene["bins"].get(c)
        if pos is not None:
            start = img_dim + pos_dim + i * 2
            obs[start]   = pos[0]
            obs[start+1] = pos[1]

    return obs


# ── 训练曲线绘图 ──────────────────────────────────────────────────────────────

def plot_training_curves(log_dir: str, output_dir: str):
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("[Plot] 请安装matplotlib：pip install matplotlib")
        return

    log_dir    = Path(log_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = {}
    for f in sorted(log_dir.rglob("eval_results.json")):
        try:
            # 从路径名推断epsilon（格式：eps0.2_seed0）
            parts = f.parent.name.split("_")
            eps_str = next((p for p in parts if p.startswith("eps")), "eps0.0")
            eps = float(eps_str.replace("eps",""))
            key = f"ε={eps:.1f}"

            with open(f) as fp:
                data = json.load(fp)
            steps   = [d["step"]         for d in data]
            success = [d["success_rate"] for d in data]
            if key not in runs: runs[key] = []
            runs[key].append((np.array(steps), np.array(success)))
        except Exception as e:
            print(f"[Plot] 跳过 {f}: {e}")

    if not runs:
        print("[Plot] 没找到eval_results.json文件")
        print(f"  检查路径：{log_dir}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors_map = {
        "ε=0.0": "#888780",
        "ε=0.2": "#1D9E75",
        "ε=0.5": "#534AB7",
    }

    for key, seed_runs in sorted(runs.items()):
        c = colors_map.get(key, "#333333")
        max_steps = max(r[0][-1] for r in seed_runs)
        xs = np.linspace(0, max_steps, 200)
        interps = [np.interp(xs, sr[0], sr[1]) for sr in seed_runs]
        arr  = np.array(interps)
        mean = arr.mean(0)
        std  = arr.std(0)
        ax.plot(xs/1000, mean, label=f"{key} (n={len(seed_runs)})",
                color=c, linewidth=2)
        ax.fill_between(xs/1000, mean-std, mean+std, alpha=0.15, color=c)

    ax.set_xlabel("训练步数（×10³）", fontsize=12)
    ax.set_ylabel("任务成功率", fontsize=12)
    ax.set_title("ExploRLLM 消融实验 — Sagittarius SGR532 ", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.0%}"))
    plt.tight_layout()

    out = output_dir / "training_curves.png"
    plt.savefig(str(out), dpi=150)
    print(f"[Plot] 训练曲线保存到：{out}")
    plt.close()


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    out  = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.plot:
        plot_training_curves(args.log_dir, args.output_dir)
        return

    if args.model_path is None:
        print("错误：--model-path 必填（评估模式）")
        sys.exit(1)

    if args.real_robot:
        results = eval_real_robot(
            args.model_path, args.n_episodes, args.task,
            args.colors, args.yaml_path)
    else:
        results = eval_sim(
            args.model_path, args.n_episodes, args.task,
            args.colors, args.yaml_path)

    out_file = out / "eval_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Eval] 结果保存到：{out_file}")


if __name__ == "__main__":
    main()
