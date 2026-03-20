#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval.py  (v3 — VLM 感知升级版)
================================
变化：
  - 真机部署时使用 AdaptivePerception（VLM优先，HSV fallback）
  - 新增 --vlm-api-key / --vlm-model 参数
  - 真机扫描时用高精度多帧均值模式
  - 其余逻辑与上一版本保持一致
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

import env_config  # noqa: F401 — 加载 .env
from env_config import vlm_api_key, vlm_base_url, vlm_model


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path",    type=str, default=None)
    p.add_argument("--n-episodes",    type=int, default=30)
    p.add_argument("--task",          type=str, default="short_horizon")
    p.add_argument("--pose-id-count", type=int, default=1)
    p.add_argument("--real-robot",    action="store_true")
    p.add_argument("--plot",          action="store_true")
    p.add_argument("--log-dir",       type=str, default="./logs")
    p.add_argument("--output-dir",    type=str, default="./results")
    p.add_argument("--colors",        type=str, nargs="+", default=None)
    p.add_argument("--yaml-path",     type=str, default=None)

    # VLM 感知参数（与训练用 LLM 的密钥/网关完全独立）
    p.add_argument("--vlm-api-key",   type=str,
                   default=vlm_api_key(),
                   help="VLM API Key（默认 VLM_API_KEY；真机必填，仿真可空）")
    p.add_argument("--vlm-model",     type=str, default=vlm_model(),
                   help="VLM 模型；默认 VLM_MODEL")
    p.add_argument("--vlm-base-url",  type=str,
                   default=vlm_base_url(),
                   help="VLM 网关；默认 VLM_BASE_URL")
    p.add_argument("--calib-yaml",    type=str, default=None,
                   help="Lab2 标定 yaml 路径（不指定则使用默认标定值）")
    p.add_argument("--split-x",       type=int, default=320,
                   help="图像左右分区分割线（像素），左=方块，右=桶")
    return p.parse_args()


# ── 仿真评估（不变） ───────────────────────────────────────────────────────────

def eval_sim(model_path: str, n_episodes: int, task: str,
             colors=None, yaml_path=None, pose_id_count: int = 1):
    import rospy
    from config.color_config import ColorConfig
    from envs.pick_place_env import SagittariusPickPlaceEnv
    from agents.custom_sac import ExploRLLMSAC

    if not rospy.core.is_initialized():
        rospy.init_node("explorllm_eval", anonymous=True)

    cfg = ColorConfig(yaml_path=yaml_path)
    if colors:
        cfg.colors = [c for c in colors if c in cfg.colors]
        cfg._build_index()

    model = ExploRLLMSAC.load(model_path)
    env   = SagittariusPickPlaceEnv(
        task=task, max_steps=10, color_config=cfg, pose_id_count=pose_id_count
    )

    results   = {"episodes": []}
    successes = 0
    rewards   = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        ep_r = 0.0
        done = trunc = False
        steps = 0
        print(f"\n[Eval] Episode {ep+1}/{n_episodes}  "
              f"pick={info.get('pick_color')} → place={info.get('place_color')}")

        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, step_info = env.step(action)
            ep_r  += r
            steps += 1
            prim  = int(round(float(action[0])))
            obj_i = int(round(float(action[1])))
            na    = env.n_active
            ac    = env._active_colors
            if obj_i < na:
                c, t_str = ac[obj_i], "block"
            else:
                c, t_str = ac[obj_i - na], "bin"
            if len(action) >= 5:
                pose_id = int(round(float(action[2])))
                rx, ry = float(action[3]), float(action[4])
            else:
                pose_id = 0
                rx, ry = float(action[2]), float(action[3])
            print(f"  step {steps}: {['pick','place'][prim]} {c}_{t_str}  "
                  f"pose_id={pose_id} res=({rx:.3f},{ry:.3f})  "
                  f"r={r:.3f}  ok={step_info.get('success')}")

        rewards.append(ep_r)
        if done:
            successes += 1
            print(f"  ✓ 成功  reward={ep_r:.3f}")
        else:
            print(f"  ✗ 超时  reward={ep_r:.3f}")

        results["episodes"].append({
            "ep": ep, "reward": ep_r, "success": done, "steps": steps,
            "pick_color": info.get("pick_color"),
            "place_color": info.get("place_color"),
        })

    results["success_rate"] = successes / n_episodes
    results["mean_reward"]  = float(np.mean(rewards))
    results["std_reward"]   = float(np.std(rewards))
    print(f"\n成功率: {results['success_rate']:.2%}  "
          f"平均奖励: {results['mean_reward']:.3f}")
    env.close()
    return results


# ── 真机评估（升级为 VLM 感知） ────────────────────────────────────────────────

def eval_real_robot(args, n_episodes: int, task: str,
                    colors=None, yaml_path=None):
    """
    真机部署评估。

    使用 AdaptivePerception（VLM + HSV fallback）替代纯 HSV 检测。
    VLM 扫描场景获得物体坐标，注入 observation，policy 执行动作。

    前置条件：
        roslaunch sagittarius_moveit demo_true.launch
    """
    import rospy
    from config.color_config import ColorConfig
    from envs.pick_place_env import SagittariusPickPlaceEnv
    from agents.custom_sac import ExploRLLMSAC
    from perception.camera_perception import AdaptivePerception

    if not rospy.core.is_initialized():
        rospy.init_node("explorllm_real", anonymous=True)

    cfg = ColorConfig(yaml_path=yaml_path)
    if colors:
        cfg.colors = [c for c in colors if c in cfg.colors]
        cfg._build_index()

    model = ExploRLLMSAC.load(args.model_path)
    env   = SagittariusPickPlaceEnv(task=task, max_steps=10, color_config=cfg)

    # 初始化 VLM 感知
    if not (args.vlm_api_key or "").strip():
        print("[RealEval] 警告：未设置 VLM_API_KEY / --vlm-api-key，将使用 HSV fallback（功能受限）")

    camera = AdaptivePerception(
        api_key=args.vlm_api_key or "no-key",
        vlm_model=args.vlm_model,
        base_url=args.vlm_base_url,
        split_x=args.split_x,
        color_config=cfg,
    )
    if args.calib_yaml:
        camera.load_calibration_from_yaml(args.calib_yaml)

    print(f"\n[RealEval] 摆放说明：")
    print(f"  - 方块：桌面左侧区域（约 x=0.15~0.32 m）")
    print(f"  - 方块与桶：与训练一致，均在桌面统一随机区内（见 env OBJECT_ZONE_*）")
    print(f"  - VLM 会识别任意颜色，无需提前标定 HSV 阈值")
    print(f"  - 摄像头分辨率建议 ≥ 640×480，均匀照明")
    input("\n  摆好物品后按 Enter 开始...\n")

    successes = 0
    rewards   = []

    for ep in range(n_episodes):
        print(f"\n[RealEval] Episode {ep+1}/{n_episodes}")

        # 高精度多帧扫描
        obs, info = env.reset()
        pick_c  = info.get("pick_color")
        place_c = info.get("place_color")
        print(f"  任务: pick {pick_c} block → place in {place_c} bin")
        print(f"  正在扫描场景（VLM，3帧均值）...")

        scene = camera.scan_scene_with_retry(
            target_block=pick_c,
            target_bin=place_c,
            n_frames=3,
            wait_sec=0.5,
        )

        # 打印检测结果
        bp   = scene["blocks"].get(pick_c)
        binp = scene["bins"].get(place_c)
        print(f"  {pick_c}_block  : "
              f"({bp[0]:.3f},{bp[1]:.3f})" if bp else "  目标方块未检测到！")
        print(f"  {place_c}_bin   : "
              f"({binp[0]:.3f},{binp[1]:.3f})" if binp else "  目标桶未检测到！")

        active = list(info.get("active_colors") or env._active_colors)
        obs = _inject_camera_positions(obs, scene, active)

        ep_r = 0.0
        done = trunc = False
        while not (done or trunc):
            # 每步刷新场景（方块被抓起后位置变化）
            scene = camera.scan_scene(wait_sec=0.2, n_retry=1)
            active = list(env._active_colors)
            obs   = _inject_camera_positions(obs, scene, active)

            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, step_info = env.step(action)
            active = list(step_info.get("active_colors") or env._active_colors)
            obs = _inject_camera_positions(obs, scene, active)
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
                             active_colors: list) -> np.ndarray:
    """将摄像头检测到的坐标注入 observation（顺序与 env 的 active_colors 槽位一致）。"""
    obs = obs.copy()
    N       = len(active_colors)
    img_dim = N * 3 * 28 * 28
    pos_dim = N * 2

    for i, c in enumerate(active_colors):
        pos = scene["blocks"].get(c)
        if pos is not None:
            s = img_dim + i * 2
            obs[s], obs[s+1] = pos[0], pos[1]

    for i, c in enumerate(active_colors):
        pos = scene["bins"].get(c)
        if pos is not None:
            s = img_dim + pos_dim + i * 2
            obs[s], obs[s+1] = pos[0], pos[1]

    return obs


# ── 训练曲线绘图（不变） ──────────────────────────────────────────────────────

def plot_training_curves(log_dir: str, output_dir: str):
    try:
        import matplotlib.pyplot as plt
        import matplotlib; matplotlib.use("Agg")
    except ImportError:
        print("[Plot] pip install matplotlib"); return

    log_dir    = Path(log_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = {}
    for f in sorted(log_dir.rglob("eval_results.json")):
        try:
            parts   = f.parent.name.split("_")
            eps_str = next((p for p in parts if p.startswith("eps")), "eps0.0")
            eps     = float(eps_str.replace("eps", ""))
            key     = f"ε={eps:.1f}"
            with open(f) as fp:
                data = json.load(fp)
            steps   = [d["step"]         for d in data]
            success = [d["success_rate"] for d in data]
            if key not in runs: runs[key] = []
            runs[key].append((np.array(steps), np.array(success)))
        except Exception as e:
            print(f"[Plot] 跳过 {f}: {e}")

    if not runs:
        print("[Plot] 没找到 eval_results.json"); return

    fig, ax = plt.subplots(figsize=(10, 6))
    cm = {"ε=0.0": "#888780", "ε=0.2": "#1D9E75", "ε=0.5": "#534AB7"}
    for key, seed_runs in sorted(runs.items()):
        c  = cm.get(key, "#333333")
        mx = max(r[0][-1] for r in seed_runs)
        xs = np.linspace(0, mx, 200)
        arr = np.array([np.interp(xs, sr[0], sr[1]) for sr in seed_runs])
        ax.plot(xs/1000, arr.mean(0), label=f"{key} (n={len(seed_runs)})",
                color=c, linewidth=2)
        ax.fill_between(xs/1000, arr.mean(0)-arr.std(0),
                        arr.mean(0)+arr.std(0), alpha=0.15, color=c)

    ax.set_xlabel("训练步数（×10³）", fontsize=12)
    ax.set_ylabel("任务成功率",       fontsize=12)
    ax.set_title("ExploRLLM 消融实验 — Sagittarius SGR532 (VLM感知)", fontsize=13)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    plt.tight_layout()
    out = output_dir / "training_curves.png"
    plt.savefig(str(out), dpi=150)
    print(f"[Plot] 保存到 {out}")
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
        print("错误：--model-path 必填")
        sys.exit(1)

    if args.real_robot:
        results = eval_real_robot(
            args, args.n_episodes, args.task,
            args.colors, args.yaml_path)
    else:
        results = eval_sim(
            args.model_path, args.n_episodes, args.task,
            args.colors, args.yaml_path, args.pose_id_count)

    out_file = out / "eval_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Eval] 结果保存到 {out_file}")


if __name__ == "__main__":
    main()
