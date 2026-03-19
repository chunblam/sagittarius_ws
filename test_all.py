#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_all.py  (v3 — VLM 感知升级版)
=====================================
变化：
  - Test 6 升级为 VLM 感知测试（替代 HSV 分区检测）
  - Test 7 保持 LLM API 连接测试
  - 其余测试与上一版本一致
"""

import sys
import os
import time
import argparse
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import env_config  # noqa: F401 — 加载 .env
from env_config import (
    llm_api_key,
    llm_base_url,
    llm_model,
    vlm_api_key,
    vlm_model,
)

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg): print(f"  {RED}✗{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}!{RESET} {msg}")
def info(msg): print(f"  {BLUE}→{RESET} {msg}")
def section(t):
    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}  {t}{RESET}")
    print(f"{BOLD}{'─'*60}{RESET}")


# ── Test 1: ROS 连接 ──────────────────────────────────────────────────────────
def test_1_ros_connection():
    section("Test 1: ROS连接 & rosmaster")
    try:
        import subprocess
        r = subprocess.run(["rostopic","list"],
                           capture_output=True,text=True,timeout=5)
        if r.returncode != 0:
            fail("rostopic list 失败，rosmaster 未运行")
            fail("请先运行: roslaunch sagittarius_gazebo demo_gazebo.launch")
            return False
        ok(f"rosmaster 运行中，{len(r.stdout.strip().splitlines())} 个话题")
        import rospy
        if not rospy.core.is_initialized():
            rospy.init_node("explorllm_test", anonymous=True,
                            disable_signals=True)
        ok("rospy 节点初始化成功")
        return True
    except Exception as e:
        fail(f"ROS连接失败: {e}"); return False


# ── Test 2: Gazebo 物体检测 ───────────────────────────────────────────────────
def test_2_gazebo_objects():
    section("Test 2: Gazebo物体检测（6色×2类=12个）")
    from config.color_config import get_color_config
    cfg = get_color_config()
    info(f"颜色配置: {cfg.colors}")

    expected = []
    for c in cfg.colors:
        expected.append(f"{c}_block")
        expected.append(f"{c}_bin")

    try:
        import rospy
        from gazebo_msgs.msg import ModelStates
        received = {"data": None}
        rospy.Subscriber("/gazebo/model_states", ModelStates,
                         lambda m: received.update({"data": m}), queue_size=1)
        info("等待 /gazebo/model_states（3秒）...")
        time.sleep(3.0)

        if received["data"] is None:
            fail("没有收到消息，Gazebo 未启动或未点击 ▷")
            return False

        names = received["data"].name
        ok(f"Gazebo 中共有 {len(names)} 个模型")

        all_found = True
        for obj in expected:
            if obj in names:
                ok(f"  找到: {obj}")
            else:
                fail(f"  缺少: {obj}")
                all_found = False

        if not all_found:
            warn("把 pick_place_scene.world 的 <model> 内容加入你的 world 文件")
            warn("注意：v3 的垃圾桶每个有5个 link，Gazebo 模型数会相应增加")
        return all_found
    except Exception as e:
        fail(f"失败: {e}"); traceback.print_exc(); return False


# ── Test 3: MoveIt ────────────────────────────────────────────────────────────
def test_3_moveit():
    section("Test 3: MoveIt规划组连接")
    try:
        import moveit_commander
        moveit_commander.roscpp_initialize(sys.argv)
        arm = moveit_commander.MoveGroupCommander("sagittarius_arm")
        ok(f"sagittarius_arm  参考坐标系: {arm.get_planning_frame()}")
        ok(f"  末端执行器: {arm.get_end_effector_link()}")
        pose = arm.get_current_pose()
        ok(f"  末端: x={pose.pose.position.x:.3f}, "
           f"y={pose.pose.position.y:.3f}, z={pose.pose.position.z:.3f}")
        moveit_commander.MoveGroupCommander("sagittarius_gripper")
        ok("sagittarius_gripper 连接成功")
        return True
    except Exception as e:
        fail(f"MoveIt 连接失败: {e}")
        warn("等待 MoveIt 完全加载可能需要 10-20 秒，请稍后重试")
        return False


# ── Test 4: 颜色配置 ──────────────────────────────────────────────────────────
def test_4_color_config():
    section("Test 4: 颜色配置（ColorConfig）")
    try:
        from config.color_config import get_color_config
        cfg = get_color_config()
        ok(f"颜色配置: {cfg.colors}  ({cfg.n_colors} 种)")
        for c in cfg.colors:
            idx = cfg.color_to_idx(c)
            assert cfg.idx_to_color(idx) == c
            ok(f"  {c:10s} → idx={idx} → {cfg.idx_to_color(idx)} ✓")
        enc = cfg.encode_task(cfg.colors[0], cfg.colors[1])
        ok(f"  任务编码: {enc}")
        return True
    except Exception as e:
        fail(f"颜色配置失败: {e}"); traceback.print_exc(); return False


# ── Test 5: 环境 reset/step ───────────────────────────────────────────────────
def test_5_env_reset_step():
    section("Test 5: 环境 reset() 和 step()")
    try:
        from envs.pick_place_env import SagittariusPickPlaceEnv
        from config.color_config import get_color_config
        import numpy as np

        cfg = get_color_config()
        N   = cfg.n_colors
        env = SagittariusPickPlaceEnv(task="short_horizon", max_steps=3)
        ok("环境创建成功")
        ok(f"  obs_dim={env.obs_dim}  action_dim=4")

        t0 = time.time()
        obs, info_dict = env.reset()
        ok(f"reset() 成功 ({time.time()-t0:.1f}s)  "
           f"pick={info_dict.get('pick_color')}, "
           f"place={info_dict.get('place_color')}")

        if obs.shape[0] != env.obs_dim:
            fail(f"obs 维度 {obs.shape[0]} ≠ {env.obs_dim}"); return False
        ok(f"  obs 维度正确: {env.obs_dim}")

        img_dim = N*3*28*28
        block_pos = obs[img_dim:img_dim+N*2].reshape(N,2)
        bin_pos   = obs[img_dim+N*2:img_dim+N*4].reshape(N,2)
        info(f"  方块位置[0]: {block_pos[0]}")
        info(f"  桶位置[0]:   {bin_pos[0]}")

        if np.all(block_pos==0): warn("方块位置全是 0，检查 Gazebo 物体")
        if np.all(bin_pos==0):   warn("桶位置全是 0，检查 Gazebo 桶模型")

        info("调用 env.step()（约 10-20 秒）...")
        obs2, r, done, trunc, info2 = env.step(
            np.array([0.,0.,0.,0.], dtype=np.float32))
        ok(f"step() 返回  reward={r:.4f}  success={info2.get('success')}")

        env.close()
        return True
    except Exception as e:
        fail(f"失败: {e}"); traceback.print_exc(); return False


# ── Test 6: VLM 感知（真机可选） ─────────────────────────────────────────────
def test_6_vlm_perception():
    section("Test 6: VLM 感知测试（真机可选）")

    api_key   = vlm_api_key()
    _vlm_m    = vlm_model()

    if not api_key:
        warn("未设置 VLM_API_KEY，跳过 VLM 感知测试")
        warn("在 .env 中填写 VLM_API_KEY（与训练用 LLM_API_KEY 独立）")
        warn("可用 VLM_MODEL 切换视觉模型")
        return True

    # 检查摄像头
    try:
        import subprocess
        r = subprocess.run(["rostopic","list"],
                           capture_output=True,text=True,timeout=3)
        if "/usb_cam/image_raw" not in r.stdout:
            warn("摄像头话题不存在，跳过（仿真训练不需要摄像头）")
            warn("真机部署时需要先启动摄像头节点")
            return True
    except Exception:
        warn("无法检查摄像头，跳过")
        return True

    try:
        from perception.camera_perception import AdaptivePerception
        from config.color_config import get_color_config

        cfg    = get_color_config()
        camera = AdaptivePerception(
            api_key=api_key, vlm_model=_vlm_m, color_config=cfg)

        info(f"VLM 模型: {_vlm_m}")
        info("等待图像并调用 VLM 扫描（约 3-8 秒）...")
        time.sleep(2.0)

        if camera._latest_image is None:
            warn("没有收到图像，跳过 VLM 调用")
            return True

        ok(f"图像尺寸: {camera._latest_image.shape}")

        t0    = time.time()
        scene = camera.scan_scene(wait_sec=0.5, n_retry=1)
        elapsed = time.time() - t0

        ok(f"VLM 扫描完成，耗时 {elapsed:.1f}s")

        n_blocks = sum(1 for v in scene["blocks"].values() if v is not None)
        n_bins   = sum(1 for v in scene["bins"].values()   if v is not None)
        ok(f"检测到方块 {n_blocks} 个，垃圾桶 {n_bins} 个")

        # 打印所有检测到的物体
        for c, pos in scene["blocks"].items():
            if pos is not None:
                ok(f"  block/{c:10s}: ({pos[0]:.3f}, {pos[1]:.3f}) m")
        for c, pos in scene["bins"].items():
            if pos is not None:
                ok(f"  bin/{c:10s}:   ({pos[0]:.3f}, {pos[1]:.3f}) m")

        # 保存调试图
        try:
            import cv2
            debug = camera.get_debug_image()
            if debug is not None:
                path = "/tmp/vlm_perception_debug.jpg"
                cv2.imwrite(path, debug)
                ok(f"调试图像保存到: {path}")
                info("请打开该图像检查：绿色=方块，蓝色=桶，黄线=分区线")
        except Exception:
            pass

        if n_blocks == 0 and n_bins == 0:
            warn("什么都没检测到，可能原因：")
            warn("  1. 桌面没有放物体")
            warn("  2. 图像质量不佳（光线不足/模糊）")
            warn("  3. VLM 模型没有视觉能力（需要带 -vl 或 -vision 的模型）")

        return True

    except Exception as e:
        fail(f"VLM 感知测试失败: {e}")
        traceback.print_exc()
        return False


# ── Test 7: LLM API ───────────────────────────────────────────────────────────
def test_7_llm_api():
    section("Test 7: LLM API 连接（探索策略）")
    api_key = llm_api_key()
    if not api_key:
        warn("未设置 LLM_API_KEY，跳过"); return True

    model = llm_model()
    info(f"探索策略模型: {model}（.env / LLM_MODEL）")
    try:
        from llm.llm_policy import LLMExplorationPolicy
        from config.color_config import get_color_config

        cfg    = get_color_config()
        policy = LLMExplorationPolicy(
            api_key=api_key, base_url=llm_base_url(), model=model,
            epsilon=1.0, n_candidates=1, color_config=cfg)

        positions = {}
        for i, c in enumerate(cfg.colors):
            positions[f"{c}_block"] = [0.18+i*0.02, -0.1+i*0.05]
            positions[f"{c}_bin"]   = [0.35+i*0.01,  0.0+i*0.06]

        obs_dict = {
            "positions":   positions, "gripper": "open",
            "pick_color":  cfg.colors[0], "place_color": cfg.colors[1],
            "held_object": None,
        }

        t0 = time.time()
        prim, obj_idx = policy.call_high_level(obs_dict)
        ok(f"LLM API 调用成功 ({time.time()-t0:.1f}s)  "
           f"primitive={prim}  obj_idx={obj_idx}")

        N = cfg.n_colors
        if obj_idx < N:
            ok(f"  → {cfg.idx_to_color(obj_idx)}_block")
        else:
            ok(f"  → {cfg.idx_to_color(obj_idx-N)}_bin")

        if prim == 0:
            ok("返回值合理（夹爪 open → 应 pick）")
        else:
            warn(f"预期 pick，返回了 {prim}，检查 prompt")
        return True
    except Exception as e:
        fail(f"LLM API 失败: {e}")
        if "authentication" in str(e).lower(): fail("可能 API Key 错误")
        traceback.print_exc()
        return False


# ── 主函数 ────────────────────────────────────────────────────────────────────
TESTS = {
    1: ("ROS连接 & rosmaster",               test_1_ros_connection),
    2: ("Gazebo物体检测（12个）",             test_2_gazebo_objects),
    3: ("MoveIt规划组连接",                   test_3_moveit),
    4: ("颜色配置（ColorConfig）",            test_4_color_config),
    5: ("环境 reset() 和 step()",             test_5_env_reset_step),
    6: ("VLM 感知 [真机可选]",               test_6_vlm_perception),
    7: ("LLM API 连接（探索策略）",           test_7_llm_api),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test", type=int, default=None)
    args = p.parse_args()

    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"{BOLD}  ExploRLLM Sagittarius  — 环境验证测试{RESET}")
    print(f"{BOLD}  [VLM 感知升级版]{RESET}")
    print(f"{BOLD}{'═'*60}{RESET}\n")
    print("  前置条件：")
    print("  1. roslaunch sagittarius_gazebo demo_gazebo.launch")
    print("  2. 在 Gazebo 点击 ▷")
    print("  3. pick_place_scene.world 内容已加入 world 文件")
    print("  4. .env：Test6 需 VLM_*，Test7 需 LLM_*（密钥与网关均独立）")
    print("  5. LLM_MODEL / VLM_MODEL 可在 .env 中分别配置\n")
    input("  准备好后按 Enter 开始...\n")

    if args.test is not None:
        if args.test not in TESTS:
            print(f"无效编号 {args.test}，范围 1-{len(TESTS)}")
            sys.exit(1)
        name, fn = TESTS[args.test]
        result   = fn()
        print(f"\n  结果: {GREEN+'PASS'+RESET if result else RED+'FAIL'+RESET}")
        sys.exit(0 if result else 1)

    results = {}
    for num in sorted(TESTS.keys()):
        name, fn = TESTS[num]
        try:
            passed = fn()
        except KeyboardInterrupt:
            print(f"\n{YELLOW}中断{RESET}"); break
        except Exception as e:
            print(f"\n{RED}Test {num} 异常: {e}{RESET}"); passed = False
        results[num] = (name, passed)
        if not passed and num <= 3:
            print(f"\n{RED}{BOLD}基础测试失败，停止。{RESET}"); break
        time.sleep(0.8)

    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"{BOLD}  结果汇总{RESET}")
    print(f"{BOLD}{'═'*60}{RESET}")
    for num, (name, passed) in results.items():
        s = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"  Test {num}: {s}  {name}")
    print(f"{BOLD}{'═'*60}{RESET}\n")

    all_ok = all(p for _, p in results.values())
    if all_ok:
        print(f"{GREEN}{BOLD}全部通过！{RESET}")
        print("  训练（Gazebo GT感知，不需要VLM Key）：")
        print("    python train.py --epsilon 0.0")
        print("    python train.py --epsilon 0.2 --api-key sk-xxx")
        print("  真机评估（VLM感知）：")
        print("    python eval.py --model-path logs/.../final_model.zip \\")
        print("                   --real-robot  # VLM 用 .env 中 VLM_* 或命令行传参")
    else:
        failed = [n for n,(_, p) in results.items() if not p]
        print(f"{RED}失败: {failed}{RESET}")
    print()


if __name__ == "__main__":
    main()
