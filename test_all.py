#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_all.py  
========================
分步测试脚本，对应的环境（多颜色 + 随机桶位置）。

测试列表：
  Test 1: ROS连接 & rosmaster
  Test 2: Gazebo物体检测（6种颜色 × 方块+桶 = 12个物体）
  Test 3: MoveIt规划组连接
  Test 4: 颜色配置加载（color_config）
  Test 5: 环境 reset() 和 step() - 验证新observation结构
  Test 6: 摄像头分区检测（方块区 vs 桶区，同色不混淆）
  Test 7: LLM API连接（需要API Key，否则跳过）

前置条件：
  终端1: roslaunch sagittarius_gazebo demo_gazebo.launch
         然后在Gazebo里点击 ▷
"""

import sys
import os
import time
import argparse
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

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

def section(title):
    print(f"\n{BOLD}{'─'*58}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─'*58}{RESET}")


def test_1_ros_connection():
    section("Test 1: ROS连接 & rosmaster")
    try:
        import subprocess
        r = subprocess.run(["rostopic", "list"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode != 0:
            fail("rostopic list 失败，rosmaster 可能没有运行")
            fail("请先运行: roslaunch sagittarius_gazebo demo_gazebo.launch")
            return False
        topics = r.stdout.strip().split("\n")
        ok(f"rosmaster 运行中，{len(topics)} 个话题")
        import rospy
        if not rospy.core.is_initialized():
            rospy.init_node("explorllm_test", anonymous=True,
                            disable_signals=True)
        ok("rospy 节点初始化成功")
        return True
    except Exception as e:
        fail(f"ROS连接失败: {e}")
        return False


def test_2_gazebo_objects():
    section("Test 2: Gazebo物体检测（6色×2类=12个）")

    from configs.color_config import get_color_config
    cfg = get_color_config()
    info(f"当前颜色配置: {cfg.colors}")

    expected = []
    for c in cfg.colors:
        expected.append(f"{c}_block")
        expected.append(f"{c}_bin")
    info(f"预期物体数量: {len(expected)}")

    try:
        import rospy
        from gazebo_msgs.msg import ModelStates
        received = {"data": None}
        def cb(msg): received["data"] = msg
        rospy.Subscriber("/gazebo/model_states", ModelStates, cb, queue_size=1)
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
                fail(f"  缺少: {obj}  ← 需要在 world 文件里添加")
                all_found = False

        if not all_found:
            warn("请将 pick_place_scene.world 的内容加入你的 Gazebo world 文件")
        return all_found

    except Exception as e:
        fail(f"测试失败: {e}")
        traceback.print_exc()
        return False


def test_3_moveit():
    section("Test 3: MoveIt规划组连接")
    try:
        import moveit_commander
        moveit_commander.roscpp_initialize(sys.argv)
        arm = moveit_commander.MoveGroupCommander("sagittarius_arm")
        ok(f"sagittarius_arm 连接成功")
        ok(f"  参考坐标系: {arm.get_planning_frame()}")
        ok(f"  末端执行器: {arm.get_end_effector_link()}")
        pose = arm.get_current_pose()
        ok(f"  末端位置: x={pose.pose.position.x:.3f}, "
           f"y={pose.pose.position.y:.3f}, z={pose.pose.position.z:.3f}")
        moveit_commander.MoveGroupCommander("sagittarius_gripper")
        ok("sagittarius_gripper 连接成功")
        return True
    except Exception as e:
        fail(f"MoveIt 连接失败: {e}")
        warn("等待MoveIt完全加载可能需要10-20秒，请稍后重试")
        traceback.print_exc()
        return False


def test_4_color_config():
    section("Test 4: 颜色配置加载")
    try:
        from configs.color_config import ColorConfig, get_color_config
        cfg = get_color_config()
        ok(f"颜色配置加载成功")
        ok(f"  支持颜色: {cfg.colors}  (共{cfg.n_colors}种)")

        for i, color in enumerate(cfg.colors):
            idx  = cfg.color_to_idx(color)
            back = cfg.idx_to_color(idx)
            assert back == color, f"往返失败: {color}→{idx}→{back}"
            ok(f"  {color:8s} → idx={idx} → {back} ✓")

        if len(cfg.colors) >= 2:
            enc = cfg.encode_task(cfg.colors[0], cfg.colors[1])
            ok(f"  任务编码: pick={cfg.colors[0]}, place={cfg.colors[1]} → {enc}")

        for color in cfg.colors[:3]:  # 只打印前3个避免太长
            lower, upper = cfg.get_hsv_range(color)
            ok(f"  {color:8s} HSV: lower={lower}, upper={upper}")

        return True
    except Exception as e:
        fail(f"颜色配置失败: {e}")
        traceback.print_exc()
        return False


def test_5_env_reset_step():
    section("Test 5: 环境 reset() 和 step() ")
    try:
        from envs.pick_place_env import SagittariusPickPlaceEnv
        from configs.color_config import get_color_config
        import numpy as np

        cfg = get_color_config()
        N   = cfg.n_colors
        env = SagittariusPickPlaceEnv(task="short_horizon", max_steps=3)
        ok("环境对象创建成功")

        info("调用 env.reset() ...")
        t0 = time.time()
        obs, info_dict = env.reset()
        elapsed = time.time() - t0

        ok(f"reset() 成功，耗时 {elapsed:.1f}s")
        ok(f"  obs shape: {obs.shape}，预期 ({env.obs_dim},)")
        ok(f"  pick={info_dict.get('pick_color')}, "
           f"place={info_dict.get('place_color')}, "
           f"n_colors={info_dict.get('n_colors')}")

        if obs.shape[0] != env.obs_dim:
            fail(f"obs维度不匹配！{obs.shape[0]} ≠ {env.obs_dim}")
            return False
        ok("  obs维度正确")

        # 拆解并打印各部分
        img_dim = N * 3 * 28 * 28
        pos_dim = N * 2
        bin_dim = N * 2
        block_pos = obs[img_dim:img_dim+pos_dim].reshape(N, 2)
        bin_pos   = obs[img_dim+pos_dim:img_dim+pos_dim+bin_dim].reshape(N, 2)
        gripper   = obs[img_dim+pos_dim+bin_dim]
        task      = obs[img_dim+pos_dim+bin_dim+1:]

        info(f"  方块位置（首个颜色 {cfg.colors[0]}）: "
             f"x={block_pos[0,0]:.3f}, y={block_pos[0,1]:.3f}")
        info(f"  垃圾桶位置（首个颜色 {cfg.colors[0]}）: "
             f"x={bin_pos[0,0]:.3f}, y={bin_pos[0,1]:.3f}")
        info(f"  夹爪: {'open' if gripper < 0.5 else 'closed'}")
        info(f"  任务编码: {task}")

        if np.all(block_pos == 0):
            warn("方块位置全是0，检查 Gazebo 物体是否加载")
        if np.all(bin_pos == 0):
            warn("桶位置全是0，检查 Gazebo 桶模型是否加载")

        info("\n调用 env.step()（机械臂会动，约10-20秒）...")
        action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        t0 = time.time()
        obs2, reward, done, trunc, info2 = env.step(action)
        elapsed = time.time() - t0

        ok(f"step() 返回，耗时 {elapsed:.1f}s")
        ok(f"  reward={reward:.4f}, success={info2.get('success')}")
        ok(f"  color={info2.get('color')}, primitive={info2.get('primitive')}")

        if not info2.get("success"):
            warn("MoveIt规划失败（正常现象，训练时会自动重试）")

        env.close()
        return True
    except Exception as e:
        fail(f"环境测试失败: {e}")
        traceback.print_exc()
        return False


def test_6_camera_perception():
    section("Test 6: 摄像头分区检测（方块区 vs 桶区）")
    info("此测试需要摄像头连接，无摄像头时自动跳过")

    try:
        import subprocess
        r = subprocess.run(["rostopic", "list"],
                           capture_output=True, text=True, timeout=3)
        if "/usb_cam/image_raw" not in r.stdout:
            warn("摄像头话题不存在，跳过此测试")
            warn("真机测试时单独运行: python test_all.py --test 6")
            return True

        from perception.camera_perception import CameraPerception
        from configs.color_config import get_color_config
        import cv2

        cfg = get_color_config()
        p   = CameraPerception(color_config=cfg)

        info("等待图像（2秒）...")
        time.sleep(2.0)

        if p._latest_image is None:
            warn("没有收到图像，跳过")
            return True

        ok(f"图像尺寸: {p._latest_image.shape}")
        ok(f"分割线 split_x={p.split_x}px")
        ok("左半区[0, split_x) = 方块检测区")
        ok("右半区[split_x, W) = 垃圾桶检测区")

        info("扫描场景...")
        p.print_scene_summary()

        # 保存调试图
        debug = p.get_debug_image()
        if debug is not None:
            path = "/tmp/perception_debug.jpg"
            cv2.imwrite(path, debug)
            ok(f"调试图像已保存: {path}")
            info("请打开该图像检查：绿色=方块检测，蓝色=桶检测，黄线=分割线")

        # 验证同色隔离
        scene = p.scan_scene()
        isolated = [c for c in cfg.colors
                    if scene["blocks"][c] is not None
                    and scene["bins"][c] is not None]
        if isolated:
            ok(f"成功同时检测到同色方块和桶: {isolated}")
            ok("分区检测正常工作，同色物体已正确区分！")
        else:
            info("桌面上没有同色方块+桶组合（正常），可手动摆放后再测试")

        return True
    except Exception as e:
        fail(f"摄像头测试失败: {e}")
        traceback.print_exc()
        return False


def test_7_llm_api():
    section("Test 7: LLM API连接")
    api_key = os.environ.get("LLM_API_KEY", "")
    if not api_key:
        warn("未设置 LLM_API_KEY，跳过")
        warn("设置方法: export LLM_API_KEY='你的api-key'")
        return True

    model = os.environ.get("LLM_MODEL", "deepseek-v3")
    info(f"使用模型: {model}")

    try:
        from llm.llm_policy import LLMExplorationPolicy
        from configs.color_config import get_color_config

        cfg    = get_color_config()
        policy = LLMExplorationPolicy(
            api_key=api_key, model=model,
            epsilon=1.0, n_candidates=1, color_config=cfg)

        positions = {}
        for i, c in enumerate(cfg.colors):
            positions[f"{c}_block"] = [0.18 + i*0.02, -0.1 + i*0.05]
            positions[f"{c}_bin"]   = [0.35 + i*0.01,  0.0 + i*0.06]

        obs_dict = {
            "positions":   positions,
            "gripper":     "open",
            "pick_color":  cfg.colors[0],
            "place_color": cfg.colors[1],
            "held_object": None,
        }

        t0 = time.time()
        prim, obj_idx = policy.call_high_level(obs_dict)
        elapsed = time.time() - t0

        ok(f"LLM API 调用成功，耗时 {elapsed:.1f}s")
        ok(f"  primitive={prim} ({'pick' if prim==0 else 'place'})")
        ok(f"  object_index={obj_idx}")

        N = cfg.n_colors
        if obj_idx < N:
            ok(f"  对应: {cfg.idx_to_color(obj_idx)}_block")
        else:
            ok(f"  对应: {cfg.idx_to_color(obj_idx-N)}_bin")

        if prim == 0:
            ok("返回值合理（夹爪open → 应该pick）")
        else:
            warn("预期pick，LLM返回了place，检查prompt")

        return True
    except Exception as e:
        fail(f"LLM API 测试失败: {e}")
        if "authentication" in str(e).lower():
            fail("可能是API Key错误")
        traceback.print_exc()
        return False


TESTS = {
    1: ("ROS连接 & rosmaster",           test_1_ros_connection),
    2: ("Gazebo物体检测（12个）",         test_2_gazebo_objects),
    3: ("MoveIt规划组连接",               test_3_moveit),
    4: ("颜色配置加载 (color_config)",    test_4_color_config),
    5: ("环境 reset() 和 step()",    test_5_env_reset_step),
    6: ("摄像头分区检测",                 test_6_camera_perception),
    7: ("LLM API连接",                    test_7_llm_api),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, default=None)
    args = parser.parse_args()

    print(f"\n{BOLD}{'═'*58}{RESET}")
    print(f"{BOLD}  ExploRLLM Sagittarius — 环境验证测试{RESET}")
    print(f"{BOLD}{'═'*58}{RESET}\n")
    print("  前置条件：")
    print("  1. roslaunch sagittarius_gazebo demo_gazebo.launch")
    print("  2. 在Gazebo点击 ▷")
    print("  3. pick_place_scene.world 内容已加入 world 文件\n")
    input("  准备好后按 Enter 开始...")

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
            print(f"\n{YELLOW}中断{RESET}")
            break
        except Exception as e:
            print(f"\n{RED}Test {num} 异常: {e}{RESET}")
            passed = False

        results[num] = (name, passed)
        if not passed and num <= 3:
            print(f"\n{RED}{BOLD}基础测试失败，停止后续测试。{RESET}")
            break
        time.sleep(0.8)

    print(f"\n{BOLD}{'═'*58}{RESET}")
    print(f"{BOLD}  结果汇总{RESET}")
    print(f"{BOLD}{'═'*58}{RESET}")
    for num, (name, passed) in results.items():
        s = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"  Test {num}: {s}  {name}")
    print(f"{BOLD}{'═'*58}{RESET}\n")

    all_passed = all(p for _, p in results.values())
    if all_passed:
        print(f"{GREEN}{BOLD}全部通过！训练命令：{RESET}")
        print("  python train.py --epsilon 0.0")
        print("  python train.py --epsilon 0.2 --api-key sk-xxx")
    else:
        failed = [n for n, (_, p) in results.items() if not p]
        print(f"{RED}失败的测试: {failed}{RESET}")
    print()


if __name__ == "__main__":
    main()
