#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_all.py
===========

每个测试独立运行，失败时告诉你具体是哪里出问题。
不需要API Key，不需要开始训练，只是验证环境是否搭建正确。

使用方法：
    # 全部测试按顺序跑
    python test_all.py

    # 只跑某一项
    python test_all.py --test 1   # 只测ROS连接
    python test_all.py --test 2   # 只测Gazebo物体
    python test_all.py --test 3   # 只测MoveIt
    python test_all.py --test 4   # 只测Env的reset和step
    python test_all.py --test 5   # 只测observation的形状和内容
    python test_all.py --test 6   # 只测LLM API连接（需要API Key）

前置条件（运行测试前先在另一个终端执行）：
    roslaunch sagittarius_gazebo demo_gazebo.launch
    # 然后在Gazebo界面点击 ▷ 开始仿真
"""

import sys
import os
import time
import argparse
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ── ANSI颜色 ──────────────────────────────────────────────────────────────────
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
    print(f"\n{BOLD}{'─'*55}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─'*55}{RESET}")


# ════════════════════════════════════════════════════════════════════════
#  TEST 1: ROS连接测试
#  测试内容：rospy能否初始化、rosmaster是否在运行
#  如果失败：说明你没有运行 roslaunch，或者ROS环境没source
# ════════════════════════════════════════════════════════════════════════

def test_1_ros_connection():
    section("Test 1: ROS连接 & rosmaster")

    info("正在尝试连接rosmaster...")

    try:
        import rospy
    except ImportError:
        fail("rospy导入失败。请确认已执行：source ~/sagittarius_ws/devel/setup.bash")
        return False

    try:
        import subprocess
        result = subprocess.run(
            ["rostopic", "list"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            fail("rostopic list失败。rosmaster可能没有运行。")
            fail("请先在另一个终端运行：roslaunch sagittarius_gazebo demo_gazebo.launch")
            return False
        ok("rosmaster正在运行")

        topics = result.stdout.strip().split("\n")
        info(f"当前活跃的话题数量：{len(topics)}")

    except subprocess.TimeoutExpired:
        fail("rostopic list超时。rosmaster没有响应。")
        return False
    except FileNotFoundError:
        fail("找不到rostopic命令。ROS环境是否已source？")
        return False

    try:
        rospy.init_node("explorllm_test", anonymous=True, disable_signals=True)
        ok("rospy节点初始化成功")
    except Exception as e:
        fail(f"rospy初始化失败：{e}")
        return False

    return True


# ════════════════════════════════════════════════════════════════════════
#  TEST 2: Gazebo物体检测测试
#  测试内容：/gazebo/model_states话题里是否有red_block等6个物体
#  如果失败：说明.world文件里没有加入物体模型，或Gazebo没有启动
# ════════════════════════════════════════════════════════════════════════

def test_2_gazebo_objects():
    section("Test 2: Gazebo物体检测")

    REQUIRED_OBJECTS = [
        "red_block", "green_block", "blue_block",
        "red_bowl",  "green_bowl",  "blue_bowl",
    ]

    info("订阅 /gazebo/model_states 话题，等待3秒...")

    try:
        import rospy
        from gazebo_msgs.msg import ModelStates

        received = {"data": None}

        def cb(msg):
            received["data"] = msg

        rospy.Subscriber("/gazebo/model_states", ModelStates, cb, queue_size=1)
        time.sleep(3.0)

        if received["data"] is None:
            fail("/gazebo/model_states 没有收到任何消息。")
            fail("请确认Gazebo已启动并点击了▷开始仿真。")
            return False

        model_names = received["data"].name
        ok(f"收到model_states，共有 {len(model_names)} 个模型")
        info(f"所有模型名称：{model_names}")

        all_found = True
        for obj in REQUIRED_OBJECTS:
            if obj in model_names:
                ok(f"找到：{obj}")
            else:
                fail(f"缺少：{obj}  ← 需要在.world文件里添加这个模型")
                all_found = False

        if not all_found:
            warn("请将 pick_place_scene.world 的内容加入你的Gazebo world文件。")
            warn("具体操作见 README.md 里的'HOW TO USE'说明。")

        return all_found

    except Exception as e:
        fail(f"测试失败：{e}")
        traceback.print_exc()
        return False


# ════════════════════════════════════════════════════════════════════════
#  TEST 3: MoveIt连接测试
#  测试内容：能否连接MoveIt，sagittarius_arm规划组是否可用
#  如果失败：说明MoveIt没有启动（demo_gazebo.launch应该会自动启动）
# ════════════════════════════════════════════════════════════════════════

def test_3_moveit():
    section("Test 3: MoveIt规划组连接")

    info("尝试连接MoveIt规划组 sagittarius_arm ...")

    try:
        import moveit_commander
        moveit_commander.roscpp_initialize(sys.argv)

        arm = moveit_commander.MoveGroupCommander("sagittarius_arm")
        ok(f"sagittarius_arm 连接成功")
        ok(f"  规划参考坐标系：{arm.get_planning_frame()}")
        ok(f"  末端执行器：{arm.get_end_effector_link()}")
        ok(f"  当前关节角（度）：{[round(j*57.3,1) for j in arm.get_current_joint_values()]}")

        gripper = moveit_commander.MoveGroupCommander("sagittarius_gripper")
        ok(f"sagittarius_gripper 连接成功")

        # 测试能否获取当前位姿（不执行运动）
        pose = arm.get_current_pose()
        ok(f"  当前末端位置：x={pose.pose.position.x:.3f}, "
           f"y={pose.pose.position.y:.3f}, z={pose.pose.position.z:.3f}")

        return True

    except Exception as e:
        fail(f"MoveIt连接失败：{e}")
        warn("请确认 demo_gazebo.launch 已完整启动（包括MoveIt节点）。")
        warn("等待MoveIt完全加载可能需要10-20秒，请稍后再试。")
        traceback.print_exc()
        return False


# ════════════════════════════════════════════════════════════════════════
#  TEST 4: 环境reset和step测试
#  测试内容：
#    - env.reset() 能否正常返回observation
#    - env.step(action) 能否执行并返回 (obs, reward, done, truncated, info)
#    - 机械臂能否执行一个随机动作（pick原语）
#  如果失败：
#    - 如果reset失败：通常是Gazebo物体或MoveIt问题
#    - 如果step失败：通常是MoveIt规划失败（正常现象，会返回False但不会崩溃）
# ════════════════════════════════════════════════════════════════════════

def test_4_env_reset_step():
    section("Test 4: 环境 reset() 和 step()")

    info("初始化 SagittariusPickPlaceEnv ...")

    try:
        from envs.pick_place_env import SagittariusPickPlaceEnv
        import numpy as np

        env = SagittariusPickPlaceEnv(
            task="short_horizon",
            max_steps=3,          # 只测3步，不用跑完整episode
        )
        ok("环境对象创建成功")

        # ── 测试 reset() ──────────────────────────────────────────────
        info("调用 env.reset() ...")
        t0 = time.time()
        obs, info_dict = env.reset()
        elapsed = time.time() - t0

        ok(f"reset() 成功，耗时 {elapsed:.1f}s")
        ok(f"  observation shape : {obs.shape}")
        ok(f"  observation dtype : {obs.dtype}")
        ok(f"  pick_color        : {info_dict.get('pick_color')}")
        ok(f"  place_color       : {info_dict.get('place_color')}")

        # 检查observation维度
        expected_dim = 14131  # IMG_DIM + POS_DIM + GRIP_DIM + LANG_DIM
        if obs.shape == (expected_dim,):
            ok(f"  observation维度正确：({expected_dim},)")
        else:
            warn(f"  observation维度为 {obs.shape}，预期 ({expected_dim},)")
            warn("  如果你修改了N_TOTAL或CROP_SIZE，需要同步更新custom_sac.py里的常量。")

        # 检查observation数值是否合理（不是全0或NaN）
        import numpy as np
        pos_start = 14112   # IMG_DIM
        pos_data  = obs[pos_start : pos_start + 12].reshape(6, 2)
        info(f"  物体位置数据（前3个）：{pos_data[:3].tolist()}")

        if np.all(pos_data == 0):
            warn("  所有位置都是0。Gazebo物体可能没有正确加载。")
        elif np.any(np.isnan(pos_data)):
            fail("  位置数据包含NaN！请检查Gazebo连接。")
            return False
        else:
            ok("  位置数据看起来正常")

        # ── 测试 step() ──────────────────────────────────────────────
        info("\n调用 env.step(action) ...")
        info("  这会控制机械臂执行一次pick动作，耗时约10-20秒...")

        # 构造一个pick动作：primitive=0(pick), obj=0(red_block), res=(0,0)
        action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        t0 = time.time()
        obs2, reward, done, truncated, info2 = env.step(action)
        elapsed = time.time() - t0

        ok(f"step() 返回，耗时 {elapsed:.1f}s")
        ok(f"  reward     : {reward:.4f}")
        ok(f"  done       : {done}")
        ok(f"  truncated  : {truncated}")
        ok(f"  success    : {info2.get('success')}  "
           f"(False=规划失败，正常现象)")
        ok(f"  primitive  : {info2.get('primitive')}  (0=pick)")
        ok(f"  obj_idx    : {info2.get('obj_idx')}")
        ok(f"  target_xy  : {info2.get('target_xy')}")

        if info2.get('success') is False:
            warn("  MoveIt规划失败。原因可能是：")
            warn("  1. 目标位置超出机械臂工作空间")
            warn("  2. MoveIt还未完全初始化（稍等再试）")
            warn("  3. 积木位置不在TABLE_X/Y的范围内（检查.world文件坐标）")
            warn("  这不是致命错误，训练时会自动重试。")

        env.close()
        return True

    except Exception as e:
        fail(f"环境测试失败：{e}")
        traceback.print_exc()
        return False


# ════════════════════════════════════════════════════════════════════════
#  TEST 5: Observation内容详细检查
#  测试内容：
#    - 把observation的各个部分拆开来检查
#    - 确认图像crops、位置、夹爪状态、语言目标的格式正确
#    - 不需要机械臂运动，只检查数据格式
#  如果失败：通常是数组切片索引问题，需要检查pick_place_env.py的常量
# ════════════════════════════════════════════════════════════════════════

def test_5_observation_structure():
    section("Test 5: Observation结构详细检查")

    try:
        from envs.pick_place_env import (
            SagittariusPickPlaceEnv, IMG_DIM, POS_DIM,
            GRIP_DIM, LANG_DIM, N_TOTAL, CROP_SIZE
        )
        import numpy as np

        env = SagittariusPickPlaceEnv(task="short_horizon", max_steps=2)
        obs, info_dict = env.reset()

        print(f"\n  观测向量总长度: {len(obs)}")
        print(f"  组成部分：")
        print(f"    图像patches : {IMG_DIM:5d}  = {N_TOTAL}个物体 × 3通道 × {CROP_SIZE}×{CROP_SIZE}像素")
        print(f"    物体位置    :    {POS_DIM:2d}  = {N_TOTAL}个物体 × 2坐标(x,y)")
        print(f"    夹爪状态    :     {GRIP_DIM}  = 0.0(open) or 1.0(closed)")
        print(f"    语言目标    :     {LANG_DIM}  = one-hot [pick_r,pick_g,pick_b,place_r,place_g,place_b]")

        # 拆解各部分
        img_part  = obs[:IMG_DIM]
        pos_part  = obs[IMG_DIM : IMG_DIM+POS_DIM].reshape(N_TOTAL, 2)
        grip_part = obs[IMG_DIM+POS_DIM]
        lang_part = obs[IMG_DIM+POS_DIM+GRIP_DIM :]

        # 图像检查
        img_arr = img_part.reshape(N_TOTAL, 3, CROP_SIZE, CROP_SIZE)
        img_max = img_arr.max()
        img_min = img_arr.min()
        print(f"\n  图像patches:")
        print(f"    shape: {img_arr.shape}")
        print(f"    值域: [{img_min:.3f}, {img_max:.3f}]  "
              f"(应在 [0.0, 1.0] 内)")
        if img_max <= 1.0 and img_min >= 0.0:
            ok("图像数值范围正确")
        else:
            warn("图像数值超出[0,1]范围，检查pick_place_env.py里的归一化")

        # 位置检查
        from envs.pick_place_env import ALL_OBJECTS
        print(f"\n  物体位置（带噪声）:")
        for i, name in enumerate(ALL_OBJECTS):
            print(f"    {name:15s}: x={pos_part[i,0]:.4f}, y={pos_part[i,1]:.4f}")

        # 夹爪检查
        print(f"\n  夹爪状态: {grip_part:.1f}  "
              f"({'open' if grip_part < 0.5 else 'closed'})")

        # 语言目标检查
        colors = ["red", "green", "blue"]
        pick_idx  = int(np.argmax(lang_part[:3]))
        place_idx = int(np.argmax(lang_part[3:]))
        print(f"\n  语言目标one-hot: {lang_part}")
        print(f"    pick  → {colors[pick_idx]}   (与info_dict.pick_color={info_dict.get('pick_color')} 对应)")
        print(f"    place → {colors[place_idx]}  (与info_dict.place_color={info_dict.get('place_color')} 对应)")

        if colors[pick_idx] == info_dict.get('pick_color'):
            ok("语言目标编码与info_dict一致")
        else:
            fail("语言目标编码与info_dict不一致！检查_build_lang_onehot函数。")

        env.close()
        return True

    except Exception as e:
        fail(f"Observation结构检查失败：{e}")
        traceback.print_exc()
        return False


# ════════════════════════════════════════════════════════════════════════
#  TEST 6: LLM API连接测试
#  测试内容：能否调用LLM API，高层策略是否返回合理结果
#  需要API Key，如果没有API Key就跳过
# ════════════════════════════════════════════════════════════════════════

def test_6_llm_api():
    section("Test 6: LLM API连接")

    api_key = os.environ.get("LLM_API_KEY", "")
    if not api_key:
        warn("未设置 LLM_API_KEY 环境变量，跳过LLM测试。")
        warn("设置方法：export LLM_API_KEY='你的api-key'")
        return True  # 不算失败

    model = os.environ.get("LLM_MODEL", "deepseek-v3")
    info(f"使用模型：{model}")
    info("调用高层LLM策略（πH）...")

    try:
        from llm.llm_policy import LLMExplorationPolicy

        policy = LLMExplorationPolicy(
            api_key=api_key,
            model=model,
            epsilon=1.0,   # 测试时强制使用LLM
            n_candidates=1,
        )

        # 构造一个测试场景
        obs_dict = {
            "positions": {
                "red_block":   [0.25, -0.05],
                "green_block": [0.28,  0.08],
                "blue_block":  [0.22,  0.12],
                "red_bowl":    [0.35, -0.15],
                "green_bowl":  [0.35,  0.00],
                "blue_bowl":   [0.35,  0.15],
            },
            "gripper":     "open",
            "pick_color":  "red",
            "place_color": "blue",
            "held_object": None,
        }

        import numpy as np
        import time
        t0 = time.time()
        primitive, obj_idx = policy.call_high_level(obs_dict)
        elapsed = time.time() - t0

        ok(f"LLM API调用成功，耗时 {elapsed:.1f}s")
        ok(f"  返回：primitive={primitive}（0=pick,1=place），obj_idx={obj_idx}")

        # 检查返回值合理性
        if primitive == 0 and 0 <= obj_idx <= 2:
            ok("  返回值合理：应该pick（夹爪是open且没有holding）")
        elif primitive == 0:
            warn(f"  primitive=pick正确，但obj_idx={obj_idx}超出block范围[0,2]")
        else:
            warn(f"  预期primitive=0(pick)，但LLM返回了{primitive}。检查prompt。")

        return True

    except Exception as e:
        fail(f"LLM API测试失败：{e}")
        if "authentication" in str(e).lower() or "api" in str(e).lower():
            fail("可能是API Key错误，或base_url不正确。")
        traceback.print_exc()
        return False


# ════════════════════════════════════════════════════════════════════════
#  主函数
# ════════════════════════════════════════════════════════════════════════

TESTS = {
    1: ("ROS连接 & rosmaster",          test_1_ros_connection),
    2: ("Gazebo物体检测",                test_2_gazebo_objects),
    3: ("MoveIt规划组连接",              test_3_moveit),
    4: ("环境 reset() 和 step()",        test_4_env_reset_step),
    5: ("Observation结构详细检查",       test_5_observation_structure),
    6: ("LLM API连接",                   test_6_llm_api),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, default=None,
                        help="只运行指定编号的测试（1-6），不指定则全部运行")
    args = parser.parse_args()

    print(f"\n{BOLD}{'═'*55}{RESET}")
    print(f"{BOLD}  ExploRLLM Sagittarius — 环境验证测试{RESET}")
    print(f"{BOLD}{'═'*55}{RESET}")
    print()
    print("  前置条件：")
    print("  1. 在另一个终端运行：")
    print("     $ cd ~/sagittarius_ws && source devel/setup.bash")
    print("     $ roslaunch sagittarius_gazebo demo_gazebo.launch")
    print("  2. 在Gazebo界面里点击 ▷ 开始仿真")
    print("  3. 确认 pick_place_scene.world 的内容已加入 Gazebo world 文件")
    print()
    input("  准备好后按 Enter 开始测试...")

    if args.test is not None:
        # 只跑指定测试
        if args.test not in TESTS:
            print(f"无效测试编号：{args.test}，有效范围 1-{len(TESTS)}")
            sys.exit(1)
        name, fn = TESTS[args.test]
        result = fn()
        status = f"{GREEN}PASS{RESET}" if result else f"{RED}FAIL{RESET}"
        print(f"\n  结果：{status}")
        sys.exit(0 if result else 1)

    # 全部测试
    results = {}
    for num in sorted(TESTS.keys()):
        name, fn = TESTS[num]
        try:
            passed = fn()
        except KeyboardInterrupt:
            print(f"\n{YELLOW}[中断] 测试被用户终止{RESET}")
            break
        except Exception as e:
            print(f"\n{RED}[错误] 测试 {num} 发生未捕获异常：{e}{RESET}")
            passed = False

        results[num] = (name, passed)

        if not passed and num <= 3:
            print(f"\n{RED}{BOLD}基础测试失败，停止后续测试。{RESET}")
            print("  请先修复上面的问题再继续。")
            break

        time.sleep(1.0)

    # 总结
    print(f"\n\n{BOLD}{'═'*55}{RESET}")
    print(f"{BOLD}  测试结果汇总{RESET}")
    print(f"{BOLD}{'═'*55}{RESET}")
    for num, (name, passed) in results.items():
        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"  Test {num}: {status}  {name}")
    print(f"{BOLD}{'═'*55}{RESET}\n")

    all_passed = all(p for _, p in results.values())
    if all_passed:
        print(f"{GREEN}{BOLD}所有测试通过！可以开始训练：{RESET}")
        print("  python train.py --epsilon 0.0            # 纯SAC测试训练")
        print("  python train.py --epsilon 0.2 --api-key sk-xxx  # 完整ExploRLLM")
    else:
        failed = [num for num, (_, p) in results.items() if not p]
        print(f"{RED}{BOLD}以下测试失败：{failed}{RESET}")
        print("  请根据每个测试的提示修复问题后重新运行。")
    print()


if __name__ == "__main__":
    main()
