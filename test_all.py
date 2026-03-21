#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_all.py  (v3 — VLM 感知升级版)
=====================================
变化：
  - Test 5：A reset 体检 → B 奖励方向（正确/错误 pick 间释放持块）→ D 理想分阶段 → C 闭环；可加 --test5-auto
  - Test 6 升级为 VLM 感知测试（替代 HSV 分区检测）
  - Test 7 保持 LLM API 连接测试
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
        mg_kw = env_config.moveit_move_group_commander_kwargs()
        if mg_kw:
            arm = moveit_commander.MoveGroupCommander("sagittarius_arm", **mg_kw)
            grip = moveit_commander.MoveGroupCommander("sagittarius_gripper", **mg_kw)
        else:
            arm = moveit_commander.MoveGroupCommander("sagittarius_arm")
            grip = moveit_commander.MoveGroupCommander("sagittarius_gripper")
        ns = env_config.moveit_commander_ns()
        info(
            f"MoveIt ns={repr(ns) if ns else '/ (root)'}  "
            f"robot_description={env_config.moveit_robot_description_param()}"
        )
        ok(f"sagittarius_arm  参考坐标系: {arm.get_planning_frame()}")
        ok(f"  末端执行器: {arm.get_end_effector_link()}")
        pose = arm.get_current_pose()
        ok(f"  末端: x={pose.pose.position.x:.3f}, "
           f"y={pose.pose.position.y:.3f}, z={pose.pose.position.z:.3f}")
        _ = grip  # 连接成功即可
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


def _test5_pause(interactive: bool, msg: str) -> None:
    """交互模式下在终端按 Enter 再进入下一步；--test5-auto 时跳过。"""
    if not interactive:
        return
    print(f"\n{BLUE}{'─'*56}{RESET}")
    print(f"  {YELLOW}{msg}{RESET}")
    input(f"  {BOLD}按 Enter 继续…{RESET} ")


def _test5_release_arm_after_pick(env):
    """
    B 段「错误 pick_block」测试前：若上一步 pick_and_place 已持块（middle，无 PlanningScene attach）
    或抓取成功但未入桶，直接再 env.step 可能碰撞。开爪并回 home，恢复无持块状态。
    """
    env._holding_color = None
    try:
        env._open_gripper()
    except Exception:
        pass
    time.sleep(0.35)
    try:
        env._return_home()
    except Exception:
        pass
    time.sleep(0.5)


def _test5_run_ideal_phases(env, ppe_mod, interactive: bool) -> bool:
    """
    与 pick_place_env._execute_pick_and_place 内部运动分解等价的理想分阶段流程
    （单一 pick_and_place 原语；内含 INTER_ROBOT_STAGE_PAUSE_S 间隔）：
    开爪 → 方块上方 → 预降 → 抓取高度 → 夹持(middle) → 垂直抬至 APPROACH_H → 同高平移至桶上方
    → 开爪释放 → 等待 PLACE_DROP_SETTLE_S（不 attach / 不下放到 PLACE_H）
    使用 Gazebo GT；每步前可暂停观察。

    调用方应在进入 D 段前已执行 env.reset()（Test5 在 B 段结束后会 reset）。
    """
    import math

    env._refresh_positions()
    pick_c = env.pick_color
    place_c = env.place_color
    tz = float(ppe_mod.TABLE_Z)
    info(f"\n{BOLD}── D：理想分阶段操控（与 env 原语一致）──{RESET}")
    info(f"  任务 pick={pick_c} → place={place_c}")

    pb = env._get_pose(ppe_mod.block_name(pick_c))
    bx, by = float(pb[0]), float(pb[1])
    gx, gy = env._grasp_tcp_xy_from_block_center(bx, by)
    yaw = math.atan2(gy - ppe_mod.ARM_BASE_Y, gx - ppe_mod.ARM_BASE_X)
    pre_z = tz + ppe_mod.BLOCK_H + ppe_mod.PRE_GRASP_CLEAR_Z
    lift_verify_z = float(ppe_mod.PICK_LIFT_VERIFY_Z)

    def _m(label: str, fn) -> bool:
        _test5_pause(interactive, label)
        ok_flag = fn()
        if ok_flag:
            ok(f"  {label} → OK")
        else:
            warn(f"  {label} → 失败")
        return ok_flag

    def _do_open():
        env._open_gripper()
        return True

    def _do_close():
        env._close_gripper()
        return True

    if not _m("D-1 开爪", _do_open):
        return False

    if not _m(
        f"D-2 移至方块上方安全高度 z={tz + ppe_mod.APPROACH_H:.3f}",
        lambda: env._move_to_xy(
            gx, gy, tz + ppe_mod.APPROACH_H, yaw,
            ignore_block_color=pick_c, orientation_mode="horizontal",
        ),
    ):
        return False
    if not _m(
        f"D-3 预降至方块顶面上方 z={pre_z:.3f}",
        lambda: env._move_to_xy(
            gx, gy, pre_z, yaw,
            ignore_block_color=pick_c, orientation_mode="horizontal",
        ),
    ):
        return False
    if not _m(
        f"D-4 落至抓取高度 z={tz + ppe_mod.GRASP_H:.3f}",
        lambda: env._move_to_xy(
            gx, gy, tz + ppe_mod.GRASP_H, yaw,
            ignore_block_color=pick_c, orientation_mode="horizontal",
        ),
    ):
        return False
    if not _m("D-5 夹爪夹持（middle）", _do_close):
        return False
    if not _m(
        f"D-6 抬升至安全高度 z={tz + ppe_mod.APPROACH_H:.3f}",
        lambda: env._move_to_xy(
            gx, gy, tz + ppe_mod.APPROACH_H, yaw,
            ignore_block_color=pick_c, orientation_mode="horizontal",
        ),
    ):
        env._open_gripper()
        return False

    block_z = float(env._get_pose(ppe_mod.block_name(pick_c))[2])
    if block_z < lift_verify_z:
        warn(f"  D-7 抓取验证失败: block z={block_z:.4f} < {lift_verify_z:.4f}")
        env._open_gripper()
        env._holding_color = None
        return False
    env._holding_color = pick_c
    ok(f"  D-7 抓取验证通过 (z={block_z:.4f})，持块（middle，无 attach）")

    bin_xyz = env._get_pose(ppe_mod.bin_name(place_c))
    px, py = float(bin_xyz[0]), float(bin_xyz[1])
    yaw_p = math.atan2(py - ppe_mod.ARM_BASE_Y, px - ppe_mod.ARM_BASE_X)

    if not _m(
        f"D-8 移至目标桶上方同安全高度 z={tz + ppe_mod.APPROACH_H:.3f}",
        lambda: env._move_to_xy(
            px, py, tz + ppe_mod.APPROACH_H, yaw_p, orientation_mode="horizontal",
            ignore_block_color=pick_c,
        ),
    ):
        return False

    _test5_pause(interactive, "D-9 桶口上方开爪释放（与 env 原语一致）")
    env._open_gripper()
    env._holding_color = None
    ok("  D-9 已开爪")
    time.sleep(float(getattr(ppe_mod, "PLACE_DROP_SETTLE_S", 1.0)))
    ok(f"  D 段结束（已等待 {getattr(ppe_mod, 'PLACE_DROP_SETTLE_S', 1.0)}s 供方块落入桶）")
    return True


# ── Test 5: 环境 reset/step ───────────────────────────────────────────────────
def test_5_env_reset_step(interactive: bool = True):
    """
    仅 **3 回合闭环测试**：每回合 env.reset → 按任务对齐发送 7 维 pick_and_place，
    直到 done 或 max_steps 截断。

    interactive：Enter 逐步；--test5-auto 连续运行。

    （原 A/B/D 分阶段与奖励对比已移除；若需逐步拆 MoveIt，仍可用本文件中的
    `_test5_run_ideal_phases` 在交互式会话里单独调用。）
    """
    section("Test 5: 闭环（3 回合 pick_and_place）")
    try:
        from envs.pick_place_env import SagittariusPickPlaceEnv
        from config.color_config import get_color_config
        import numpy as np

        cfg = get_color_config()
        env = SagittariusPickPlaceEnv(task="short_horizon", max_steps=8)
        na = env.n_active
        ok("环境创建成功")
        ok(f"  obs_dim={env.obs_dim}  n_active={na}  action_dim=7 (pick_and_place)")

        info("每步：任务对齐 pick_idx + place_idx，单步 pick_and_place 直至 done/trunc。")
        closed_loop_ok = 0
        closed_loop_trials = 3
        pose_id_compat = 0.0

        for k in range(closed_loop_trials):
            _test5_pause(
                interactive,
                f"第 {k+1}/{closed_loop_trials} 回合（将 env.reset）",
            )
            obs, info_dict = env.reset()
            if obs.shape[0] != env.obs_dim:
                fail(f"obs 维度 {obs.shape[0]} ≠ {env.obs_dim}")
                env.close()
                return False

            pick_idx = env._active_block_colors.index(
                info_dict.get("pick_color"))
            place_idx = env._active_bin_colors.index(
                info_dict.get("place_color"))
            info(f"  任务 pick={info_dict.get('pick_color')} → "
                 f"place={info_dict.get('place_color')}  "
                 f"idx=({pick_idx},{place_idx})")

            ep_done = False
            ep_trunc = False
            ep_steps = 0
            while not (ep_done or ep_trunc):
                ep_steps += 1
                action = np.array(
                    [
                        float(pick_idx),
                        float(place_idx),
                        pose_id_compat,
                        0.0, 0.0, 0.0, 0.0,
                    ],
                    dtype=np.float32,
                )
                _test5_pause(
                    interactive,
                    f"回合{k+1} step{ep_steps}  env.step(pick_and_place)",
                )
                t_step = time.time()
                _, r_ep, ep_done, ep_trunc, info_ep = env.step(action)
                info(
                    f"    pick_xy={info_ep.get('pick_xy')}  "
                    f"place_xy={info_ep.get('place_xy')}  "
                    f"metrics={info_ep.get('metrics')}"
                )
                info(
                    f"  回合{k+1} step{ep_steps:02d}  "
                    f"success={info_ep.get('success')}  "
                    f"done={ep_done} trunc={ep_trunc}  "
                    f"r={r_ep:.3f} ({time.time()-t_step:.1f}s)"
                )

            if ep_done:
                closed_loop_ok += 1
                ok(f"  回合{k+1} 完成（done=True, steps={ep_steps}）")
            else:
                warn(f"  回合{k+1} 截断（trunc=True, steps={ep_steps}）")

        if closed_loop_ok > 0:
            ok(f"闭环：{closed_loop_ok}/{closed_loop_trials} 回合完成任务")
        else:
            warn("闭环：0 回合 done。若 pick_ok=True 但 place 失败，检查 PlanningScene "
                 "持块–桶碰撞（见 pick_place_env._move_to_xy 内 sync 后 allow_collisions）。")

        return True
    except Exception as e:
        fail(f"失败: {e}"); traceback.print_exc(); return False
    finally:
        try:
            if "env" in locals():
                env.close()
        except Exception:
            pass


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
        from envs.pick_place_env import SLOT_COUNT
        na = SLOT_COUNT
        if cfg.n_colors < 2:
            warn("ColorConfig 颜色数 < 2，跳过 LLM 物体索引语义检查")
            return True
        bc = list(cfg.colors[:2])
        bn = list(cfg.colors[:2])

        policy = LLMExplorationPolicy(
            api_key=api_key, base_url=llm_base_url(), model=model,
            epsilon=1.0, n_candidates=1, color_config=cfg,
            n_active=na)

        positions = {}
        for i, c in enumerate(bc):
            positions[f"{c}_block"] = [0.18 + i * 0.02, -0.1 + i * 0.05]
        for i, c in enumerate(bn):
            positions[f"{c}_bin"] = [0.35 + i * 0.01, 0.0 + i * 0.06]

        obs_dict = {
            "positions":   positions, "gripper": "open",
            "pick_color":  bc[0], "place_color": bn[1],
            "held_object": None,
            "active_colors": sorted(set(bc) | set(bn)),
            "n_active":    na,
            "n_blocks":    len(bc),
            "n_bins":      len(bn),
            "active_block_colors": bc,
            "active_bin_colors":   bn,
        }

        t0 = time.time()
        pbi, pli = policy.call_high_level(obs_dict)
        ok(f"LLM API 调用成功 ({time.time()-t0:.1f}s)  "
            f"pick_block_index={pbi}  place_bin_index={pli}")

        if 0 <= pbi < len(bc):
            ok(f"  pick → {bc[pbi]}_block")
        if 0 <= pli < len(bn):
            ok(f"  place → {bn[pli]}_bin")
        ok("返回值应同时包含 pick_block_index 与 place_bin_index（单原语 pick_and_place）")
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
    p.add_argument(
        "--test5-auto",
        action="store_true",
        help="仅 Test5：不暂停，连续跑完（跑全部测试时请加上，避免每步按 Enter）",
    )
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
        if args.test == 5:
            result = test_5_env_reset_step(interactive=not args.test5_auto)
        else:
            result = fn()
        print(f"\n  结果: {GREEN+'PASS'+RESET if result else RED+'FAIL'+RESET}")
        sys.exit(0 if result else 1)

    results = {}
    for num in sorted(TESTS.keys()):
        name, fn = TESTS[num]
        try:
            if num == 5:
                passed = test_5_env_reset_step(interactive=False)
            else:
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
