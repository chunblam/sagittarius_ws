#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pick_place_env.py
====================
升级版环境，相比原版的核心变化：

  变化1：支持任意多种颜色（从color_config动态读取，不再写死3种）
  变化2：垃圾桶位置随机（每次episode随机传送到桌面不同位置）
  变化3：observation向量用颜色index代替one-hot（支持任意颜色数）
  变化4：observation包含桶的位置（因为桶不再固定，必须告诉policy桶在哪）

Observation向量新布局：
  [img_patches | block_positions | bin_positions | gripper | task_encoding]
   14112          N_COLORS*2        N_COLORS*2      1         2

  其中 task_encoding = [pick_color_idx, place_color_idx]，整数。

Action向量不变：
  [primitive(0-1), obj_index(0-2*N_COLORS-1), res_x, res_y]
  obj_index 现在覆盖：前N_COLORS个 = 方块，后N_COLORS个 = 垃圾桶
"""

import os
import sys
import time
from typing import Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import env_config  # noqa: F401 — MoveIt 命名空间 EXPLORELLM_MOVEIT_NS
import rospy
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from std_srvs.srv import Empty

import moveit_commander
from geometry_msgs.msg import PoseStamped

from config.color_config import ColorConfig, get_color_config


# ── 场景参数 ──────────────────────────────────────────────────────────────────

# 方块随机化区域（桌面左半部分）
BLOCK_ZONE_X = (0.15, 0.30)
BLOCK_ZONE_Y = (-0.18, 0.18)

# 垃圾桶随机化区域（桌面右半部分）
BIN_ZONE_X   = (0.28, 0.40)
BIN_ZONE_Y   = (-0.18, 0.18)

TABLE_Z       = 0.02   # 桌面高度
BLOCK_H       = 0.04   # 方块高度
BIN_H         = 0.09   # 垃圾桶高度（重心在 TABLE_Z + BIN_H/2）

# 运动高度
APPROACH_H    = 0.13
GRASP_H       = 0.005
PLACE_H       = 0.07

# 图像crop参数
CROP_SIZE     = 28
POSITION_NOISE_SIGMA = 0.035

# MoveIt SRDF 中的 group_state 名称（与 RViz MotionPlanning 一致，区分大小写）
MOVEIT_ARM_HOME_STATE = "home"       # 常见另有 up / sleep，勿写成 Home
MOVEIT_GRIPPER_OPEN_STATE = "open"
MOVEIT_GRIPPER_CLOSE_STATE = "close"

# Gazebo模型名称约定：{color}_block, {color}_bin
def block_name(color: str) -> str: return f"{color}_block"
def bin_name(color: str)   -> str: return f"{color}_bin"


class SagittariusPickPlaceEnv(gym.Env):
    """
    升级版 Sagittarius pick-and-place 环境。

    支持：
      - 任意多种颜色（由ColorConfig决定）
      - 方块和垃圾桶每个episode都随机位置
      - 自然语言指令：指定pick哪种颜色、place进哪种颜色的桶
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self,
                 task: str = "short_horizon",
                 max_steps: int = 10,
                 noise_sigma: float = POSITION_NOISE_SIGMA,
                 color_config: ColorConfig = None,
                 yaml_path: str = None):

        super().__init__()
        self.task        = task
        self.max_steps   = max_steps
        self.noise_sigma = noise_sigma
        self.color_cfg   = color_config or get_color_config(yaml_path)

        N = self.color_cfg.n_colors   # 颜色总数（动态）

        # 当前episode的任务颜色
        self.pick_color:  str = self.color_cfg.colors[0]
        self.place_color: str = self.color_cfg.colors[1]
        self._step_count   = 0
        self._gripper_open = True
        self._holding_color: str = None   # 当前夹着哪种颜色的方块

        # ── 观测空间 ──────────────────────────────────────────────────────
        # img: N_colors * 3 * 28 * 28
        # block_positions: N_colors * 2
        # bin_positions:   N_colors * 2
        # gripper: 1
        # task: 2 (pick_idx, place_idx，整数但存为float)
        img_dim   = N * 3 * CROP_SIZE * CROP_SIZE
        pos_dim   = N * 2   # blocks
        bin_dim   = N * 2   # bins
        grip_dim  = 1
        task_dim  = 2
        obs_dim   = img_dim + pos_dim + bin_dim + grip_dim + task_dim

        self.img_dim  = img_dim
        self.pos_dim  = pos_dim
        self.bin_dim  = bin_dim
        self.obs_dim  = obs_dim
        self.N        = N

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )

        # ── 动作空间 ──────────────────────────────────────────────────────
        # [primitive(0-1), obj_index(0 to 2N-1), res_x, res_y]
        # obj_index 0..N-1 = 方块索引（对应color_cfg.colors）
        # obj_index N..2N-1 = 桶索引
        self.action_space = spaces.Box(
            low=np.array( [0.0, 0.0, -0.05, -0.05], dtype=np.float32),
            high=np.array([1.0, float(2*N-1), 0.05, 0.05], dtype=np.float32)
        )

        # ── 内部状态 ──────────────────────────────────────────────────────
        self._ros_initialized = False
        self._moveit_arm      = None
        self._moveit_gripper  = None
        self._model_states    = None
        self._latest_image    = None
        self._bridge          = None

        # 当前episode中物体位置缓存（含噪声）
        self._block_positions: dict = {}   # color → np.array([x,y])
        self._bin_positions:   dict = {}   # color → np.array([x,y])

    # ── ROS初始化 ──────────────────────────────────────────────────────────

    def _init_ros(self):
        if self._ros_initialized:
            return

        if not rospy.core.is_initialized():
            rospy.init_node("explorllm_env", anonymous=True)

        moveit_commander.roscpp_initialize(sys.argv)
        mg_kw = env_config.moveit_move_group_commander_kwargs()
        if mg_kw:
            # 绝对参数路径 + ns：与 /sgr532/robot_description、/sgr532/move_group 一致
            self._moveit_arm = moveit_commander.MoveGroupCommander(
                "sagittarius_arm", **mg_kw)
            self._moveit_gripper = moveit_commander.MoveGroupCommander(
                "sagittarius_gripper", **mg_kw)
        else:
            self._moveit_arm = moveit_commander.MoveGroupCommander("sagittarius_arm")
            self._moveit_gripper = moveit_commander.MoveGroupCommander(
                "sagittarius_gripper")

        self._moveit_arm.set_goal_position_tolerance(0.005)
        self._moveit_arm.set_goal_orientation_tolerance(0.02)
        self._moveit_arm.set_max_velocity_scaling_factor(0.4)
        self._moveit_arm.set_max_acceleration_scaling_factor(0.4)
        self._moveit_arm.allow_replanning(True)
        self._moveit_gripper.set_goal_joint_tolerance(0.001)
        self._moveit_gripper.set_max_velocity_scaling_factor(0.5)

        rospy.Subscriber("/gazebo/model_states", ModelStates,
                         self._model_cb, queue_size=1)

        try:
            from sensor_msgs.msg import Image
            from cv_bridge import CvBridge
            self._bridge = CvBridge()
            rospy.Subscriber("/usb_cam/image_raw", Image,
                             self._img_cb, queue_size=1)
        except ImportError:
            pass

        rospy.wait_for_service("/gazebo/set_model_state", timeout=10)
        self._set_model_state = rospy.ServiceProxy(
            "/gazebo/set_model_state", SetModelState)

        try:
            rospy.wait_for_service("/gazebo/unpause_physics", timeout=5)
            rospy.ServiceProxy("/gazebo/unpause_physics", Empty)()
        except Exception:
            pass

        self._ros_initialized = True
        rospy.loginfo("[Env] ROS初始化完成。支持颜色: %s",
                      self.color_cfg.colors)

    def _model_cb(self, msg):
        self._model_states = msg

    def _img_cb(self, msg):
        if self._bridge:
            try:
                self._latest_image = self._bridge.imgmsg_to_cv2(msg, "bgr8")
            except Exception:
                pass

    # ── Gazebo物体操作 ───────────────────────────────────────────────────────

    def _get_pose(self, name: str) -> np.ndarray:
        """从model_states获取物体GT坐标(x,y,z)。"""
        if self._model_states is None:
            return np.zeros(3, dtype=np.float32)
        try:
            idx = self._model_states.name.index(name)
            p = self._model_states.pose[idx].position
            return np.array([p.x, p.y, p.z], dtype=np.float32)
        except ValueError:
            return np.zeros(3, dtype=np.float32)

    def _teleport(self, name: str, x: float, y: float, z: float):
        state = ModelState()
        state.model_name = name
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = z
        state.pose.orientation.w = 1.0
        state.reference_frame = "world"
        try:
            self._set_model_state(state)
        except Exception as e:
            rospy.logwarn(f"[Env] Teleport {name} 失败: {e}")

    def _randomize_scene(self):
        """
        随机化所有方块和垃圾桶的位置，保证同类物体互不重叠。

        修复说明（对应截图中桶大量重叠的根因）：
          1. 桶最小间距 0.10 → 0.16 m
             空心长方体外廓对角线 ≈ sqrt(0.07²+0.07²) ≈ 0.099 m；
             0.16 m 确保两桶边缘之间有约 6 cm 安全间距，不会物理穿透。
          2. 桶的 y 区域扩展到 ±0.22 m（原 ±0.18 m 只有 0.36 m，
             6 个桶需要约 6×0.16 = 0.96 m，空间根本不够）。
          3. 尝试次数 50 → 100，减少随机失败概率。
          4. 100 次仍失败时 fallback 到均匀网格摆放，
             杜绝静默跳过 teleport（静默跳过是桶堆在 world 默认位置的直接原因）。
          5. 方块间距保持 0.08 m（5 cm 方块，足够）。
        """
        rng = np.random.default_rng()

        # ── 辅助：生成均匀网格位置（fallback 用） ────────────────────────
        def _grid_positions(n: int, zone_x: tuple, zone_y: tuple,
                            gap: float) -> list:
            """在 zone 内生成最多 n 个均匀分布的网格坐标。"""
            cols = max(1, int((zone_x[1] - zone_x[0]) / gap))
            rows = max(1, int((zone_y[1] - zone_y[0]) / gap))
            xs = np.linspace(zone_x[0] + gap / 2,
                             zone_x[1] - gap / 2, min(cols, n))
            ys = np.linspace(zone_y[0] + gap / 2,
                             zone_y[1] - gap / 2,
                             max(1, int(np.ceil(n / max(len(xs), 1)))))
            pts = [(float(x), float(y)) for y in ys for x in xs]
            return pts[:n]

        # ── 随机化方块 ────────────────────────────────────────────────────
        placed_blocks = []
        BLOCK_MIN_GAP = 0.08
        for idx, color in enumerate(self.color_cfg.colors):
            name = block_name(color)
            placed = False
            for _ in range(100):
                x = rng.uniform(*BLOCK_ZONE_X)
                y = rng.uniform(*BLOCK_ZONE_Y)
                if all(np.linalg.norm([x - px, y - py]) > BLOCK_MIN_GAP
                       for px, py in placed_blocks):
                    placed_blocks.append((x, y))
                    self._teleport(name, x, y, TABLE_Z + BLOCK_H / 2)
                    placed = True
                    break

            if not placed:
                # Fallback：网格位置，不能静默跳过
                grid = _grid_positions(
                    len(self.color_cfg.colors),
                    BLOCK_ZONE_X, BLOCK_ZONE_Y, BLOCK_MIN_GAP)
                if idx < len(grid):
                    x, y = grid[idx]
                    placed_blocks.append((x, y))
                    self._teleport(name, x, y, TABLE_Z + BLOCK_H / 2)
                    rospy.logwarn(
                        f"[Env] 方块 {color} 随机摆放失败，fallback 网格 "
                        f"({x:.3f}, {y:.3f})")

        # ── 随机化垃圾桶 ──────────────────────────────────────────────────
        # 关键参数：
        #   BIN_MIN_GAP = 0.16 m（中心间距），空心长方体 0.07 m 边长，留 ~6 cm 边缘间隙
        #   BIN_ZONE_Y_WIDE = ±0.22 m，扩展 y 方向以容纳 6 个桶
        BIN_MIN_GAP    = 0.16
        BIN_ZONE_Y_WIDE = (-0.22, 0.22)

        placed_bins = []
        for idx, color in enumerate(self.color_cfg.colors):
            name = bin_name(color)
            placed = False
            for _ in range(100):
                x = rng.uniform(*BIN_ZONE_X)
                y = rng.uniform(*BIN_ZONE_Y_WIDE)
                if all(np.linalg.norm([x - px, y - py]) > BIN_MIN_GAP
                       for px, py in placed_bins):
                    placed_bins.append((x, y))
                    self._teleport(name, x, y, TABLE_Z + BIN_H / 2)
                    placed = True
                    break

            if not placed:
                # Fallback：网格位置，绝不静默跳过
                grid = _grid_positions(
                    len(self.color_cfg.colors),
                    BIN_ZONE_X, BIN_ZONE_Y_WIDE, BIN_MIN_GAP)
                if idx < len(grid):
                    x, y = grid[idx]
                    placed_bins.append((x, y))
                    self._teleport(name, x, y, TABLE_Z + BIN_H / 2)
                    rospy.logwarn(
                        f"[Env] 垃圾桶 {color} 随机摆放失败，fallback 网格 "
                        f"({x:.3f}, {y:.3f})")
                else:
                    rospy.logerr(
                        f"[Env] 垃圾桶 {color} 无法摆放！网格也满了。"
                        f"请减少颜色数量或扩大 BIN_ZONE。")

    # ── 带噪声的位置获取 ─────────────────────────────────────────────────────

    def _refresh_positions(self):
        """刷新所有物体的带噪声位置缓存（每次step调用）。"""
        noise = lambda: np.random.normal(0, self.noise_sigma, 2)
        for color in self.color_cfg.colors:
            xyz = self._get_pose(block_name(color))
            self._block_positions[color] = xyz[:2] + noise()
            xyz = self._get_pose(bin_name(color))
            self._bin_positions[color]   = xyz[:2] + noise()

    def _get_block_pos(self, color: str) -> np.ndarray:
        return self._block_positions.get(
            color, np.zeros(2, dtype=np.float32))

    def _get_bin_pos(self, color: str) -> np.ndarray:
        return self._bin_positions.get(
            color, np.zeros(2, dtype=np.float32))

    # ── Observation构建 ──────────────────────────────────────────────────────

    def _build_crops(self) -> np.ndarray:
        """
        提取每种颜色对应的图像crop。
        顺序：[color_0_block, color_1_block, ..., color_0_bin, ...]
        等等——其实observation里我们只放方块的crop，桶的位置用坐标表示。
        返回形状 (N, 3, 28, 28)
        """
        import cv2
        crops = []
        img = self._latest_image

        for color in self.color_cfg.colors:
            block_pos = self._get_block_pos(color)   # (x,y) in robot frame
            if img is not None:
                u, v = self._robot_to_pixel(block_pos)
                h, w = img.shape[:2]
                u, v = int(np.clip(u, 14, w-14)), int(np.clip(v, 14, h-14))
                crop = img[v-14:v+14, u-14:u+14]
                if crop.shape[:2] != (28, 28):
                    crop = np.zeros((28, 28, 3), dtype=np.uint8)
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop = crop.astype(np.float32) / 255.0
            else:
                crop = np.zeros((28, 28, 3), dtype=np.float32)
            crops.append(crop.transpose(2, 0, 1))

        return np.array(crops, dtype=np.float32)  # (N, 3, 28, 28)

    def _robot_to_pixel(self, xy: np.ndarray) -> Tuple[float, float]:
        """机械臂坐标 → 像素坐标（需要标定值）。"""
        from perception.camera_perception import CameraPerception
        # 用默认标定值，真机部署时会被替换
        kx, bx = -0.00029, 0.31084
        ky, by  =  0.00030, 0.09080
        v = (xy[0] - bx) / kx if abs(kx) > 1e-9 else 240
        u = (xy[1] - by) / ky if abs(ky) > 1e-9 else 320
        return float(u), float(v)

    def _build_observation(self) -> np.ndarray:
        """
        构建完整observation向量。

        新布局（相比原版增加了bin_positions和改了task编码）：
          [crops(N*3*28*28) | block_pos(N*2) | bin_pos(N*2) | gripper(1) | task(2)]
        """
        self._refresh_positions()

        crops = self._build_crops()   # (N, 3, 28, 28)

        # 方块位置
        block_pos = np.array(
            [self._get_block_pos(c) for c in self.color_cfg.colors],
            dtype=np.float32).flatten()   # (N*2,)

        # 桶位置（重要升级：现在包含在observation里）
        bin_pos = np.array(
            [self._get_bin_pos(c) for c in self.color_cfg.colors],
            dtype=np.float32).flatten()   # (N*2,)

        gripper = np.array(
            [0.0 if self._gripper_open else 1.0], dtype=np.float32)

        # 任务编码：两个整数index（不再是one-hot，支持任意颜色数）
        task = np.array([
            float(self.color_cfg.color_to_idx(self.pick_color)),
            float(self.color_cfg.color_to_idx(self.place_color)),
        ], dtype=np.float32)

        obs = np.concatenate([
            crops.flatten(),   # N*3*28*28
            block_pos,         # N*2
            bin_pos,           # N*2
            gripper,           # 1
            task,              # 2
        ]).astype(np.float32)

        assert obs.shape == (self.obs_dim,), \
            f"obs shape {obs.shape} != expected {(self.obs_dim,)}"

        return obs

    # ── MoveIt动作原语 ──────────────────────────────────────────────────────

    def _open_gripper(self):
        self._moveit_gripper.set_named_target(MOVEIT_GRIPPER_OPEN_STATE)
        self._moveit_gripper.go(wait=True)
        self._gripper_open = True
        time.sleep(0.3)

    def _close_gripper(self):
        self._moveit_gripper.set_named_target(MOVEIT_GRIPPER_CLOSE_STATE)
        self._moveit_gripper.go(wait=True)
        self._gripper_open = False
        time.sleep(0.3)

    def _move_to_xy(self, x: float, y: float, z: float) -> bool:
        """
        规划并执行到目标位姿。
        plan 失败直接返回 False。
        execute 失败（异常或返回 False）也返回 False，
        避免将"规划成功但执行失败"误报为 success=True 污染奖励信号。
        """
        target = PoseStamped()
        target.header.frame_id = "world"
        target.pose.position.x = x
        target.pose.position.y = y
        target.pose.position.z = z
        target.pose.orientation.w = 1.0
        self._moveit_arm.set_start_state_to_current_state()
        self._moveit_arm.set_pose_target(
            target, self._moveit_arm.get_end_effector_link())
        plan_ok, traj, _, _ = self._moveit_arm.plan()
        if not plan_ok:
            return False
        try:
            exec_result = self._moveit_arm.execute(traj, wait=True)
            time.sleep(0.2)
            # 旧版 MoveIt Python API execute() 可能返回 None（视为成功）
            return exec_result if isinstance(exec_result, bool) else True
        except Exception as e:
            rospy.logwarn(f"[Env] execute 失败: {e}")
            return False

    def _execute_pick(self, x: float, y: float, color: str) -> bool:
        """
        完整 pick 原语：开爪 → 接近 → 下降 → 关爪 → 物理验证 → 抬起。

        抓取物理验证：
          关爪后等 Gazebo 物理引擎更新，检查目标方块 z 坐标。
          被夹起时方块随末端抬升，z 应明显高于 TABLE_Z + BLOCK_H。
          验证失败（方块仍在桌面）则开爪放弃，返回 False。
          这确保 success=True 时方块确实被夹住，奖励信号可信。
        """
        # 抓取验证阈值：TABLE_Z + BLOCK_H + 0.025 m 保守裕量
        GRASP_VERIFY_Z = TABLE_Z + BLOCK_H + 0.025

        self._open_gripper()
        if not self._move_to_xy(x, y, TABLE_Z + APPROACH_H): return False
        if not self._move_to_xy(x, y, TABLE_Z + GRASP_H):    return False
        self._close_gripper()

        # ── 物理验证：方块是否被抬起 ───────────────────────────────────
        time.sleep(0.25)   # 等 Gazebo 物理引擎稳定
        block_z = self._get_pose(block_name(color))[2]
        if block_z < GRASP_VERIFY_Z:
            rospy.logwarn(
                f"[Env] 抓取验证失败 {color}_block: z={block_z:.4f} "
                f"< 阈值 {GRASP_VERIFY_Z:.4f}，开爪放弃")
            self._open_gripper()
            return False

        self._holding_color = color
        if not self._move_to_xy(x, y, TABLE_Z + APPROACH_H):
            self._open_gripper()
            self._holding_color = None
            return False
        return True

    def _execute_place(self, x: float, y: float) -> bool:
        if not self._move_to_xy(x, y, TABLE_Z + APPROACH_H): return False
        if not self._move_to_xy(x, y, TABLE_Z + PLACE_H):    return False
        self._open_gripper()
        self._holding_color = None
        self._move_to_xy(x, y, TABLE_Z + APPROACH_H)
        return True

    def _return_home(self):
        try:
            self._moveit_arm.set_named_target(MOVEIT_ARM_HOME_STATE)
            self._moveit_arm.go(wait=True)
            self._open_gripper()
        except Exception:
            pass

    # ── 奖励函数 ──────────────────────────────────────────────────────────────

    def _compute_reward(self, primitive: int, color: str,
                        action_xy: np.ndarray, success: bool) -> float:
        """
        奖励函数（重写版）。

        完整成功链 = 选对颜色方块 → 执行+抓取验证通过 → 放入正确桶 → 任务完成。
        每个环节都有对应的奖励/惩罚，让 policy 清楚"差在哪一步"。

        奖励结构
        ──────────────────────────────────────────────────────────────────────
        PICK 原语 (primitive == 0):
          -dist(action_xy, pick_color 方块位置)   密集距离 shaping（始终对目标颜色算）
          -0.5   color != pick_color              选了错误颜色方块，明确惩罚
          +0.5   success and color==pick_color    正确颜色且物理验证通过
          -0.2   not success                      运动/抓取失败

        PLACE 原语 (primitive == 1):
          -dist(action_xy, place_color 桶位置)    密集距离 shaping（始终对目标桶算）
          -0.5   color != place_color             朝错误桶方向运动，惩罚
          +2.0   success and 方块进入目标桶        放置成功
          -0.2   not success                      运动失败
          +3.0   _check_done()                    任务完成 terminal bonus（独立叠加）
        ──────────────────────────────────────────────────────────────────────
        """
        r = 0.0

        if primitive == 0:  # ── PICK ──────────────────────────────────────
            # 距离 shaping：action 相对目标方块（始终用任务指定颜色）
            target_block_pos = self._get_block_pos(self.pick_color)
            dist = float(np.linalg.norm(action_xy - target_block_pos))
            r -= dist

            if color != self.pick_color:
                # 选了错误颜色的方块：明确惩罚（原来只是少拿 0.2，现在主动扣）
                r -= 0.5
            else:
                if success:
                    # 正确颜色 + 物理验证通过（方块真的被抬起）
                    r += 0.5
                else:
                    # 正确颜色但运动/抓取失败
                    r -= 0.2

        elif primitive == 1:  # ── PLACE ────────────────────────────────────
            # 距离 shaping：action 相对目标桶（始终用任务指定颜色）
            target_bin_pos = self._get_bin_pos(self.place_color)
            dist = float(np.linalg.norm(action_xy - target_bin_pos))
            r -= dist

            if color != self.place_color:
                # 朝错误颜色的桶方向运动：惩罚
                r -= 0.5

            if not success:
                r -= 0.2
            else:
                # 运动成功：检查方块是否真的进入目标桶（0.07 m 半径内）
                block_pos  = self._get_pose(block_name(self.pick_color))[:2]
                bin_gt_pos = self._get_pose(bin_name(self.place_color))[:2]
                if float(np.linalg.norm(block_pos - bin_gt_pos)) < 0.07:
                    r += 2.0   # 放置成功

            # Terminal bonus：任务完全完成时单独叠加（与中间 shaping 量级分开）
            if self._check_done():
                r += 3.0

        return float(r)

    def _check_done(self) -> bool:
        """检查当前任务是否完成。"""
        block_pos = self._get_pose(block_name(self.pick_color))[:2]
        bin_pos   = self._get_pose(bin_name(self.place_color))[:2]
        return np.linalg.norm(block_pos - bin_pos) < 0.06

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_ros()
        self._step_count   = 0
        self._gripper_open = True
        self._holding_color = None

        # 随机采样任务颜色（确保pick和place是不同颜色）
        rng = np.random.default_rng(seed)
        colors = self.color_cfg.colors
        self.pick_color  = rng.choice(colors)
        remaining        = [c for c in colors if c != self.pick_color]
        self.place_color = rng.choice(remaining)

        # 随机化场景
        self._randomize_scene()
        time.sleep(0.5)   # 等Gazebo物理引擎稳定
        self._return_home()

        obs  = self._build_observation()
        info = {
            "pick_color":  self.pick_color,
            "place_color": self.place_color,
            "n_colors":    self.N,
        }
        return obs, info

    def step(self, action: np.ndarray):
        self._step_count += 1

        # 解码动作
        primitive = int(np.round(np.clip(action[0], 0, 1)))
        obj_idx   = int(np.round(np.clip(action[1], 0, 2*self.N - 1)))
        res_xy    = np.clip(action[2:4], -0.05, 0.05)

        # obj_idx 0..N-1 = 方块，N..2N-1 = 桶
        if obj_idx < self.N:
            color    = self.color_cfg.idx_to_color(obj_idx)
            base_xy  = self._get_block_pos(color)
        else:
            color    = self.color_cfg.idx_to_color(obj_idx - self.N)
            base_xy  = self._get_bin_pos(color)

        target_xy = np.clip(
            base_xy + res_xy,
            [min(BLOCK_ZONE_X[0], BIN_ZONE_X[0]),
             min(BLOCK_ZONE_Y[0], BIN_ZONE_Y[0])],
            [max(BLOCK_ZONE_X[1], BIN_ZONE_X[1]),
             max(BLOCK_ZONE_Y[1], BIN_ZONE_Y[1])],
        )

        # 执行动作原语
        if primitive == 0:
            success = self._execute_pick(
                target_xy[0], target_xy[1], color)
        else:
            success = self._execute_place(
                target_xy[0], target_xy[1])

        reward     = self._compute_reward(primitive, color, target_xy, success)
        terminated = self._check_done()
        truncated  = self._step_count >= self.max_steps

        obs = self._build_observation()
        info = {
            "primitive":  primitive,
            "color":      color,
            "obj_idx":    obj_idx,
            "target_xy":  target_xy.tolist(),
            "success":    success,
            "step":       self._step_count,
            "pick_color": self.pick_color,
            "place_color":self.place_color,
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        if self._ros_initialized:
            try:
                self._return_home()
                moveit_commander.roscpp_shutdown()
            except Exception:
                pass
