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

Observation向量新布局（N = n_active，每回合从 color_cfg 随机抽 ACTIVE_COLORS_PER_EPISODE 种）：
  [img_patches | block_positions | bin_positions | gripper | task_encoding]
   N*3*28*28      N*2              N*2             1         2

  task_encoding = [pick_local_idx, place_local_idx]，在 _active_colors 中的下标 0..N-1。

Action向量：
  [primitive(0-1), obj_index(0-2*N-1), res_x, res_y]
  前 N 个 = 激活颜色方块，后 N 个 = 激活颜色垃圾桶（顺序与 _active_colors 一致）。
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

# 每个 episode 实际激活的颜色数量
# 从全部颜色中随机选取这么多种，同时出现在桌面上
# 设为 3 的理由：
#   - 3 个桶在扩大后的区域里轻松不重叠（间距充足）
#   - 减少 observation 中无用颜色槽位的噪声
#   - 训练收敛更快，泛化性不受影响（每次颜色随机不同）
ACTIVE_COLORS_PER_EPISODE = 3

# 方块随机化区域（桌面左半部分）
# x: 0.15~0.32（17cm），y: ±0.22（44cm）
# 3 个方块（5cm，间距 0.09m）完全放得下
BLOCK_ZONE_X = (0.15, 0.32)
BLOCK_ZONE_Y = (-0.22, 0.22)
BLOCK_MIN_GAP = 0.09   # 5cm方块 + 4cm间隙，互不碰撞

# 垃圾桶随机化区域（桌面右半部分）
# x: 0.30~0.50（20cm，2列），y: ±0.24（48cm，3行）
# 最多 2×3=6 个桶，间距 0.16m 时 3 个桶绰绰有余
BIN_ZONE_X   = (0.30, 0.50)
BIN_ZONE_Y   = (-0.24, 0.24)
BIN_MIN_GAP  = 0.16   # 7cm桶对角线0.099m，0.16m确保边缘间距≥6cm

# 高度参数（z轴，单位米）
# ──────────────────────────────────────────────────────────
# 坐标系约定（与 world 文件一致）：
#   桌面顶面 z = 0.00（world文件桌面顶面在z=0）
#   方块底面 z = 0.00，方块重心 z = BLOCK_H/2 = 0.02
#   桶底面   z = 0.00，桶重心   z = BIN_H/2   = 0.06
#
# GRASP_H 修正：
#   原来 TABLE_Z + GRASP_H = 0.00 + 0.005 = 0.005m
#   方块顶面在 z = TABLE_Z + BLOCK_H = 0.00 + 0.04 = 0.04m
#   末端只到 0.025m，在桌面里，触发碰撞检测 → CONTROL_FAILED
#   正确值：末端应到达方块中部稍上方
#   TABLE_Z + GRASP_H = TABLE_Z + BLOCK_H/2 + 0.005 = 0.025m
#   → GRASP_H = BLOCK_H/2 + 0.005 = 0.025m
#
# APPROACH_H 修正：
#   接近高度必须明显高于最高物体（桶高0.12m）加安全裕量
#   TABLE_Z + APPROACH_H = 0.18m（高于桶顶2cm，便于抬起后验证）
# ──────────────────────────────────────────────────────────
TABLE_Z       = 0.00   # 桌面顶面（world文件里桌面顶面在 z=0）
BLOCK_H       = 0.04   # 方块高度
BLOCK_W       = 0.05   # 方块边长（x/y）
BIN_H         = 0.12   # 垃圾桶高度（外高）
BIN_INNER_W   = 0.065  # 垃圾桶内腔宽度（x/y）
BIN_WALL_T    = 0.0025 # 壁厚/底厚（与world一致）

APPROACH_H    = 0.18   # 接近高度（末端 z = 0.18m）
GRASP_H       = 0.025  # 抓取高度（末端 z = 0.025m ≈ 方块中部）
PLACE_H       = 0.11   # 放置高度（末端 z = 0.11m，接近桶口内部）

# 图像crop参数
CROP_SIZE     = 28
POSITION_NOISE_SIGMA = 0.030   # 稍微降低噪声，6→3颜色后精度可以提高

# MoveIt SRDF 中的 group_state 名称（与 RViz MotionPlanning 一致，区分大小写）
MOVEIT_ARM_HOME_STATE = "home"
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

        # 每个 episode 激活的颜色数（从全部颜色中随机抽取）
        # N_active <= color_cfg.n_colors
        self.n_active = min(ACTIVE_COLORS_PER_EPISODE,
                            self.color_cfg.n_colors)
        # 当前 episode 激活的颜色子集（每次 reset 重新抽取）
        self._active_colors: list = self.color_cfg.colors[:self.n_active]

        N = self.n_active   # observation/action 维度基于激活数量

        # 当前episode的任务颜色
        self.pick_color:  str = self._active_colors[0]
        self.place_color: str = self._active_colors[1]
        self._step_count   = 0
        self._gripper_open = True
        self._holding_color: str = None

        # ── 观测空间 ──────────────────────────────────────────────────────
        # img: n_active * 3 * 28 * 28
        # block_positions: n_active * 2
        # bin_positions:   n_active * 2
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
        self.N        = N   # = n_active

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )

        # ── 动作空间 ──────────────────────────────────────────────────────
        # [primitive(0-1), obj_index(0 to 2N-1), res_x, res_y]
        # obj_index 0..N-1 = 方块（本回合 _active_colors 顺序）
        # obj_index N..2N-1 = 桶索引（同上）
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
        随机化场景：
          1. 从全部颜色中随机抽取 n_active 种颜色作为本 episode 的激活颜色
          2. 把激活颜色的方块随机摆在左半区，桶随机摆在右半区
          3. 非激活颜色的物体传送到桌面外的停靠区，不干扰仿真
          4. 保证同区域内物体互不重叠，100次失败后使用确定性网格 fallback

        网格 fallback 算法（修正版）：
          之前的 linspace 方案在 x 方向宽度不足时生成的网格间距不满足要求。
          新方案：先算出每行能放多少个（floor(zone_x_width / gap)），
          再按行优先枚举（x 先排满一行再换 y），保证每个点都满足间距。
        """
        rng = np.random.default_rng()

        # ── Step 1: 抽取本 episode 的激活颜色 ─────────────────────────────
        all_colors = list(self.color_cfg.colors)
        rng.shuffle(all_colors)
        self._active_colors = all_colors[: self.n_active]
        inactive_colors     = all_colors[self.n_active :]

        # 更新任务颜色（确保在激活颜色里）
        self.pick_color  = self._active_colors[0]
        remaining        = [c for c in self._active_colors
                            if c != self.pick_color]
        self.place_color = rng.choice(remaining) if remaining else self._active_colors[0]

        # ── Step 2: 把非激活颜色的物体挪走（停靠区，不干扰桌面） ──────────
        PARK_X, PARK_Y_START = 2.0, 0.0   # 远离桌面
        for i, color in enumerate(inactive_colors):
            park_y = PARK_Y_START + i * 0.2
            self._teleport(block_name(color), PARK_X, park_y, 0.02)
            self._teleport(bin_name(color),   PARK_X + 0.5, park_y, 0.05)

        # ── Step 3: 正确的网格生成（保证间距 ≥ gap） ─────────────────────
        def _deterministic_grid(n: int, zone_x: tuple,
                                zone_y: tuple, gap: float) -> list:
            """
            生成最多 n 个位置的均匀网格，保证任意两点间距 ≥ gap。
            按行优先枚举：先沿 x 方向排满一行，再移到下一行。
            """
            cols = max(1, int((zone_x[1] - zone_x[0]) / gap))
            rows = max(1, int((zone_y[1] - zone_y[0]) / gap))
            pts  = []
            for row in range(rows):
                for col in range(cols):
                    cx = zone_x[0] + gap * (col + 0.5)
                    cy = zone_y[0] + gap * (row + 0.5)
                    if cx <= zone_x[1] and cy <= zone_y[1]:
                        pts.append((cx, cy))
                    if len(pts) >= n:
                        return pts
            return pts[:n]

        # ── Step 4: 随机化方块 ────────────────────────────────────────────
        placed_blocks = []
        for idx, color in enumerate(self._active_colors):
            name   = block_name(color)
            placed = False
            for _ in range(150):
                x = rng.uniform(*BLOCK_ZONE_X)
                y = rng.uniform(*BLOCK_ZONE_Y)
                if all(np.linalg.norm([x - px, y - py]) > BLOCK_MIN_GAP
                       for px, py in placed_blocks):
                    placed_blocks.append((x, y))
                    self._teleport(name, x, y, BLOCK_H / 2)
                    placed = True
                    break

            if not placed:
                grid = _deterministic_grid(
                    self.n_active, BLOCK_ZONE_X, BLOCK_ZONE_Y, BLOCK_MIN_GAP)
                if idx < len(grid):
                    x, y = grid[idx]
                    placed_blocks.append((x, y))
                    self._teleport(name, x, y, BLOCK_H / 2)
                    rospy.logwarn(
                        f"[Env] 方块 {color} 随机摆放失败，"
                        f"fallback 网格 ({x:.3f}, {y:.3f})")

        # ── Step 5: 随机化垃圾桶 ──────────────────────────────────────────
        placed_bins = []
        for idx, color in enumerate(self._active_colors):
            name   = bin_name(color)
            placed = False
            for _ in range(150):
                x = rng.uniform(*BIN_ZONE_X)
                y = rng.uniform(*BIN_ZONE_Y)
                if all(np.linalg.norm([x - px, y - py]) > BIN_MIN_GAP
                       for px, py in placed_bins):
                    placed_bins.append((x, y))
                    self._teleport(name, x, y, BIN_H / 2)
                    placed = True
                    break

            if not placed:
                grid = _deterministic_grid(
                    self.n_active, BIN_ZONE_X, BIN_ZONE_Y, BIN_MIN_GAP)
                if idx < len(grid):
                    x, y = grid[idx]
                    placed_bins.append((x, y))
                    self._teleport(name, x, y, BIN_H / 2)
                    rospy.logwarn(
                        f"[Env] 垃圾桶 {color} 随机摆放失败，"
                        f"fallback 网格 ({x:.3f}, {y:.3f})")
                else:
                    rospy.logerr(
                        f"[Env] 垃圾桶 {color} 无法摆放！"
                        f"请扩大 BIN_ZONE 或减少 n_active。")

    # ── 带噪声的位置获取 ─────────────────────────────────────────────────────

    def _refresh_positions(self):
        """刷新激活颜色物体的带噪声位置缓存（每次step调用）。"""
        noise = lambda: np.random.normal(0, self.noise_sigma, 2)
        for color in self._active_colors:
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
        提取激活颜色方块对应的图像 crop。
        返回形状 (n_active, 3, 28, 28)
        """
        import cv2
        crops = []
        img = self._latest_image

        for color in self._active_colors:
            block_pos = self._get_block_pos(color)
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

        return np.array(crops, dtype=np.float32)  # (n_active, 3, 28, 28)

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
        构建完整 observation 向量（仅包含激活颜色）。
        布局：[crops(N*3*28*28) | block_pos(N*2) | bin_pos(N*2) | gripper(1) | task(2)]
        N = n_active（本 episode 激活的颜色数）
        """
        self._refresh_positions()

        crops = self._build_crops()   # (n_active, 3, 28, 28)

        block_pos = np.array(
            [self._get_block_pos(c) for c in self._active_colors],
            dtype=np.float32).flatten()

        bin_pos = np.array(
            [self._get_bin_pos(c) for c in self._active_colors],
            dtype=np.float32).flatten()

        gripper = np.array(
            [0.0 if self._gripper_open else 1.0], dtype=np.float32)

        # 任务编码：在 _active_colors 列表中的局部 index（0~n_active-1）
        pick_local  = self._active_colors.index(self.pick_color)
        place_local = self._active_colors.index(self.place_color)
        task = np.array([float(pick_local), float(place_local)],
                        dtype=np.float32)

        obs = np.concatenate([
            crops.flatten(), block_pos, bin_pos, gripper, task,
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
        # 抬起验证阈值：必须高于桶口，再额外留 2cm 裕量
        LIFT_VERIFY_Z = TABLE_Z + BIN_H + 0.02

        self._open_gripper()
        if not self._move_to_xy(x, y, TABLE_Z + APPROACH_H): return False
        if not self._move_to_xy(x, y, TABLE_Z + GRASP_H):    return False
        self._close_gripper()
        if not self._move_to_xy(x, y, TABLE_Z + APPROACH_H):
            self._open_gripper()
            self._holding_color = None
            return False

        # ── 物理验证：抬起动作结束后再检查 z（不依赖固定 sleep） ─────────
        block_z = self._get_pose(block_name(color))[2]
        if block_z < LIFT_VERIFY_Z:
            rospy.logwarn(
                f"[Env] 抓取验证失败 {color}_block: z={block_z:.4f} "
                f"< 阈值 {LIFT_VERIFY_Z:.4f}，开爪放弃")
            self._open_gripper()
            self._holding_color = None
            return False

        self._holding_color = color
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
                # 运动成功：检查方块是否真的进入目标桶内腔
                if self._is_block_in_target_bin():
                    r += 2.0   # 放置成功

            # Terminal bonus：任务完全完成时单独叠加（与中间 shaping 量级分开）
            if self._check_done():
                r += 3.0

        return float(r)

    def _is_block_in_target_bin(self) -> bool:
        """
        判断目标方块是否进入目标桶内腔。
        条件：
          1) 方块中心在桶内腔投影范围内（并扣除方块半宽，避免贴壁误判）；
          2) 方块中心 z 在桶口以下。
        """
        block_xyz = self._get_pose(block_name(self.pick_color))
        bin_xyz   = self._get_pose(bin_name(self.place_color))

        dx = float(abs(block_xyz[0] - bin_xyz[0]))
        dy = float(abs(block_xyz[1] - bin_xyz[1]))

        inner_half = BIN_INNER_W / 2.0
        block_half = BLOCK_W / 2.0
        margin_xy  = max(inner_half - block_half, 0.0)

        inside_xy = (dx <= margin_xy) and (dy <= margin_xy)
        below_rim = float(block_xyz[2]) < (TABLE_Z + BIN_H)
        return bool(inside_xy and below_rim)

    def _check_done(self) -> bool:
        """检查当前任务是否完成（目标方块确实在目标桶内腔）。"""
        return self._is_block_in_target_bin()

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_ros()
        self._step_count    = 0
        self._gripper_open  = True
        self._holding_color = None

        # _randomize_scene 内部会重新抽取 _active_colors 和任务颜色
        self._randomize_scene()
        time.sleep(0.5)
        self._return_home()

        obs  = self._build_observation()
        info = {
            "pick_color":    self.pick_color,
            "place_color":   self.place_color,
            "active_colors": list(self._active_colors),
            "n_colors":      self.n_active,
        }
        return obs, info

    def step(self, action: np.ndarray):
        self._step_count += 1

        # 解码动作
        primitive = int(np.round(np.clip(action[0], 0, 1)))
        obj_idx   = int(np.round(np.clip(action[1], 0, 2*self.N - 1)))
        res_xy    = np.clip(action[2:4], -0.05, 0.05)

        # obj_idx 0..N-1 = 方块，N..2N-1 = 桶（N = n_active）
        # 颜色映射基于本 episode 的 _active_colors（而非全部颜色）
        if obj_idx < self.N:
            color   = self._active_colors[obj_idx]
            base_xy = self._get_block_pos(color)
        else:
            color   = self._active_colors[obj_idx - self.N]
            base_xy = self._get_bin_pos(color)

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
            "primitive":     primitive,
            "color":         color,
            "obj_idx":       obj_idx,
            "target_xy":     target_xy.tolist(),
            "success":       success,
            "step":          self._step_count,
            "pick_color":    self.pick_color,
            "place_color":   self.place_color,
            "active_colors": list(self._active_colors),
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        if self._ros_initialized:
            try:
                self._return_home()
                moveit_commander.roscpp_shutdown()
            except Exception:
                pass
