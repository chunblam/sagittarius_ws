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
import numpy as np
import gymnasium as gym
from gymnasium import spaces

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import rospy
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from std_srvs.srv import Empty

import moveit_commander
from geometry_msgs.msg import PoseStamped

from configs.color_config import ColorConfig, get_color_config


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
        self._moveit_arm     = moveit_commander.MoveGroupCommander("sagittarius_arm")
        self._moveit_gripper = moveit_commander.MoveGroupCommander("sagittarius_gripper")

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
        随机化所有方块和垃圾桶的位置。

        方块放在桌面左半区，桶放在右半区。
        两个区域不重叠，互相不会碰撞。
        同区域内的物体保持最小间距。
        """
        rng = np.random.default_rng()

        # ── 随机化方块 ────────────────────────────────────────────────────
        placed_blocks = []
        for color in self.color_cfg.colors:
            name = block_name(color)
            for _ in range(50):
                x = rng.uniform(*BLOCK_ZONE_X)
                y = rng.uniform(*BLOCK_ZONE_Y)
                if all(np.linalg.norm([x-px, y-py]) > 0.08
                       for px, py in placed_blocks):
                    placed_blocks.append((x, y))
                    self._teleport(name, x, y, TABLE_Z + BLOCK_H / 2)
                    break

        # ── 随机化垃圾桶 ──────────────────────────────────────────────────
        placed_bins = []
        for color in self.color_cfg.colors:
            name = bin_name(color)
            for _ in range(50):
                x = rng.uniform(*BIN_ZONE_X)
                y = rng.uniform(*BIN_ZONE_Y)
                if all(np.linalg.norm([x-px, y-py]) > 0.10
                       for px, py in placed_bins):
                    placed_bins.append((x, y))
                    self._teleport(name, x, y, TABLE_Z + BIN_H / 2)
                    break

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

    def _robot_to_pixel(self, xy: np.ndarray) -> Tuple:
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
        self._moveit_gripper.set_named_target("open")
        self._moveit_gripper.go(wait=True)
        self._gripper_open = True
        time.sleep(0.3)

    def _close_gripper(self):
        self._moveit_gripper.set_named_target("close")
        self._moveit_gripper.go(wait=True)
        self._gripper_open = False
        time.sleep(0.3)

    def _move_to_xy(self, x: float, y: float, z: float) -> bool:
        target = PoseStamped()
        target.header.frame_id = "world"
        target.pose.position.x = x
        target.pose.position.y = y
        target.pose.position.z = z
        target.pose.orientation.w = 1.0
        self._moveit_arm.set_start_state_to_current_state()
        self._moveit_arm.set_pose_target(
            target, self._moveit_arm.get_end_effector_link())
        ok, traj, _, _ = self._moveit_arm.plan()
        if ok:
            self._moveit_arm.execute(traj, wait=True)
            time.sleep(0.2)
        return ok

    def _execute_pick(self, x: float, y: float, color: str) -> bool:
        self._open_gripper()
        if not self._move_to_xy(x, y, TABLE_Z + APPROACH_H): return False
        if not self._move_to_xy(x, y, TABLE_Z + GRASP_H):    return False
        self._close_gripper()
        self._holding_color = color
        if not self._move_to_xy(x, y, TABLE_Z + APPROACH_H): return False
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
            self._moveit_arm.set_named_target("Home")
            self._moveit_arm.go(wait=True)
            self._open_gripper()
        except Exception:
            pass

    # ── 奖励函数 ──────────────────────────────────────────────────────────────

    def _compute_reward(self, primitive: int, color: str,
                        action_xy: np.ndarray, success: bool) -> float:
        r = 0.0

        if primitive == 0:  # pick
            target_pos = self._get_block_pos(color)
            dist = np.linalg.norm(action_xy - target_pos)
            r += -dist
            if success and color == self.pick_color:
                r += 0.2   # 小bonus：抓对了颜色

        elif primitive == 1:  # place
            target_pos = self._get_bin_pos(self.place_color)
            dist = np.linalg.norm(action_xy - target_pos)
            r += -dist
            if success:
                # 检查方块是否真的进桶了
                block_pos  = self._get_pose(block_name(self.pick_color))[:2]
                bin_gt_pos = self._get_pose(bin_name(self.place_color))[:2]
                if np.linalg.norm(block_pos - bin_gt_pos) < 0.06:
                    r += 1.0   # 任务成功

        if not success:
            r -= 0.2

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
