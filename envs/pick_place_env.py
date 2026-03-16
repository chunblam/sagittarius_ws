#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pick_place_env.py
=================
Sagittarius 拾放任务的 Gazebo gym.Env 封装。

本环境桥接：
  - stable-baselines3（SAC 训练循环）
  - ROS / MoveIt（运动执行）
  - Gazebo（仿真状态与物理）

观测空间：
  - image_patches : (N_obj, 3, 28, 28)  每个物体的 RGB 裁剪块
  - obj_positions : (N_obj, 2)           带高斯噪声的 x,y 位置
  - gripper_state : (1,)                 0=张开，1=闭合
  - lang_goal     : (N_obj*2,)           抓取/放置目标的 one-hot 编码

动作空间（残差、以物体为中心）：
  - primitive  : int  0=抓取，1=放置
  - obj_index  : int  对哪个物体操作
  - residual_xy: (2,) 相对物体中心的偏移（米）
"""

import os
import time
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import rospy
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetModelState, SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Twist
from std_srvs.srv import Empty

import moveit_commander
import moveit_msgs.msg
from geometry_msgs.msg import PoseStamped
import sys


# ── 场景配置 ─────────────────────────────────────────────────────────────────

# Gazebo 中出现的物体名称
BLOCK_NAMES  = ["red_block",   "green_block",  "blue_block"]
BOWL_NAMES   = ["red_bowl",    "green_bowl",   "blue_bowl"]
ALL_OBJECTS  = BLOCK_NAMES + BOWL_NAMES

# 用于语言编码索引的颜色
COLOR_INDEX  = {"red": 0, "green": 1, "blue": 2}

# 桌面工作范围（米，机器人基坐标系）
TABLE_X_MIN, TABLE_X_MAX = 0.15, 0.40
TABLE_Y_MIN, TABLE_Y_MAX = -0.20, 0.20
TABLE_Z = 0.02  # 桌面相对基座高度

# 抓取/放置高度
APPROACH_HEIGHT  = 0.12   # 预抓取时桌面以上高度
GRASP_HEIGHT     = 0.005  # 实际抓取时物体中心以上高度
PLACE_HEIGHT     = 0.06   # 在碗中心上方释放的高度

# 图像裁剪尺寸
CROP_SIZE = 28

# 位置噪声 sigma：约等于裁剪半径的一半（米，参考 ExploRLLM 论文）
# 假设每个裁剪覆盖约 8cm 半径 → σ = 0.04
POSITION_NOISE_SIGMA = 0.04

# 跟踪的物体数量（抓取仅用块，放置时参考全部）
N_OBJECTS = len(BLOCK_NAMES)      # 3
N_TARGETS = len(BOWL_NAMES)       # 3
N_TOTAL   = N_OBJECTS + N_TARGETS # 6


# ── 环境类 ───────────────────────────────────────────────────────────────────

class SagittariusPickPlaceEnv(gym.Env):
    """
    Sagittarius SGR532 的单步抓取或放置环境。

    每次 step() 只执行一个原语（抓取 或 放置）。
    智能体需串联 抓取 → 放置 才能完成一次任务。
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self,
                 task: str = "short_horizon",
                 max_steps: int = 10,
                 noise_sigma: float = POSITION_NOISE_SIGMA,
                 render_mode: str = None):
        """
        Args:
            task        : "short_horizon"（抓一个放一个）或
                          "long_horizon"（把所有块按颜色放入对应碗）
            max_steps   : 每个 episode 最大原语步数
            noise_sigma : 加到真值位置上的高斯噪声标准差
            render_mode : gymnasium 渲染模式（未用，由 Gazebo 负责显示）
        """
        super().__init__()

        self.task         = task
        self.max_steps    = max_steps
        self.noise_sigma  = noise_sigma
        self.render_mode  = render_mode
        self._step_count  = 0
        self._gripper_open = True

        # 当前 episode 的语言目标
        self.pick_color   = None   # 例如 "red"
        self.place_color  = None   # 例如 "blue"

        # ── Gymnasium 空间定义 ─────────────────────────────────────────────────
        # 观测：扁平 Box，便于 stable-baselines3，由 CustomSACPolicy 再拆开
        # 布局: [img_patches(N_total*3*28*28), positions(N_total*2),
        #        gripper(1), lang_onehot(6)]
        img_dim   = N_TOTAL * 3 * CROP_SIZE * CROP_SIZE  # 6*3*28*28 = 14112
        pos_dim   = N_TOTAL * 2                           # 12
        grip_dim  = 1
        lang_dim  = N_OBJECTS + N_TARGETS                 # 6  (抓/放 one-hot)
        obs_dim   = img_dim + pos_dim + grip_dim + lang_dim

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )

        # 动作: [primitive(1), obj_index(1), residual_x(1), residual_y(1)]
        # 此处 primitive、obj_index 为连续值，通过四舍五入离散化
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, -0.05, -0.05], dtype=np.float32),
            high=np.array([1.0, float(N_TOTAL-1), 0.05, 0.05], dtype=np.float32)
        )

        # ── ROS / MoveIt 初始化（延后到首次 reset 再执行）──────────────────────
        self._ros_initialized = False
        self._moveit_arm      = None
        self._moveit_gripper  = None
        self._model_states    = None  # 最新的 /gazebo/model_states 消息
        self._image_cache     = {}    # 物体名 -> 最新 np.ndarray (H,W,3)

        # 延后调用 _init_ros()，在首次 reset() 时执行，避免多 env 时 rospy.init_node 冲突

    # ── ROS 初始化 ────────────────────────────────────────────────────────────

    def _init_ros(self):
        """初始化 ROS 节点、MoveIt 及话题订阅。"""
        if self._ros_initialized:
            return

        rospy.loginfo("[Env] Initialising ROS node...")
        # 仅在尚未初始化时创建节点（允许外部先初始化）
        if not rospy.core.is_initialized():
            rospy.init_node("explorllm_env", anonymous=True)

        # MoveIt 控制组
        moveit_commander.roscpp_initialize(sys.argv)
        self._moveit_arm = moveit_commander.MoveGroupCommander("sagittarius_arm")
        self._moveit_gripper = moveit_commander.MoveGroupCommander("sagittarius_gripper")

        # 位姿容差与速度
        self._moveit_arm.set_goal_position_tolerance(0.005)
        self._moveit_arm.set_goal_orientation_tolerance(0.02)
        self._moveit_arm.set_max_velocity_scaling_factor(0.4)
        self._moveit_arm.set_max_acceleration_scaling_factor(0.4)
        self._moveit_arm.allow_replanning(True)

        self._moveit_gripper.set_goal_joint_tolerance(0.001)
        self._moveit_gripper.set_max_velocity_scaling_factor(0.5)
        self._moveit_gripper.set_max_acceleration_scaling_factor(0.5)

        # 订阅 Gazebo 模型状态（真值位置）
        rospy.Subscriber("/gazebo/model_states", ModelStates,
                         self._model_states_cb, queue_size=1)

        # 订阅相机图像（评估时用于裁剪）
        try:
            from sensor_msgs.msg import Image
            from cv_bridge import CvBridge
            self._bridge = CvBridge()
            self._latest_image = None
            rospy.Subscriber("/usb_cam/image_raw", Image,
                             self._image_cb, queue_size=1)
            rospy.loginfo("[Env] Camera subscriber registered.")
        except ImportError:
            rospy.logwarn("[Env] cv_bridge not found; using blank image crops.")
            self._bridge = None
            self._latest_image = None

        # Gazebo 服务
        rospy.wait_for_service("/gazebo/set_model_state", timeout=10)
        self._set_model_state = rospy.ServiceProxy(
            "/gazebo/set_model_state", SetModelState)

        # 若物理仿真暂停则恢复
        rospy.wait_for_service("/gazebo/unpause_physics", timeout=5)
        unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        try:
            unpause()
        except rospy.ServiceException:
            pass

        self._ros_initialized = True
        rospy.loginfo("[Env] ROS initialisation complete.")

    def _model_states_cb(self, msg: ModelStates):
        self._model_states = msg

    def _image_cb(self, msg):
        if self._bridge is not None:
            import cv2
            try:
                self._latest_image = self._bridge.imgmsg_to_cv2(msg, "bgr8")
            except Exception:
                pass

    # ── Gazebo 物体管理 ───────────────────────────────────────────────────────

    def _get_object_pose(self, name: str) -> np.ndarray:
        """
        从最新 ModelStates 中取指定物体的 (x, y, z)。
        若未找到则返回零向量。
        """
        if self._model_states is None:
            return np.zeros(3, dtype=np.float32)
        try:
            idx = self._model_states.name.index(name)
            p = self._model_states.pose[idx].position
            return np.array([p.x, p.y, p.z], dtype=np.float32)
        except ValueError:
            rospy.logwarn(f"[Env] Object '{name}' not in model_states.")
            return np.zeros(3, dtype=np.float32)

    def _randomize_objects(self):
        """将块随机传送到桌面上；碗放在固定区域。"""
        rng = np.random.default_rng()

        # 随机块位置（防碰撞：至少相距 8cm）
        placed = []
        for name in BLOCK_NAMES:
            for _ in range(50):  # 最大尝试次数
                x = rng.uniform(TABLE_X_MIN + 0.05, TABLE_X_MAX - 0.05)
                y = rng.uniform(TABLE_Y_MIN + 0.05, TABLE_Y_MAX - 0.05)
                if all(np.linalg.norm([x - px, y - py]) > 0.08
                       for (px, py) in placed):
                    placed.append((x, y))
                    self._teleport(name, x, y, TABLE_Z + 0.02)
                    break

        # 碗固定在指定位置（在块随机区域之外）
        bowl_positions = [
            (0.35, -0.15, TABLE_Z),   # red bowl
            (0.35,  0.00, TABLE_Z),   # green bowl
            (0.35,  0.15, TABLE_Z),   # blue bowl
        ]
        for name, (x, y, z) in zip(BOWL_NAMES, bowl_positions):
            self._teleport(name, x, y, z)

    def _teleport(self, name: str, x: float, y: float, z: float):
        """将 Gazebo 模型瞬间移动到 (x,y,z)。"""
        state = ModelState()
        state.model_name = name
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = z
        state.pose.orientation.w = 1.0
        state.reference_frame = "world"
        try:
            self._set_model_state(state)
        except rospy.ServiceException as e:
            rospy.logwarn(f"[Env] Teleport failed for {name}: {e}")

    # ── 观测构建 ───────────────────────────────────────────────────────────────

    def _get_positions_with_noise(self) -> np.ndarray:
        """
        返回带高斯噪声的 (N_total, 2) 的 x,y 位置数组。
        用于训练时模拟真实相机检测不确定性。
        """
        positions = []
        for name in ALL_OBJECTS:
            xyz = self._get_object_pose(name)
            noise = np.random.normal(0, self.noise_sigma, size=2)
            positions.append(xyz[:2] + noise)
        return np.array(positions, dtype=np.float32)  # (N_total, 2)

    def _get_image_crops(self, positions: np.ndarray) -> np.ndarray:
        """
        在每个物体位置周围裁出 28x28 的 RGB 块。
        若无相机图像则使用空白块。

        Args:
            positions: (N_total, 2) 机器人坐标系下的物体位置

        Returns:
            crops: (N_total, 3, 28, 28) float32，取值 [0,1]
        """
        import cv2
        crops = []
        img = self._latest_image

        for i, name in enumerate(ALL_OBJECTS):
            if img is not None:
                # 将机器人坐标系 (x,y) 投影到图像像素 (u,v)
                # 使用 Lab2 camera_calibration_hsv 的标定结果
                # 训练时采用简单透视近似
                u, v = self._robot_to_pixel(positions[i])
                h, w = img.shape[:2]
                u, v = int(np.clip(u, 14, w-14)), int(np.clip(v, 14, h-14))
                crop = img[v-14:v+14, u-14:u+14]
                if crop.shape[:2] != (28, 28):
                    crop = np.zeros((28, 28, 3), dtype=np.uint8)
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop = crop.astype(np.float32) / 255.0
            else:
                # 纯 Gazebo 训练无相机时使用空白块
                crop = np.zeros((28, 28, 3), dtype=np.float32)

            # (H,W,C) → (C,H,W) 符合 PyTorch 约定
            crops.append(crop.transpose(2, 0, 1))

        return np.array(crops, dtype=np.float32)  # (N_total, 3, 28, 28)

    def _robot_to_pixel(self, xy: np.ndarray):
        """
        将机器人坐标系 (x,y) 转换为相机像素 (u,v)。
        使用 Lab2 标定得到的线性回归系数。
        运行 camera_calibration_hsv.launch 后请替换为实际标定值。

        占位：假设俯视相机约 60cm 高度。
        """
        # 以下为标定脚本输出的 k,b，运行 Lab2 标定后请用 vision_config.yaml 中的值替换
        kx, bx = -0.00029, 0.31084   # x_robot = kx*v_pixel + bx
        ky, by =  0.00030, 0.09080   # y_robot = ky*u_pixel + by

        # 反解得到从机器人坐标到像素
        if abs(kx) > 1e-9:
            v = (xy[0] - bx) / kx
        else:
            v = 240
        if abs(ky) > 1e-9:
            u = (xy[1] - by) / ky
        else:
            u = 320
        return float(u), float(v)

    def _build_lang_onehot(self) -> np.ndarray:
        """
        构建 6 维 one-hot：[pick_r, pick_g, pick_b, place_r, place_g, place_b]
        """
        vec = np.zeros(N_OBJECTS + N_TARGETS, dtype=np.float32)
        if self.pick_color  in COLOR_INDEX:
            vec[COLOR_INDEX[self.pick_color]]  = 1.0
        if self.place_color in COLOR_INDEX:
            vec[N_OBJECTS + COLOR_INDEX[self.place_color]] = 1.0
        return vec

    def _build_observation(self) -> np.ndarray:
        """
        组装完整的扁平观测向量。
        布局: [img_patches | positions | gripper | lang]
        """
        positions = self._get_positions_with_noise()    # (N_total, 2)
        crops     = self._get_image_crops(positions)    # (N_total, 3, 28, 28)
        gripper   = np.array([0.0 if self._gripper_open else 1.0],
                              dtype=np.float32)
        lang      = self._build_lang_onehot()           # (6,)

        obs = np.concatenate([
            crops.flatten(),       # 14112
            positions.flatten(),   # 12
            gripper,               # 1
            lang,                  # 6
        ]).astype(np.float32)
        return obs

    # ── MoveIt 运动原语 ───────────────────────────────────────────────────────

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

    def _move_to_pose(self, x: float, y: float, z: float,
                      qx=0.0, qy=0.0, qz=0.0, qw=1.0) -> bool:
        """
        将机械臂末端移动到笛卡尔位姿 (x,y,z) 及四元数。
        规划并执行成功返回 True。
        """
        target = PoseStamped()
        target.header.frame_id = "world"
        target.pose.position.x = x
        target.pose.position.y = y
        target.pose.position.z = z
        target.pose.orientation.x = qx
        target.pose.orientation.y = qy
        target.pose.orientation.z = qz
        target.pose.orientation.w = qw

        self._moveit_arm.set_start_state_to_current_state()
        self._moveit_arm.set_pose_target(
            target, self._moveit_arm.get_end_effector_link())

        success, traj, _, err = self._moveit_arm.plan()
        if success:
            self._moveit_arm.execute(traj, wait=True)
            time.sleep(0.2)
        return success

    def _execute_pick(self, x: float, y: float) -> bool:
        """
        完整抓取原语：
          1. 张开夹爪
          2. 移动到目标上方（approach 高度）
          3. 下到抓取高度
          4. 闭合夹爪
          5. 抬回
        全部成功返回 True。
        """
        self._open_gripper()

        # 从上方接近
        ok = self._move_to_pose(x, y, TABLE_Z + APPROACH_HEIGHT)
        if not ok:
            rospy.logwarn("[Env] Pick approach planning failed.")
            return False

        # 下降
        ok = self._move_to_pose(x, y, TABLE_Z + GRASP_HEIGHT)
        if not ok:
            rospy.logwarn("[Env] Pick descend planning failed.")
            return False

        # 抓取
        self._close_gripper()

        # 抬起
        ok = self._move_to_pose(x, y, TABLE_Z + APPROACH_HEIGHT)
        if not ok:
            rospy.logwarn("[Env] Pick lift planning failed.")
            return False

        return True

    def _execute_place(self, x: float, y: float) -> bool:
        """
        完整放置原语：
          1. 移动到目标碗上方（approach 高度）
          2. 下到放置高度
          3. 张开夹爪
          4. 抬回
        全部成功返回 True。
        """
        # 移动到碗上方
        ok = self._move_to_pose(x, y, TABLE_Z + APPROACH_HEIGHT)
        if not ok:
            rospy.logwarn("[Env] Place approach planning failed.")
            return False

        # 略下降
        ok = self._move_to_pose(x, y, TABLE_Z + PLACE_HEIGHT)
        if not ok:
            rospy.logwarn("[Env] Place descend planning failed.")
            return False

        # 释放
        self._open_gripper()

        # 抬起
        self._move_to_pose(x, y, TABLE_Z + APPROACH_HEIGHT)

        return True

    def _return_home(self):
        """将机械臂回到安全的 'Home' 命名位姿。"""
        try:
            self._moveit_arm.set_named_target("Home")
            self._moveit_arm.go(wait=True)
            self._open_gripper()
        except Exception as e:
            rospy.logwarn(f"[Env] Failed to return home: {e}")

    # ── 奖励计算 ───────────────────────────────────────────────────────────────

    def _compute_reward(self, primitive: int, obj_idx: int,
                        action_xy: np.ndarray,
                        success: bool) -> float:
        """
        奖励 = 稠密距离项 + 稀疏成功项。

        稠密：末端到目标的负距离（鼓励靠近）
        稀疏：放置后块在正确碗内则 +1.0
        """
        r = 0.0

        if primitive == 0:  # pick
            # 稠密：朝目标块移动给奖励
            target_name = BLOCK_NAMES[obj_idx % N_OBJECTS]
            target_pos  = self._get_object_pose(target_name)[:2]
            dist = np.linalg.norm(action_xy - target_pos)
            r += -dist  # 越近越好

        elif primitive == 1:  # place
            # 稠密：靠近目标碗给奖励
            bowl_name  = f"{self.place_color}_bowl"
            bowl_pos   = self._get_object_pose(bowl_name)[:2]
            dist = np.linalg.norm(action_xy - bowl_pos)
            r += -dist

            # 稀疏：检查块是否在碗内
            if success:
                block_name = f"{self.pick_color}_block"
                block_pos  = self._get_object_pose(block_name)[:2]
                bowl_pos2  = self._get_object_pose(bowl_name)[:2]
                if np.linalg.norm(block_pos - bowl_pos2) < 0.05:
                    r += 1.0  # 任务成功

        if not success:
            r -= 0.2  # 运动规划失败惩罚

        return float(r)

    def _check_task_done(self) -> bool:
        """若 episode 级任务目标已完成则返回 True。"""
        if self.task == "short_horizon":
            # 当 pick_color 块进入 place_color 碗时视为完成
            block_pos = self._get_object_pose(f"{self.pick_color}_block")[:2]
            bowl_pos  = self._get_object_pose(f"{self.place_color}_bowl")[:2]
            return np.linalg.norm(block_pos - bowl_pos) < 0.05

        elif self.task == "long_horizon":
            # 当所有块都在对应颜色的碗内时完成
            for color in ["red", "green", "blue"]:
                block_pos = self._get_object_pose(f"{color}_block")[:2]
                bowl_pos  = self._get_object_pose(f"{color}_bowl")[:2]
                if np.linalg.norm(block_pos - bowl_pos) >= 0.05:
                    return False
            return True

        return False

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """
        重置 episode：
          1. 在 Gazebo 中随机物体位置
          2. 采样新的语言目标
          3. 机械臂回 Home
          4. 构建初始观测
        """
        super().reset(seed=seed)
        self._init_ros()  # 首次之后调用为 no-op

        self._step_count = 0
        self._gripper_open = True

        # 采样语言目标
        rng = np.random.default_rng(seed)
        colors = list(COLOR_INDEX.keys())
        if self.task == "short_horizon":
            self.pick_color  = rng.choice(colors)
            remaining = [c for c in colors if c != self.pick_color]
            self.place_color = rng.choice(remaining)
        else:
            # long_horizon：所有块放入对应颜色碗
            self.pick_color  = None
            self.place_color = None

        # 随机化 Gazebo 场景
        self._randomize_objects()
        time.sleep(0.5)  # 等待 Gazebo 物理稳定

        # 机械臂回 Home
        self._return_home()

        obs  = self._build_observation()
        info = {
            "pick_color":  self.pick_color,
            "place_color": self.place_color,
            "task":        self.task,
        }
        return obs, info

    def step(self, action: np.ndarray):
        """
        执行一个原语动作。

        Args:
            action: [primitive(0-1), obj_index(0-5), res_x, res_y]

        Returns:
            obs, reward, terminated, truncated, info
        """
        self._step_count += 1

        # 解析动作
        primitive = int(np.round(np.clip(action[0], 0, 1)))
        obj_idx   = int(np.round(np.clip(action[1], 0, N_TOTAL - 1)))
        res_xy    = np.clip(action[2:4], -0.05, 0.05)

        # 以带噪的物体位置为基准，加上残差得到目标点
        positions = self._get_positions_with_noise()
        base_xy   = positions[obj_idx]
        target_xy = base_xy + res_xy
        target_xy = np.clip(target_xy,
                            [TABLE_X_MIN, TABLE_Y_MIN],
                            [TABLE_X_MAX, TABLE_Y_MAX])

        # 执行原语
        if primitive == 0:
            success = self._execute_pick(target_xy[0], target_xy[1])
        else:
            success = self._execute_place(target_xy[0], target_xy[1])

        # 计算奖励
        reward = self._compute_reward(primitive, obj_idx, target_xy, success)

        # 判断是否结束
        terminated = self._check_task_done()
        truncated  = self._step_count >= self.max_steps

        # 重新构建观测
        obs = self._build_observation()

        info = {
            "primitive":  primitive,
            "obj_idx":    obj_idx,
            "target_xy":  target_xy.tolist(),
            "success":    success,
            "step":       self._step_count,
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        """释放 MoveIt 与 ROS 资源。"""
        if self._ros_initialized:
            try:
                self._return_home()
                moveit_commander.roscpp_shutdown()
            except Exception:
                pass
