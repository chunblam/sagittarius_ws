#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pick_place_env.py
=================
Gazebo gym.Env wrapper for Sagittarius pick-and-place task.

This is the core environment that bridges:
  - stable-baselines3 (SAC training loop)
  - ROS / MoveIt (motion execution)
  - Gazebo (simulation state & physics)

Observation space:
  - image_patches : (N_obj, 3, 28, 28)  per-object RGB crops
  - obj_positions : (N_obj, 2)           x,y positions with Gaussian noise
  - gripper_state : (1,)                 0=open, 1=closed
  - lang_goal     : (N_obj*2,)           one-hot encoding of pick/place targets

Action space (residual, object-centric):
  - primitive  : int  0=pick, 1=place
  - obj_index  : int  which object to act on
  - residual_xy: (2,) offset from object center (meters)
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


# ── Scene configuration ──────────────────────────────────────────────────────

# Object names as they appear in Gazebo
BLOCK_NAMES  = ["red_block",   "green_block",  "blue_block"]
BOWL_NAMES   = ["red_bowl",    "green_bowl",   "blue_bowl"]
ALL_OBJECTS  = BLOCK_NAMES + BOWL_NAMES

# Colors used for language encoding index
COLOR_INDEX  = {"red": 0, "green": 1, "blue": 2}

# Table workspace bounds (meters, in robot base frame)
TABLE_X_MIN, TABLE_X_MAX = 0.15, 0.40
TABLE_Y_MIN, TABLE_Y_MAX = -0.20, 0.20
TABLE_Z = 0.02  # surface height above base

# Grasp/place heights
APPROACH_HEIGHT  = 0.12   # meters above table for pre-grasp
GRASP_HEIGHT     = 0.005  # meters above object center for actual grasp
PLACE_HEIGHT     = 0.06   # meters above bowl center for release

# Image crop size
CROP_SIZE = 28

# Noise sigma: half the crop radius in meters (as per ExploRLLM paper)
# Assumes each crop covers ~8cm radius → σ = 0.04
POSITION_NOISE_SIGMA = 0.04

# Number of objects tracked (blocks only for pick, all for place reference)
N_OBJECTS = len(BLOCK_NAMES)      # 3
N_TARGETS = len(BOWL_NAMES)       # 3
N_TOTAL   = N_OBJECTS + N_TARGETS # 6


# ── Environment ───────────────────────────────────────────────────────────────

class SagittariusPickPlaceEnv(gym.Env):
    """
    Single-step pick-or-place environment for Sagittarius SGR532.

    Each call to step() executes ONE primitive (pick OR place).
    The agent must chain pick → place to complete a task.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self,
                 task: str = "short_horizon",
                 max_steps: int = 10,
                 noise_sigma: float = POSITION_NOISE_SIGMA,
                 render_mode: str = None):
        """
        Args:
            task        : "short_horizon" (pick one, place one) or
                          "long_horizon" (sort all blocks)
            max_steps   : max primitive steps per episode
            noise_sigma : std of Gaussian noise added to GT positions
            render_mode : gymnasium render mode (unused, Gazebo handles viz)
        """
        super().__init__()

        self.task         = task
        self.max_steps    = max_steps
        self.noise_sigma  = noise_sigma
        self.render_mode  = render_mode
        self._step_count  = 0
        self._gripper_open = True

        # Current episode language goal
        self.pick_color   = None   # e.g. "red"
        self.place_color  = None   # e.g. "blue"

        # ── Gymnasium spaces ──────────────────────────────────────────────────
        # Observation: flat dict-like Box for stable-baselines3 compatibility
        #   We flatten everything; CustomSACPolicy will un-flatten.
        #
        # Layout: [img_patches(N_total*3*28*28), positions(N_total*2),
        #          gripper(1), lang_onehot(6)]
        img_dim   = N_TOTAL * 3 * CROP_SIZE * CROP_SIZE  # 6*3*28*28 = 14112
        pos_dim   = N_TOTAL * 2                           # 12
        grip_dim  = 1
        lang_dim  = N_OBJECTS + N_TARGETS                 # 6  (one-hot pick + place)
        obs_dim   = img_dim + pos_dim + grip_dim + lang_dim

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )

        # Action: [primitive(1), obj_index(1), residual_x(1), residual_y(1)]
        # primitive and obj_index are continuous here; we discretize via rounding.
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, -0.05, -0.05], dtype=np.float32),
            high=np.array([1.0, float(N_TOTAL-1), 0.05, 0.05], dtype=np.float32)
        )

        # ── ROS / MoveIt init ─────────────────────────────────────────────────
        self._ros_initialized = False
        self._moveit_arm      = None
        self._moveit_gripper  = None
        self._model_states    = None  # latest /gazebo/model_states message
        self._image_cache     = {}    # object_name -> latest np.ndarray (H,W,3)

        # Deferred: call _init_ros() on first reset() to avoid
        # rospy.init_node conflicts when running multiple envs.

    # ── ROS initialisation ────────────────────────────────────────────────────

    def _init_ros(self):
        """Initialise ROS node, MoveIt, and topic subscriptions."""
        if self._ros_initialized:
            return

        rospy.loginfo("[Env] Initialising ROS node...")
        # Only init node if not already running (allows external init)
        if not rospy.core.is_initialized():
            rospy.init_node("explorllm_env", anonymous=True)

        # MoveIt commanders
        moveit_commander.roscpp_initialize(sys.argv)
        self._moveit_arm = moveit_commander.MoveGroupCommander("sagittarius_arm")
        self._moveit_gripper = moveit_commander.MoveGroupCommander("sagittarius_gripper")

        # Tolerances & speed
        self._moveit_arm.set_goal_position_tolerance(0.005)
        self._moveit_arm.set_goal_orientation_tolerance(0.02)
        self._moveit_arm.set_max_velocity_scaling_factor(0.4)
        self._moveit_arm.set_max_acceleration_scaling_factor(0.4)
        self._moveit_arm.allow_replanning(True)

        self._moveit_gripper.set_goal_joint_tolerance(0.001)
        self._moveit_gripper.set_max_velocity_scaling_factor(0.5)
        self._moveit_gripper.set_max_acceleration_scaling_factor(0.5)

        # Subscribe to Gazebo model states (ground-truth positions)
        rospy.Subscriber("/gazebo/model_states", ModelStates,
                         self._model_states_cb, queue_size=1)

        # Subscribe to camera image (for crop extraction at eval time)
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

        # Gazebo services
        rospy.wait_for_service("/gazebo/set_model_state", timeout=10)
        self._set_model_state = rospy.ServiceProxy(
            "/gazebo/set_model_state", SetModelState)

        # Unpause physics if paused
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

    # ── Gazebo object management ───────────────────────────────────────────────

    def _get_object_pose(self, name: str) -> np.ndarray:
        """
        Return (x, y, z) of named object from latest ModelStates.
        Returns zeros if object not found.
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
        """Teleport blocks to random positions on the table; bowls to fixed zones."""
        rng = np.random.default_rng()

        # Randomise block positions (with collision avoidance: min 8cm apart)
        placed = []
        for name in BLOCK_NAMES:
            for _ in range(50):  # max attempts
                x = rng.uniform(TABLE_X_MIN + 0.05, TABLE_X_MAX - 0.05)
                y = rng.uniform(TABLE_Y_MIN + 0.05, TABLE_Y_MAX - 0.05)
                if all(np.linalg.norm([x - px, y - py]) > 0.08
                       for (px, py) in placed):
                    placed.append((x, y))
                    self._teleport(name, x, y, TABLE_Z + 0.02)
                    break

        # Bowls at fixed positions (outside block randomisation zone)
        bowl_positions = [
            (0.35, -0.15, TABLE_Z),   # red bowl
            (0.35,  0.00, TABLE_Z),   # green bowl
            (0.35,  0.15, TABLE_Z),   # blue bowl
        ]
        for name, (x, y, z) in zip(BOWL_NAMES, bowl_positions):
            self._teleport(name, x, y, z)

    def _teleport(self, name: str, x: float, y: float, z: float):
        """Move a Gazebo model to (x,y,z) instantly."""
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

    # ── Observation building ───────────────────────────────────────────────────

    def _get_positions_with_noise(self) -> np.ndarray:
        """
        Returns (N_total, 2) array of x,y positions with Gaussian noise.
        This simulates real-camera detection uncertainty during training.
        """
        positions = []
        for name in ALL_OBJECTS:
            xyz = self._get_object_pose(name)
            noise = np.random.normal(0, self.noise_sigma, size=2)
            positions.append(xyz[:2] + noise)
        return np.array(positions, dtype=np.float32)  # (N_total, 2)

    def _get_image_crops(self, positions: np.ndarray) -> np.ndarray:
        """
        Extract 28x28 RGB crops around each object position.
        Falls back to blank crops if camera image unavailable.

        Args:
            positions: (N_total, 2) object positions in robot frame

        Returns:
            crops: (N_total, 3, 28, 28) float32 in [0,1]
        """
        import cv2
        crops = []
        img = self._latest_image

        for i, name in enumerate(ALL_OBJECTS):
            if img is not None:
                # Project robot-frame (x,y) to image pixel (u,v)
                # This uses the calibration from Lab2's camera_calibration_hsv
                # For training we use a simple perspective approximation
                u, v = self._robot_to_pixel(positions[i])
                h, w = img.shape[:2]
                u, v = int(np.clip(u, 14, w-14)), int(np.clip(v, 14, h-14))
                crop = img[v-14:v+14, u-14:u+14]
                if crop.shape[:2] != (28, 28):
                    crop = np.zeros((28, 28, 3), dtype=np.uint8)
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop = crop.astype(np.float32) / 255.0
            else:
                # Blank crop during pure Gazebo training (no camera needed)
                crop = np.zeros((28, 28, 3), dtype=np.float32)

            # (H,W,C) → (C,H,W) for PyTorch convention
            crops.append(crop.transpose(2, 0, 1))

        return np.array(crops, dtype=np.float32)  # (N_total, 3, 28, 28)

    def _robot_to_pixel(self, xy: np.ndarray):
        """
        Convert robot-frame (x,y) to camera pixel (u,v).
        Uses linear regression coefficients from Lab2 calibration.
        Replace with your actual calibrated values after running
        camera_calibration_hsv.launch.

        Placeholder: assumes a top-down camera at ~60cm height.
        """
        # These values come from the k,b output of the calibration script.
        # After running Lab2 calibration, replace with vision_config.yaml values.
        kx, bx = -0.00029, 0.31084   # x_robot = kx*v_pixel + bx
        ky, by =  0.00030, 0.09080   # y_robot = ky*u_pixel + by

        # Invert to get pixel from robot coords
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
        Build 6-dim one-hot: [pick_r, pick_g, pick_b, place_r, place_g, place_b]
        """
        vec = np.zeros(N_OBJECTS + N_TARGETS, dtype=np.float32)
        if self.pick_color  in COLOR_INDEX:
            vec[COLOR_INDEX[self.pick_color]]  = 1.0
        if self.place_color in COLOR_INDEX:
            vec[N_OBJECTS + COLOR_INDEX[self.place_color]] = 1.0
        return vec

    def _build_observation(self) -> np.ndarray:
        """
        Assemble full flat observation vector.
        Layout: [img_patches | positions | gripper | lang]
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

    # ── MoveIt motion primitives ───────────────────────────────────────────────

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
        Move arm end-effector to Cartesian pose (x,y,z) + quaternion.
        Returns True if plan + execution succeeded.
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
        Full pick primitive:
          1. Open gripper
          2. Move above target (approach height)
          3. Descend to grasp height
          4. Close gripper
          5. Lift back up
        Returns True if all steps succeeded.
        """
        self._open_gripper()

        # Approach from above
        ok = self._move_to_pose(x, y, TABLE_Z + APPROACH_HEIGHT)
        if not ok:
            rospy.logwarn("[Env] Pick approach planning failed.")
            return False

        # Descend
        ok = self._move_to_pose(x, y, TABLE_Z + GRASP_HEIGHT)
        if not ok:
            rospy.logwarn("[Env] Pick descend planning failed.")
            return False

        # Grasp
        self._close_gripper()

        # Lift
        ok = self._move_to_pose(x, y, TABLE_Z + APPROACH_HEIGHT)
        if not ok:
            rospy.logwarn("[Env] Pick lift planning failed.")
            return False

        return True

    def _execute_place(self, x: float, y: float) -> bool:
        """
        Full place primitive:
          1. Move above target bowl (approach height)
          2. Descend to place height
          3. Open gripper
          4. Lift back up
        Returns True if all steps succeeded.
        """
        # Move above bowl
        ok = self._move_to_pose(x, y, TABLE_Z + APPROACH_HEIGHT)
        if not ok:
            rospy.logwarn("[Env] Place approach planning failed.")
            return False

        # Descend slightly
        ok = self._move_to_pose(x, y, TABLE_Z + PLACE_HEIGHT)
        if not ok:
            rospy.logwarn("[Env] Place descend planning failed.")
            return False

        # Release
        self._open_gripper()

        # Lift
        self._move_to_pose(x, y, TABLE_Z + APPROACH_HEIGHT)

        return True

    def _return_home(self):
        """Return arm to safe 'Home' named pose."""
        try:
            self._moveit_arm.set_named_target("Home")
            self._moveit_arm.go(wait=True)
            self._open_gripper()
        except Exception as e:
            rospy.logwarn(f"[Env] Failed to return home: {e}")

    # ── Reward computation ────────────────────────────────────────────────────

    def _compute_reward(self, primitive: int, obj_idx: int,
                        action_xy: np.ndarray,
                        success: bool) -> float:
        """
        Reward = dense distance component + sparse success component.

        Dense: negative distance from end-effector to target (encourages progress)
        Sparse: +1.0 if block is inside correct bowl after place action
        """
        r = 0.0

        if primitive == 0:  # pick
            # Dense: reward for moving toward the target block
            target_name = BLOCK_NAMES[obj_idx % N_OBJECTS]
            target_pos  = self._get_object_pose(target_name)[:2]
            dist = np.linalg.norm(action_xy - target_pos)
            r += -dist  # closer is better

        elif primitive == 1:  # place
            # Dense: reward for proximity to correct bowl
            bowl_name  = f"{self.place_color}_bowl"
            bowl_pos   = self._get_object_pose(bowl_name)[:2]
            dist = np.linalg.norm(action_xy - bowl_pos)
            r += -dist

            # Sparse: check if block is inside the bowl
            if success:
                block_name = f"{self.pick_color}_block"
                block_pos  = self._get_object_pose(block_name)[:2]
                bowl_pos2  = self._get_object_pose(bowl_name)[:2]
                if np.linalg.norm(block_pos - bowl_pos2) < 0.05:
                    r += 1.0  # task success!

        if not success:
            r -= 0.2  # penalty for motion planning failure

        return float(r)

    def _check_task_done(self) -> bool:
        """Return True if the episode-level task goal is completed."""
        if self.task == "short_horizon":
            # Done when pick_color block is inside place_color bowl
            block_pos = self._get_object_pose(f"{self.pick_color}_block")[:2]
            bowl_pos  = self._get_object_pose(f"{self.place_color}_bowl")[:2]
            return np.linalg.norm(block_pos - bowl_pos) < 0.05

        elif self.task == "long_horizon":
            # Done when all blocks are in their matching bowls
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
        Reset episode:
          1. Randomise object positions in Gazebo
          2. Sample a new language goal
          3. Return home
          4. Build initial observation
        """
        super().reset(seed=seed)
        self._init_ros()  # no-op after first call

        self._step_count = 0
        self._gripper_open = True

        # Sample language goal
        rng = np.random.default_rng(seed)
        colors = list(COLOR_INDEX.keys())
        if self.task == "short_horizon":
            self.pick_color  = rng.choice(colors)
            remaining = [c for c in colors if c != self.pick_color]
            self.place_color = rng.choice(remaining)
        else:
            # long_horizon: all blocks to matching bowls
            self.pick_color  = None
            self.place_color = None

        # Randomise Gazebo scene
        self._randomize_objects()
        time.sleep(0.5)  # let Gazebo settle physics

        # Return arm to home
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
        Execute one primitive action.

        Args:
            action: [primitive(0-1), obj_index(0-5), res_x, res_y]

        Returns:
            obs, reward, terminated, truncated, info
        """
        self._step_count += 1

        # Decode action
        primitive = int(np.round(np.clip(action[0], 0, 1)))
        obj_idx   = int(np.round(np.clip(action[1], 0, N_TOTAL - 1)))
        res_xy    = np.clip(action[2:4], -0.05, 0.05)

        # Get object position (noisy) as base, add residual
        positions = self._get_positions_with_noise()
        base_xy   = positions[obj_idx]
        target_xy = base_xy + res_xy
        target_xy = np.clip(target_xy,
                            [TABLE_X_MIN, TABLE_Y_MIN],
                            [TABLE_X_MAX, TABLE_Y_MAX])

        # Execute primitive
        if primitive == 0:
            success = self._execute_pick(target_xy[0], target_xy[1])
        else:
            success = self._execute_place(target_xy[0], target_xy[1])

        # Compute reward
        reward = self._compute_reward(primitive, obj_idx, target_xy, success)

        # Check termination
        terminated = self._check_task_done()
        truncated  = self._step_count >= self.max_steps

        # Rebuild observation
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
        """Cleanup MoveIt and ROS resources."""
        if self._ros_initialized:
            try:
                self._return_home()
                moveit_commander.roscpp_shutdown()
            except Exception:
                pass
