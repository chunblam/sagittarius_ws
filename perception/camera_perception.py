#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
camera_perception.py
====================
真实相机感知模块，用于 sim-to-real 部署。

订阅 Lab2 的 HSV 颜色检测（sagittarius_object_color_detector），
在机器人基座坐标系下提供物体位置。

在真实机械臂上评估时，用本模块替代 Gazebo 真值+噪声的感知。
"""

import numpy as np
import rospy
from typing import Dict, Optional, Tuple


# ── 默认标定（请用 Lab2 标定得到的实际值替换）──────────────────────────────────
# 来自 camera_calibration_hsv.launch 的输出
# 格式：y_robot = ky * x_camera + by,  x_robot = kx * y_camera + bx
DEFAULT_CALIB = {
    "kx":  -0.00029,
    "bx":   0.31084,
    "ky":   0.00030,
    "by":   0.09080,
}

# 颜色名列表（须与 pick_place_env.py 一致）
COLOR_NAMES = ["red", "green", "blue"]


class CameraPerception:
    """
    将 Lab2 的 HSV 颜色检测与 RL 环境桥接。

    Lab2 检测器发布的是像素坐标，本类用 Lab2 学到的标定
    将其转换为机器人基座坐标系下的 (x, y)。

    订阅的话题（来自 sagittarius_object_color_detector）：
        /object_detector/red_object   (geometry_msgs/Point)
        /object_detector/green_object (geometry_msgs/Point)
        /object_detector/blue_object  (geometry_msgs/Point)

    若使用自定义检测话题，请修改 _setup_subscribers()。
    """

    def __init__(self, calib: Dict = None):
        self.calib = calib or DEFAULT_CALIB
        self._detections = {}   # 颜色 → (pixel_x, pixel_y) 或 None

        self._setup_subscribers()
        rospy.loginfo("[CameraPerception] Initialised. Waiting for detections...")

    def _setup_subscribers(self):
        """订阅 HSV 检测器输出的话题。"""
        try:
            from geometry_msgs.msg import Point

            for color in COLOR_NAMES:
                topic = f"/object_detector/{color}_object"
                # 用闭包正确捕获 color
                def make_cb(c):
                    def cb(msg):
                        self._detections[c] = (float(msg.x), float(msg.y))
                    return cb
                rospy.Subscriber(topic, Point, make_cb(color), queue_size=1)

            rospy.loginfo("[CameraPerception] Subscribed to detection topics.")

        except Exception as e:
            rospy.logwarn(f"[CameraPerception] Subscriber setup failed: {e}. "
                          f"Detection topics may not be available yet.")

    def pixel_to_robot(self, px: float, py: float) -> Tuple[float, float]:
        """
        用线性回归标定将相机像素坐标 (px, py) 转换为
        机器人基座坐标系下的 (x, y)。

        Lab2 标定关系：
            x_robot = kx * py + bx
            y_robot = ky * px + by
        """
        x = self.calib["kx"] * py + self.calib["bx"]
        y = self.calib["ky"] * px + self.calib["by"]
        return float(x), float(y)

    def get_object_positions(self,
                             timeout_sec: float = 1.0) -> Dict[str, np.ndarray]:
        """
        返回所有块在机器人坐标系下的最新检测位置。

        Args:
            timeout_sec: 等待检测更新的最长时间（秒）

        Returns:
            dict: 每种颜色 {color: np.array([x, y])}；未检测到的物体为 np.zeros(2)。
        """
        deadline = rospy.Time.now() + rospy.Duration(timeout_sec)
        while rospy.Time.now() < deadline:
            if len(self._detections) >= len(COLOR_NAMES):
                break
            rospy.sleep(0.05)

        positions = {}
        for color in COLOR_NAMES:
            if color in self._detections:
                px, py = self._detections[color]
                x, y = self.pixel_to_robot(px, py)
                positions[color] = np.array([x, y], dtype=np.float32)
            else:
                rospy.logwarn(f"[CameraPerception] No detection for {color}.")
                positions[color] = np.zeros(2, dtype=np.float32)

        return positions

    def build_observation_positions(self) -> np.ndarray:
        """
        构建 RL 观测用的 (N_total, 2) 位置数组：
        块用相机检测，碗用固定位置。

        碗的位置与训练时场景一致（固定）。
        """
        from envs.pick_place_env import (BLOCK_NAMES, BOWL_NAMES,
                                         N_TOTAL, TABLE_Z)

        block_positions = self.get_object_positions()

        # 碗的固定位置（与 pick_place_env.py 中 _randomize_objects 一致）
        bowl_positions_fixed = {
            "red_bowl":   np.array([0.35, -0.15], dtype=np.float32),
            "green_bowl": np.array([0.35,  0.00], dtype=np.float32),
            "blue_bowl":  np.array([0.35,  0.15], dtype=np.float32),
        }

        all_positions = []
        for name in BLOCK_NAMES:
            color = name.split("_")[0]
            all_positions.append(block_positions.get(
                color, np.zeros(2, dtype=np.float32)))
        for name in BOWL_NAMES:
            all_positions.append(bowl_positions_fixed.get(
                name, np.zeros(2, dtype=np.float32)))

        return np.array(all_positions, dtype=np.float32)  # 形状 (N_total, 2)

    def update_calibration(self, kx: float, bx: float,
                           ky: float, by: float):
        """
        更新标定系数。
        在运行 Lab2 的 camera_calibration_hsv.launch 后调用。
        """
        self.calib = {"kx": kx, "bx": bx, "ky": ky, "by": by}
        rospy.loginfo(f"[CameraPerception] Calibration updated: {self.calib}")

    def load_calibration_from_yaml(self, yaml_path: str):
        """
        从 Lab2 保存的 vision_config.yaml 加载标定。
        Lab2 脚本将线性回归的 k、b 保存在该文件中。
        """
        import yaml
        try:
            with open(yaml_path, "r") as f:
                config = yaml.safe_load(f)
            # Lab2 标定输出格式为 kx, bx, ky, by；若 YAML 键名不同请相应修改
            self.calib["kx"] = float(config.get("kx", self.calib["kx"]))
            self.calib["bx"] = float(config.get("bx", self.calib["bx"]))
            self.calib["ky"] = float(config.get("ky", self.calib["ky"]))
            self.calib["by"] = float(config.get("by", self.calib["by"]))
            rospy.loginfo(f"[CameraPerception] Loaded calibration from {yaml_path}")
        except Exception as e:
            rospy.logwarn(f"[CameraPerception] Failed to load calibration: {e}")
