#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
camera_perception.py
====================
Real-camera perception module for sim-to-real deployment.

Subscribes to the HSV color detector from Lab2
(sagittarius_object_color_detector) and provides object positions
in robot base frame coordinates.

This replaces the Gazebo GT+noise perception during evaluation on the real robot.
"""

import numpy as np
import rospy
from typing import Dict, Optional, Tuple


# ── Default calibration (replace with your actual values from Lab2 calibration) ──
# These come from the output of camera_calibration_hsv.launch
# Format: y_robot = ky * x_camera + by,  x_robot = kx * y_camera + bx
DEFAULT_CALIB = {
    "kx":  -0.00029,
    "bx":   0.31084,
    "ky":   0.00030,
    "by":   0.09080,
}

# Color index mapping (must match pick_place_env.py)
COLOR_NAMES = ["red", "green", "blue"]


class CameraPerception:
    """
    Bridges the Lab2 HSV color detection system with the RL environment.

    The Lab2 detector publishes object detections as pixel coordinates.
    This class converts them to robot-frame (x, y) coordinates using
    the calibration learned in Lab2.

    Subscribed topics (from sagittarius_object_color_detector):
        /object_detector/red_object   (geometry_msgs/Point)
        /object_detector/green_object (geometry_msgs/Point)
        /object_detector/blue_object  (geometry_msgs/Point)

    If using a custom detection topic, update _setup_subscribers().
    """

    def __init__(self, calib: Dict = None):
        self.calib = calib or DEFAULT_CALIB
        self._detections = {}   # color → (pixel_x, pixel_y) or None

        self._setup_subscribers()
        rospy.loginfo("[CameraPerception] Initialised. Waiting for detections...")

    def _setup_subscribers(self):
        """Subscribe to HSV detector output topics."""
        try:
            from geometry_msgs.msg import Point

            for color in COLOR_NAMES:
                topic = f"/object_detector/{color}_object"
                # Use a closure to capture color correctly
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
        Convert camera pixel coordinates (px, py) to robot base-frame
        coordinates (x, y) using the linear regression calibration.

        The calibration from Lab2 gives:
            x_robot = kx * py + bx
            y_robot = ky * px + by
        """
        x = self.calib["kx"] * py + self.calib["bx"]
        y = self.calib["ky"] * px + self.calib["by"]
        return float(x), float(y)

    def get_object_positions(self,
                             timeout_sec: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Return the latest detected positions for all blocks in robot frame.

        Args:
            timeout_sec: how long to wait for a detection update

        Returns:
            dict: {color: np.array([x, y])} for each detected color
                  Returns np.zeros(2) for undetected objects.
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
        Build the (N_total, 2) positions array for the RL observation,
        using camera detections for blocks and fixed positions for bowls.

        Bowl positions are fixed in the scene (same as training setup).
        """
        from envs.pick_place_env import (BLOCK_NAMES, BOWL_NAMES,
                                         N_TOTAL, TABLE_Z)

        block_positions = self.get_object_positions()

        # Fixed bowl positions (match _randomize_objects in pick_place_env.py)
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

        return np.array(all_positions, dtype=np.float32)  # (N_total, 2)

    def update_calibration(self, kx: float, bx: float,
                           ky: float, by: float):
        """
        Update calibration coefficients.
        Call this after running Lab2's camera_calibration_hsv.launch.
        """
        self.calib = {"kx": kx, "bx": bx, "ky": ky, "by": by}
        rospy.loginfo(f"[CameraPerception] Calibration updated: {self.calib}")

    def load_calibration_from_yaml(self, yaml_path: str):
        """
        Load calibration from the vision_config.yaml saved by Lab2.
        The Lab2 script saves linear regression k,b values there.
        """
        import yaml
        try:
            with open(yaml_path, "r") as f:
                config = yaml.safe_load(f)
            # The Lab2 calibration output format stores kx, bx, ky, by
            # Adjust keys if your YAML has different names
            self.calib["kx"] = float(config.get("kx", self.calib["kx"]))
            self.calib["bx"] = float(config.get("bx", self.calib["bx"]))
            self.calib["ky"] = float(config.get("ky", self.calib["ky"]))
            self.calib["by"] = float(config.get("by", self.calib["by"]))
            rospy.loginfo(f"[CameraPerception] Loaded calibration from {yaml_path}")
        except Exception as e:
            rospy.logwarn(f"[CameraPerception] Failed to load calibration: {e}")
