#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
camera_perception.py
=======================
升级版感知模块，相比原版的核心变化：

  原版：只检测方块（blocks），桶坐标硬编码
  新版：同时检测方块和垃圾桶，两者坐标都是动态的

工作原理：
  同一套HSV检测代码，对6种颜色各跑一次。
  每种颜色检测出来的最大连通域：
    - 如果面积小（<阈值）→ 方块
    - 如果面积大（>阈值）→ 垃圾桶
  因为桶的开口面积明显大于方块的顶面面积。

这样就不需要摄像头刻意区分方块和桶，用同一个检测流程完成。

真机部署时的使用方式：
  1. 把方块和桶随意摆在桌面
  2. 调用 scan_scene() 获取所有物体的当前位置
  3. 把位置注入observation向量
  4. policy执行动作
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point

try:
    from cv_bridge import CvBridge
    HAS_CV_BRIDGE = True
except ImportError:
    HAS_CV_BRIDGE = False

from configs.color_config import ColorConfig, get_color_config


# ── 面积阈值（像素数） ──────────────────────────────────────────────────────
# 方块顶面在摄像头视野里大概是 30×30 = 900 像素
# 垃圾桶开口在摄像头视野里大概是 60×60 = 3600 像素
# 阈值取中间值
BLOCK_MAX_AREA  = 2500   # 小于这个 → 方块
BIN_MIN_AREA    = 2500   # 大于这个 → 垃圾桶
# 两者有重叠区间时，按距离图像中心远近区分（桶通常在桌面外侧）


class DetectedObject:
    """单个检测到的物体。"""
    def __init__(self, color: str, obj_type: str,
                 pixel_xy: Tuple[float, float],
                 robot_xy: Tuple[float, float],
                 area: float):
        self.color     = color       # 颜色名称
        self.obj_type  = obj_type    # "block" 或 "bin"
        self.pixel_xy  = pixel_xy    # 图像像素坐标
        self.robot_xy  = robot_xy    # 机械臂基坐标系(x,y)
        self.area      = area        # 检测区域面积（像素）

    def __repr__(self):
        return (f"DetectedObject({self.color} {self.obj_type} "
                f"robot_xy=({self.robot_xy[0]:.3f},{self.robot_xy[1]:.3f}))")


class CameraPerception:
    """
    升级版摄像头感知模块。

    同时检测方块和垃圾桶，支持任意颜色。
    """

    def __init__(self,
                 color_config: Optional[ColorConfig] = None,
                 calib: Optional[Dict] = None):
        """
        Args:
            color_config: ColorConfig实例，None时用全局单例
            calib: 标定参数 {kx, bx, ky, by}，None时用默认值
        """
        self.color_cfg = color_config or get_color_config()
        self.calib = calib or {
            "kx": -0.00029,
            "bx":  0.31084,
            "ky":  0.00030,
            "by":  0.09080,
        }

        self._latest_image: Optional[np.ndarray] = None
        self._bridge = CvBridge() if HAS_CV_BRIDGE else None

        # 订阅摄像头
        rospy.Subscriber("/usb_cam/image_raw", Image,
                         self._image_cb, queue_size=1)
        rospy.loginfo("[Perception] 初始化完成，等待图像...")

    def _image_cb(self, msg: Image):
        if self._bridge:
            try:
                self._latest_image = self._bridge.imgmsg_to_cv2(msg, "bgr8")
            except Exception:
                pass

    def pixel_to_robot(self, px: float, py: float) -> Tuple[float, float]:
        """像素坐标 → 机械臂基坐标系(x, y)。"""
        x = self.calib["kx"] * py + self.calib["bx"]
        y = self.calib["ky"] * px + self.calib["by"]
        return float(x), float(y)

    def update_calibration(self, kx: float, bx: float,
                           ky: float, by: float):
        self.calib = {"kx": kx, "bx": bx, "ky": ky, "by": by}

    def _detect_color_in_image(
            self,
            img: np.ndarray,
            color: str) -> List[Tuple[float, float, float]]:
        """
        在图像中检测指定颜色的所有连通域。

        Returns:
            List of (center_x, center_y, area) in pixel coordinates,
            sorted by area descending
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower, upper = self.color_cfg.get_hsv_range(color)

        if self.color_cfg.needs_wrap(color):
            # 红色/粉色跨越色相环：需要两段mask
            lower1 = np.array([0,       lower[1], lower[2]], dtype=np.uint8)
            upper1 = np.array([upper[0], upper[1], upper[2]], dtype=np.uint8)
            lower2 = np.array([lower[0], lower[1], lower[2]], dtype=np.uint8)
            upper2 = np.array([180,      upper[1], upper[2]], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
        else:
            mask = cv2.inRange(hsv, lower, upper)

        # 形态学去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 找轮廓
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:  # 过滤噪点
                continue
            M = cv2.moments(cnt)
            if M["m00"] < 1e-6:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            detections.append((cx, cy, area))

        # 按面积从大到小排序
        detections.sort(key=lambda x: x[2], reverse=True)
        return detections

    def scan_scene(self,
                   wait_sec: float = 0.5
                   ) -> Dict[str, Dict[str, Optional[np.ndarray]]]:
        """
        扫描当前场景，返回所有检测到的方块和垃圾桶位置。

        等待 wait_sec 秒确保图像是最新的。

        Returns:
            {
              "blocks": {color: np.array([x,y]) or None},
              "bins":   {color: np.array([x,y]) or None},
            }
            坐标都是机械臂基坐标系下的(x,y)，单位米。
        """
        import time
        time.sleep(wait_sec)

        result = {
            "blocks": {c: None for c in self.color_cfg.colors},
            "bins":   {c: None for c in self.color_cfg.colors},
        }

        if self._latest_image is None:
            rospy.logwarn("[Perception] 没有收到图像！返回空结果。")
            return result

        img = self._latest_image.copy()
        h, w = img.shape[:2]
        img_center_x = w / 2

        for color in self.color_cfg.colors:
            detections = self._detect_color_in_image(img, color)

            if not detections:
                continue

            # 按面积把检测结果分为方块和桶
            # 策略：最大的连通域如果很大 → 桶，较小的 → 方块
            # 但同一颜色可能同时有方块和桶存在（比如绿色方块 + 绿色桶）
            blocks_found = []
            bins_found   = []

            for cx, cy, area in detections:
                if area >= BIN_MIN_AREA:
                    bins_found.append((cx, cy, area))
                else:
                    blocks_found.append((cx, cy, area))

            # 取最大面积的方块（面积最大 = 摄像头看得最清楚）
            if blocks_found:
                cx, cy, _ = blocks_found[0]
                rx, ry = self.pixel_to_robot(cx, cy)
                result["blocks"][color] = np.array([rx, ry], dtype=np.float32)

            # 取最大面积的桶
            if bins_found:
                cx, cy, _ = bins_found[0]
                rx, ry = self.pixel_to_robot(cx, cy)
                result["bins"][color] = np.array([rx, ry], dtype=np.float32)

        return result

    def get_block_position(self, color: str) -> Optional[np.ndarray]:
        """快捷方法：获取指定颜色方块的位置。"""
        scene = self.scan_scene()
        return scene["blocks"].get(color)

    def get_bin_position(self, color: str) -> Optional[np.ndarray]:
        """快捷方法：获取指定颜色垃圾桶的位置。"""
        scene = self.scan_scene()
        return scene["bins"].get(color)

    def get_debug_image(self, color: str) -> Optional[np.ndarray]:
        """返回带有检测框标注的调试图像（用于验证检测效果）。"""
        if self._latest_image is None:
            return None
        img = self._latest_image.copy()
        detections = self._detect_color_in_image(img, color)
        for cx, cy, area in detections:
            obj_type = "bin" if area >= BIN_MIN_AREA else "block"
            color_bgr = (0, 255, 0) if obj_type == "block" else (255, 0, 0)
            cv2.circle(img, (int(cx), int(cy)), 10, color_bgr, 2)
            cv2.putText(img, f"{color} {obj_type} {area:.0f}",
                        (int(cx)+12, int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1)
        return img
