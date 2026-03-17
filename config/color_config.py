#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
color_config.py
===============
统一的颜色配置模块。

解决的核心问题：
  原版代码把颜色写死为 [red, green, blue] 三种。
  这个模块让系统动态支持任意多种颜色，颜色列表从 Lab2 的
  vision_config.yaml 里自动读取，不需要改代码。

颜色如何编码进观测向量：
  原来：3维one-hot [red, green, blue]
  现在：N维 embedding index，N = 实际颜色数量
        用整数index而不是one-hot，这样颜色数量变化时网络不需要重新设计

支持的颜色（只要Lab2标定了阈值就能用）：
  red, green, blue, yellow, pink, orange, purple, cyan, ...
"""

import os
import yaml
from typing import Dict, List, Optional, Tuple
import numpy as np


# ── 默认颜色集（没有yaml时的后备方案） ──────────────────────────────────────
# 包含实验室常见的所有方块颜色
DEFAULT_COLORS = [
    "red",
    "green",
    "blue",
    "yellow",
    "pink",
    "orange",
]

# ── HSV阈值（默认值，会被yaml覆盖） ─────────────────────────────────────────
# 格式：{color_name: {hmin, hmax, smin, smax, vmin, vmax}}
DEFAULT_HSV_THRESHOLDS = {
    "red":    {"hmin": 0,   "hmax": 10,  "smin": 100, "smax": 255, "vmin": 80,  "vmax": 255},
    "green":  {"hmin": 40,  "hmax": 80,  "smin": 60,  "smax": 255, "vmin": 40,  "vmax": 255},
    "blue":   {"hmin": 100, "hmax": 130, "smin": 80,  "smax": 255, "vmin": 50,  "vmax": 255},
    "yellow": {"hmin": 20,  "hmax": 35,  "smin": 100, "smax": 255, "vmin": 100, "vmax": 255},
    "pink":   {"hmin": 140, "hmax": 170, "smin": 50,  "smax": 255, "vmin": 100, "vmax": 255},
    "orange": {"hmin": 10,  "hmax": 25,  "smin": 120, "smax": 255, "vmin": 80,  "vmax": 255},
}

# 红色特殊处理：色相环在0/360处回绕，红色需要两段范围
RED_WRAP_COLORS = {"red", "pink"}


class ColorConfig:
    """
    颜色配置管理器。

    负责：
      1. 从 vision_config.yaml 加载颜色列表和HSV阈值
      2. 提供颜色名称 ↔ 整数index的双向映射
      3. 为观测向量生成颜色编码
      4. 为HSV检测提供阈值查询
    """

    def __init__(self, yaml_path: Optional[str] = None):
        """
        Args:
            yaml_path: Lab2 vision_config.yaml 的路径。
                       如果为None，自动搜索常见路径。
                       如果找不到，使用默认颜色集。
        """
        self.hsv_thresholds: Dict[str, dict] = {}
        self.colors: List[str] = []
        self._color_to_idx: Dict[str, int] = {}
        self._idx_to_color: Dict[int, str] = {}

        # 尝试加载yaml
        if yaml_path is None:
            yaml_path = self._find_yaml()

        if yaml_path and os.path.exists(yaml_path):
            self._load_from_yaml(yaml_path)
            print(f"[ColorConfig] 从yaml加载颜色配置：{yaml_path}")
            print(f"[ColorConfig] 支持的颜色：{self.colors}")
        else:
            self._use_defaults()
            print(f"[ColorConfig] 使用默认颜色配置（未找到yaml）")
            print(f"[ColorConfig] 支持的颜色：{self.colors}")

        self._build_index()

    def _find_yaml(self) -> Optional[str]:
        """自动搜索 vision_config.yaml 的常见路径。"""
        candidates = [
            os.path.expanduser(
                "~/sagittarius_ws/src/sagittarius_arm_ros/"
                "sagittarius_object_color_detector/config/vision_config.yaml"),
            os.path.expanduser(
                "~/sagittarius_ws/src/"
                "sagittarius_object_color_detector/config/vision_config.yaml"),
            "./configs/vision_config.yaml",
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    def _load_from_yaml(self, yaml_path: str):
        """从 Lab2 的 vision_config.yaml 读取颜色和HSV阈值。"""
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        # yaml结构：每个颜色名是一个key，值是 hmin/hmax/smin/smax/vmin/vmax 字典
        # 跳过非颜色key（如 calibration, customize 等）
        non_color_keys = {"calibration", "customize", "kx", "bx", "ky", "by"}

        loaded_colors = []
        for key, val in data.items():
            if key in non_color_keys:
                continue
            if not isinstance(val, dict):
                continue
            # 判断是否是颜色阈值块（包含hmin/hmax等字段）
            if "hmin" in val or "hmax" in val:
                color_name = key.lower().strip()
                self.hsv_thresholds[color_name] = {
                    "hmin": float(val.get("hmin", 0)),
                    "hmax": float(val.get("hmax", 360)),
                    "smin": float(val.get("smin", 0)),
                    "smax": float(val.get("smax", 255)),
                    "vmin": float(val.get("vmin", 0)),
                    "vmax": float(val.get("vmax", 255)),
                }
                loaded_colors.append(color_name)

        if not loaded_colors:
            print("[ColorConfig] yaml里没找到颜色阈值，使用默认配置。")
            self._use_defaults()
            return

        self.colors = sorted(loaded_colors)  # 排序确保index稳定

    def _use_defaults(self):
        """使用内置默认颜色和阈值。"""
        self.colors = list(DEFAULT_COLORS)
        self.hsv_thresholds = dict(DEFAULT_HSV_THRESHOLDS)

    def _build_index(self):
        """构建颜色名称 ↔ 整数index的映射。"""
        self._color_to_idx = {c: i for i, c in enumerate(self.colors)}
        self._idx_to_color = {i: c for i, c in enumerate(self.colors)}

    # ── 公共接口 ──────────────────────────────────────────────────────────────

    @property
    def n_colors(self) -> int:
        """支持的颜色总数。"""
        return len(self.colors)

    def color_to_idx(self, color: str) -> int:
        """颜色名称 → 整数index。未知颜色返回0。"""
        return self._color_to_idx.get(color.lower(), 0)

    def idx_to_color(self, idx: int) -> str:
        """整数index → 颜色名称。"""
        return self._idx_to_color.get(idx, self.colors[0])

    def encode_task(self, pick_color: str, place_color: str) -> np.ndarray:
        """
        将任务描述编码为观测向量的语言目标部分。

        返回形状 (2,) 的整数数组：[pick_idx, place_idx]
        网络的embedding层会把这两个整数映射为向量。
        """
        return np.array([
            self.color_to_idx(pick_color),
            self.color_to_idx(place_color),
        ], dtype=np.int32)

    def get_hsv_range(self, color: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取颜色的HSV检测范围（用于cv2.inRange）。

        返回：(lower_bound, upper_bound)，均为 np.array([H, S, V])
        注意：HSV范围是OpenCV格式 H∈[0,180]，S/V∈[0,255]
        """
        if color not in self.hsv_thresholds:
            # 未知颜色返回全范围（会检测所有颜色，不实用但不崩溃）
            return np.array([0, 0, 0]), np.array([180, 255, 255])

        t = self.hsv_thresholds[color]
        # Lab2的yaml里H是0-360，OpenCV里H是0-180，需要除以2
        lower = np.array([t["hmin"] / 2, t["smin"], t["vmin"]], dtype=np.uint8)
        upper = np.array([t["hmax"] / 2, t["smax"], t["vmax"]], dtype=np.uint8)
        return lower, upper

    def needs_wrap(self, color: str) -> bool:
        """红色/粉色的色相跨越0/180边界，需要两段检测。"""
        return color.lower() in RED_WRAP_COLORS

    def add_custom_color(self, name: str, hmin: float, hmax: float,
                         smin: float = 50, smax: float = 255,
                         vmin: float = 50, vmax: float = 255):
        """
        运行时添加自定义颜色（不需要重启，不需要改yaml）。

        用于实验室里有新颜色方块但暂时懒得重跑Lab2标定的情况。
        注意：H值用Lab2格式（0-360），会自动转换。
        """
        name = name.lower()
        self.hsv_thresholds[name] = {
            "hmin": hmin, "hmax": hmax,
            "smin": smin, "smax": smax,
            "vmin": vmin, "vmax": vmax,
        }
        if name not in self.colors:
            self.colors.append(name)
            self.colors.sort()
            self._build_index()
        print(f"[ColorConfig] 添加颜色：{name}")

    def save_to_yaml(self, path: str):
        """把当前颜色配置保存为yaml（方便下次直接加载）。"""
        data = {}
        for color, thresh in self.hsv_thresholds.items():
            data[color] = thresh
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
        print(f"[ColorConfig] 保存到：{path}")


# 全局单例（导入时自动初始化）
_global_config: Optional[ColorConfig] = None


def get_color_config(yaml_path: Optional[str] = None) -> ColorConfig:
    """获取全局颜色配置单例。"""
    global _global_config
    if _global_config is None:
        _global_config = ColorConfig(yaml_path)
    return _global_config
