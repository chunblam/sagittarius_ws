#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibration_guide.py
====================
运行Lab2标定程序后，如何找到k、b值并填入项目代码。

直接运行这个脚本可以自动读取Lab2保存的标定结果，并输出需要填入的代码。
"""

# ══════════════════════════════════════════════════════════════════════════════
#  第一步：在哪里找k、b值
# ══════════════════════════════════════════════════════════════════════════════
#
#  运行Lab2的标定程序后：
#    $ roslaunch sagittarius_object_color_detector camera_calibration_hsv.launch
#
#  标定结束时，终端会打印类似这样的内容：
#
#    Linear Regression for x and yc is :  x = -0.00029yc + (0.31084)
#    Linear Regression for y and xc is :  y =  0.00030xc + (0.09080)
#
#  其中：
#    x      = 机械臂基坐标系下的 x 坐标（前后方向，米）
#    y      = 机械臂基坐标系下的 y 坐标（左右方向，米）
#    xc     = 摄像头图像的像素列坐标（0~640）
#    yc     = 摄像头图像的像素行坐标（0~480）
#
#  所以对应关系是：
#    x_robot = kx * yc_pixel + bx      →   kx = -0.00029,  bx = 0.31084
#    y_robot = ky * xc_pixel + by      →   ky =  0.00030,  by = 0.09080
#
#  同时，这些值也会自动保存到配置文件：
#    ~/sagittarius_ws/src/sagittarius_arm_ros/
#        sagittarius_object_color_detector/config/vision_config.yaml
#
# ══════════════════════════════════════════════════════════════════════════════
#  第二步：标定值保存在哪个文件里
# ══════════════════════════════════════════════════════════════════════════════
#
#  vision_config.yaml 的内容结构大致如下：
#
#    calibration:
#      kx: -0.00029
#      bx:  0.31084
#      ky:  0.00030
#      by:  0.09080
#    blue:
#      hmax: 238
#      hmin: 176
#      ...
#
#  注意：不同版本的代码保存的key名称可能略有不同。
#  如果yaml里没有 kx/bx/ky/by，就直接从终端打印的公式里读取数字。
#
# ══════════════════════════════════════════════════════════════════════════════
#  第三步：填入项目代码
# ══════════════════════════════════════════════════════════════════════════════
#
#  打开  perception/camera_perception.py
#  找到  DEFAULT_CALIB  这个字典（大约在第30行），修改为你的实际值：
#
#    DEFAULT_CALIB = {
#        "kx": -0.00029,   # ← 替换为你终端打印的 kx
#        "bx":  0.31084,   # ← 替换为你终端打印的 bx（括号里的数）
#        "ky":  0.00030,   # ← 替换为你终端打印的 ky
#        "by":  0.09080,   # ← 替换为你终端打印的 by（括号里的数）
#    }
#
# ══════════════════════════════════════════════════════════════════════════════

import os
import sys

# ── 自动读取脚本（运行这个文件可以直接输出需要的代码） ────────────────────────

VISION_CONFIG_CANDIDATES = [
    # 常见路径，按优先级排列
    os.path.expanduser(
        "~/sagittarius_ws/src/sagittarius_arm_ros/"
        "sagittarius_object_color_detector/config/vision_config.yaml"),
    os.path.expanduser(
        "~/sagittarius_ws/src/"
        "sagittarius_object_color_detector/config/vision_config.yaml"),
    # 如果你的工作空间名字不同，在这里加路径
]


def find_vision_config():
    for path in VISION_CONFIG_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def read_calibration_from_yaml(yaml_path: str) -> dict:
    """从vision_config.yaml读取标定值，返回 {kx, bx, ky, by}。"""
    import yaml
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    calib = {}

    # 方式1：yaml里有专门的calibration节
    if "calibration" in data:
        c = data["calibration"]
        calib = {
            "kx": float(c.get("kx", 0.0)),
            "bx": float(c.get("bx", 0.0)),
            "ky": float(c.get("ky", 0.0)),
            "by": float(c.get("by", 0.0)),
        }
    # 方式2：直接在顶层
    elif "kx" in data:
        calib = {
            "kx": float(data["kx"]),
            "bx": float(data["bx"]),
            "ky": float(data["ky"]),
            "by": float(data["by"]),
        }
    else:
        print("[WARNING] 在yaml文件里没找到标定值（kx/bx/ky/by）。")
        print("  请从终端打印的公式里手动读取，然后直接编辑 perception/camera_perception.py。")
        return {}

    return calib


def print_code_snippet(calib: dict):
    """打印需要复制粘贴到camera_perception.py的代码片段。"""
    print("\n" + "="*60)
    print("  复制下面的代码，替换 camera_perception.py 里的 DEFAULT_CALIB：")
    print("="*60)
    print(f"""
DEFAULT_CALIB = {{
    "kx": {calib['kx']},
    "bx": {calib['bx']},
    "ky": {calib['ky']},
    "by": {calib['by']},
}}
""")
    print("="*60)
    print("  文件位置：perception/camera_perception.py  第30行左右")
    print("="*60 + "\n")


def update_camera_perception_file(calib: dict):
    """自动将标定值写入 camera_perception.py（可选，谨慎使用）。"""
    target = os.path.join(os.path.dirname(__file__),
                          "perception", "camera_perception.py")
    if not os.path.exists(target):
        print(f"[ERROR] 找不到文件：{target}")
        return False

    with open(target, "r") as f:
        content = f.read()

    # 替换DEFAULT_CALIB块
    import re
    new_block = (
        f'DEFAULT_CALIB = {{\n'
        f'    "kx": {calib["kx"]},\n'
        f'    "bx": {calib["bx"]},\n'
        f'    "ky": {calib["ky"]},\n'
        f'    "by": {calib["by"]},\n'
        f'}}'
    )
    pattern = r'DEFAULT_CALIB\s*=\s*\{[^}]+\}'
    if re.search(pattern, content, re.DOTALL):
        new_content = re.sub(pattern, new_block, content, flags=re.DOTALL)
        with open(target, "w") as f:
            f.write(new_content)
        print(f"[OK] 已自动更新：{target}")
        return True
    else:
        print("[WARNING] 没找到DEFAULT_CALIB块，请手动编辑。")
        return False


if __name__ == "__main__":
    print("\n=== 标定值读取工具 ===\n")

    # 1. 尝试找到vision_config.yaml
    yaml_path = find_vision_config()

    if yaml_path:
        print(f"[OK] 找到标定文件：{yaml_path}")
        calib = read_calibration_from_yaml(yaml_path)
        if calib:
            print_code_snippet(calib)

            ans = input("是否自动写入 perception/camera_perception.py？(y/n) ")
            if ans.strip().lower() == "y":
                update_camera_perception_file(calib)
        else:
            print("[提示] yaml文件存在但没有标定数值。")
            print("  请从运行标定程序时终端输出的两行公式里手动读取 kx/bx/ky/by。")
    else:
        print("[提示] 没找到 vision_config.yaml 文件。")
        print("  请先运行Lab2的标定程序：")
        print("  $ roslaunch sagittarius_object_color_detector camera_calibration_hsv.launch")
        print()
        print("  标定完成后，从终端打印内容里找这两行：")
        print("  Linear Regression for x and yc is :  x = kx*yc + (bx)")
        print("  Linear Regression for y and xc is :  y = ky*xc + (by)")
        print()
        print("  然后运行标定读取工具（本脚本）或手动填入 perception/camera_perception.py。")

    print()
