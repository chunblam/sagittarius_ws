#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集中加载 .env 并读取 LLM / VLM 两套配置（互不混用 key 与 base_url）。

为何单独文件而不是在每个脚本里各写 load_dotenv + os.getenv？
  - import 本模块一次即完成 load_dotenv，train / eval / test_all 无需重复粘贴；
  - 6 个环境变量名与默认值只维护此处，避免拼写不一致或只改了一处代码。

环境变量：
  LLM_* / VLM_*（各 3 个）— 见下方常量名
  EXPLORELLM_MOVEIT_NS — 机器人/MoveIt 所在 ROS 命名空间（默认 sgr532）
  EXPLORELLM_MOVEIT_WAIT — 连接 move_group action 超时秒数（默认 30）
  EXPLORELLM_MOVEIT_PLANNING_TIME_S — 单次 OMPL 规划时间上限（秒，默认 8，与 pick_place_env 一致）
  EXPLORELLM_GAZEBO_RESET_SIMULATION_ON_HOME_FAIL — MoveIt 回 home 失败后是否调用
      /gazebo/reset_simulation 恢复物理（默认 1，训练推荐；无 Gazebo 时自动跳过）
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

# ── 环境变量名（供文档与其它模块引用）────────────────────────────────────────
LLM_API_KEY_ENV = "LLM_API_KEY"
LLM_BASE_URL_ENV = "LLM_BASE_URL"
LLM_MODEL_ENV = "LLM_MODEL"
VLM_API_KEY_ENV = "VLM_API_KEY"
VLM_BASE_URL_ENV = "VLM_BASE_URL"
VLM_MODEL_ENV = "VLM_MODEL"

DEFAULT_LLM_MODEL = "deepseek-v3"
DEFAULT_VLM_MODEL = "qwen-vl"

# Sagittarius Gazebo 常见 launch：参数在 /sgr532/robot_description，move_group 在 /sgr532/move_group
MOVEIT_NS_ENV = "EXPLORELLM_MOVEIT_NS"
MOVEIT_WAIT_ENV = "EXPLORELLM_MOVEIT_WAIT"
MOVEIT_PLANNING_TIME_ENV = "EXPLORELLM_MOVEIT_PLANNING_TIME_S"
GAZEBO_RESET_SIM_ON_HOME_FAIL_ENV = "EXPLORELLM_GAZEBO_RESET_SIMULATION_ON_HOME_FAIL"
DEFAULT_MOVEIT_NS = "sgr532"
DEFAULT_MOVEIT_WAIT_S = 15.0


def _strip_or_none(key: str) -> Optional[str]:
    v = os.environ.get(key, "")
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def llm_api_key() -> str:
    return os.environ.get(LLM_API_KEY_ENV, "") or ""


def llm_base_url() -> Optional[str]:
    return _strip_or_none(LLM_BASE_URL_ENV)


def llm_model() -> str:
    v = _strip_or_none(LLM_MODEL_ENV)
    return v if v else DEFAULT_LLM_MODEL


def vlm_api_key() -> str:
    return os.environ.get(VLM_API_KEY_ENV, "") or ""


def vlm_base_url() -> Optional[str]:
    return _strip_or_none(VLM_BASE_URL_ENV)


def vlm_model() -> str:
    v = _strip_or_none(VLM_MODEL_ENV)
    return v if v else DEFAULT_VLM_MODEL


def moveit_commander_ns() -> str:
    """
    MoveIt 命名空间短名（无斜杠），例如 sgr532。
    与 rosparam 前缀一致：/sgr532/robot_description、话题 /sgr532/move_group/...
    若 launch 使用根命名空间，将 EXPLORELLM_MOVEIT_NS 设为 root / none / 空。
    """
    raw = os.environ.get(MOVEIT_NS_ENV, DEFAULT_MOVEIT_NS)
    s = str(raw).strip()
    if s.lower() in ("", "/", ".", "root", "none", "~", "global"):
        return ""
    return s.strip("/")


def moveit_planning_time_s() -> float:
    """
    单次 MoveIt plan() 的 OMPL 时间上限（秒）。
    默认 4.0；可通过 EXPLORELLM_MOVEIT_PLANNING_TIME_S 覆盖（例如 2 加快失败、8 提高难场景成功率）。
    """
    raw = os.environ.get(MOVEIT_PLANNING_TIME_ENV, "").strip()
    if not raw:
        return 4
    try:
        t = float(raw)
        return max(0.5, min(t, 120.0))
    except ValueError:
        return 4


def moveit_robot_description_param() -> str:
    """robot_description 在参数服务器上的绝对路径，如 /sgr532/robot_description。"""
    ns = moveit_commander_ns()
    if not ns:
        return "/robot_description"
    return f"/{ns}/robot_description"


def gazebo_reset_simulation_on_home_fail() -> bool:
    """
    MoveIt 回 named home 失败时，是否调用 Gazebo /gazebo/reset_simulation。
    默认 True：碰撞后关节/物理卡住时仍可恢复训练；随后 reset() 仍会 _randomize_scene 摆放物体。
    设为 0/false 可关闭（例如调试 MoveIt）。无 Gazebo 时服务不可用，环境会忽略。
    """
    raw = os.environ.get(GAZEBO_RESET_SIM_ON_HOME_FAIL_ENV, "1")
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def moveit_move_group_commander_kwargs() -> dict:
    """
    传给 MoveGroupCommander(..., **kwargs) 的额外关键字。

    在命名空间下必须同时：
      - robot_description 用绝对路径（否则 C++ 侧解析不到 /ns/robot_description）；
      - ns 与 move_group action 所在空间一致（否则连不上 /ns/move_group）。
    根命名空间时返回空 dict，由调用方使用默认 MoveGroupCommander(name)。
    """
    ns = moveit_commander_ns()
    if not ns:
        return {}
    raw_wait = os.environ.get(MOVEIT_WAIT_ENV, "")
    if raw_wait.strip():
        try:
            wait_s = float(raw_wait.strip())
        except ValueError:
            wait_s = DEFAULT_MOVEIT_WAIT_S
    else:
        wait_s = DEFAULT_MOVEIT_WAIT_S
    return {
        "robot_description": f"/{ns}/robot_description",
        "ns": ns,
        "wait_for_servers": wait_s,
    }
