#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集中加载 .env 并读取 LLM / VLM 两套配置（互不混用 key 与 base_url）。

为何单独文件而不是在每个脚本里各写 load_dotenv + os.getenv？
  - import 本模块一次即完成 load_dotenv，train / eval / test_all 无需重复粘贴；
  - 6 个环境变量名与默认值只维护此处，避免拼写不一致或只改了一处代码。

环境变量：
  LLM_* / VLM_*（各 3 个）— 见下方常量名
  EXPLORELLM_MOVEIT_NS — MoveIt / robot_description 所在 ROS 命名空间（默认 sgr532）
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

# Sagittarius Gazebo 常见 launch 将 URDF/MoveIt 挂在 /sgr532/ 下（如 /sgr532/robot_description）
MOVEIT_NS_ENV = "EXPLORELLM_MOVEIT_NS"
DEFAULT_MOVEIT_NS = "sgr532"


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
    供 moveit_commander.MoveGroupCommander(..., ns=...) 使用。
    与 rosparam 中 robot_description 前缀一致，例如 /sgr532/robot_description → 返回 \"sgr532\"。
    若 launch 把参数放在根命名空间，可将 EXPLORELLM_MOVEIT_NS 设为 root / none / 空。
    """
    raw = os.environ.get(MOVEIT_NS_ENV, DEFAULT_MOVEIT_NS)
    s = str(raw).strip()
    if s.lower() in ("", "/", ".", "root", "none", "~", "global"):
        return ""
    return s.strip("/")
