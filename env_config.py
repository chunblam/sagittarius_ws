#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集中加载 .env 并读取 LLM / VLM 两套配置（互不混用 key 与 base_url）。

为何单独文件而不是在每个脚本里各写 load_dotenv + os.getenv？
  - import 本模块一次即完成 load_dotenv，train / eval / test_all 无需重复粘贴；
  - 6 个环境变量名与默认值只维护此处，避免拼写不一致或只改了一处代码。

环境变量（共 6 个，与 .env.example 一致）：
  LLM_API_KEY, LLM_BASE_URL, LLM_MODEL   — 训练探索策略（llm_policy）
  VLM_API_KEY, VLM_BASE_URL, VLM_MODEL   — 真机视觉感知（camera_perception）
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
