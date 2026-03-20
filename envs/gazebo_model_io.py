#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 pick_place_scene.world 提取 <model> 片段，供 Gazebo SpawnModel 使用。
避免在代码里维护第二份完整 SDF；spawn 前将 <static>true</static> 改为 false，
便于 SetModelState 与物理一致。
"""
from __future__ import annotations

import os
import re
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORLD_PATH = os.path.join(ROOT, "pick_place_scene.world")

_world_text_cache: Optional[str] = None


def _world_text() -> str:
    global _world_text_cache
    if _world_text_cache is None:
        with open(WORLD_PATH, "r", encoding="utf-8") as f:
            _world_text_cache = f.read()
    return _world_text_cache


def extract_model_inner_sdf(model_name: str) -> Optional[str]:
    """
    返回 world 文件中 <model name="...">...</model> 片段（不含外层 <sdf>）。
    """
    text = _world_text()
    # 非贪婪匹配到第一个闭合 </model>
    pat = rf'<model name="{re.escape(model_name)}">.*?</model>'
    m = re.search(pat, text, re.DOTALL)
    if not m:
        return None
    inner = m.group(0)
    # 训练时由环境 teleport，spawn 时姿态用服务 initial_pose；模型内 pose 归零避免叠加
    inner = re.sub(
        r"<pose>[^<]*</pose>",
        "<pose>0 0 0 0 0 0</pose>",
        inner,
        count=1,
    )
    inner = inner.replace("<static>true</static>", "<static>false</static>")
    return inner


def model_xml_for_spawn(model_name: str) -> Optional[str]:
    inner = extract_model_inner_sdf(model_name)
    if inner is None:
        return None
    return (
        '<?xml version="1.0"?>\n'
        '<sdf version="1.6">\n'
        f"{inner}\n"
        "</sdf>\n"
    )
