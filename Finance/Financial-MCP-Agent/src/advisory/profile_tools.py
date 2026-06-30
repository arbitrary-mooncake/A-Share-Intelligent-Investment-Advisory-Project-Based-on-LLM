"""
MCP 工具 — 供 LLM 调用的用户画像工具与 Function Calling Schema。

提供:
- get_profile_manager: 懒加载单例工厂
- update_user_profile_tool: 更新用户画像的 MCP 工具函数
- get_user_profile_tool: 获取用户画像 JSON 的 MCP 工具函数
- PROFILE_TOOLS_SCHEMA: OpenAI Function Call 格式的工具定义列表
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src.advisory.user_profile import UserProfileManager

# ── 全局单例（懒加载） ──

_manager_instance: Optional[UserProfileManager] = None


def get_profile_manager() -> UserProfileManager:
    """懒加载单例工厂，返回全局唯一的 UserProfileManager 实例。"""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = UserProfileManager()
    return _manager_instance


def update_user_profile_tool(
    risk_tolerance: Optional[str] = None,
    investment_horizon: Optional[str] = None,
    favorite_sectors: Optional[List[str]] = None,
    avoid_sectors: Optional[List[str]] = None,
    investment_style: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """MCP 工具：更新用户投资画像。

    Args:
        risk_tolerance: 风险承受能力 (Conservative/Balanced/Aggressive/Unknown)
        investment_horizon: 投资周期偏好 (Short/Medium/Long/Unknown)
        favorite_sectors: 偏好板块列表
        avoid_sectors: 回避板块列表
        investment_style: 投资风格描述
        **kwargs: 其余参数合并到 custom_preferences

    Returns:
        描述变更结果的中文字符串。
    """
    manager = get_profile_manager()
    return manager.update_profile(
        risk_tolerance=risk_tolerance,
        investment_horizon=investment_horizon,
        favorite_sectors=favorite_sectors,
        avoid_sectors=avoid_sectors,
        investment_style=investment_style,
        **kwargs,
    )


def get_user_profile_tool() -> str:
    """MCP 工具：获取当前用户投资画像的完整 JSON 字符串。"""
    manager = get_profile_manager()
    return json.dumps(manager.get_profile(), ensure_ascii=False, indent=2)


# ── OpenAI Function Call Schema ──

PROFILE_TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "update_user_profile",
            "description": "更新用户投资画像（风险承受能力、投资周期、偏好/回避板块等）",
            "parameters": {
                "type": "object",
                "properties": {
                    "risk_tolerance": {
                        "type": "string",
                        "enum": [
                            "Conservative",
                            "Balanced",
                            "Aggressive",
                            "Unknown",
                        ],
                        "description": "风险承受能力",
                    },
                    "investment_horizon": {
                        "type": "string",
                        "enum": ["Short", "Medium", "Long", "Unknown"],
                        "description": "投资周期偏好",
                    },
                    "favorite_sectors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "偏好板块列表",
                    },
                    "avoid_sectors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "回避板块列表",
                    },
                    "investment_style": {
                        "type": "string",
                        "description": "投资风格描述",
                    },
                },
                "additionalProperties": True,
                "description": "可接收任意额外参数，自动合并到 custom_preferences",
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_profile",
            "description": "获取当前用户投资画像的完整 JSON",
            "parameters": {
                "type": "object",
                "properties": {},
                "description": "无需参数",
            },
        },
    },
]
