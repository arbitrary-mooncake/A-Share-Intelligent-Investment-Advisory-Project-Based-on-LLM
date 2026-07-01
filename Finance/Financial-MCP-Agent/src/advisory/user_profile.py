"""
用户画像管理 — 用户投资偏好的 JSON 持久化存储与 LangGraph State 注入。

提供 UserProfileManager 类，负责用户画像的加载、保存、更新，
以及生成格式化的文本摘要和 LangGraph State 注入上下文。
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# ── 项目路径 ──

_BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_DEFAULT_PROFILE_PATH = os.path.join(
    _BASE_DIR, "data", "user_profiles", "default_profile.json"
)

# ── 有效枚举值 ──

_VALID_RISK_TOLERANCES = frozenset({"Conservative", "Balanced", "Aggressive", "Unknown"})
_VALID_HORIZONS = frozenset({"Short", "Medium", "Long", "Unknown"})


class UserProfileManager:
    """用户投资画像管理器，提供 JSON 持久化与 state 注入能力。"""

    def __init__(self, file_path: Optional[str] = None) -> None:
        """初始化管理器。

        Args:
            file_path: 画像文件路径，默认为 data/user_profiles/default_profile.json。
        """
        self._file_path: str = file_path or _DEFAULT_PROFILE_PATH
        self._profile: Dict[str, Any] = self._load_profile()

    # ── 内部方法 ──

    def _load_profile(self) -> Dict[str, Any]:
        """从文件加载画像，文件不存在或解析失败时返回默认值。"""
        if not os.path.exists(self._file_path):
            return self._default_profile()
        try:
            with open(self._file_path, "r", encoding="utf-8") as f:
                data: Dict[str, Any] = json.load(f)
            # 确保所有必要字段都存在
            default = self._default_profile()
            for key in default:
                if key not in data:
                    data[key] = default[key]
            return data
        except Exception:
            return self._default_profile()

    @staticmethod
    def _default_profile() -> Dict[str, Any]:
        """返回默认画像字典。"""
        return {
            "risk_tolerance": "Unknown",
            "investment_horizon": "Unknown",
            "favorite_sectors": [],
            "avoid_sectors": [],
            "investment_style": "",
            "custom_preferences": {},
            "updated_at": "",
        }

    def _save_profile(self) -> None:
        """保存画像到 JSON 文件，更新 updated_at 为当前 ISO 时间。"""
        self._profile["updated_at"] = datetime.now(timezone.utc).isoformat()
        base_dir = os.path.dirname(self._file_path)
        os.makedirs(base_dir, exist_ok=True)
        with open(self._file_path, "w", encoding="utf-8") as f:
            json.dump(self._profile, f, ensure_ascii=False, indent=2)

    # ── 公共方法 ──

    def update_profile(
        self,
        risk_tolerance: Optional[str] = None,
        investment_horizon: Optional[str] = None,
        favorite_sectors: Optional[List[str]] = None,
        avoid_sectors: Optional[List[str]] = None,
        investment_style: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """更新用户画像字段，只更新非 None 值。

        risk_tolerance 仅在 {"Conservative", "Balanced", "Aggressive", "Unknown"} 内才更新。
        investment_horizon 仅在 {"Short", "Medium", "Long", "Unknown"} 内才更新。
        其余 kwargs 全部合并到 custom_preferences 中。

        Returns:
            描述变更结果的中文字符串。
        """
        changes: List[str] = []

        if risk_tolerance is not None:
            if risk_tolerance in _VALID_RISK_TOLERANCES:
                self._profile["risk_tolerance"] = risk_tolerance
                changes.append(f"风险承受能力 -> {risk_tolerance}")
            else:
                changes.append(
                    f"风险承受能力 跳过(无效值: {risk_tolerance})"
                )

        if investment_horizon is not None:
            if investment_horizon in _VALID_HORIZONS:
                self._profile["investment_horizon"] = investment_horizon
                changes.append(f"投资周期 -> {investment_horizon}")
            else:
                changes.append(f"投资周期 跳过(无效值: {investment_horizon})")

        if favorite_sectors is not None:
            self._profile["favorite_sectors"] = favorite_sectors
            changes.append(f"偏好板块 [{len(favorite_sectors)}项]")

        if avoid_sectors is not None:
            self._profile["avoid_sectors"] = avoid_sectors
            changes.append(f"回避板块 [{len(avoid_sectors)}项]")

        if investment_style is not None:
            self._profile["investment_style"] = investment_style
            changes.append(f"投资风格 -> {investment_style}")

        if kwargs:
            self._profile["custom_preferences"].update(kwargs)
            for k in kwargs:
                changes.append(f"自定义参数 {k} -> {kwargs[k]}")

        self._save_profile()
        return f"用户画像已更新: {'; '.join(changes)}"

    def get_profile_summary(self) -> str:
        """返回格式化的多行文本摘要。"""
        p = self._profile
        lines: List[str] = [
            f"风险承受能力: {p.get('risk_tolerance', 'Unknown')}",
            f"投资周期偏好: {p.get('investment_horizon', 'Unknown')}",
            f"投资风格: {p.get('investment_style', '') or '未设置'}",
            f"偏好板块: {', '.join(p.get('favorite_sectors', [])) or '无'}",
            f"回避板块: {', '.join(p.get('avoid_sectors', [])) or '无'}",
        ]
        custom = p.get("custom_preferences", {})
        if custom:
            for k, v in custom.items():
                lines.append(f"自定义 - {k}: {v}")
        updated = p.get("updated_at", "")
        if updated:
            lines.append(f"最后更新: {updated}")
        return "\n".join(lines)

    def get_profile(self) -> Dict[str, Any]:
        """返回完整画像字典的副本。"""
        return dict(self._profile)

    def to_state_context(self) -> str:
        """返回 [USER_PROFILE] 格式的注入文本，用于 LangGraph State 注入。"""
        p = self._profile
        lines: List[str] = ["[USER_PROFILE]"]
        lines.append(f"risk_tolerance={p.get('risk_tolerance', 'Unknown')}")
        lines.append(f"investment_horizon={p.get('investment_horizon', 'Unknown')}")
        lines.append(f"investment_style={p.get('investment_style', '')}")
        fav = p.get("favorite_sectors", [])
        lines.append(f"favorite_sectors={','.join(fav) if fav else 'None'}")
        avd = p.get("avoid_sectors", [])
        lines.append(f"avoid_sectors={','.join(avd) if avd else 'None'}")
        custom = p.get("custom_preferences", {})
        if custom:
            lines.append("custom_preferences:")
            for k, v in custom.items():
                lines.append(f"  {k}={v}")
        lines.append("[/USER_PROFILE]")
        return "\n".join(lines)
