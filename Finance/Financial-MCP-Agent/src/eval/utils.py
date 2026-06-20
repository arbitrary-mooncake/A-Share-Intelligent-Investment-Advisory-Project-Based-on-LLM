"""
评测系统工具函数 — 总纲 §17.1
"""
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional


def safe_json_load(path: str, default: Any = None) -> Any:
    """安全加载JSON文件"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def safe_json_dump(data: Any, path: str) -> bool:
    """安全写入JSON文件"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def format_pct(value: float, decimals: int = 2) -> str:
    """格式化百分比"""
    return f"{value * 100:.{decimals}f}%"


def format_money(value: float) -> str:
    """格式化金额"""
    if abs(value) >= 1e8:
        return f"{value / 1e8:.2f}亿"
    if abs(value) >= 1e4:
        return f"{value / 1e4:.0f}万"
    return f"{value:.2f}"


def trading_days_between(start: str, end: str) -> int:
    """估算两个日期之间的交易日数 (简化: 排除周末)"""
    from datetime import timedelta
    d1 = datetime.strptime(start[:10], "%Y-%m-%d")
    d2 = datetime.strptime(end[:10], "%Y-%m-%d")
    days = (d2 - d1).days
    weeks = days // 7
    trading_days = days - weeks * 2
    # Adjust for partial week
    for i in range(days % 7):
        check = d1 + timedelta(days=weeks * 7 + i + 1)
        if check.weekday() < 5:
            trading_days = min(trading_days + 1, days)
    return max(0, min(trading_days, days))
