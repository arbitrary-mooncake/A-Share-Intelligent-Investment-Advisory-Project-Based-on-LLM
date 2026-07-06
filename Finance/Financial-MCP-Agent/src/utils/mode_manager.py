"""
双版本模式管理器：检测当前运行模式（Lite / Full）。

Lite 模式：
  - 仅需 1 个 DeepSeek API Key
  - 免费 Tushare (120 积分) + AKShare 数据回退
  - 5/7 功能页面开放（禁用"模拟分析与迭代"和"智能投顾"）

Full 模式（默认）：
  - 6 个专用 LLM 模型
  - Tushare 5000+ 积分（付费会员）
  - 全部 7 个功能页面开放

模式由 .env 中的 APP_MODE 变量控制。
"""
import os
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv

load_dotenv(override=True)

APP_MODE_LITE = "lite"
APP_MODE_FULL = "full"

# Lite 模式下禁用的页面
LITE_DISABLED_PAGES = [
    "pages/06_模拟分析与迭代.py",
    "pages/07_智能投顾.py",
]


def get_mode() -> Literal["lite", "full"]:
    """
    获取当前运行模式。

    Returns:
        "lite" 或 "full"（默认）
    """
    mode = os.getenv("APP_MODE", APP_MODE_FULL).strip().lower()
    if mode == APP_MODE_LITE:
        return APP_MODE_LITE
    return APP_MODE_FULL


def is_lite_mode() -> bool:
    """检查是否处于 Lite 模式。"""
    return get_mode() == APP_MODE_LITE


def is_full_mode() -> bool:
    """检查是否处于 Full 模式。"""
    return get_mode() == APP_MODE_FULL


def is_page_enabled(page_path: str) -> bool:
    """
    检查指定页面在当前模式下是否可用。

    Args:
        page_path: 页面相对路径（如 "pages/06_模拟分析与迭代.py"）

    Returns:
        True 表示可用，False 表示被禁用
    """
    if is_full_mode():
        return True
    return page_path not in LITE_DISABLED_PAGES


def get_tushare_points() -> int:
    """
    检测当前 Tushare 积分等级。

    Returns:
        积分数值，检测失败返回 0
    """
    try:
        from src.utils.tushare_client import _call
        result = _call("user_info", {})
        if result:
            return result.get("points", 0)
    except Exception:
        pass
    return 0


def set_mode(mode: str) -> bool:
    """
    切换模式（修改 .env 文件中的 APP_MODE）。

    Args:
        mode: "lite" 或 "full"

    Returns:
        True 表示成功，False 表示失败
    """
    if mode not in (APP_MODE_LITE, APP_MODE_FULL):
        return False

    env_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", ".env"
    )
    env_path = os.path.normpath(env_path)

    if not os.path.exists(env_path):
        return False

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        new_lines = []
        found = False
        for line in lines:
            if line.strip().startswith("APP_MODE="):
                new_lines.append(f"APP_MODE={mode}\n")
                found = True
            else:
                new_lines.append(line)

        if not found:
            new_lines.insert(0, f"APP_MODE={mode}\n")

        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

        os.environ["APP_MODE"] = mode

        return True
    except Exception:
        return False
