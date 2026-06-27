"""
共享工具函数 — 安全类型转换、时间格式化、字符串处理
所有组件和页面统一使用这些函数，消除各处重复定义。
"""


def safe_str(val) -> str:
    """安全转换：None/空/占位文本 → 'N/A'，否则返回清理后的值"""
    if val is None:
        return "N/A"
    s = str(val).strip()
    if s == "" or s == "None" or s == "数据待查询":
        return "N/A"
    s = s.replace("…", "").replace("...", "").strip()
    return s if s else "N/A"


def safe_float(val):
    """安全转换为 float，失败返回 None"""
    if val is None:
        return None
    try:
        return float(str(val).replace("%", "").replace("倍", "").strip())
    except (ValueError, TypeError):
        return None


def fmt_time(t: str) -> str:
    """把时间格式化为 yyyy-mm-dd hh:mm，去掉 T 和微秒"""
    if not t:
        return "—"
    t = t.replace("T", " ")
    if "." in t:
        t = t.split(".")[0]
    if " " in t:
        date_part, time_part = t.split(" ", 1)
        time_parts = time_part.split(":")
        if len(time_parts) >= 2:
            return f"{date_part} {time_parts[0]}:{time_parts[1]}"
    return t


def strip_exchange_prefix(code: str) -> str:
    """去除交易所前缀，如 sh.688256 → 688256"""
    if code.startswith("sh.") or code.startswith("sz."):
        return code[3:]
    return code


def format_price_change(val) -> str:
    """格式化涨跌幅，统一保留一位小数"""
    if val is None or val == "N/A" or val == "" or val == "数据待查询":
        return "N/A"
    s = str(val).replace("…", "").replace("...", "").strip()
    if s == "数据待查询":
        return "N/A"
    try:
        num = float(s.replace("%", ""))
        return f"{num:.1f}%"
    except (ValueError, TypeError):
        return s if s else "N/A"


def format_turnover(val) -> str:
    """格式化换手率"""
    if val is None or val == "N/A" or val == "" or val == "数据待查询":
        return "N/A"
    s = str(val).replace("…", "").replace("...", "").strip()
    return s if s else "N/A"


def price_color(val) -> str:
    """涨跌幅正负颜色"""
    if val == "N/A":
        return "#888"
    try:
        num = float(val.replace("%", ""))
        return "#dc3545" if num < 0 else "#28a745" if num > 0 else "#888"
    except (ValueError, TypeError):
        return "#888"
