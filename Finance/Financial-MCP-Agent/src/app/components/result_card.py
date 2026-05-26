"""
查询结果展示组件 — 股票信息卡片、涨跌幅、行业、投资建议、行业估值基准
"""

import streamlit as st


def _safe_str(val) -> str:
    """安全转换：None/空/占位文本 → "N/A"，否则返回清理后的值"""
    if val is None:
        return "N/A"
    s = str(val).strip()
    if s == "" or s == "None" or s == "数据待查询":
        return "N/A"
    s = s.replace("…", "").replace("...", "").strip()
    return s if s else "N/A"


def _safe_float(val):
    """安全转换为 float，失败返回 None"""
    if val is None:
        return None
    try:
        return float(str(val).replace("%", "").replace("倍", "").strip())
    except (ValueError, TypeError):
        return None


def _strip_exchange_prefix(code: str) -> str:
    """去除交易所前缀，如 sh.688256 → 688256"""
    if code.startswith("sh.") or code.startswith("sz."):
        return code[3:]
    return code


def _price_color(val) -> str:
    """涨跌幅正负颜色"""
    if val == "N/A":
        return "#888"
    try:
        num = float(val.replace("%", ""))
        return "#dc3545" if num < 0 else "#28a745" if num > 0 else "#888"
    except (ValueError, TypeError):
        return "#888"


def _valuation_level_html(current_val, cheap_threshold, expensive_threshold) -> str:
    """生成估值水平 HTML 标签"""
    if current_val is None or cheap_threshold is None or expensive_threshold is None:
        return ""
    if current_val < cheap_threshold:
        return (
            '<span style="color:#28a745;font-weight:600;font-size:0.85em;">'
            '⬇ 低于行业低估线</span>'
        )
    elif current_val > expensive_threshold:
        return (
            '<span style="color:#dc3545;font-weight:600;font-size:0.85em;">'
            '⬆ 高于行业高估线</span>'
        )
    else:
        return (
            '<span style="color:#ffc107;font-weight:600;font-size:0.85em;">'
            '↔ 处于行业合理区间</span>'
        )


def render_result_card(data: dict) -> None:
    """渲染查询结果卡片

    Args:
        data: 快速查询返回的字典，包含以下字段：
            - stock_code, stock_name
            - market_cap, pb, pe, turnover_rate
            - price_changes: {1d, 5d, 1m, 3m, 6m, 1y, 3y}
            - industry, industry_intro, company_intro
            - industry_benchmark: {industry_name, pe_reasonable_range, pb_reasonable_range, ...}
    """
    stock_code = _strip_exchange_prefix(data.get("stock_code", ""))
    stock_name = data.get("stock_name", "未知")

    if stock_name.isdigit() or not stock_name:
        display_text = f'<span style="font-size:1.25rem;font-weight:700;">{stock_code}</span>'
    else:
        display_text = (
            f'<span style="font-size:1.25rem;font-weight:700;">{stock_name}</span>'
            f'<span style="font-size:1rem;color:#888;margin-left:0.5rem;">{stock_code}</span>'
        )

    st.markdown(
        f'<div style="margin-bottom:0.6rem;">{display_text}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── 基本信息 ──
    st.markdown("#### 基本信息")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("总市值", _safe_str(data.get("market_cap")))
    col2.metric("市盈率 PE", _safe_str(data.get("pe")))
    col3.metric("市净率 PB", _safe_str(data.get("pb")))
    col4.metric("换手率", _format_turnover(data.get("turnover_rate")))

    # ── 行业 + 估值基准 + 公司简介 ──
    benchmark = data.get("industry_benchmark") or {}
    raw_pe = _safe_float(data.get("pe"))
    raw_pb = _safe_float(data.get("pb"))

    if benchmark and benchmark.get("pe_reasonable_range"):
        # 有行业基准 → 三栏布局：行业 | PE/PB 估值基准 | 公司简介
        col_a, col_b, col_c = st.columns([2.5, 2.5, 3])
        with col_a:
            st.markdown("**行业**")
            st.caption(data.get("industry", "N/A"))
            if data.get("industry_intro"):
                st.caption(data["industry_intro"])
        with col_b:
            st.markdown("**行业估值基准**")
            pe_range = benchmark.get("pe_reasonable_range", "")
            pb_range = benchmark.get("pb_reasonable_range", "")
            pe_status = _valuation_level_html(
                raw_pe,
                benchmark.get("pe_cheap_threshold"),
                benchmark.get("pe_expensive_threshold"),
            )
            pb_status = _valuation_level_html(
                raw_pb,
                benchmark.get("pb_cheap_threshold"),
                benchmark.get("pb_expensive_threshold"),
            )
            st.markdown(
                f'<div style="font-size:0.9em;line-height:1.6;">'
                f'PE 正常区间：<b>{pe_range}</b> {pe_status}<br>'
                f'PB 正常区间：<b>{pb_range}</b> {pb_status}'
                f'</div>',
                unsafe_allow_html=True,
            )
            primary = benchmark.get("primary_valuation", "")
            if primary:
                st.caption(f"主要估值方法：{primary}")
        with col_c:
            st.markdown("**公司简介**")
            st.caption(data.get("company_intro", "N/A"))
    else:
        # 无行业基准 → 原始双栏布局
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**行业**")
            st.caption(data.get("industry", "N/A"))
            if data.get("industry_intro"):
                st.caption(data["industry_intro"])
        with col_b:
            st.markdown("**公司简介**")
            st.caption(data.get("company_intro", "N/A"))

    # ── 涨跌幅 ──
    st.markdown("#### 涨跌幅")
    price_changes = data.get("price_changes", {})
    periods = [
        ("1d", "近1日"), ("5d", "近5日"), ("1m", "近1月"),
        ("3m", "近3月"), ("6m", "近6月"), ("1y", "近1年"), ("3y", "近3年"),
    ]
    cols = st.columns([1, 1, 1, 1, 1, 1.35, 1])
    for idx, (key, label) in enumerate(periods):
        val = price_changes.get(key)
        display_val = _format_price_change(val)
        color = _price_color(display_val)
        cols[idx].markdown(
            f'<div style="font-size:0.8em;color:#888;margin-bottom:2px;">{label}</div>'
            f'<div style="font-size:1.05em;font-weight:600;color:{color};">{display_val}</div>',
            unsafe_allow_html=True,
        )



def _format_price_change(val) -> str:
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


def _format_turnover(val) -> str:
    """格式化换手率"""
    if val is None or val == "N/A" or val == "" or val == "数据待查询":
        return "N/A"
    s = str(val).replace("…", "").replace("...", "").strip()
    return s if s else "N/A"
