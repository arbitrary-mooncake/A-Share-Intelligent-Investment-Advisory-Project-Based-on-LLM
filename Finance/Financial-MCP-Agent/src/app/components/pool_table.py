"""
股票池组件 — 上方显示行（交互） + 下方密集列表（点击选中） + 分页
严格按照设计稿视觉规范：蓝色主题、div布局、仅水平分割线
"""

from typing import Optional, Dict, Any, List

import streamlit as st
from theme import score_color, score_bg
from components.common import fmt_time


def render_add_stock_form(
    on_add: callable,
    disabled: bool = False,
    key_prefix: str = "",
) -> None:
    """渲染添加股票到当前池的表单"""
    k = f"{key_prefix}_" if key_prefix else ""

    # 标题带蓝色左边框，更醒目
    st.markdown(
        '<div style="border-left:4px solid #2563eb;padding-left:10px;margin-bottom:6px;">'
        '<span style="font-size:1.15em;font-weight:700;color:#1e3a5f;">添加股票到当前池</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    col_code, col_name, col_btn, _ = st.columns([2, 3, 1, 1])

    with col_code:
        stock_code = st.text_input(
            label="股票代码",
            placeholder="例如：603871",
            label_visibility="collapsed",
            key=f"{k}pool_add_code",
            disabled=disabled,
        )

    with col_name:
        stock_name = st.text_input(
            label="股票名称",
            placeholder="例如：嘉友国际",
            label_visibility="collapsed",
            key=f"{k}pool_add_name",
            disabled=disabled,
        )

    with col_btn:
        clicked = st.button(
            "添加",
            type="primary",
            use_container_width=True,
            disabled=disabled or not stock_code.strip() or not stock_name.strip(),
            key=f"{k}pool_add_btn",
        )

    if clicked:
        on_add(stock_code.strip(), stock_name.strip())


def render_display_row(
    stock: dict,
    term_key: str,
    on_score: callable,
    on_report: callable,
    on_delete: callable,
    on_deselect: callable,
    disabled: bool = False,
) -> None:
    """渲染上方显示行 — 精致蓝色卡片，股票信息在上，按钮在下"""
    code = stock.get("stock_code", "")
    name = stock.get("company_name", "")
    score = stock.get("score")
    score_time = fmt_time(stock.get("score_time", ""))

    color = score_color(score)
    bg = score_bg(score)

    # ── 精致卡片容器 ──
    st.markdown(
        '<div style="'
        'border:1.5px solid #bfdbfe;'
        'border-radius:14px;'
        'padding:18px 22px;'
        'background: linear-gradient(145deg, #ffffff 0%, #f0f7ff 100%);'
        'box-shadow: 0 4px 16px rgba(37,99,235,0.08);'
        'margin-bottom:10px;'
        '">',
        unsafe_allow_html=True,
    )

    # ── 股票信息（上方） ──
    info_cols = st.columns([2.2, 1.3, 1.8, 3])
    with info_cols[0]:
        st.markdown(
            f'<div style="font-size:1.35em;font-weight:800;color:#0f172a;letter-spacing:-0.3px;">{name}</div>',
            unsafe_allow_html=True,
        )
    with info_cols[1]:
        st.markdown(
            f'<div style="font-size:0.9em;color:#64748b;margin-top:6px;">代码：{code}</div>',
            unsafe_allow_html=True,
        )
    with info_cols[2]:
        if score is not None:
            st.markdown(
                f'<div style="display:inline-block;padding:5px 18px;'
                f'background:{bg};color:{color};border-radius:20px;'
                f'font-weight:800;font-size:1.25em;border:1.5px solid {color}30;">'
                f'{float(score):.1f}<span style="font-size:0.7em;font-weight:600;"> 分</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="display:inline-block;padding:5px 18px;'
                'background:#f1f5f9;color:#94a3b8;border-radius:20px;'
                'font-weight:700;font-size:1.1em;">未打分</div>',
                unsafe_allow_html=True,
            )
    with info_cols[3]:
        time_text = f"打分时间：{score_time}" if score_time != "—" else ""
        if time_text:
            st.markdown(
                f'<div style="font-size:0.82em;color:#94a3b8;margin-top:8px;text-align:right;">'
                f'🕐 {time_text}</div>',
                unsafe_allow_html=True,
            )

    # ── 优雅分割线 ──
    st.markdown(
        '<div style="height:1px;background:linear-gradient(90deg,transparent,#dbeafe,transparent);'
        'margin:12px 0 10px 0;"></div>',
        unsafe_allow_html=True,
    )

    # ── 操作按钮（下方） ──
    btn_cols = st.columns([1, 1, 1, 0.5, 3])
    with btn_cols[0]:
        if st.button("生成报告", key=f"display_report_{term_key}_{code}",
                     disabled=disabled, use_container_width=True):
            on_report(code, name)
    with btn_cols[1]:
        if st.button("打分", key=f"display_score_{term_key}_{code}",
                     disabled=disabled, use_container_width=True):
            on_score(code)
    with btn_cols[2]:
        if st.button("删除", key=f"display_delete_{term_key}_{code}",
                     disabled=disabled, use_container_width=True):
            on_delete(code)
    with btn_cols[3]:
        if st.button("✕", key=f"display_close_{term_key}_{code}",
                     disabled=disabled, use_container_width=True):
            on_deselect()

    st.markdown('</div>', unsafe_allow_html=True)


def _render_score_badge(score, label=""):
    """渲染单个分数徽章"""
    if score is None:
        st.markdown(
            '<div style="display:inline-block;padding:2px 10px;'
            'background:#f1f5f9;color:#cbd5e1;border-radius:10px;'
            'font-weight:600;font-size:0.85em;">—</div>',
            unsafe_allow_html=True,
        )
        return
    color = score_color(score)
    bg = score_bg(score)
    try:
        s = float(score)
    except (ValueError, TypeError):
        s = 0
    text = f"{s:.0f}" if label else f"{s:.0f}分"
    if label:
        text = f"{label} {s:.0f}"
    st.markdown(
        f'<div style="display:inline-block;padding:2px 10px;'
        f'background:{bg};color:{color};border-radius:10px;'
        f'font-weight:700;font-size:0.85em;border:1px solid {color}25;">'
        f'{text}</div>',
        unsafe_allow_html=True,
    )


def render_fine_display_row(
    stock: dict,
    term_key: str,
    on_score: callable,
    on_report: callable,
    on_delete: callable,
    on_deselect: callable,
    disabled: bool = False,
) -> None:
    """精筛显示行 — 三期限分数并列 + 打分/报告/删除按钮"""
    code = stock.get("stock_code", "")
    name = stock.get("company_name", "")
    scores = stock.get("scores", {})
    short_s = scores.get("short", {})
    medium_s = scores.get("medium", {})
    long_s = scores.get("long", {})
    score_time = fmt_time(stock.get("last_updated", ""))

    st.markdown(
        '<div style="'
        'border:1.5px solid #bfdbfe;'
        'border-radius:14px;'
        'padding:18px 22px;'
        'background: linear-gradient(145deg, #ffffff 0%, #f0f7ff 100%);'
        'box-shadow: 0 4px 16px rgba(37,99,235,0.08);'
        'margin-bottom:10px;'
        '">',
        unsafe_allow_html=True,
    )

    # 股票信息行
    info_cols = st.columns([2.2, 1.3, 2.5, 2.5])
    with info_cols[0]:
        st.markdown(
            f'<div style="font-size:1.35em;font-weight:800;color:#0f172a;letter-spacing:-0.3px;">{name}</div>',
            unsafe_allow_html=True,
        )
    with info_cols[1]:
        st.markdown(
            f'<div style="font-size:0.9em;color:#64748b;margin-top:6px;">代码：{code}</div>',
            unsafe_allow_html=True,
        )
    with info_cols[2]:
        time_text = f"🕐 {score_time}" if score_time and score_time != "—" else ""
        if time_text:
            st.markdown(
                f'<div style="font-size:0.82em;color:#94a3b8;margin-top:8px;">{time_text}</div>',
                unsafe_allow_html=True,
            )

    # 三分数行
    st.markdown('<div style="margin-top:6px;"></div>', unsafe_allow_html=True)
    score_cols = st.columns([1, 1, 1, 1.5])
    with score_cols[0]:
        _render_score_badge(short_s.get("score"), "短线")
    with score_cols[1]:
        _render_score_badge(medium_s.get("score"), "中线")
    with score_cols[2]:
        _render_score_badge(long_s.get("score"), "长线")

    # 分割线
    st.markdown(
        '<div style="height:1px;background:linear-gradient(90deg,transparent,#dbeafe,transparent);'
        'margin:10px 0 8px 0;"></div>',
        unsafe_allow_html=True,
    )

    # 按钮
    btn_cols = st.columns([1, 1, 1, 0.5, 3])
    with btn_cols[0]:
        if st.button("生成报告", key=f"fine_display_report_{term_key}_{code}",
                     disabled=disabled, use_container_width=True):
            on_report(code, name)
    with btn_cols[1]:
        if st.button("打分", key=f"fine_display_score_{term_key}_{code}",
                     disabled=disabled, use_container_width=True):
            on_score(code)
    with btn_cols[2]:
        if st.button("删除", key=f"fine_display_delete_{term_key}_{code}",
                     disabled=disabled, use_container_width=True):
            on_delete(code)
    with btn_cols[3]:
        if st.button("✕", key=f"fine_display_close_{term_key}_{code}",
                     disabled=disabled, use_container_width=True):
            on_deselect()

    st.markdown('</div>', unsafe_allow_html=True)


def render_fine_pool_list(
    stocks: list,
    term_key: str,
    selected_code: str,
    on_select: callable,
    page: int = 0,
    page_size: int = 20,
) -> None:
    """精筛股票列表 — 名称 | 打分时间 | 短线 | 中线 | 长线 | 操作"""
    if not stocks:
        return

    total_pages = max(1, (len(stocks) + page_size - 1) // page_size)
    page = min(page, total_pages - 1)
    start_idx = page * page_size
    end_idx = start_idx + page_size
    page_stocks = stocks[start_idx:end_idx]

    for i, s in enumerate(page_stocks):
        code = s.get("stock_code", "")
        name = s.get("company_name", "")
        scores = s.get("scores", {})
        last_updated = s.get("last_updated", "")
        score_date = (last_updated or "")[:10] if last_updated else ""

        is_selected = (code == selected_code)

        if i > 0:
            st.markdown(
                '<div style="height:1px;background:#e2e8f0;margin:2px 0 2px 0;"></div>',
                unsafe_allow_html=True,
            )

        if is_selected:
            st.markdown(
                '<div style="background:#f0f7ff;border-left:3px solid #2563eb;'
                'border-radius:6px;padding:6px 10px;">',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="border-left:3px solid transparent;padding:6px 10px;">',
                unsafe_allow_html=True,
            )

        cols = st.columns([2, 1, 1, 1, 1, 0.5])

        with cols[0]:
            st.markdown(
                f'<div style="font-weight:700;font-size:1.05em;color:#0f172a;">{name}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="font-size:0.78em;color:#94a3b8;margin-top:1px;">{code}</div>',
                unsafe_allow_html=True,
            )

        with cols[1]:
            if score_date:
                st.markdown(
                    f'<div style="font-size:0.78em;color:#94a3b8;margin-top:4px;">{score_date}</div>',
                    unsafe_allow_html=True,
                )

        for j, (term, label) in enumerate([("short", "短线"), ("medium", "中线"), ("long", "长线")]):
            with cols[2 + j]:
                ts = scores.get(term, {})
                sc = ts.get("score")
                if sc is not None:
                    color = score_color(sc)
                    bg = score_bg(sc)
                    st.markdown(
                        f'<div style="display:inline-block;padding:2px 8px;'
                        f'background:{bg};color:{color};border-radius:10px;'
                        f'font-weight:700;font-size:0.82em;border:1px solid {color}20;">'
                        f'{label} {float(sc):.0f}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div style="font-size:0.78em;color:#cbd5e1;">—</div>',
                        unsafe_allow_html=True,
                    )

        with cols[5]:
            btn_label = "✓" if is_selected else "查看"
            btn_type = "primary" if is_selected else "secondary"
            if st.button(btn_label, key=f"fine_sel_{term_key}_{code}",
                         type=btn_type, use_container_width=True):
                on_select(code)

        st.markdown('</div>', unsafe_allow_html=True)


def render_pool_list(
    stocks: list,
    term_key: str,
    selected_code: str,
    on_select: callable,
    page: int = 0,
    page_size: int = 20,
) -> None:
    """渲染下方密集股票列表，分栏布局：名称 | 分数 | 时间 | 操作。

    视觉规范：
    - 名称/分数/时间分栏，间距拉开
    - 分数按高低显示颜色（绿/黄/红）
    - 选中行蓝色左边框高亮
    - 无垂直边框，仅有水平分割线
    """
    if not stocks:
        return

    total_pages = max(1, (len(stocks) + page_size - 1) // page_size)
    page = min(page, total_pages - 1)
    start_idx = page * page_size
    end_idx = start_idx + page_size
    page_stocks = stocks[start_idx:end_idx]

    for i, s in enumerate(page_stocks):
        code = s.get("stock_code", "")
        name = s.get("company_name", "")
        score = s.get("score")
        score_time = fmt_time(s.get("score_time", ""))

        is_selected = (code == selected_code)

        # 顶部细分割线（第一条之前不加）
        if i > 0:
            st.markdown(
                '<div style="height:1px;background:#e2e8f0;margin:2px 0 2px 0;"></div>',
                unsafe_allow_html=True,
            )

        # 选中态：淡蓝背景 + 蓝色左边框
        if is_selected:
            st.markdown(
                '<div style="'
                'background:#f0f7ff;'
                'border-left:3px solid #2563eb;'
                'border-radius:6px;'
                'padding:6px 10px;'
                '">',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="'
                'border-left:3px solid transparent;'
                'padding:6px 10px;'
                '">',
                unsafe_allow_html=True,
            )

        # 分栏：[名称2.5 | 分数1 | 时间1.8 | 按钮0.6]
        cols = st.columns([2.5, 1, 1.8, 0.6])

        with cols[0]:
            st.markdown(
                f'<div style="font-weight:700;font-size:1.05em;color:#0f172a;">{name}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="font-size:0.78em;color:#94a3b8;margin-top:1px;">{code}</div>',
                unsafe_allow_html=True,
            )

        with cols[1]:
            if score is not None:
                color = score_color(score)
                bg = score_bg(score)
                st.markdown(
                    f'<div style="display:inline-block;padding:3px 12px;'
                    f'background:{bg};color:{color};border-radius:12px;'
                    f'font-weight:700;font-size:0.95em;border:1px solid {color}25;">'
                    f'{float(score):.1f}<span style="font-size:0.75em;">分</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div style="display:inline-block;padding:3px 12px;'
                    'background:#f1f5f9;color:#cbd5e1;border-radius:12px;'
                    'font-weight:600;font-size:0.9em;">未打分</div>',
                    unsafe_allow_html=True,
                )

        with cols[2]:
            if score_time != "—":
                st.markdown(
                    f'<div style="font-size:0.8em;color:#94a3b8;padding-top:6px;">🕐 {score_time}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div style="font-size:0.8em;color:#cbd5e1;padding-top:6px;">—</div>',
                    unsafe_allow_html=True,
                )

        with cols[3]:
            btn_label = "✓" if is_selected else "查看"
            btn_type = "primary" if is_selected else "secondary"
            if st.button(btn_label, key=f"pool_sel_{term_key}_{code}",
                         type=btn_type, use_container_width=True):
                on_select(code)

        st.markdown('</div>', unsafe_allow_html=True)


def render_pagination(
    term_key: str,
    total_stocks: int,
    page: int,
    page_size: int,
) -> tuple:
    """渲染分页控件，返回 (new_page, new_page_size)"""
    if total_stocks == 0:
        return 0, page_size

    total_pages = max(1, (total_stocks + page_size - 1) // page_size)

    cols = st.columns([0.8, 0.8, 3, 1, 0.8])

    with cols[0]:
        if st.button("◀ 上一页", key=f"prev_{term_key}",
                     disabled=page <= 0, use_container_width=True):
            return max(0, page - 1), page_size

    with cols[1]:
        if st.button("下一页 ▶", key=f"next_{term_key}",
                     disabled=page >= total_pages - 1, use_container_width=True):
            return min(total_pages - 1, page + 1), page_size

    with cols[2]:
        st.markdown(
            f'<div style="text-align:center;padding-top:6px;font-size:13px;color:#64748b;font-weight:500;">'
            f'第 <span style="color:#2563eb;font-weight:700;">{page + 1}</span>/{total_pages} 页'
            f' · 共 <span style="color:#2563eb;font-weight:700;">{total_stocks}</span> 只股票'
            f'</div>',
            unsafe_allow_html=True,
        )

    with cols[3]:
        new_size = st.selectbox(
            "每页",
            options=[10, 20, 30, 50, 100],
            index=[10, 20, 30, 50, 100].index(page_size) if page_size in [10, 20, 30, 50, 100] else 1,
            key=f"page_size_{term_key}",
            label_visibility="collapsed",
        )
        if new_size != page_size:
            return 0, new_size

    with cols[4]:
        st.markdown(
            f'<div style="text-align:right;padding-top:6px;font-size:12px;color:#94a3b8;">共{total_stocks}只</div>',
            unsafe_allow_html=True,
        )

    return page, page_size
