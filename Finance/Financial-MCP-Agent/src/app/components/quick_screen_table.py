"""
快筛股票池组件 — 上方显示行 + 下方密集列表（点击选中） + 分页
严格按照设计稿视觉规范：蓝色主题、div布局、仅水平分割线
"""

from typing import Optional, Dict, Any, List

import streamlit as st
from theme import score_color, score_bg
from components.common import fmt_time


def render_quick_display_row(
    stock: dict,
    term_key: str,
    on_score: callable,
    on_delete: callable,
    on_deselect: callable,
    disabled: bool = False,
) -> None:
    """渲染快筛上方显示行 — 精致蓝色卡片，股票信息在上，按钮在下"""
    code = stock.get("stock_code", "")
    name = stock.get("company_name", "")
    scores = stock.get("scores", {})

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
    st.markdown(
        f'<div style="font-size:1.35em;font-weight:800;color:#0f172a;letter-spacing:-0.3px;margin-bottom:2px;">'
        f'{name}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="font-size:0.9em;color:#64748b;margin-bottom:10px;">代码：{code}</div>',
        unsafe_allow_html=True,
    )

    # ── 优雅分割线 ──
    st.markdown(
        '<div style="height:1px;background:linear-gradient(90deg,transparent,#dbeafe,transparent);'
        'margin:8px 0 12px 0;"></div>',
        unsafe_allow_html=True,
    )

    # ── 三个期限分数（中间） ──
    sc_cols = st.columns([2, 2, 2])
    for idx, (t, label) in enumerate([("short", "短线"), ("medium", "中线"), ("long", "长线")]):
        ts = scores.get(t, {})
        sv = ts.get("score")
        st_time = fmt_time(ts.get("score_time", ""))
        color = score_color(sv)
        bg = score_bg(sv)

        with sc_cols[idx]:
            if sv is not None:
                st.markdown(
                    f'<div style="text-align:center;padding:8px 4px;background:{bg};'
                    f'border-radius:12px;border:1.5px solid {color}25;">'
                    f'<div style="font-size:0.85em;color:#64748b;font-weight:600;margin-bottom:3px;">{label}</div>'
                    f'<div style="font-size:1.4em;font-weight:800;color:{color};">{float(sv):.1f}<span style="font-size:0.6em;">分</span></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div style="text-align:center;padding:8px 4px;background:#f8fafc;'
                    f'border-radius:12px;border:1.5px solid #e2e8f0;">'
                    f'<div style="font-size:0.85em;color:#94a3b8;font-weight:600;margin-bottom:3px;">{label}</div>'
                    f'<div style="font-size:1.15em;font-weight:700;color:#cbd5e1;">未打分</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            if st_time != "—":
                st.markdown(
                    f'<div style="text-align:center;font-size:0.78em;color:#94a3b8;margin-top:5px;">'
                    f'🕐 {st_time}</div>',
                    unsafe_allow_html=True,
                )

    # ── 操作按钮（下方） ──
    st.markdown(
        '<div style="height:1px;background:linear-gradient(90deg,transparent,#dbeafe,transparent);'
        'margin:14px 0 10px 0;"></div>',
        unsafe_allow_html=True,
    )

    sc_cols2 = st.columns([2, 2, 2])
    for idx, (t, label) in enumerate([("short", "短线"), ("medium", "中线"), ("long", "长线")]):
        with sc_cols2[idx]:
            sv = scores.get(t, {}).get("score")
            btn_label = "重新打分" if sv is not None else f"{label}打分"
            if st.button(btn_label, key=f"qs_display_btn_{t}_{term_key}_{code}",
                        disabled=disabled, use_container_width=True):
                on_score(t, code)

    # 删除和关闭按钮
    btn_cols = st.columns([3, 1, 1])
    with btn_cols[1]:
        if st.button("删除", key=f"qs_display_delete_{term_key}_{code}",
                     disabled=disabled, use_container_width=True):
            on_delete(code)
    with btn_cols[2]:
        if st.button("✕ 关闭", key=f"qs_display_close_{term_key}_{code}",
                     disabled=disabled, use_container_width=True):
            on_deselect()

    # 打分详情展开
    has_detail = any(
        scores.get(t, {}).get("reasoning") or
        scores.get(t, {}).get("risk_warning") or
        scores.get(t, {}).get("suggested_action")
        for t in ("short", "medium", "long")
    )
    if has_detail:
        with st.expander("📋 打分详情"):
            for t, label in [("short", "短线"), ("medium", "中线"), ("long", "长线")]:
                ts = scores.get(t, {})
                if ts:
                    st.markdown(
                        f'<div style="font-size:0.95em;font-weight:700;color:#1e40af;margin:8px 0 4px 0;">'
                        f'{label}</div>',
                        unsafe_allow_html=True,
                    )
                    if ts.get("suggested_action"):
                        st.caption(f"💡 {ts['suggested_action']}")
                    if ts.get("reasoning"):
                        st.caption(f"📊 {ts['reasoning']}")
                    if ts.get("risk_warning"):
                        st.caption(f"⚠️ {ts['risk_warning']}")

    st.markdown('</div>', unsafe_allow_html=True)


def render_quick_list(
    stocks: list,
    term_key: str,
    selected_code: str,
    on_select: callable,
    page: int = 0,
    page_size: int = 20,
) -> None:
    """渲染快筛密集列表，分栏布局：名称 | 打分时间 | 短线 | 中线 | 长线 | 操作。"""
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

        # 分栏：[名称2 | 时间1 | 短线0.9 | 中线0.9 | 长线0.9 | 按钮0.5]
        cols = st.columns([2, 1, 0.9, 0.9, 0.9, 0.5])

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

        for idx, (t, label) in enumerate([("short", "短"), ("medium", "中"), ("long", "长")]):
            with cols[idx + 2]:
                sv = scores.get(t, {}).get("score")
                if sv is not None:
                    color = score_color(sv)
                    bg = score_bg(sv)
                    st.markdown(
                        f'<div style="display:inline-block;padding:3px 10px;'
                        f'background:{bg};color:{color};border-radius:10px;'
                        f'font-weight:700;font-size:0.88em;border:1px solid {color}25;">'
                        f'{label} {float(sv):.1f}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div style="display:inline-block;padding:3px 10px;'
                        f'background:#f1f5f9;color:#cbd5e1;border-radius:10px;'
                        f'font-weight:600;font-size:0.85em;">'
                        f'{label} —</div>',
                        unsafe_allow_html=True,
                    )

        with cols[5]:
            btn_label = "✓" if is_selected else "查看"
            btn_type = "primary" if is_selected else "secondary"
            if st.button(btn_label, key=f"qs_sel_{term_key}_{code}",
                         type=btn_type, use_container_width=True):
                on_select(code)

        st.markdown('</div>', unsafe_allow_html=True)
