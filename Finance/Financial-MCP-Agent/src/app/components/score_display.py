"""
分数展示组件 — 分数 + 时间戳的可视化展示
"""

import streamlit as st


def render_score_card(
    score: float,
    score_time: str = "",
    term_label: str = "",
    prefix: str = "",
) -> None:
    """渲染单个分数卡片

    Args:
        score: 分数值 (0-100)
        score_time: 打分时间戳
        term_label: 期限标签（短线/中线/长线）
        prefix: 分数前缀文本（如"当前分数："）
    """
    color, label = _score_color_label(score)

    st.markdown(
        f'<div style="padding: 0.8rem 1.2rem; border-radius: 10px; '
        f'background-color: {color}15; border-left: 4px solid {color}; '
        f'margin-bottom: 0.5rem;">'
        f'<span style="font-size: 1.6em; font-weight: 700; color: {color};">'
        f'{prefix}{score:.1f}'
        f'</span>'
        f'{"&nbsp;&nbsp;" + term_label if term_label else ""}'
        f'<br/>'
        f'<span style="font-size: 0.85em; color: #6c757d;">'
        f'{score_time or "未打分"}'
        f'</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_multi_score(scores: dict, term_label: str = "") -> None:
    """渲染多期限分数对比卡片

    Args:
        scores: 包含 short/medium/long 三个期限分数的字典
        term_label: 当前选中的期限标签（高亮显示）
    """
    cols = st.columns(3)
    terms = [
        ("short", "短线"),
        ("medium", "中线"),
        ("long", "长线"),
    ]

    for idx, (key, label) in enumerate(terms):
        score = scores.get(key, {}).get("score")
        time_str = scores.get(key, {}).get("score_time", "")

        with cols[idx]:
            is_active = (key == term_label or
                         (term_label == "short" and key == "short") or
                         (term_label == "medium" and key == "medium") or
                         (term_label == "long" and key == "long"))

            if score is not None:
                color, _ = _score_color_label(score)
                border_style = (
                    f"border: 2px solid {color};" if is_active
                    else "border: 1px solid #e0e0e0;"
                )
                st.markdown(
                    f'<div style="padding: 0.6rem; border-radius: 8px; '
                    f'{border_style} text-align: center; '
                    f'background-color: {color}08;">'
                    f'<div style="font-size: 0.8em; color: #6c757d;">{label}</div>'
                    f'<div style="font-size: 1.4em; font-weight: 700; color: {color};">'
                    f'{score:.1f}'
                    f'</div>'
                    f'<div style="font-size: 0.7em; color: #999;">{time_str}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div style="padding: 0.6rem; border-radius: 8px; '
                    f'border: 1px solid #e0e0e0; text-align: center; '
                    f'background-color: #f8f9fa;">'
                    f'<div style="font-size: 0.8em; color: #6c757d;">{label}</div>'
                    f'<div style="font-size: 1.4em; font-weight: 700; color: #999;">'
                    f'未打分'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


def _score_color_label(score: float) -> tuple:
    """根据分数返回颜色和评级标签"""
    if score >= 85:
        return "#28a745", "优秀"
    elif score >= 70:
        return "#20c997", "良好"
    elif score >= 60:
        return "#ffc107", "中等"
    elif score >= 40:
        return "#fd7e14", "偏弱"
    else:
        return "#dc3545", "较差"
