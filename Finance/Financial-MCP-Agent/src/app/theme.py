"""
统一主题/样式系统 — 全局 CSS 注入 + 共享样式工具函数

配色体系:
  主色（品牌蓝）: #1e40af / #2563eb / #dbeafe
  成功（绿）:     #059669 / #d1fae5
  警告（黄/橙）:  #d97706 / #fef3c7
  危险（红）:     #dc2626 / #fee2e2
  信息（青）:     #0891b2 / #ccfbf1
  中性灰:         #0f172a(文字) / #64748b(次要) / #94a3b8(辅助) / #e2e8f0(分割)
"""

import streamlit as st

# ── 颜色常量 ──────────────────────────────────
COLORS = {
    "primary": "#2563eb",
    "primary_dark": "#1e40af",
    "primary_light": "#dbeafe",
    "success": "#059669",
    "success_bg": "#d1fae5",
    "warning": "#d97706",
    "warning_bg": "#fef3c7",
    "danger": "#dc2626",
    "danger_bg": "#fee2e2",
    "info": "#0891b2",
    "info_bg": "#ccfbf1",
    "text": "#0f172a",
    "text_secondary": "#64748b",
    "text_muted": "#94a3b8",
    "border": "#e2e8f0",
    "bg": "#f8fafc",
    "white": "#ffffff",
}

# ── 卡片阴影 ──────────────────────────────────
SHADOWS = {
    1: "0 1px 3px rgba(0,0,0,0.06)",
    2: "0 4px 12px rgba(0,0,0,0.08)",
    3: "0 8px 24px rgba(37,99,235,0.10)",
}


def inject_global_styles():
    """注入全局自定义 CSS（在页面 st.set_page_config 之后调用）"""
    st.markdown(f"""
    <style>
    /* ── 全局字体优化（中文优先字体栈） ── */
    html, body, [class*="css"] {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
            "Hiragino Sans GB", "Microsoft YaHei", Roboto, "Helvetica Neue", Arial, sans-serif;
    }}

    /* ── 平滑滚动（减少页面跳转的生硬感） ── */
    html {{
        scroll-behavior: smooth;
    }}

    /* ── 文本选中色 ── */
    ::selection {{
        background: #bfdbfe;
        color: #1e3a8a;
    }}

    /* ── 全局滚动条精修 ── */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    ::-webkit-scrollbar-thumb {{
        background: #cbd5e1;
        border-radius: 8px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: #94a3b8;
    }}
    ::-webkit-scrollbar-track {{
        background: transparent;
    }}

    /* ── 卡片容器（L2 浮起卡片，默认样式） ── */
    .theme-card {{
        background: {COLORS["white"]};
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
        border: 1px solid {COLORS["border"]};
        box-shadow: {SHADOWS[2]};
    }}

    /* ── L1 基础卡片 ── */
    .theme-card-l1 {{
        background: {COLORS["white"]};
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
        border: 1px solid {COLORS["border"]};
        box-shadow: {SHADOWS[1]};
    }}

    /* ── L3 重点卡片（蓝色渐变） ── */
    .theme-card-l3 {{
        background: linear-gradient(145deg, #ffffff 0%, #f0f7ff 100%);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
        border: 1.5px solid #bfdbfe;
        box-shadow: {SHADOWS[3]};
    }}

    /* ── 优雅分割线 ── */
    .theme-divider {{
        height: 1px;
        background: linear-gradient(90deg, transparent, #dbeafe, transparent);
        margin: 12px 0 10px 0;
    }}

    /* ── 细分割线 ── */
    .theme-divider-thin {{
        height: 1px;
        background: {COLORS["border"]};
        margin: 2px 0;
    }}

    /* ── 区块标题 ── */
    .theme-section-title {{
        font-size: 1.15em;
        font-weight: 700;
        color: {COLORS["primary_dark"]};
        border-left: 4px solid {COLORS["primary"]};
        padding-left: 10px;
        margin-bottom: 8px;
    }}

    /* ── 页面大标题 ── */
    .theme-page-title {{
        font-size: 1.7em;
        font-weight: 800;
        color: {COLORS["text"]};
        margin-bottom: 4px;
    }}

    /* ── Metric 美化 ── */
    .stMetric {{
        background: {COLORS["white"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
        box-shadow: {SHADOWS[1]};
        transition: box-shadow 0.18s ease, transform 0.18s ease;
    }}
    .stMetric:hover {{
        box-shadow: 0 4px 12px rgba(37,99,235,0.10);
        transform: translateY(-1px);
    }}

    /* ── Button 全局精修：圆角 + 过渡 + 悬停浮起 ── */
    .stButton > button, .stDownloadButton > button {{
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.18s ease;
    }}
    .stButton > button:hover:not(:disabled),
    .stDownloadButton > button:hover:not(:disabled) {{
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37,99,235,0.16);
        border-color: #93c5fd;
    }}
    .stButton > button:active:not(:disabled),
    .stDownloadButton > button:active:not(:disabled) {{
        transform: translateY(0);
        box-shadow: 0 1px 4px rgba(37,99,235,0.12);
    }}
    .stButton > button:disabled {{
        opacity: 0.55;
    }}

    /* ── Button 主色覆盖（微妙渐变 + 悬停加深阴影） ── */
    .stButton > button[kind="primary"] {{
        background-color: {COLORS["primary"]} !important;
        background-image: linear-gradient(160deg, #3b82f6 0%, #2563eb 55%, #1d4ed8 100%) !important;
        border-color: {COLORS["primary"]} !important;
    }}
    .stButton > button[kind="primary"]:hover:not(:disabled) {{
        box-shadow: 0 6px 16px rgba(37,99,235,0.30) !important;
        border-color: {COLORS["primary_dark"]} !important;
    }}

    /* ── 进度条精修：圆角胶囊 + 渐变填充 + 平滑宽度动画 ── */
    [data-testid="stProgress"] > div > div {{
        border-radius: 999px;
        background-color: #e2e8f0;
    }}
    [data-testid="stProgress"] [role="progressbar"] {{
        border-radius: 999px;
        background-image: linear-gradient(90deg, #60a5fa 0%, #2563eb 100%);
        transition: width 0.4s ease;
    }}

    /* ── Tabs 精修 ── */
    [data-testid="stTabs"] [data-baseweb="tab"] {{
        font-weight: 600;
        border-radius: 8px 8px 0 0;
        transition: background-color 0.15s ease, color 0.15s ease;
    }}
    [data-testid="stTabs"] [data-baseweb="tab"]:hover {{
        background-color: #eff6ff;
        color: {COLORS["primary_dark"]};
    }}

    /* ── Expander 卡片化 ── */
    [data-testid="stExpander"] {{
        border: 1px solid {COLORS["border"]};
        border-radius: 10px;
        background: {COLORS["white"]};
        box-shadow: {SHADOWS[1]};
        overflow: hidden;
    }}

    /* ── Alert 提示框圆角 ── */
    [data-testid="stAlert"] {{
        border-radius: 10px;
    }}

    /* ── DataFrame 圆角裁切 ── */
    [data-testid="stDataFrame"] {{
        border-radius: 10px;
        overflow: hidden;
    }}

    /* ── 侧边栏链接过渡 ── */
    [data-testid="stSidebar"] a {{
        border-radius: 8px;
        transition: background-color 0.15s ease;
    }}

    /* ── 选中行高亮 ── */
    .theme-row-selected {{
        background: #f0f7ff;
        border-left: 3px solid {COLORS["primary"]};
        border-radius: 6px;
        padding: 6px 10px;
    }}

    /* ── 普通行 ── */
    .theme-row-normal {{
        border-left: 3px solid transparent;
        padding: 6px 10px;
    }}

    /* ── Expander 内列表间距 ── */
    .stExpander ul ul, .stExpander ol ul, .stExpander ul ol {{
        margin-top: 0.25rem;
        margin-bottom: 0.25rem;
        padding-left: 1.5rem;
    }}
    .stExpander li {{
        margin-bottom: 0.2rem;
    }}
    .stExpander li > p {{
        margin-bottom: 0.1rem;
    }}

    /* ── 删除确认框 ── */
    .theme-confirm-box {{
        border: 1.5px solid #fca5a5;
        border-radius: 12px;
        padding: 16px 20px;
        background: linear-gradient(145deg, #fff5f5 0%, #fef2f2 100%);
        box-shadow: 0 4px 12px rgba(220,38,38,0.08);
        margin-bottom: 10px;
    }}

    /* ── 选中操作行背景 ── */
    .theme-action-row {{
        background: #f0f7ff;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        border: 1px solid #b8daff;
    }}
    </style>
    """, unsafe_allow_html=True)


def card_container(shadow: int = 2):
    """返回卡片容器的开始 div HTML"""
    if shadow == 1:
        return f'<div class="theme-card-l1">'
    elif shadow == 3:
        return f'<div class="theme-card-l3">'
    return f'<div class="theme-card">'


def card_end():
    """返回卡片容器的结束 div"""
    return '</div>'


def divider():
    """返回优雅分割线 HTML"""
    return '<div class="theme-divider"></div>'


def section_title(text: str):
    """返回区块标题 HTML"""
    return f'<div class="theme-section-title">{text}</div>'


def score_badge_html(score, label: str = "") -> str:
    """统一分数徽章 HTML。
    分数 >= 80: 绿色, >= 60: 黄色, < 60: 红色, None: 灰色。
    """
    if score is None:
        color = COLORS["text_muted"]
        bg = "#f1f5f9"
        text = "N/A"
    else:
        try:
            s = float(score)
        except (ValueError, TypeError):
            color = COLORS["text_muted"]
            bg = "#f1f5f9"
            text = str(score)
            return f'<span style="display:inline-block;padding:3px 12px;background:{bg};color:{color};border-radius:10px;font-weight:700;font-size:0.88em;border:1px solid {color}30;">{text}</span>'

        if s >= 80:
            color = COLORS["success"]
            bg = COLORS["success_bg"]
        elif s >= 60:
            color = COLORS["warning"]
            bg = COLORS["warning_bg"]
        else:
            color = COLORS["danger"]
            bg = COLORS["danger_bg"]
        text = f"{label} {s:.0f}" if label else f"{s:.0f}"

    return (
        f'<span style="display:inline-block;padding:3px 12px;'
        f'background:{bg};color:{color};border-radius:10px;'
        f'font-weight:700;font-size:0.88em;border:1px solid {color}25;">'
        f'{text}</span>'
    )


def score_color(score) -> str:
    """返回分数对应的颜色值"""
    if score is None:
        return COLORS["text_muted"]
    try:
        s = float(score)
    except (ValueError, TypeError):
        return COLORS["text_muted"]
    if s >= 80:
        return COLORS["success"]
    elif s >= 60:
        return COLORS["warning"]
    return COLORS["danger"]


def score_bg(score) -> str:
    """返回分数对应的背景色"""
    if score is None:
        return "#f3f4f6"
    try:
        s = float(score)
    except (ValueError, TypeError):
        return "#f3f4f6"
    if s >= 80:
        return COLORS["success_bg"]
    elif s >= 60:
        return COLORS["warning_bg"]
    return COLORS["danger_bg"]


def page_title(title: str):
    """渲染统一的页面标题"""
    st.markdown(
        f'<div class="theme-page-title">{title}</div>',
        unsafe_allow_html=True,
    )
