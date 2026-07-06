"""
共享侧边栏组件 — 在所有页面统一渲染导航。

配合 .streamlit/config.toml 的 [client] showSidebarNavigation = false 使用，
替代 Streamlit 自动生成的多页导航，避免与 Home.py 手动导航重复。

双版本架构：Lite 模式下「模拟分析与迭代」和「智能投顾」灰色禁用。
"""
import streamlit as st


NAV_ITEMS = [
    ("Home.py", "🏠 首页"),
    ("pages/01_股票查询.py", "🔍 股票查询"),
    ("pages/02_股票池.py", "📊 股票池"),
    ("pages/03_批量打分.py", "📋 批量打分"),
    ("pages/04_智能问答.py", "💬 智能问答"),
    ("pages/07_智能投顾.py", "🤖 智能投顾"),
    ("pages/05_基金专区.py", "🏦 基金专区"),
    ("pages/06_模拟分析与迭代.py", "📈 模拟分析与迭代"),
]

# Lite 模式下禁用的页面路径
_LITE_DISABLED = {
    "pages/06_模拟分析与迭代.py",
    "pages/07_智能投顾.py",
}


def render_sidebar():
    """渲染统一侧边栏导航（支持双版本模式）。"""
    # 延迟导入避免循环依赖
    from src.utils.mode_manager import is_lite_mode

    lite_mode = is_lite_mode()

    with st.sidebar:
        st.markdown(
            '<div style="padding:0.3rem 0 0.6rem 0;">'
            '<div style="font-size:1.15em;font-weight:800;color:#1e40af;letter-spacing:0.5px;">'
            '📈 AI 投资研究助手'
            '</div>'
            '<div style="font-size:0.72em;color:#94a3b8;margin-top:2px;">A股智能分析平台</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="height:1px;background:linear-gradient(90deg,transparent,#dbeafe,transparent);margin:6px 0 10px 0;"></div>',
            unsafe_allow_html=True,
        )
        for page, label in NAV_ITEMS:
            if lite_mode and page in _LITE_DISABLED:
                st.markdown(
                    f'<div style="padding:6px 12px;color:#9ca3af;font-size:0.9em;'
                    f'opacity:0.6;cursor:not-allowed;">{label} 🔒</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.page_link(page, label=label, use_container_width=True)
        st.markdown("---")
        st.caption("💡 提示：本平台仅供研究参考，不构成投资建议")

        # 渲染模式指示器
        from src.app.components.mode_panel import render_mode_indicator, render_switch_dialog
        render_mode_indicator()
        render_switch_dialog()
