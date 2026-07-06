"""
模式面板组件：在侧边栏显示当前模式指示器和切换按钮。

Lite 模式：显示"⚡ 精简模式" + Tushare 积分 + DeepSeek 标识
Full 模式：显示"🚀 完整模式" + "全部功能已解锁"
"""
import streamlit as st


def render_mode_indicator():
    """在侧边栏底部渲染当前模式指示器"""
    from src.utils.mode_manager import get_mode, is_lite_mode

    mode = get_mode()

    with st.sidebar:
        st.markdown(
            '<div style="height:1px;background:linear-gradient(90deg,transparent,#dbeafe,transparent);margin:6px 0 10px 0;"></div>',
            unsafe_allow_html=True,
        )

        if is_lite_mode():
            st.markdown(
                '<div style="padding:8px 12px;background:#eff6ff;border-radius:8px;border:1px solid #bfdbfe;">'
                '<div style="font-size:0.95em;font-weight:700;color:#1e40af;">⚡ 精简模式</div>'
                '<div style="font-size:0.75em;color:#64748b;margin-top:2px;">'
                'LLM: DeepSeek | 数据: Tushare + AKShare'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            if st.button("🔄 切换到完整版", key="switch_to_full", use_container_width=True):
                st.session_state["show_switch_dialog"] = True
        else:
            st.markdown(
                '<div style="padding:8px 12px;background:#fefce8;border-radius:8px;border:1px solid #fde68a;">'
                '<div style="font-size:0.95em;font-weight:700;color:#92400e;">🚀 完整模式</div>'
                '<div style="font-size:0.75em;color:#64748b;margin-top:2px;">'
                '全部功能已解锁 | 6 模型 + Tushare 5000+'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            if st.button("🔄 切换到精简版", key="switch_to_lite", use_container_width=True):
                st.session_state["show_switch_dialog"] = True


def render_switch_dialog():
    """模式切换确认对话框"""
    if not st.session_state.get("show_switch_dialog"):
        return

    from src.utils.mode_manager import get_mode, set_mode

    current = get_mode()
    target = "full" if current == "lite" else "lite"

    @st.dialog(f"切换到{'完整版' if target == 'full' else '精简版'}")
    def _dialog():
        if target == "full":
            st.warning("切换到完整版需要配置 6 个 LLM API Key 和 Tushare 5000+ 积分。")
            st.markdown("您需要手动编辑 `.env` 文件，填入所有 Key，然后重启应用。")
        else:
            st.info("切换到精简版将使用 DeepSeek 单一模型。")
            st.markdown("「模拟分析与迭代」和「智能投顾」功能将不可用。")
            st.markdown("请确保已配置 `DEEPSEEK_API_KEY`。")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("确认切换", type="primary"):
                success = set_mode(target)
                if success:
                    st.session_state["show_switch_dialog"] = False
                    st.toast(f"已切换到{'完整版' if target == 'full' else '精简版'}，请重启应用。")
                    st.rerun()
                else:
                    st.error("切换失败，请手动编辑 .env 文件中的 APP_MODE 变量。")
        with col2:
            if st.button("取消"):
                st.session_state["show_switch_dialog"] = False
                st.rerun()

    _dialog()
