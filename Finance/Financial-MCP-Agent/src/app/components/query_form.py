"""
查询表单组件 — 股票输入框 + 查询/生成报告按钮
"""

import streamlit as st


def render_query_form(
    on_query: callable,
    on_report: callable,
    disabled: bool = False,
) -> None:
    """渲染查询表单

    Args:
        on_query: 点击"查询"按钮时的回调函数
        on_report: 点击"生成报告"按钮时的回调函数
        disabled: 是否禁用按钮（正在执行中时）
    """
    st.markdown(
        """
        <style>
        .query-section {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 1.5rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
        }
        </style>
        <div class="query-section"></div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 输入股票代码或名称")

    col_input, col_btn1, col_btn2, col_spacer = st.columns(
        [6, 1.5, 2, 0.5]
    )

    with col_input:
        stock_input = st.text_input(
            label="股票代码/名称",
            placeholder="例如：603871 或 嘉友国际 或 分析嘉友国际",
            label_visibility="collapsed",
            key="query_stock_input",
            disabled=disabled,
        )

    with col_btn1:
        query_clicked = st.button(
            "查 询",
            type="primary",
            use_container_width=True,
            disabled=disabled or not stock_input.strip(),
            key="btn_quick_query",
        )

    with col_btn2:
        report_clicked = st.button(
            "生成报告",
            type="secondary",
            use_container_width=True,
            disabled=disabled or not stock_input.strip(),
            key="btn_generate_report",
        )

    # 按钮点击触发回调
    if query_clicked:
        on_query(stock_input.strip())

    if report_clicked:
        on_report(stock_input.strip())
