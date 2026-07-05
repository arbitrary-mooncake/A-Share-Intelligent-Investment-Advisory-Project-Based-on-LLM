"""
股票查询页面 — 快速查询 + 深度报告生成
"""

import asyncio
import os
import sys

# 将 app/ 目录加入 sys.path，使其下的模块可被导入
_app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

import streamlit as st

from components.query_form import render_query_form
from components.result_card import render_result_card
from components.shared_sidebar import render_sidebar
from api_client import (
    APIError,
    quick_query,
    trigger_report,
    poll_report_status,
)
from theme import inject_global_styles

# ──────────────────────────────────────────────
# 页面配置
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="股票查询",
    page_icon="🔍",
    layout="wide",
)

inject_global_styles()
render_sidebar()


def handle_quick_query(stock_input: str):
    """同步包装器 — 快速查询"""
    async def _run():
        return await quick_query(stock_input)
    return asyncio.run(_run())


def handle_trigger_report(stock_input: str):
    """同步包装器 — 触发报告并返回 task_id"""
    async def _run():
        return await trigger_report(stock_input)
    return asyncio.run(_run())


def handle_poll_report(task_id: str, on_progress=None):
    """同步包装器 — 轮询报告状态"""
    async def _run():
        return await poll_report_status(task_id, progress_callback=on_progress)
    return asyncio.run(_run())


# ──────────────────────────────────────────────
# 页面标题
# ──────────────────────────────────────────────
st.title("🔍 股票查询")
st.caption("输入股票代码或名称，快速获取分析或生成深度报告")


# ──────────────────────────────────────────────
# 查询表单
# ──────────────────────────────────────────────
is_busy = st.session_state.get("_query_busy", False)


def on_query_click(stock_input):
    st.session_state["_query_busy"] = True
    st.session_state["_query_result"] = None
    st.session_state["_query_error"] = None
    st.session_state["_query_input"] = stock_input

    try:
        result = handle_quick_query(stock_input)
        st.session_state["_query_result"] = result
    except APIError as e:
        st.session_state["_query_error"] = str(e)
    except Exception as e:
        st.session_state["_query_error"] = f"查询异常: {e}"
    finally:
        st.session_state["_query_busy"] = False


def on_report_click(stock_input):
    st.session_state["_query_busy"] = True
    st.session_state["_report_task_id"] = None
    st.session_state["_report_result"] = None
    st.session_state["_report_error"] = None
    st.session_state["_report_input"] = stock_input

    try:
        task_id = handle_trigger_report(stock_input)
        st.session_state["_report_task_id"] = task_id
    except APIError as e:
        st.session_state["_report_error"] = str(e)
        st.session_state["_query_busy"] = False
    except Exception as e:
        st.session_state["_report_error"] = f"触发报告异常: {e}"
        st.session_state["_query_busy"] = False


render_query_form(
    on_query=lambda s: on_query_click(s),
    on_report=lambda s: on_report_click(s),
    disabled=is_busy,
)

# ──────────────────────────────────────────────
# 显示快速查询结果
# ──────────────────────────────────────────────
if st.session_state.get("_query_result"):
    render_result_card(st.session_state["_query_result"])

if st.session_state.get("_query_error"):
    st.error(st.session_state["_query_error"])

# ──────────────────────────────────────────────
# 报告生成状态
# ──────────────────────────────────────────────
task_id = st.session_state.get("_report_task_id")

# 有 task_id 且尚未获得最终结果 → 轮询
if task_id and not st.session_state.get("_report_result") and not st.session_state.get("_report_error"):
    input_label = st.session_state.get("_report_input", "该股票")

    with st.status(
        f"正在为 **{input_label}** 生成深度分析报告...",
        expanded=True,
        state="running",
    ) as status:
        progress = st.progress(0, text="初始化分析任务...")

        def on_progress(real_progress):
            pct = min(real_progress, 1.0)
            stage = "正在执行多维分析..." if pct < 0.6 else "正在生成报告..."
            progress.progress(pct, text=f"{stage} ({pct:.0%})")

        try:
            result = handle_poll_report(task_id, on_progress=on_progress)

            if result.get("status") == "completed":
                progress.progress(1.0, text="报告生成完成！")
                status.update(label="报告已生成", state="complete", expanded=False)
                st.session_state["_report_result"] = result
            else:
                err = result.get("error", "未知错误")
                status.update(label="报告生成失败", state="error", expanded=False)
                st.session_state["_report_error"] = err
        except APIError as e:
            status.update(label="报告生成失败", state="error", expanded=False)
            st.session_state["_report_error"] = str(e)
        except Exception as e:
            status.update(label="报告生成失败", state="error", expanded=False)
            st.session_state["_report_error"] = f"轮询异常: {e}"
        finally:
            st.session_state["_query_busy"] = False

# 报告生成完成 → 显示下载按钮
if st.session_state.get("_report_result"):
    result = st.session_state["_report_result"]
    st.success("深度分析报告已生成！")
    report_content = result.get("report_content", "")
    stock_name = result.get("company_name", "股票")
    stock_code = result.get("stock_code", "")

    # 生成文件名
    from datetime import datetime
    date_str = datetime.now().strftime("%Y%m%d")

    pdf_path = result.get("report_pdf_path")
    if pdf_path and os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()
        pdf_filename = f"{stock_name}_分析报告_{date_str}.pdf"
        st.download_button(
            label="📄 下载 PDF 报告",
            data=pdf_data,
            file_name=pdf_filename,
            mime="application/pdf",
            type="primary",
        )
    else:
        md_filename = f"{stock_name}_分析报告_{date_str}.md"
        st.download_button(
            label="下载报告 (Markdown)",
            data=report_content,
            file_name=md_filename,
            mime="text/markdown",
            type="primary",
        )

    # 折叠展示报告内容
    with st.expander("预览报告内容"):
        st.markdown(report_content)

    # 清除按钮，允许用户生成新报告
    if st.button("清除结果，开始新查询"):
        for key in [
            "_report_task_id", "_report_result", "_report_error",
            "_report_input", "_query_result", "_query_error", "_query_input",
        ]:
            st.session_state.pop(key, None)

# 报告错误
if st.session_state.get("_report_error"):
    st.error(st.session_state["_report_error"])
    if st.button("清除错误，重试"):
        for key in [
            "_report_task_id", "_report_result", "_report_error", "_report_input",
        ]:
            st.session_state.pop(key, None)
