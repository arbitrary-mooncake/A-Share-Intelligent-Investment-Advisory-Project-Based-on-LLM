"""
智能问答页面 — 自然语言开放式A股分析对话
"""
import asyncio
import os
import sys

_app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

import streamlit as st
from api_client import qa_ask_stream, APIError

st.set_page_config(
    page_title="智能问答",
    page_icon="💬",
    layout="wide",
)

# ──────────────────────────────────────────────
# 样式
# ──────────────────────────────────────────────
st.markdown("""
<style>
.chat-meta {
    font-size: 0.8rem;
    color: #888;
    margin-bottom: 0.3rem;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# 页面标题
# ──────────────────────────────────────────────
st.title("💬 智能问答")
st.caption("与资深A股投研分析师直接对话，支持自然语言开放式交流与多轮追问")

# ──────────────────────────────────────────────
# 侧边栏：会话管理 + 提示
# ──────────────────────────────────────────────
with st.sidebar:
    st.subheader("会话管理")

    if st.button("🆕 新建会话", use_container_width=True):
        st.session_state.pop("qa_session_id", None)
        st.session_state.pop("qa_messages", None)
        st.rerun()

    session_id = st.session_state.get("qa_session_id", "")
    if session_id:
        st.info(f"会话ID: `{session_id}`")
        msg_count = len(st.session_state.get("qa_messages", []))
        st.caption(f"当前对话: {msg_count} 条消息")
    else:
        st.caption("发送第一条消息后自动创建会话")

    st.divider()
    st.caption("**提问示例：**")
    st.caption("• 为什么茅台今天大涨？")
    st.caption("• 宁德时代PE高不高？")
    st.caption("• 利润增长了股价却不涨，原因是什么？")
    st.caption("• 半导体行业目前怎么看？")
    st.caption("• 黄金走势会怎么变化？")
    st.caption("• 把比亚迪和宁德时代做个对比")
    st.caption("• 你对目前中国经济形势怎么看？")

    st.divider()
    st.caption("**对话技巧：**")
    st.caption("• 问题越具体，回答越精准")
    st.caption("• 可以追问「展开讲讲」「为什么」")
    st.caption("• 提及股票代码获得更准确的数据")
    st.caption("• 复杂问题会自动启用深度分析")

# ──────────────────────────────────────────────
# 聊天消息历史
# ──────────────────────────────────────────────
if "qa_messages" not in st.session_state:
    st.session_state.qa_messages = []

# 渲染历史消息
for msg in st.session_state.qa_messages:
    with st.chat_message(msg["role"]):
        if msg.get("meta"):
            st.caption(msg["meta"], unsafe_allow_html=False)
        st.markdown(msg["content"])

# ──────────────────────────────────────────────
# 输入框
# ──────────────────────────────────────────────
if prompt := st.chat_input("输入您的投资分析问题..."):
    # 添加用户消息
    st.session_state.qa_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 异步处理回答
    async def _process():
        full_answer = ""
        meta_info = ""
        status_text = ""

        # 创建占位区域
        answer_container = st.chat_message("assistant")
        with answer_container:
            status_placeholder = st.empty()
            meta_placeholder = st.empty()
            answer_placeholder = st.empty()

        try:
            async for event in qa_ask_stream(
                question=prompt,
                session_id=st.session_state.get("qa_session_id"),
            ):
                event_type = event.get("type")
                event_data = event.get("data")

                if event_type == "meta":
                    sid = event_data.get("session_id", "")
                    complexity = event_data.get("complexity", "")
                    stock = event_data.get("company_name", "")
                    if sid:
                        st.session_state["qa_session_id"] = sid
                    parts = [f"复杂度: {complexity}"]
                    if stock:
                        parts.append(f"股票: {stock}")
                    meta_info = " | ".join(parts)
                    meta_placeholder.caption(meta_info)

                elif event_type == "status":
                    status_text = event_data.get("message", "处理中...")
                    status_placeholder.caption(f"⏳ {status_text}")

                elif event_type == "answer":
                    status_placeholder.empty()
                    full_answer += str(event_data)
                    answer_placeholder.markdown(full_answer + "▌")

                elif event_type == "clarify":
                    status_placeholder.empty()
                    answer_placeholder.warning(event_data.get("message", "需要更多信息"))
                    full_answer = f"(_需要澄清: {event_data.get('message', '')}_)"

                elif event_type == "error":
                    status_placeholder.empty()
                    answer_placeholder.error(event_data)
                    full_answer = f"(_请求失败: {event_data}_)"

                elif event_type == "done":
                    pass

        except Exception as e:
            answer_placeholder.error(f"请求失败: {e}")
            full_answer = f"(_请求失败: {e}_)"

        # 清理并保存
        status_placeholder.empty()
        if full_answer:
            answer_placeholder.markdown(full_answer)
            st.session_state.qa_messages.append({
                "role": "assistant",
                "content": full_answer,
                "meta": meta_info,
            })

    asyncio.run(_process())
    st.rerun()

# ──────────────────────────────────────────────
# 底部操作
# ──────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🗑️ 清除当前对话", use_container_width=True):
        st.session_state.qa_messages = []
        st.rerun()
with col2:
    if st.button("🔄 新建会话并清除", use_container_width=True):
        st.session_state.pop("qa_session_id", None)
        st.session_state.pop("qa_messages", None)
        st.rerun()
with col3:
    if st.button("📋 显示最后回答原文", use_container_width=True):
        if st.session_state.qa_messages:
            last = [m for m in st.session_state.qa_messages if m["role"] == "assistant"]
            if last:
                with st.expander("最后回答原文", expanded=True):
                    st.code(last[-1]["content"], language=None)
