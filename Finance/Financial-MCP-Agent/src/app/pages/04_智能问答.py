"""
智能问答页面 — 多会话窗口 + 自然语言开放式A股分析对话
"""
import asyncio
import os
import sys
from datetime import datetime

_app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

import streamlit as st
from api_client import (
    qa_ask_stream, qa_list_sessions, qa_create_session,
    qa_delete_session, qa_rename_session, APIError,
)

st.set_page_config(
    page_title="智能问答",
    page_icon="💬",
    layout="wide",
)

st.markdown("""
<style>
.chat-meta { font-size: 0.8rem; color: #888; margin-bottom: 0.3rem; }
.session-item { padding: 0.5rem 0.3rem; border-radius: 6px; margin: 2px 0; cursor: pointer; }
.session-item:hover { background: #f0f0f0; }
.session-item.active { background: #e3f2fd; border-left: 3px solid #1976d2; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# 刷新会话列表
# ──────────────────────────────────────────────
def _refresh_sessions():
    try:
        st.session_state["qa_sessions"] = asyncio.run(qa_list_sessions())
    except Exception:
        if "qa_sessions" not in st.session_state:
            st.session_state["qa_sessions"] = []


if "qa_sessions" not in st.session_state:
    _refresh_sessions()
if "qa_active_session" not in st.session_state:
    st.session_state["qa_active_session"] = None
if "qa_messages" not in st.session_state:
    st.session_state.qa_messages = []
if "qa_rename_target" not in st.session_state:
    st.session_state.qa_rename_target = None


# ──────────────────────────────────────────────
# 侧边栏：会话窗口列表
# ──────────────────────────────────────────────
with st.sidebar:
    st.subheader("💬 会话窗口")

    # 新建会话
    new_name = st.text_input("新窗口名称", placeholder="输入名称（可选）", key="new_session_name",
                             label_visibility="collapsed")
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("➕ 新建窗口", use_container_width=True):
            name = new_name.strip() or "新对话"
            try:
                result = asyncio.run(qa_create_session(name))
                sid = result.get("session_id", "")
                st.session_state["qa_active_session"] = sid
                st.session_state.qa_messages = []
                _refresh_sessions()
                st.rerun()
            except Exception as e:
                st.error(f"创建失败: {e}")
    with c2:
        if st.button("🔄 刷新", use_container_width=True):
            _refresh_sessions()
            st.rerun()

    st.divider()

    # 会话列表
    sessions = st.session_state.get("qa_sessions", [])
    active_id = st.session_state.get("qa_active_session")

    if not sessions:
        st.caption("暂无会话窗口，点击上方按钮创建")

    for sess in sessions:
        sid = sess["session_id"]
        is_active = (sid == active_id)
        name = sess.get("name", "新对话")
        msg_count = sess.get("message_count", 0)
        updated = sess.get("updated_at", 0)
        time_str = datetime.fromtimestamp(updated).strftime("%m/%d %H:%M") if updated else ""

        # 当前正在重命名的会话
        renaming = (st.session_state.qa_rename_target == sid)

        col_main, col_del = st.columns([4, 1])

        with col_main:
            if renaming:
                rename_input = st.text_input("新名称", value=name, key=f"rename_input_{sid}",
                                             label_visibility="collapsed")
                rc1, rc2 = st.columns(2)
                with rc1:
                    if st.button("✅", key=f"confirm_{sid}", help="确认"):
                        new_n = rename_input.strip()
                        if new_n and new_n != name:
                            try:
                                asyncio.run(qa_rename_session(sid, new_n))
                                st.session_state.qa_rename_target = None
                                _refresh_sessions()
                                st.rerun()
                            except Exception as e:
                                st.error(f"失败: {e}")
                with rc2:
                    if st.button("❌", key=f"cancel_{sid}", help="取消"):
                        st.session_state.qa_rename_target = None
                        st.rerun()
            else:
                style = "font-weight:bold;" if is_active else ""
                btn_label = f"{'🔵 ' if is_active else ''}{name}"
                if st.button(btn_label, key=f"sel_{sid}", use_container_width=True,
                             help=f"{msg_count}条消息 · {time_str}"):
                    st.session_state["qa_active_session"] = sid
                    st.session_state.qa_messages = []
                    st.rerun()
                st.caption(f"{msg_count}条 · {time_str}")

        with col_del:
            if not renaming:
                if st.button("⋯", key=f"menu_{sid}", help="重命名/删除"):
                    pass  # 展开操作菜单

        # 操作菜单（重命名 + 删除）
        if not renaming:
            with st.expander("", expanded=False, expanded_icon="⚙"):
                if st.button("✏️ 重命名", key=f"rename_btn_{sid}"):
                    st.session_state.qa_rename_target = sid
                    st.rerun()
                if st.button("🗑️ 删除窗口", key=f"del_btn_{sid}"):
                    try:
                        asyncio.run(qa_delete_session(sid))
                        if active_id == sid:
                            st.session_state["qa_active_session"] = None
                            st.session_state.qa_messages = []
                        st.session_state.qa_rename_target = None
                        _refresh_sessions()
                        st.rerun()
                    except Exception as e:
                        st.error(f"删除失败: {e}")

    st.divider()

    # 提问示例
    st.caption("**支持的问题类型：**")
    st.caption("• 单只股票分析（估值/走势/财务）")
    st.caption("• 多标的对比（行业/跨行业）")
    st.caption("• 行业/板块/宏观研判")
    st.caption("• 情景推演与策略建议")
    st.caption("• 黄金/商品/经济形势")

    st.divider()
    st.caption("**技巧：** 提及股票代码更准确；可追问「展开讲讲」")


# ──────────────────────────────────────────────
# 主区域
# ──────────────────────────────────────────────
active_id = st.session_state.get("qa_active_session")

if active_id:
    # 从sessions列表找当前会话名
    current_name = "对话"
    for s in sessions:
        if s["session_id"] == active_id:
            current_name = s.get("name", "对话")
            break
    st.title(f"💬 {current_name}")
    st.caption(f"会话ID: `{active_id}` | 与资深A股投研分析师直接对话")
else:
    st.title("💬 智能问答")
    st.caption("在左侧选择一个会话窗口，或创建新窗口开始对话")

# ──────────────────────────────────────────────
# 渲染历史消息
# ──────────────────────────────────────────────
for msg in st.session_state.qa_messages:
    with st.chat_message(msg["role"]):
        if msg.get("meta"):
            st.caption(msg["meta"])
        st.markdown(msg["content"])


# ──────────────────────────────────────────────
# 输入框（有活跃会话时才可用）
# ──────────────────────────────────────────────
if active_id:
    prompt_placeholder = "输入您的投资分析问题..."
else:
    prompt_placeholder = "请先在左侧创建或选择一个会话窗口"

if prompt := st.chat_input(prompt_placeholder, disabled=(not active_id)):
    # 添加用户消息
    st.session_state.qa_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    maybe_rerun = False

    async def _process():
        nonlocal maybe_rerun
        full_answer = ""
        meta_info = ""

        answer_container = st.chat_message("assistant")
        with answer_container:
            status_placeholder = st.empty()
            meta_placeholder = st.empty()
            answer_placeholder = st.empty()

        try:
            async for event in qa_ask_stream(
                question=prompt,
                session_id=active_id,
            ):
                event_type = event.get("type")
                event_data = event.get("data")

                if event_type == "meta":
                    sid = event_data.get("session_id", "")
                    complexity = event_data.get("complexity", "")
                    stock = event_data.get("company_name", "")
                    if sid and sid != active_id:
                        st.session_state["qa_active_session"] = sid
                        maybe_rerun = True
                    parts = [f"复杂度: {complexity}"]
                    if stock:
                        parts.append(f"股票: {stock}")
                    meta_info = " | ".join(parts)
                    meta_placeholder.caption(meta_info)

                elif event_type == "status":
                    status_placeholder.caption(f"⏳ {event_data.get('message', '处理中...')}")

                elif event_type == "answer":
                    status_placeholder.empty()
                    full_answer += str(event_data)
                    answer_placeholder.markdown(full_answer + "▌")

                elif event_type == "clarify":
                    status_placeholder.empty()
                    answer_placeholder.warning(event_data.get("message", "需要更多信息"))
                    full_answer = f"(_需澄清: {event_data.get('message', '')}_)"

                elif event_type == "error":
                    status_placeholder.empty()
                    answer_placeholder.error(event_data)
                    full_answer = f"(_请求失败: {event_data}_)"

                elif event_type == "done":
                    pass

        except Exception as e:
            answer_placeholder.error(f"请求失败: {e}")
            full_answer = f"(_请求失败: {e}_)"

        status_placeholder.empty()
        if full_answer:
            answer_placeholder.markdown(full_answer)
            st.session_state.qa_messages.append({
                "role": "assistant",
                "content": full_answer,
                "meta": meta_info,
            })
            _refresh_sessions()

    asyncio.run(_process())
    st.rerun()


# ──────────────────────────────────────────────
# 底部操作
# ──────────────────────────────────────────────
if active_id and st.session_state.qa_messages:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ 清空当前对话", use_container_width=True):
            st.session_state.qa_messages = []
            st.rerun()
    with col2:
        if st.button("📋 显示最后回答", use_container_width=True):
            last = [m for m in st.session_state.qa_messages if m["role"] == "assistant"]
            if last:
                st.code(last[-1]["content"], language=None)
