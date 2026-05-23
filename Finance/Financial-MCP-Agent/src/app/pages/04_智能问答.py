"""
智能问答页面 — 两栏布局：可收起会话列表 + 对话区
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
    qa_delete_session, qa_rename_session, qa_get_session, APIError,
)

st.set_page_config(
    page_title="智能问答",
    page_icon="💬",
    layout="wide",
)

# ──────────────────────────────────────────────
# 初始状态
# ──────────────────────────────────────────────
if "qa_sessions" not in st.session_state:
    st.session_state.qa_sessions = []
if "qa_active_session" not in st.session_state:
    st.session_state.qa_active_session = None
if "qa_messages" not in st.session_state:
    st.session_state.qa_messages = []
if "qa_sidebar_collapsed" not in st.session_state:
    st.session_state.qa_sidebar_collapsed = False
if "qa_rename_target" not in st.session_state:
    st.session_state.qa_rename_target = None
if "qa_show_rename_input" not in st.session_state:
    st.session_state.qa_show_rename_input = False
if "qa_rename_session_id" not in st.session_state:
    st.session_state.qa_rename_session_id = None


def _refresh_sessions():
    try:
        st.session_state["qa_sessions"] = asyncio.run(qa_list_sessions())
    except Exception as e:
        if st.session_state.get("qa_sessions") is None:
            st.session_state["qa_sessions"] = []


def _load_session_history(session_id: str):
    """从后端加载指定会话的完整历史"""
    if not session_id:
        return
    try:
        data = asyncio.run(qa_get_session(session_id))
        history = data.get("history", [])
        st.session_state.qa_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in history
        ]
    except Exception:
        st.session_state.qa_messages = []


if not st.session_state.qa_sessions:
    _refresh_sessions()

active_id = st.session_state.get("qa_active_session")

# 页面首次加载时，自动恢复活跃会话的历史消息
if active_id and not st.session_state.get("qa_messages"):
    _load_session_history(active_id)

collapsed = st.session_state.get("qa_sidebar_collapsed", False)
sessions = st.session_state.get("qa_sessions", [])

# ──────────────────────────────────────────────
# 自定义样式
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* 会话列表条目 */
.session-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.6rem 0.8rem; margin: 2px 0; border-radius: 8px;
    cursor: pointer; transition: background 0.15s;
}
.session-row:hover { background: #f0f2f5; }
.session-row.active { background: #e3f2fd; border-left: 3px solid #1976d2; }
.session-name { font-size: 0.92rem; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 180px; }
.session-meta { font-size: 0.72rem; color: #999; }
/* 收起/展开按钮 */
.toggle-btn { font-size: 1.1rem; padding: 0.2rem 0.5rem; border: 1px solid #ddd; border-radius: 6px; background: #fff; cursor: pointer; }
.toggle-btn:hover { background: #f0f0f0; }
.toggle-btn.floating { position: fixed; left: 10px; top: 50%; z-index: 100; }
/* 对话区域 */
.chat-area { padding: 0 1rem; }
/* 创建按钮 */
.new-session-btn { width: 100%; padding: 0.5rem; border: 1px dashed #ccc; border-radius: 8px; background: transparent; color: #666; cursor: pointer; text-align: center; }
.new-session-btn:hover { border-color: #1976d2; color: #1976d2; background: #f5f8ff; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# 主区域两栏布局
# ──────────────────────────────────────────────

if collapsed:
    # ── 收起状态：对话区占满 + 浮动展开按钮 ──
    col_main = st.container()

    with col_main:
        ctl, _ = st.columns([0.05, 0.95])
        with ctl:
            if st.button("▶", key="expand_btn", help="展开会话列表"):
                st.session_state.qa_sidebar_collapsed = False
                st.rerun()

        st.title("💬 智能问答")
        if active_id:
            current_name = "对话"
            for s in sessions:
                if s["session_id"] == active_id:
                    current_name = s.get("name", "对话")
                    break
            st.caption(f"当前窗口: {current_name} | ID: `{active_id}`")
        else:
            st.caption("请展开会话列表选择一个窗口")

        # 消息历史
        for msg in st.session_state.qa_messages:
            with st.chat_message(msg["role"]):
                if msg.get("meta"):
                    st.caption(msg["meta"])
                st.markdown(msg["content"])

        # 输入框
        if active_id:
            if prompt := st.chat_input("输入您的投资分析问题...", key="chat_input_collapsed"):
                _handle_chat_input(prompt, active_id)
        else:
            st.chat_input("请先选择会话窗口", disabled=True, key="chat_disabled_collapsed")

else:
    # ── 展开状态：左窄列 + 右对话区 ──
    col_left, col_right = st.columns([0.22, 0.78])

    with col_left:
        st.markdown("### 💬 会话列表")

        # 收起按钮
        if st.button("◀ 收起", key="collapse_btn", use_container_width=True):
            st.session_state.qa_sidebar_collapsed = True
            st.rerun()

        st.divider()

        # 会话列表
        for sess in sessions:
            sid = sess["session_id"]
            is_active = (sid == active_id)
            name = sess.get("name", "新对话")
            msg_count = sess.get("message_count", 0)
            updated = sess.get("updated_at", 0)
            time_str = datetime.fromtimestamp(updated).strftime("%m/%d %H:%M") if updated else ""

            # 重命名状态
            if st.session_state.qa_rename_session_id == sid:
                new_name = st.text_input("新名称", value=name, key=f"rn_{sid}", label_visibility="collapsed")
                rc1, rc2 = st.columns(2)
                with rc1:
                    if st.button("✓", key=f"ok_{sid}"):
                        if new_name.strip() and new_name.strip() != name:
                            try:
                                asyncio.run(qa_rename_session(sid, new_name.strip()))
                                _refresh_sessions()
                            except Exception as e:
                                st.error(f"失败: {e}")
                        st.session_state.qa_rename_session_id = None
                        st.rerun()
                with rc2:
                    if st.button("✗", key=f"cx_{sid}"):
                        st.session_state.qa_rename_session_id = None
                        st.rerun()
            else:
                # 会话条目
                row_c1, row_c2 = st.columns([5, 1])
                with row_c1:
                    label = f"{'🔵 ' if is_active else '  '}{name}"
                    if st.button(label, key=f"sel_{sid}", use_container_width=True,
                                 help=f"{msg_count}条消息 · {time_str}"):
                        st.session_state["qa_active_session"] = sid
                        _load_session_history(sid)
                        st.rerun()
                with row_c2:
                    if st.button("⋯", key=f"mu_{sid}", help="操作菜单"):
                        st.session_state[f"qa_menu_open_{sid}"] = not st.session_state.get(f"qa_menu_open_{sid}", False)

                # 操作菜单
                if st.session_state.get(f"qa_menu_open_{sid}", False):
                    mc1, mc2 = st.columns(2)
                    with mc1:
                        if st.button("✏️", key=f"ed_{sid}", help="重命名"):
                            st.session_state.qa_rename_session_id = sid
                            st.session_state[f"qa_menu_open_{sid}"] = False
                            st.rerun()
                    with mc2:
                        if st.button("🗑️", key=f"dl_{sid}", help="删除"):
                            try:
                                asyncio.run(qa_delete_session(sid))
                                if active_id == sid:
                                    st.session_state["qa_active_session"] = None
                                    st.session_state.qa_messages = []
                                _refresh_sessions()
                                st.rerun()
                            except Exception as e:
                                st.error(f"删除失败: {e}")

                st.caption(f"  {msg_count}条 · {time_str}")

        st.divider()

        # 新建窗口
        new_name = st.text_input("窗口名称", placeholder="输入名称（可选）", key="new_name", label_visibility="collapsed")
        if st.button("＋ 新建窗口", use_container_width=True):
            name = new_name.strip() or "新对话"
            try:
                result = asyncio.run(qa_create_session(name))
                st.session_state["qa_active_session"] = result.get("session_id", "")
                st.session_state.qa_messages = []
                _refresh_sessions()
                st.rerun()
            except Exception as e:
                st.error(f"创建失败: {e}")

        # 刷新
        if st.button("🔄 刷新列表", use_container_width=True):
            _refresh_sessions()
            st.rerun()

    with col_right:
        if active_id:
            current_name = "对话"
            for s in sessions:
                if s["session_id"] == active_id:
                    current_name = s.get("name", "对话")
                    break

            st.markdown(f"### 💬 {current_name}")
            st.caption(f"ID: `{active_id}`")

            # 消息历史
            for msg in st.session_state.qa_messages:
                with st.chat_message(msg["role"]):
                    if msg.get("meta"):
                        st.caption(msg["meta"])
                    st.markdown(msg["content"])

            # 输入框
            if prompt := st.chat_input("输入您的投资分析问题...", key="chat_input_expanded"):
                _handle_chat_input(prompt, active_id)

        else:
            st.markdown("### 💬 智能问答")
            st.info("👈 在左侧选择一个会话窗口，或创建新窗口开始对话")


# ──────────────────────────────────────────────
# 对话输入处理函数
# ──────────────────────────────────────────────

def _handle_chat_input(prompt: str, session_id: str):
    """处理用户输入并流式展示回答"""
    st.session_state.qa_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    async def _process():
        full_answer = ""
        meta_info = ""

        answer_container = st.chat_message("assistant")
        with answer_container:
            status_placeholder = st.empty()
            meta_placeholder = st.empty()
            answer_placeholder = st.empty()

        try:
            async for event in qa_ask_stream(question=prompt, session_id=session_id):
                event_type = event.get("type")
                event_data = event.get("data")

                if event_type == "meta":
                    sid = event_data.get("session_id", "")
                    complexity = event_data.get("complexity", "")
                    stock = event_data.get("company_name", "")
                    if sid and sid != session_id:
                        st.session_state["qa_active_session"] = sid
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
