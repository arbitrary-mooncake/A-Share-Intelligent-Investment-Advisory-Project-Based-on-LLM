"""
AI 智能问答 — ChatGPT 风格：左侧会话列表 + 底部输入框 + 自动命名
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

st.set_page_config(page_title="AI 智能问答", page_icon="💬", layout="wide")

# ──────────────────────────────────────────────
# 样式
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* 让主内容区填满高度 */
.main .block-container { padding-top: 1rem; padding-bottom: 0; }
/* 输入框固定在底部 */
.stChatInput { position: fixed; bottom: 0; background: white; padding: 1rem 0; z-index: 100; }
/* 给消息区留出输入框空间 */
.stChatMessage { margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────

def _sync_call(coro):
    """Streamlit 安全的同步调用异步函数"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result(timeout=30)
        return asyncio.run(coro)
    except RuntimeError:
        return asyncio.run(coro)


def _refresh_sessions():
    try:
        st.session_state["qa_sessions"] = _sync_call(qa_list_sessions())
    except Exception:
        if "qa_sessions" not in st.session_state:
            st.session_state["qa_sessions"] = []


def _load_history(session_id: str):
    """加载会话历史到 qa_messages"""
    if not session_id:
        st.session_state["qa_messages"] = []
        return
    try:
        data = _sync_call(qa_get_session(session_id))
        history = data.get("history", [])
        st.session_state["qa_messages"] = [
            {"role": m["role"], "content": m["content"]}
            for m in history
        ]
    except Exception:
        if "qa_messages" not in st.session_state:
            st.session_state["qa_messages"] = []


def _auto_name(question: str) -> str:
    """根据首条问题自动生成会话名"""
    q = question.strip()
    # 提取核心内容，截断
    for prefix in ["分析", "帮我", "请", "我想", "如何", "怎么"]:
        if q.startswith(prefix):
            q = q[len(prefix):]
            break
    return q[:20] + ("..." if len(q) > 20 else "")


# ──────────────────────────────────────────────
# 初始化状态
# ──────────────────────────────────────────────
if "qa_sessions" not in st.session_state:
    _refresh_sessions()
if "qa_active_session" not in st.session_state:
    st.session_state["qa_active_session"] = None
if "qa_messages" not in st.session_state:
    st.session_state["qa_messages"] = []
if "qa_sidebar_collapsed" not in st.session_state:
    st.session_state["qa_sidebar_collapsed"] = False
if "qa_rename_target" not in st.session_state:
    st.session_state["qa_rename_target"] = None
if "qa_pending_rename" not in st.session_state:
    st.session_state["qa_pending_rename"] = None

sessions = st.session_state.get("qa_sessions", [])
active_id = st.session_state.get("qa_active_session")
collapsed = st.session_state.get("qa_sidebar_collapsed", False)
current_name = "AI 投资研究助手"

# 找到当前会话名
for s in sessions:
    if s["session_id"] == active_id:
        current_name = s.get("name", "对话")
        break

# ──────────────────────────────────────────────
# 两栏布局
# ──────────────────────────────────────────────

if collapsed:
    # 收起：全宽对话
    if st.button("▶ 展开会话列表", key="expand_top"):
        st.session_state.qa_sidebar_collapsed = False
        st.rerun()

    st.markdown(f"### 💬 {current_name}")
    st.caption(f"AI 投资研究助手 · ID: `{active_id or '未选择'}`")

    _render_messages()
    _render_chat_input(active_id, current_name)

else:
    left, right = st.columns([0.22, 0.78])

    with left:
        st.markdown("### 会话列表")
        if st.button("◀ 收起", use_container_width=True):
            st.session_state.qa_sidebar_collapsed = True
            st.rerun()
        st.divider()

        # 会话项
        for sess in sessions:
            sid = sess["session_id"]
            is_active = (sid == active_id)
            name = sess.get("name", "新对话")
            msg_count = sess.get("message_count", 0)
            updated = sess.get("updated_at", 0)
            time_str = datetime.fromtimestamp(updated).strftime("%m/%d %H:%M") if updated else ""

            # 重命名中
            if st.session_state.qa_pending_rename == sid:
                new_name = st.text_input("新名称", value=name, key=f"rn_{sid}",
                                         label_visibility="collapsed")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("✓", key=f"ok_{sid}"):
                        nn = new_name.strip()
                        if nn:
                            _sync_call(qa_rename_session(sid, nn))
                            st.session_state.qa_pending_rename = None
                            _refresh_sessions()
                            st.rerun()
                with c2:
                    if st.button("✗", key=f"cx_{sid}"):
                        st.session_state.qa_pending_rename = None
                        st.rerun()
            else:
                # 会话条目按钮
                label = f"{'🔵 ' if is_active else '  '}{name[:18]}"
                if st.button(label, key=f"sess_{sid}", use_container_width=True,
                             help=f"{msg_count}条 · {time_str}"):
                    st.session_state["qa_active_session"] = sid
                    _load_history(sid)
                    st.rerun()

                # 元信息
                st.caption(f"  {msg_count}条 · {time_str}")

                # 操作菜单
                with st.expander("⚙", expanded=False):
                    if st.button("✏️ 重命名", key=f"ed_{sid}"):
                        st.session_state.qa_pending_rename = sid
                        st.rerun()
                    if st.button("🗑️ 删除", key=f"dl_{sid}"):
                        _sync_call(qa_delete_session(sid))
                        if active_id == sid:
                            st.session_state["qa_active_session"] = None
                            st.session_state["qa_messages"] = []
                        _refresh_sessions()
                        st.rerun()

        st.divider()

        # 新建
        new_name = st.text_input("新窗口名", key="new_name", label_visibility="collapsed",
                                 placeholder="留空则自动命名")
        if st.button("＋ 新建窗口", use_container_width=True):
            name = new_name.strip() or "新对话"
            result = _sync_call(qa_create_session(name))
            sid = result.get("session_id", "")
            st.session_state["qa_active_session"] = sid
            st.session_state["qa_messages"] = []
            _refresh_sessions()
            st.rerun()

        if st.button("🔄 刷新", use_container_width=True):
            _refresh_sessions()
            st.rerun()

    with right:
        st.markdown(f"### 💬 {current_name}")
        st.caption(f"AI 投资研究助手 · ID: `{active_id or '未选择'}`")

        _render_messages()
        _render_chat_input(active_id, current_name)


# ──────────────────────────────────────────────
# 渲染消息
# ──────────────────────────────────────────────

def _render_messages():
    for msg in st.session_state.get("qa_messages", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


# ──────────────────────────────────────────────
# 底部输入 + 发送逻辑
# ──────────────────────────────────────────────

def _render_chat_input(session_id, session_name):
    if not session_id:
        st.chat_input("请先在左侧选择或创建会话窗口", disabled=True)
        return

    prompt = st.chat_input("输入您的投资分析问题...", key="main_input")

    if not prompt:
        return

    prompt = prompt.strip()
    if not prompt:
        return

    # 自动命名：首条消息后使用问题摘要
    msg_count = len(st.session_state.get("qa_messages", []))
    if msg_count == 0 and session_name in ("新对话", "对话"):
        auto_name = _auto_name(prompt)
        _sync_call(qa_rename_session(session_id, auto_name))
        _refresh_sessions()

    # 添加用户消息
    st.session_state["qa_messages"].append({"role": "user", "content": prompt})

    # 流式获取回答
    full_answer = ""
    meta_info = ""

    answer_placeholder = st.chat_message("assistant")
    with answer_placeholder:
        status_el = st.empty()
        answer_el = st.empty()

    try:
        async def _stream():
            nonlocal full_answer, meta_info
            async for event in qa_ask_stream(question=prompt, session_id=session_id):
                etype = event.get("type")
                edata = event.get("data")

                if etype == "meta":
                    complexity = edata.get("complexity", "")
                    stock = edata.get("company_name", "")
                    parts = [f"复杂度:{complexity}"]
                    if stock:
                        parts.append(f"股票:{stock}")
                    meta_info = " | ".join(parts)

                elif etype == "status":
                    status_el.caption(f"⏳ {edata.get('message', '')}")

                elif etype == "answer":
                    status_el.empty()
                    full_answer += str(edata)
                    answer_el.markdown(full_answer + "▌")

                elif etype == "clarify":
                    status_el.empty()
                    answer_el.warning(edata.get("message", ""))
                    full_answer = f"(_需澄清: {edata.get('message', '')}_)"

                elif etype == "error":
                    status_el.empty()
                    answer_el.error(str(edata))
                    full_answer = f"(_请求失败: {edata}_)"

                elif etype == "done":
                    pass

        _sync_call(_stream())

    except Exception as e:
        answer_el.error(f"请求失败: {e}")
        full_answer = f"(_请求失败: {e}_)"

    status_el.empty()
    if full_answer:
        answer_el.markdown(full_answer)
        st.session_state["qa_messages"].append({
            "role": "assistant",
            "content": full_answer,
            "meta": meta_info,
        })
        _refresh_sessions()

    st.rerun()
