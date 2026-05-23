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
/* 页面撑满视口，flex列布局 */
.main .block-container {
    display: flex !important;
    flex-direction: column !important;
    min-height: calc(100vh - 4rem) !important;
    padding-bottom: 0 !important;
}
/* 消息区自动撑满剩余空间 */
.element-container:has(.stChatMessage) {
    flex: 1;
    overflow-y: auto;
}
/* 输入框固定在底部 */
.stChatInput {
    position: sticky !important;
    bottom: 0 !important;
    background: white !important;
    padding: 0.75rem 0 1rem 0 !important;
    border-top: 1px solid #eee !important;
    z-index: 10 !important;
}
.stChatInput textarea {
    min-height: 56px !important;
    font-size: 0.95rem !important;
}
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
# 渲染函数（必须在主流程之前定义）
# ──────────────────────────────────────────────

def _render_messages():
    for msg in st.session_state.get("qa_messages", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


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

    # 自动命名
    msg_count = len(st.session_state.get("qa_messages", []))
    if msg_count == 0 and session_name in ("新对话", "对话"):
        auto_name = _auto_name(prompt)
        _sync_call(qa_rename_session(session_id, auto_name))
        _refresh_sessions()

    st.session_state["qa_messages"].append({"role": "user", "content": prompt})

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
                    parts = [f"复杂度:{edata.get('complexity', '')}"]
                    if edata.get("company_name"):
                        parts.append(f"股票:{edata['company_name']}")
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
            "role": "assistant", "content": full_answer, "meta": meta_info,
        })
        _refresh_sessions()

    st.rerun()


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

            # 每一行：名称按钮 + 删除按钮
            row_l, row_r = st.columns([5, 1])

            with row_l:
                # 重命名模式：点击标题触发
                if st.session_state.qa_pending_rename == sid:
                    new_name = st.text_input("", value=name, key=f"rn_{sid}",
                                             label_visibility="collapsed")
                    # 不用嵌套列，并排按钮
                    if st.button("✓ 确认", key=f"ok_{sid}"):
                        nn = new_name.strip()
                        if nn:
                            _sync_call(qa_rename_session(sid, nn))
                        st.session_state.qa_pending_rename = None
                        _refresh_sessions()
                        st.rerun()
                    if st.button("✗ 取消", key=f"cx_{sid}"):
                        st.session_state.qa_pending_rename = None
                        st.rerun()
                else:
                    label = f"{'🔵 ' if is_active else '  '}{name[:20]}"
                    if st.button(label, key=f"sess_{sid}", use_container_width=True,
                                 help=f"{msg_count}条 · {time_str} · 点击标题可重命名"):
                        # 如果是活跃会话 → 触发重命名；否则 → 切换
                        if is_active:
                            st.session_state.qa_pending_rename = sid
                            st.rerun()
                        else:
                            st.session_state["qa_active_session"] = sid
                            _load_history(sid)
                            st.rerun()

            with row_r:
                # 删除按钮 + 确认
                if f"qa_confirm_del_{sid}" not in st.session_state:
                    st.session_state[f"qa_confirm_del_{sid}"] = False

                if st.session_state[f"qa_confirm_del_{sid}"]:
                    if st.button("✅确认", key=f"cfm_{sid}", help="确认删除"):
                        _sync_call(qa_delete_session(sid))
                        st.session_state[f"qa_confirm_del_{sid}"] = False
                        if active_id == sid:
                            _refresh_sessions()
                            remaining = [s for s in st.session_state["qa_sessions"] if s["session_id"] != sid]
                            nxt = remaining[0]["session_id"] if remaining else None
                            st.session_state["qa_active_session"] = nxt
                            if nxt:
                                _load_history(nxt)
                            else:
                                st.session_state["qa_messages"] = []
                        else:
                            _refresh_sessions()
                        st.rerun()
                    if st.button("❌取消", key=f"ccl_{sid}", help="取消删除"):
                        st.session_state[f"qa_confirm_del_{sid}"] = False
                        st.rerun()
                else:
                    if st.button("🗑", key=f"dl_{sid}", help="删除此会话"):
                        st.session_state[f"qa_confirm_del_{sid}"] = True
                        st.rerun()

            st.caption(f"  {msg_count}条 · {time_str}")

        st.divider()

        # 新建
        if st.button("＋ 新建窗口", use_container_width=True):
            result = _sync_call(qa_create_session("新对话"))
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


