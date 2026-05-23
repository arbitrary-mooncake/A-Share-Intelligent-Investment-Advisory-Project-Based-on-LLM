"""
AI 智能问答 — ChatGPT 风格：左侧会话列表 + 底部输入框 + 自动命名
"""
import os
import sys
from datetime import datetime

_app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

import streamlit as st

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
# 辅助函数（全部同步 HTTP，零 async 开销）
# ──────────────────────────────────────────────

import requests as _requests
_API = "http://127.0.0.1:8000"

def _api_get(path):
    try:
        return _requests.get(f"{_API}{path}", timeout=5).json()
    except Exception:
        return None

def _api_post(path, json_data):
    try:
        return _requests.post(f"{_API}{path}", json=json_data, timeout=5).json()
    except Exception:
        return None

def _api_delete(path):
    try:
        return _requests.delete(f"{_API}{path}", timeout=5).json()
    except Exception:
        return None

def _api_patch(path, json_data):
    try:
        return _requests.patch(f"{_API}{path}", json=json_data, timeout=5).json()
    except Exception:
        return None


def _refresh_sessions():
    data = _api_get("/api/qa/sessions")
    if data is not None:
        st.session_state["qa_sessions"] = data
    elif "qa_sessions" not in st.session_state:
        st.session_state["qa_sessions"] = []


def _load_history(session_id: str):
    if not session_id:
        st.session_state["qa_messages"] = []
        return
    data = _api_get(f"/api/qa/sessions/{session_id}")
    if data:
        st.session_state["qa_messages"] = [
            {"role": m["role"], "content": m["content"]}
            for m in data.get("history", [])
        ]
    elif "qa_messages" not in st.session_state:
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


def _complexity_label(level: str) -> str:
    labels = {"L0": "L0 快速应答", "L1": "L1 快问快答",
              "L2": "L2 标准分析", "L3": "L3 深度分析", "L4": "L4 专业研报"}
    return labels.get(level, f"{level} 分析")


# ──────────────────────────────────────────────
# 渲染函数（必须在主流程之前定义）
# ──────────────────────────────────────────────

def _render_messages():
    for msg in st.session_state.get("qa_messages", []):
        with st.chat_message(msg["role"]):
            # AI消息显示复杂度标签
            if msg["role"] == "assistant" and msg.get("meta"):
                st.caption(msg["meta"])
            st.markdown(msg["content"])


def _render_chat_input(session_id, session_name):
    """Phase 1: 仅提交用户消息，设置 pending 标记后立刻 rerun"""
    if not session_id:
        st.chat_input("请先在左侧选择或创建会话窗口", disabled=True)
        return

    prompt = st.chat_input("输入您的投资分析问题...", key="main_input")
    if not prompt or not prompt.strip():
        return

    prompt = prompt.strip()

    # 自动命名
    msg_count = len(st.session_state.get("qa_messages", []))
    if msg_count == 0 and session_name in ("新对话", "对话"):
        auto_name = _auto_name(prompt)
        _api_patch(f"/api/qa/sessions/{session_id}", {"name": auto_name})
        _refresh_sessions()

    # 立即写入用户消息 + 设置 pending 标记
    st.session_state["qa_messages"].append({"role": "user", "content": prompt})
    st.session_state["_qa_process"] = {"prompt": prompt, "session_id": session_id}
    st.rerun()


def _process_pending():
    """Phase 2: 有 pending 消息时，流式获取 AI 回复并实时更新"""
    pending = st.session_state.get("_qa_process")
    if not pending:
        return
    # 重试保护：失败超过3次放弃
    retry = pending.get("_retry", 0)
    if retry > 2:
        st.session_state["_qa_process"] = None
        st.error("多次尝试获取回答失败，请检查后端服务后重试")
        return
    pending["_retry"] = retry + 1
    st.session_state["_qa_process"] = pending

    prompt = pending["prompt"]
    session_id = pending["session_id"]
    full_answer = ""
    meta_info = ""

    # AI 对话框立刻出现，显示"思考中"
    with st.chat_message("assistant"):
        status_el = st.empty()
        answer_el = st.empty()
        status_el.caption("⏳ 思考中...")

    # 同步 SSE 流式读取（避免 asyncio 阻塞问题）
    import requests
    from src.app.config import API_BASE_URL

    try:
        resp = requests.post(
            f"{API_BASE_URL}/api/qa/ask",
            json={"question": prompt, "session_id": session_id},
            stream=True,
            timeout=180,
        )
        resp.raise_for_status()

        current_event = None
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("event: "):
                current_event = line[7:].strip()
            elif line.startswith("data: "):
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                if data_str.startswith("[ERROR]"):
                    status_el.empty()
                    answer_el.error(data_str[8:])
                    full_answer = f"(_请求失败: {data_str[8:]}_)"
                    break

                if current_event == "meta":
                    try:
                        import json
                        edata = json.loads(data_str)
                    except Exception:
                        edata = {}
                    level = edata.get("complexity", "")
                    stock = edata.get("company_name", "")
                    level_label = _complexity_label(level)
                    meta_info = f"{level_label}"
                    if stock:
                        meta_info += f" · {stock}"

                elif current_event == "status":
                    status_el.caption(f"⏳ {data_str}")

                elif current_event == "answer_start":
                    # 元数据事件，不显示，重置事件类型让后续文本正常流入
                    current_event = None

                elif current_event in (None, ""):
                    # 普通文本块
                    status_el.empty()
                    full_answer += data_str
                    answer_el.markdown(full_answer + "▌")

                elif current_event == "clarify":
                    status_el.empty()
                    try:
                        edata = json.loads(data_str)
                        msg = edata.get("message", data_str)
                    except Exception:
                        msg = data_str
                    answer_el.warning(msg)
                    full_answer = f"(_需澄清: {msg}_)"

    except requests.exceptions.Timeout:
        full_answer = "请求超时，请稍后重试。"
        status_el.empty()
        answer_el.error(full_answer)
    except requests.exceptions.ConnectionError:
        full_answer = "无法连接后端服务，请确认服务已启动。"
        status_el.empty()
        answer_el.error(full_answer)
    except requests.HTTPError as e:
        full_answer = f"后端返回错误（{e.response.status_code}），请稍后重试。"
        status_el.empty()
        answer_el.error(full_answer)
    except Exception as e:
        full_answer = f"请求异常: {e}"
        status_el.empty()
        answer_el.error(full_answer)

    status_el.empty()
    answer_el.markdown(full_answer)

    # 成功回答才保存到历史并清除 pending；失败保留 pending 供重试
    if full_answer and not full_answer.startswith("请求"):
        st.session_state["qa_messages"].append({
            "role": "assistant", "content": full_answer, "meta": meta_info,
        })
        st.session_state["_qa_process"] = None
        _refresh_sessions()
    # 失败不清理 _qa_process，下次渲染重试（_retry 计数器控制上限）
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

# 页面加载/切换时，从后端同步历史（无 pending 时才加载，避免覆盖流式中的消息）
if active_id and not st.session_state.get("_qa_process"):
    _load_history(active_id)

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
    _process_pending()
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
                            _api_patch(f"/api/qa/sessions/{sid}", {"name": nn})
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
                        _api_delete(f"/api/qa/sessions/{sid}")
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
            result = _api_post("/api/qa/sessions", {"name": "新对话"})
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
        _process_pending()
        _render_chat_input(active_id, current_name)


