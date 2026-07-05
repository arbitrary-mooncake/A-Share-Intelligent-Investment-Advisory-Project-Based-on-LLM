# 智能投顾多轮对话 & 历史会话 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add multi-turn conversation with persistent session history to the 智能投顾 page, including a collapsible history sidebar and global chat panel.

**Architecture:** New `AdvisorySessionManager` (modeled after `qa/session_manager.py`) provides session CRUD + auto-save to `data/advisory_sessions/sessions.json`. The existing `/api/advisory/chat` endpoint is enhanced to load session history, inject it into the system prompt, and auto-save messages after each exchange. Frontend replaces the `_render_ai_panel` right column with a persistent chat panel + collapsible history sidebar in a 3-column layout.

**Tech Stack:** Python (FastAPI, Streamlit), JSON file persistence, OpenAI-compatible LLM API

## Global Constraints

- All session data persisted to `data/advisory_sessions/sessions.json`
- Thread-safe operations (threading.Lock)
- Max 6 turns (12 messages) injected into LLM context; older messages compressed to summary
- Each message auto-saved to disk immediately after LLM response
- Empty sessions (0 messages) discarded on clear/switch, not saved to history
- Delete requires confirmation dialog (`st.dialog`)
- Tab switch auto-clears current chat and archives session to history
- Do NOT modify any existing advisory endpoints other than `/api/advisory/chat`
- Do NOT change QA session_manager.py — advisory sessions are physically isolated

---

### Task 1: Create AdvisorySessionManager Backend

**Files:**
- Create: `Finance/Financial-MCP-Agent/src/advisory/advisory_session_manager.py`
- Create: `Finance/Financial-MCP-Agent/tests/test_advisory_session.py`

**Interfaces:**
- Produces: `AdvisorySessionManager` class with methods:
  - `create_session(name="新对话") -> str` (returns session_id)
  - `get_session(session_id) -> Optional[AdvisorySession]`
  - `list_sessions() -> List[dict]` (sorted by updated_at desc)
  - `delete_session(session_id) -> bool`
  - `rename_session(session_id, new_name) -> bool`
  - `add_message(session_id, role, content, page_context="") -> None` (auto-saves)
  - `get_history_for_llm(session_id, max_turns=6) -> str` (formatted text block)
  - `get_history_messages(session_id) -> List[dict]` (full history for frontend)
- Produces: `get_advisory_session_manager() -> AdvisorySessionManager` (global singleton)

- [ ] **Step 1: Write the test file**

```python
# tests/test_advisory_session.py
"""Tests for AdvisorySessionManager."""
import os
import sys
import json
import time
import shutil
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_TEST_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "test_advisory_sessions")


@pytest.fixture(autouse=True)
def clean_test_dir():
    """Ensure clean test directory before/after each test."""
    if os.path.exists(_TEST_DIR):
        shutil.rmtree(_TEST_DIR)
    yield
    if os.path.exists(_TEST_DIR):
        shutil.rmtree(_TEST_DIR)


@pytest.fixture
def manager():
    """Create a session manager with test directory."""
    from src.advisory.advisory_session_manager import AdvisorySessionManager
    return AdvisorySessionManager(data_dir=_TEST_DIR)


def test_create_session(manager):
    sid = manager.create_session("test session")
    assert isinstance(sid, str)
    assert len(sid) == 8
    sess = manager.get_session(sid)
    assert sess is not None
    assert sess.name == "test session"
    assert sess.history == []


def test_add_message_auto_saves(manager):
    sid = manager.create_session()
    manager.add_message(sid, "user", "hello")
    manager.add_message(sid, "assistant", "hi there")
    sess = manager.get_session(sid)
    assert len(sess.history) == 2
    assert sess.history[0].role == "user"
    assert sess.history[0].content == "hello"
    assert sess.history[1].role == "assistant"
    # Verify persistence — reload from disk
    from src.advisory.advisory_session_manager import AdvisorySessionManager
    mgr2 = AdvisorySessionManager(data_dir=_TEST_DIR)
    sess2 = mgr2.get_session(sid)
    assert len(sess2.history) == 2


def test_list_sessions_sorted(manager):
    sid1 = manager.create_session("first")
    time.sleep(0.01)
    sid2 = manager.create_session("second")
    manager.add_message(sid1, "user", "msg")  # updates sid1's updated_at
    items = manager.list_sessions()
    assert len(items) == 2
    # sid1 was updated last, should be first
    assert items[0]["session_id"] == sid1


def test_delete_session(manager):
    sid = manager.create_session("to delete")
    assert manager.delete_session(sid) is True
    assert manager.get_session(sid) is None
    assert manager.delete_session("nonexistent") is False


def test_rename_session(manager):
    sid = manager.create_session("old name")
    assert manager.rename_session(sid, "new name") is True
    sess = manager.get_session(sid)
    assert sess.name == "new name"
    assert manager.rename_session(sid, "  ") is False


def test_auto_naming(manager):
    sid = manager.create_session()
    sess = manager.get_session(sid)
    assert sess.name == "新对话"
    manager.add_message(sid, "user", "帮我分析一下贵州茅台的基本面情况")
    sess = manager.get_session(sid)
    assert sess.name == "帮我分析一下贵州茅台的基本面情况"[:20]


def test_get_history_for_llm(manager):
    sid = manager.create_session()
    for i in range(8):
        manager.add_message(sid, "user", f"question {i}")
        manager.add_message(sid, "assistant", f"answer {i}")
    history_text = manager.get_history_for_llm(sid, max_turns=6)
    assert "## 历史对话上下文" in history_text
    # Should contain recent turns but not the oldest
    assert "question 7" in history_text
    assert "answer 7" in history_text


def test_get_history_messages(manager):
    sid = manager.create_session()
    manager.add_message(sid, "user", "hi", page_context="AI顾问对话")
    manager.add_message(sid, "assistant", "hello", page_context="AI顾问对话")
    msgs = manager.get_history_messages(sid)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[0]["page_context"] == "AI顾问对话"


def test_empty_message_no_auto_name(manager):
    sid = manager.create_session()
    manager.add_message(sid, "assistant", "welcome msg")
    sess = manager.get_session(sid)
    # Should NOT auto-name from assistant messages
    assert sess.name == "新对话"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd Finance/Financial-MCP-Agent && python -m pytest tests/test_advisory_session.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.advisory.advisory_session_manager'`

- [ ] **Step 3: Implement AdvisorySessionManager**

Create `Finance/Financial-MCP-Agent/src/advisory/advisory_session_manager.py`:

```python
"""
智能投顾会话管理器 — 多轮对话的持久化存储与会话历史管理。

- 会话持久化到 data/advisory_sessions/sessions.json
- 每条消息实时写盘（自动保存）
- 线程安全
- 首条用户消息自动命名会话
- 历史压缩：超过 max_turns 轮的早期对话压缩为摘要
"""
import json
import os
import time
import uuid
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


@dataclass
class AdvisoryMessage:
    role: str          # "user" | "assistant"
    content: str
    page_context: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "page_context": self.page_context,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AdvisoryMessage":
        return cls(
            role=d["role"],
            content=d["content"],
            page_context=d.get("page_context", ""),
            timestamp=d.get("timestamp", time.time()),
        )


@dataclass
class AdvisorySession:
    session_id: str
    name: str = "新对话"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    history: List[AdvisoryMessage] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "history": [m.to_dict() for m in self.history],
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AdvisorySession":
        sess = cls(
            session_id=d["session_id"],
            name=d.get("name", "新对话"),
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
            summary=d.get("summary", ""),
        )
        sess.history = [AdvisoryMessage.from_dict(m) for m in d.get("history", [])]
        return sess

    def to_list_item(self) -> dict:
        msg_count = len(self.history)
        last_msg = ""
        if self.history:
            last_msg = self.history[-1].content[:50]
        return {
            "session_id": self.session_id,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": msg_count,
            "last_message": last_msg,
        }


class AdvisorySessionManager:
    """智能投顾会话管理器 — 磁盘持久化 + 线程安全 + 自动保存"""

    def __init__(self, data_dir: str = None):
        self._lock = threading.Lock()
        if data_dir is None:
            self._data_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "data", "advisory_sessions"
            )
        else:
            self._data_dir = data_dir
        self._sessions_file = os.path.join(self._data_dir, "sessions.json")
        self._sessions: Dict[str, AdvisorySession] = {}
        os.makedirs(self._data_dir, exist_ok=True)
        self._load()

    def _load(self):
        if os.path.exists(self._sessions_file):
            try:
                with open(self._sessions_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for item in data:
                    sess = AdvisorySession.from_dict(item)
                    self._sessions[sess.session_id] = sess
                logger.info(f"已加载 {len(self._sessions)} 个投顾会话")
            except Exception as e:
                logger.warning(f"加载投顾会话失败: {e}")

    def _save(self):
        os.makedirs(self._data_dir, exist_ok=True)
        try:
            data = [sess.to_dict() for sess in self._sessions.values()]
            with open(self._sessions_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存投顾会话失败: {e}")

    def create_session(self, name: str = "新对话") -> str:
        session_id = str(uuid.uuid4())[:8]
        with self._lock:
            self._sessions[session_id] = AdvisorySession(
                session_id=session_id, name=name
            )
        return session_id

    def get_session(self, session_id: str) -> Optional[AdvisorySession]:
        with self._lock:
            return self._sessions.get(session_id)

    def list_sessions(self) -> List[dict]:
        with self._lock:
            items = [s.to_list_item() for s in self._sessions.values()]
        items.sort(key=lambda x: x["updated_at"], reverse=True)
        return items

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            if session_id not in self._sessions:
                return False
            del self._sessions[session_id]
            self._save()
        return True

    def rename_session(self, session_id: str, new_name: str) -> bool:
        new_name = new_name.strip()
        if not new_name:
            return False
        with self._lock:
            sess = self._sessions.get(session_id)
            if not sess:
                return False
            sess.name = new_name
            sess.updated_at = time.time()
            self._save()
        return True

    def add_message(self, session_id: str, role: str, content: str, page_context: str = ""):
        with self._lock:
            sess = self._sessions.get(session_id)
            if not sess:
                return
            sess.history.append(AdvisoryMessage(
                role=role, content=content, page_context=page_context,
            ))
            sess.updated_at = time.time()
            # Auto-name from first user message
            if role == "user" and sess.name == "新对话":
                sess.name = content[:20]
            self._save()

    def get_history_for_llm(self, session_id: str, max_turns: int = 6) -> str:
        with self._lock:
            sess = self._sessions.get(session_id)
            if not sess or not sess.history:
                return ""
            all_msgs = sess.history
            cutoff = max_turns * 2
            if len(all_msgs) <= cutoff:
                recent = all_msgs
                summary_text = ""
            else:
                recent = all_msgs[-cutoff:]
                old = all_msgs[:-cutoff]
                summary_text = _compress_history(old)

            parts = ["## 历史对话上下文"]
            if summary_text:
                parts.append(f"[早期对话摘要] {summary_text}")
            for m in recent:
                label = "用户" if m.role == "user" else "AI顾问"
                truncated = m.content[:300]
                parts.append(f"{label}: {truncated}")
            return "\n".join(parts)

    def get_history_messages(self, session_id: str) -> List[dict]:
        with self._lock:
            sess = self._sessions.get(session_id)
            if not sess:
                return []
            return [m.to_dict() for m in sess.history]


def _compress_history(old_messages: list) -> str:
    import re
    stocks = set()
    questions = []
    for msg in old_messages:
        content = msg.content[:300]
        codes = re.findall(r'(?<!\d)\d{5,6}(?!\d)', content)
        stocks.update(codes)
        if msg.role == "user":
            q = content[:80].strip()
            if q:
                questions.append(q)
    parts = []
    if stocks:
        parts.append(f"涉及股票: {', '.join(sorted(stocks)[:5])}")
    if questions:
        parts.append(f"主要问题: {'; '.join(questions[-5:])}")
    parts.append(f"共{len(old_messages)}条历史消息")
    return "。".join(parts)


_global_advisory_session_manager: Optional[AdvisorySessionManager] = None


def get_advisory_session_manager() -> AdvisorySessionManager:
    global _global_advisory_session_manager
    if _global_advisory_session_manager is None:
        _global_advisory_session_manager = AdvisorySessionManager()
    return _global_advisory_session_manager
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd Finance/Financial-MCP-Agent && python -m pytest tests/test_advisory_session.py -v
```

Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/advisory/advisory_session_manager.py tests/test_advisory_session.py
git commit -m "feat(advisory): add AdvisorySessionManager for multi-turn conversation persistence"
```

---

### Task 2: Enhance Backend API Endpoints

**Files:**
- Modify: `Finance/Financial-MCP-Agent/src/api/app.py:157-160` (ChatRequest model)
- Modify: `Finance/Financial-MCP-Agent/src/api/app.py:4555-4581` (_build_advisory_system_prompt)
- Modify: `Finance/Financial-MCP-Agent/src/api/app.py:4597-4762` (advisory_chat endpoint)
- Add: New endpoints after line 4762

**Interfaces:**
- Consumes: `AdvisorySessionManager` from Task 1
- Produces: Enhanced `/api/advisory/chat` with multi-turn + auto-save
- Produces: `GET /api/advisory/sessions` — list all sessions
- Produces: `DELETE /api/advisory/sessions/{session_id}` — delete a session

- [ ] **Step 1: Add session management import in app.py**

At the top of `app.py` (near the existing imports around line 55), add:

```python
from src.advisory.advisory_session_manager import get_advisory_session_manager
```

- [ ] **Step 2: Enhance _build_advisory_system_prompt to accept history**

Replace the function at line 4555-4581 with:

```python
def _build_advisory_system_prompt(profile_summary: str, page_name: str = "", page_data: Any = None, history_text: str = "") -> str:
    """构建智能投顾 AI 顾问的 System prompt。"""
    system_parts = [
        "你是一位专业的A股智能投资顾问助手。",
        "你的职责是为用户提供股票分析、持仓管理、策略建议、市场分析等投顾服务。",
        "",
        "## 用户投资画像",
        profile_summary,
        "",
        "## 行为规范",
        "- 使用 [数据] 标签标注从工具/数据中获取的确定信息",
        "- 使用 [判断] 标签标注你的分析推理",
        "- 不确定时坦诚说明，不要编造数据",
        "- 推荐结果必须附免责声明：'以上仅为 AI 分析参考，不构成投资建议。市场有风险，投资需谨慎。'",
        "- 回答简洁专业，用中文回复",
        "",
        "## 用户画像工具",
        "当用户明确陈述偏好或你可以从对话中推断偏好时，调用 update_user_profile 工具更新画像。",
        "当需要确认当前画像时，调用 get_user_profile 工具查询。",
        "例如：用户说'我是保守型投资者' → 调用 update_user_profile(risk_tolerance='Conservative')。",
    ]
    if history_text:
        system_parts.append("")
        system_parts.append(history_text)
    if page_name:
        system_parts.append(f"\n## 当前页面上下文\n用户正在查看「{page_name}」页面。")
        if page_data:
            system_parts.append(f"页面数据: {json.dumps(page_data, ensure_ascii=False)}")
            system_parts.append("你可以就用户当前页面看到的数据进行分析和解答。")
    return "\n".join(system_parts)
```

- [ ] **Step 3: Enhance advisory_chat endpoint for multi-turn + auto-save**

Replace the `advisory_chat` function (line 4597-4762) with:

```python
@app.post("/api/advisory/chat")
async def advisory_chat(request: ChatRequest):
    """AI 顾问对话 — 多轮上下文 + 自动保存 + 用户画像工具。"""
    try:
        from src.advisory.profile_tools import PROFILE_TOOLS_SCHEMA

        modules = _init_advisory()
        upm = modules["user_profile_manager"]
        profile_summary = upm.get_profile_summary()
        session_mgr = get_advisory_session_manager()

        # 解析页面上下文
        page_name = ""
        page_data = None
        if request.page_context:
            try:
                ctx = json.loads(request.page_context)
                page_name = ctx.get("page", "")
                page_data = ctx.get("data", {})
            except (json.JSONDecodeError, TypeError):
                pass

        # 会话管理：确保 session 存在
        session_id = request.session_id
        if not session_id or session_id == "default":
            session_id = session_mgr.create_session()
        sess = session_mgr.get_session(session_id)
        if sess is None:
            session_id = session_mgr.create_session()

        # 获取历史上下文
        history_text = session_mgr.get_history_for_llm(session_id, max_turns=6)

        system_prompt = _build_advisory_system_prompt(
            profile_summary, page_name, page_data, history_text
        )

        # 保存用户消息
        session_mgr.add_message(session_id, "user", request.message, page_context=page_name)

        # 调用 M1 模型
        base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL", "")
        api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY", "")
        model = os.getenv("OPENAI_COMPATIBLE_MODEL", "mimo-v2.5-pro")

        try:
            from openai import OpenAI as _OpenAISdk
            import httpx as _httpx

            raw_client = _OpenAISdk(
                base_url=base_url,
                api_key=api_key,
                timeout=_httpx.Timeout(120, connect=30),
            )
            thinking_body = get_thinking_body(base_url, True)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.message},
            ]
            tool_schemas = [t["function"] for t in PROFILE_TOOLS_SCHEMA]
            tools_for_api = [
                {"type": "function", "function": f} for f in tool_schemas
            ]

            reply = ""
            tool_calls_log: List[Dict[str, Any]] = []
            max_rounds = 3
            tools_enabled = True

            for round_idx in range(max_rounds):
                api_kwargs = dict(model=model, messages=messages)
                if tools_enabled:
                    api_kwargs["tools"] = tools_for_api
                if thinking_body:
                    api_kwargs["extra_body"] = thinking_body

                try:
                    resp = await asyncio.wait_for(
                        asyncio.get_running_loop().run_in_executor(
                            _thread_pool,
                            lambda: raw_client.chat.completions.create(**api_kwargs),
                        ),
                        timeout=120.0,
                    )
                except Exception as call_err:
                    if tools_enabled and round_idx == 0:
                        logger.warning(f"tools 参数调用失败，降级为无 tools 模式: {call_err}")
                        tools_enabled = False
                        continue
                    raise

                choice = resp.choices[0] if resp.choices else None
                if choice is None:
                    reply = "抱歉，模型返回空结果。"
                    break

                msg = choice.message

                if tools_enabled and getattr(msg, "tool_calls", None) and len(msg.tool_calls) > 0:
                    messages.append({
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": [
                            {"id": tc.id, "type": "function",
                             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                            for tc in msg.tool_calls
                        ],
                    })
                    for tc in msg.tool_calls:
                        tool_name = tc.function.name
                        try:
                            parsed = json.loads(tc.function.arguments) if tc.function.arguments else {}
                        except (json.JSONDecodeError, TypeError):
                            parsed = {}
                        if not isinstance(parsed, dict):
                            parsed = {}
                        try:
                            tool_result = _execute_profile_tool(tool_name, parsed)
                        except Exception as te:
                            tool_result = f"工具执行失败: {te}"
                        tool_calls_log.append({
                            "tool": tool_name, "arguments": parsed,
                            "result_preview": tool_result[:200],
                        })
                        messages.append({
                            "role": "tool", "tool_call_id": tc.id, "content": tool_result,
                        })
                    if any(tc.function.name == "update_user_profile" for tc in msg.tool_calls):
                        profile_summary = upm.get_profile_summary()
                    continue

                reply = msg.content or "抱歉，我暂时无法生成回复，请稍后重试。"
                break
            else:
                reply = reply or "抱歉，工具调用轮次超限，请简化问题后重试。"

        except asyncio.TimeoutError:
            reply = "抱歉，回复生成超时（120s），请简化问题后重试。"
        except Exception as e:
            logger.warning(f"AI 对话模型调用失败: {e}")
            reply = f"抱歉，模型服务暂时不可用。（{str(e)[:100]}）"

        # 保存 AI 回复
        session_mgr.add_message(session_id, "assistant", reply, page_context=page_name)

        return {
            "status": "ok",
            "reply": reply,
            "session_id": session_id,
            "page_context": request.page_context,
            "model": model,
            "tool_calls": tool_calls_log,
        }
    except Exception as e:
        logger.error(f"{ERROR_ICON} AI 对话失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"AI 对话失败: {str(e)}")
```

- [ ] **Step 4: Add session management endpoints**

Add after the `advisory_chat` function (after line 4762):

```python
@app.get("/api/advisory/sessions")
async def list_advisory_sessions():
    """列出所有投顾会话（摘要，按更新时间倒序）"""
    try:
        session_mgr = get_advisory_session_manager()
        sessions = session_mgr.list_sessions()
        return {"status": "ok", "sessions": sessions}
    except Exception as e:
        logger.error(f"列出投顾会话失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"列出会话失败: {str(e)}")


@app.get("/api/advisory/sessions/{session_id}")
async def get_advisory_session(session_id: str):
    """获取指定会话的完整历史"""
    try:
        session_mgr = get_advisory_session_manager()
        sess = session_mgr.get_session(session_id)
        if sess is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        return {
            "status": "ok",
            "session_id": session_id,
            "name": sess.name,
            "history": [m.to_dict() for m in sess.history],
            "created_at": sess.created_at,
            "updated_at": sess.updated_at,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取投顾会话失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取会话失败: {str(e)}")


@app.delete("/api/advisory/sessions/{session_id}")
async def delete_advisory_session(session_id: str):
    """删除指定会话"""
    try:
        session_mgr = get_advisory_session_manager()
        success = session_mgr.delete_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="会话不存在")
        return {"status": "ok", "deleted": session_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除投顾会话失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"删除会话失败: {str(e)}")
```

- [ ] **Step 5: Verify API changes don't break existing functionality**

```bash
cd Finance/Financial-MCP-Agent && python -c "from src.api.app import app; print('app.py imports OK')"
```

Expected: `app.py imports OK`

- [ ] **Step 6: Commit**

```bash
git add src/api/app.py
git commit -m "feat(advisory): enhance chat API with multi-turn history + session management endpoints"
```

---

### Task 3: Frontend — 3-Column Layout + Chat Panel + History Sidebar

**Files:**
- Modify: `Finance/Financial-MCP-Agent/src/app/pages/07_智能投顾.py` (full rewrite of layout and chat/session sections)

**Interfaces:**
- Consumes: `GET /api/advisory/sessions`, `GET /api/advisory/sessions/{id}`, `DELETE /api/advisory/sessions/{id}`, `POST /api/advisory/chat` from Task 2
- Produces: 3-column layout: `[Tab Content(5)] [Chat Panel(3)] [History Sidebar(2, collapsible)]`

- [ ] **Step 1: Update _api helper to support DELETE method**

In the `_api` function (lines 28-43), add DELETE support. Replace:
```python
        if method == "GET":
            resp = requests.get(url, timeout=30)
        else:
            resp = requests.post(url, json=body, timeout=120)
```
With:
```python
        if method == "GET":
            resp = requests.get(url, timeout=30)
        elif method == "DELETE":
            resp = requests.delete(url, timeout=30)
        else:
            resp = requests.post(url, json=body, timeout=120)
```

- [ ] **Step 2: Add new session state variables**

After line 77 (after `advisory_auto_catchup_done`), add:

```python
if "advisory_current_session_id" not in st.session_state:
    st.session_state["advisory_current_session_id"] = None
if "advisory_history_expanded" not in st.session_state:
    st.session_state["advisory_history_expanded"] = False
if "advisory_delete_confirm" not in st.session_state:
    st.session_state["advisory_delete_confirm"] = None
if "advisory_session_list" not in st.session_state:
    st.session_state["advisory_session_list"] = []
if "advisory_prev_page" not in st.session_state:
    st.session_state["advisory_prev_page"] = None
```

- [ ] **Step 3: Add helper function to clear current chat**

After the state initialization block (before PAGE_CONFIG), add:

```python
def _clear_chat_for_new_session():
    """清空当前聊天，准备新会话。旧会话已在后端保存。"""
    st.session_state["advisory_chat"] = []
    st.session_state["advisory_chat_loading"] = False
    st.session_state["advisory_current_session_id"] = None


def _load_session_into_chat(session_id: str):
    """从后端加载历史会话到聊天面板"""
    result = _api(f"/api/advisory/sessions/{session_id}")
    if "error" not in result and result.get("status") == "ok":
        history = result.get("history", [])
        st.session_state["advisory_chat"] = [
            {"role": m["role"], "content": m["content"]} for m in history
        ]
        st.session_state["advisory_current_session_id"] = session_id


def _refresh_session_list():
    """刷新历史会话列表"""
    result = _api("/api/advisory/sessions")
    if "error" not in result:
        st.session_state["advisory_session_list"] = result.get("sessions", [])


def _handle_tab_switch():
    """检测 Tab 切换，自动清空聊天并归档旧会话"""
    current_page = st.session_state["advisory_page"]
    prev_page = st.session_state.get("advisory_prev_page")
    if prev_page is not None and prev_page != current_page:
        # Tab 发生了变化 — 清空聊天
        if st.session_state["advisory_chat"]:
            # 有消息的会话已在后端保存，只需前端清空
            pass
        _clear_chat_for_new_session()
    st.session_state["advisory_prev_page"] = current_page
```

- [ ] **Step 4: Replace _render_ai_chat function**

Replace the `_render_ai_chat` function (lines 94-138) with a stub that shows tab-specific content only, since the chat is now in the middle panel:

```python
def _render_ai_chat():
    """AI顾问对话 — Tab 内容区（聊天已移到中间面板）"""
    st.markdown("### 💬 AI顾问对话")
    st.caption("与智能投顾直接对话，获取投资建议与市场分析")

    if not st.session_state["advisory_chat"]:
        st.info("👋 您好！我是您的智能投顾助手，由 M1 MiMo-V2.5-Pro 模型驱动，可以为您提供:\n\n"
                "- 📊 **个股分析** — 输入股票代码或名称\n"
                "- 📈 **市场热点** — 当前板块轮动与资金流向\n"
                "- 💰 **持仓诊断** — 组合风险与收益评估\n"
                "- 🎯 **策略建议** — 根据您的风险偏好推荐策略\n\n"
                "请在右侧聊天面板开始对话。")
```

- [ ] **Step 5: Replace _render_ai_panel with _render_chat_panel**

Replace the entire `_render_ai_panel` function (lines 804-890) with:

```python
def _render_chat_panel():
    """中间常驻 AI 聊天面板 — 多轮对话 + 上下文感知快捷提问"""
    current_page_key = st.session_state["advisory_page"]
    page = PAGE_CONFIG.get(current_page_key, PAGE_CONFIG["ai_chat"])

    # 标题栏
    col_title, col_new = st.columns([4, 1])
    with col_title:
        st.markdown(
            f'<div style="text-align:center;padding:6px 0;">'
            f'<span style="font-size:1.1em;font-weight:800;color:#1e40af;">🤖 AI 顾问</span>'
            f'<span style="font-size:0.8em;color:#64748b;margin-left:8px;">'
            f'{page["icon"]} {page["label"]}</span></div>',
            unsafe_allow_html=True,
        )
    with col_new:
        if st.button("🔄", key="btn_new_chat", help="开始新对话（当前对话自动保存到历史）"):
            if st.session_state["advisory_chat"]:
                pass  # 后端已自动保存
            _clear_chat_for_new_session()
            st.rerun()

    # 上下文感知快捷提问
    quick_questions = {
        "ai_chat":    ["今天市场热点板块有哪些？", "帮我分析一下大盘走势"],
        "recommend":  ["推荐逻辑是什么？", "这些股票适合什么风险偏好？"],
        "portfolio":  ["如何优化我的持仓？", "当前组合风险等级如何？"],
        "strategies": ["哪种策略适合我？", "策略的历史表现如何？"],
        "backtest":   ["回测结果说明什么？", "如何改进策略参数？"],
        "report":     ["报告包含哪些内容？", "如何解读收益报告？"],
    }
    qqs = quick_questions.get(current_page_key, ["今天市场怎么样？", "有什么投资建议？"])
    qq_cols = st.columns(len(qqs))
    for i, q in enumerate(qqs):
        with qq_cols[i]:
            if st.button(f"💬 {q}", key=f"qq_{current_page_key}_{i}", use_container_width=True):
                st.session_state["advisory_chat"].append({"role": "user", "content": q})
                st.session_state["_pending_quick_msg"] = q
                st.rerun()

    st.divider()

    # 消息展示
    chat_container = st.container(height=500)
    with chat_container:
        for msg in st.session_state["advisory_chat"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # 底部输入
    prompt = st.chat_input("输入您的问题...", key="advisory_chat_main_input")
    if prompt and prompt.strip():
        st.session_state["advisory_chat"].append({"role": "user", "content": prompt.strip()})
        st.session_state["_pending_quick_msg"] = prompt.strip()
        st.rerun()

    # 处理待发送的消息（来自输入框或快捷按钮）
    pending = st.session_state.pop("_pending_quick_msg", None)
    if pending:
        with st.spinner("AI 顾问思考中..."):
            page_ctx = json.dumps({
                "page": PAGE_CONFIG[current_page_key]["label"],
                "data": {},
            }, ensure_ascii=False)
            session_id = st.session_state.get("advisory_current_session_id") or "default"
            result = _api("/api/advisory/chat", "POST", {
                "message": pending,
                "session_id": session_id,
                "page_context": page_ctx,
            })
            reply = result.get("reply", result.get("error", "抱歉，服务暂时不可用"))
            # 更新 session_id（后端可能在 session_id="default" 时创建了新 session）
            new_sid = result.get("session_id")
            if new_sid and new_sid != "default":
                st.session_state["advisory_current_session_id"] = new_sid
            st.session_state["advisory_chat"].append({"role": "assistant", "content": reply})
        st.rerun()

    # 空状态提示
    if not st.session_state["advisory_chat"]:
        st.caption("💡 点击上方快捷按钮或在输入框提问，开始与 AI 顾问对话")
```

- [ ] **Step 6: Add _render_history_sidebar function**

Add after `_render_chat_panel`:

```python
def _render_history_sidebar():
    """右侧可折叠历史会话侧栏"""
    expanded = st.session_state.get("advisory_history_expanded", False)

    if not expanded:
        if st.button("📋 历史", key="btn_expand_history", use_container_width=True):
            st.session_state["advisory_history_expanded"] = True
            _refresh_session_list()
            st.rerun()
        return

    # 展开状态
    col_toggle, col_title = st.columns([1, 4])
    with col_toggle:
        if st.button("◀", key="btn_collapse_history"):
            st.session_state["advisory_history_expanded"] = False
            st.rerun()
    with col_title:
        st.markdown("**📋 历史会话**")

    if st.button("＋ 新对话", key="btn_sidebar_new", use_container_width=True):
        if st.session_state["advisory_chat"]:
            pass  # 后端已自动保存
        _clear_chat_for_new_session()
        st.rerun()

    st.divider()

    # 加载/刷新列表
    _refresh_session_list()
    sessions = st.session_state.get("advisory_session_list", [])

    if not sessions:
        st.caption("暂无历史会话")
        return

    current_sid = st.session_state.get("advisory_current_session_id")

    for sess in sessions:
        sid = sess["session_id"]
        name = sess.get("name", "未命名")
        msg_count = sess.get("message_count", 0)
        is_current = (sid == current_sid)

        border = "2px solid #2563eb" if is_current else "1px solid #e2e8f0"
        bg = "#eff6ff" if is_current else "#ffffff"

        st.markdown(
            f'<div style="padding:6px 8px;border:{border};border-radius:6px;'
            f'background:{bg};margin-bottom:4px;">'
            f'<div style="font-weight:600;font-size:0.85em;overflow:hidden;'
            f'text-overflow:ellipsis;white-space:nowrap;">{name}</div>'
            f'<div style="font-size:0.75em;color:#64748b;">{msg_count}条消息</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        col_load, col_del = st.columns([3, 1])
        with col_load:
            btn_label = "📌 当前" if is_current else "💬 加载"
            if st.button(btn_label, key=f"load_{sid}", use_container_width=True, disabled=is_current):
                _load_session_into_chat(sid)
                st.rerun()
        with col_del:
            if st.button("🗑", key=f"del_{sid}"):
                st.session_state["advisory_delete_confirm"] = sid
                st.rerun()


@st.dialog("确认删除", width="small")
def _delete_confirm_dialog(session_id: str):
    """删除确认对话框"""
    st.warning("确定要删除这个历史会话吗？此操作不可撤销。")

    col_cancel, col_confirm = st.columns(2)
    with col_cancel:
        if st.button("取消", use_container_width=True):
            st.session_state["advisory_delete_confirm"] = None
            st.rerun()
    with col_confirm:
        if st.button("确认删除", type="primary", use_container_width=True):
            result = _api(f"/api/advisory/sessions/{session_id}", "DELETE")
            if "error" not in result:
                # 如果删的是当前会话，清空聊天
                if st.session_state.get("advisory_current_session_id") == session_id:
                    _clear_chat_for_new_session()
                st.session_state["advisory_delete_confirm"] = None
                _refresh_session_list()
                st.rerun()
            else:
                st.error(f"删除失败: {result['error']}")
```

- [ ] **Step 7: Rewrite the page layout section**

Replace the layout section (lines 926-941) with:

```python
# 检测 Tab 切换
_handle_tab_switch()

# 处理删除确认弹窗
if st.session_state.get("advisory_delete_confirm"):
    _delete_confirm_dialog(st.session_state["advisory_delete_confirm"])

# 三栏布局：左(5) + 中(3) + 右(2 或 0)
history_expanded = st.session_state.get("advisory_history_expanded", False)
if history_expanded:
    left_col, chat_col, history_col = st.columns([5, 3, 2])
else:
    left_col, chat_col = st.columns([5, 3])
    history_col = None

with left_col:
    {
        "ai_chat": _render_ai_chat,
        "recommend": _render_recommend,
        "portfolio": _render_portfolio,
        "strategies": _render_strategies,
        "backtest": _render_backtest,
        "report": _render_report,
    }[st.session_state["advisory_page"]]()

with chat_col:
    _render_chat_panel()

if history_col is not None:
    with history_col:
        _render_history_sidebar()
else:
    # 折叠状态下在聊天面板下方显示一个小按钮
    pass
```

- [ ] **Step 8: Verify the frontend loads without errors**

```bash
cd Finance/Financial-MCP-Agent && python -c "
import sys; sys.path.insert(0, 'src/app')
import importlib.util
spec = importlib.util.spec_from_file_location('advisory_page', 'src/app/pages/07_智能投顾.py')
mod = importlib.util.module_from_spec(spec)
# Just check syntax, don't execute
import py_compile
py_compile.compile('src/app/pages/07_智能投顾.py', doraise=True)
print('Syntax OK')
"
```

Expected: `Syntax OK`

- [ ] **Step 9: Commit**

```bash
git add src/app/pages/07_智能投顾.py
git commit -m "feat(advisory): 3-column layout with persistent chat panel + collapsible history sidebar"
```

---

### Task 4: Integration Testing & Bug Fixing

- [ ] **Step 1: Start the backend API server**

```bash
cd Finance/Financial-MCP-Agent && python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 &
```

- [ ] **Step 2: Start the Streamlit frontend**

```bash
cd Finance/Financial-MCP-Agent && python -m streamlit run src/app/Home.py --server.port 8501 &
```

- [ ] **Step 3: Test multi-turn conversation**

Open browser to http://localhost:8501, navigate to 智能投顾 page:
1. Send first message → verify AI responds and session is created
2. Send second message referencing the first → verify AI remembers context
3. Send third message → verify conversation flows naturally

- [ ] **Step 4: Test tab switching**

1. Have a conversation with 2+ messages
2. Click "股票推荐" tab → verify chat panel clears
3. Click "📋 历史" button → verify old conversation appears in history
4. Click "加载" on the old session → verify conversation is restored

- [ ] **Step 5: Test new conversation button**

1. Have a conversation
2. Click 🔄 button → verify chat clears
3. Start new conversation → verify it works independently
4. Check history → verify both conversations exist

- [ ] **Step 6: Test delete with confirmation**

1. Open history sidebar
2. Click 🗑 on a session → verify confirmation dialog appears
3. Click "取消" → verify session still exists
4. Click 🗑 again → click "确认删除" → verify session is removed

- [ ] **Step 7: Test quick question buttons**

1. Switch to each tab (推荐, 持仓, 策略, etc.)
2. Verify quick question buttons appear in chat panel
3. Click a quick question → verify AI responds with relevant answer

- [ ] **Step 8: Test persistence**

1. Have a conversation
2. Stop and restart the API server
3. Open history → verify conversation is still there
4. Load it → verify full history is restored

- [ ] **Step 9: Fix any bugs found during testing**

- [ ] **Step 10: Final commit**

```bash
git add -A
git commit -m "fix(advisory): integration fixes for multi-turn chat"
```
