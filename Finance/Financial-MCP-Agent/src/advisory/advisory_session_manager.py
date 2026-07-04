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
