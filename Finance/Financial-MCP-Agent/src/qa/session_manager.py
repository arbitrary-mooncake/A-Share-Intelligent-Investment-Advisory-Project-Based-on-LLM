"""
会话管理器：多轮对话会话的创建、更新、查询、过期清理。
纯内存存储（Phase 1），线程安全。
"""
import time
import uuid
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class QAMessage:
    """单条问答消息"""
    role: str          # "user" | "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class QASession:
    """问答会话"""
    session_id: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    history: List[QAMessage] = field(default_factory=list)
    # 上下文追踪
    last_stock_code: Optional[str] = None
    last_company_name: Optional[str] = None
    last_complexity_level: str = "L1"
    # 历史压缩摘要（Phase 3 使用）
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "history": [{"role": m.role, "content": m.content, "timestamp": m.timestamp}
                        for m in self.history],
            "last_stock_code": self.last_stock_code,
            "last_company_name": self.last_company_name,
        }

    def add_message(self, role: str, content: str):
        self.history.append(QAMessage(role=role, content=content))
        self.updated_at = time.time()

    def get_history_for_llm(self, max_turns: int = 10) -> List[dict]:
        """返回最近 N 轮对话，格式适配 LLM messages"""
        recent = self.history[-max_turns * 2:]  # 每轮=user+assistant
        return [{"role": m.role, "content": m.content} for m in recent]


class SessionManager:
    """会话管理器（线程安全的内存存储）"""

    def __init__(self, session_ttl: int = 3600):
        self._sessions: Dict[str, QASession] = {}
        self._lock = threading.Lock()
        self._session_ttl = session_ttl  # 会话过期时间（秒），默认1小时

    def create_session(self) -> str:
        """创建新会话，返回 session_id"""
        session_id = str(uuid.uuid4())[:8]
        with self._lock:
            self._sessions[session_id] = QASession(session_id=session_id)
        return session_id

    def get_session(self, session_id: str) -> Optional[QASession]:
        """获取会话，不存在时返回 None"""
        self._cleanup_expired()
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess:
                sess.updated_at = time.time()
            return sess

    def get_or_create(self, session_id: Optional[str]) -> QASession:
        """获取或创建会话"""
        if session_id:
            sess = self.get_session(session_id)
            if sess:
                return sess
        new_id = self.create_session()
        return self._sessions[new_id]

    def delete_session(self, session_id: str) -> bool:
        """删除会话，返回是否成功"""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
        return False

    def update_context(self, session_id: str, **kwargs):
        """更新会话上下文（last_stock_code, last_company_name 等）"""
        sess = self.get_session(session_id)
        if sess:
            for key, value in kwargs.items():
                if hasattr(sess, key):
                    setattr(sess, key, value)

    def _cleanup_expired(self):
        """清理过期会话"""
        now = time.time()
        with self._lock:
            expired = [
                sid for sid, sess in self._sessions.items()
                if now - sess.updated_at > self._session_ttl
            ]
            for sid in expired:
                del self._sessions[sid]

    @property
    def active_count(self) -> int:
        self._cleanup_expired()
        return len(self._sessions)


# 全局单例
_global_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    global _global_session_manager
    if _global_session_manager is None:
        _global_session_manager = SessionManager()
    return _global_session_manager
