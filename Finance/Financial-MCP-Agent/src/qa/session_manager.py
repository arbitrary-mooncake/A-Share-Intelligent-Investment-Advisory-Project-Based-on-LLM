"""
会话管理器：多轮对话会话窗口的持久化存储、CRUD、数据缓存。

- 会话窗口持久化到磁盘 JSON，可无限保留
- 每个窗口独立上下文和数据缓存，跨窗口不串通
- 数据缓存 TTL 7 天，过期自动清理
- 线程安全
"""
import json
import os
import time
import uuid
import hashlib
import threading
import glob as glob_module
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

# 持久化目录
_QA_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "qa_sessions"
)
_SESSIONS_FILE = os.path.join(_QA_DATA_DIR, "sessions.json")
_CACHE_TTL_SECONDS = 7 * 24 * 3600  # 7 天


def _ensure_dirs():
    os.makedirs(_QA_DATA_DIR, exist_ok=True)


@dataclass
class QAMessage:
    """单条问答消息"""
    role: str          # "user" | "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content, "timestamp": self.timestamp}

    @classmethod
    def from_dict(cls, d: dict) -> "QAMessage":
        return cls(role=d["role"], content=d["content"], timestamp=d.get("timestamp", time.time()))


@dataclass
class QASession:
    """问答会话窗口"""
    session_id: str
    name: str = "新对话"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    history: List[QAMessage] = field(default_factory=list)
    last_stock_code: Optional[str] = None
    last_company_name: Optional[str] = None
    last_complexity_level: str = "L1"
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "history": [m.to_dict() for m in self.history],
            "last_stock_code": self.last_stock_code,
            "last_company_name": self.last_company_name,
            "last_complexity_level": self.last_complexity_level,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "QASession":
        sess = cls(
            session_id=d["session_id"],
            name=d.get("name", "新对话"),
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
            last_stock_code=d.get("last_stock_code"),
            last_company_name=d.get("last_company_name"),
            last_complexity_level=d.get("last_complexity_level", "L1"),
        )
        sess.history = [QAMessage.from_dict(m) for m in d.get("history", [])]
        return sess

    def to_list_item(self) -> dict:
        """返回列表项摘要（不含完整历史）"""
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
            "last_stock_code": self.last_stock_code,
        }

    def add_message(self, role: str, content: str):
        self.history.append(QAMessage(role=role, content=content))
        self.updated_at = time.time()

    def get_history_for_llm(self, max_turns: int = 12) -> List[dict]:
        """
        返回对话历史用于 LLM 上下文。
        超过 max_turns 轮时，早期对话被压缩为摘要。
        """
        all_msgs = self.history
        cutoff = max_turns * 2  # 每轮=user+assistant

        if len(all_msgs) <= cutoff:
            return [{"role": m.role, "content": m.content} for m in all_msgs]

        # 压缩早期对话
        old_messages = all_msgs[:-cutoff]
        recent = all_msgs[-cutoff:]

        if not self.summary:
            self.summary = _compress_history(old_messages)

        result = [{"role": "system", "content": f"[历史对话摘要] {self.summary}"}]
        result.extend({"role": m.role, "content": m.content} for m in recent)
        return result


class SessionManager:
    """会话管理器 — 磁盘持久化 + 线程安全"""

    def __init__(self):
        self._lock = threading.Lock()
        _ensure_dirs()
        self._sessions: Dict[str, QASession] = {}
        self._load()

    def _load(self):
        """从磁盘加载所有会话"""
        if os.path.exists(_SESSIONS_FILE):
            try:
                with open(_SESSIONS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for item in data:
                    sess = QASession.from_dict(item)
                    self._sessions[sess.session_id] = sess
                logger.info(f"已加载 {len(self._sessions)} 个QA会话窗口")
            except Exception as e:
                logger.warning(f"加载QA会话失败: {e}，将使用空会话列表")

    def _save(self):
        """保存所有会话到磁盘"""
        _ensure_dirs()
        try:
            data = [sess.to_dict() for sess in self._sessions.values()]
            with open(_SESSIONS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存QA会话失败: {e}")

    # ── CRUD ──────────────────────────────────────

    def create_session(self, name: str = "新对话") -> str:
        """创建新会话窗口，返回 session_id"""
        session_id = str(uuid.uuid4())[:8]
        with self._lock:
            self._sessions[session_id] = QASession(session_id=session_id, name=name)
            self._save()
        return session_id

    def get_session(self, session_id: str) -> Optional[QASession]:
        """获取会话"""
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

    def list_sessions(self) -> List[dict]:
        """列出所有会话（摘要，按更新时间倒序）"""
        with self._lock:
            items = [s.to_list_item() for s in self._sessions.values()]
        items.sort(key=lambda x: x["updated_at"], reverse=True)
        return items

    def rename_session(self, session_id: str, new_name: str) -> bool:
        """重命名会话"""
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

    def delete_session(self, session_id: str) -> bool:
        """删除会话及其数据缓存"""
        with self._lock:
            if session_id not in self._sessions:
                return False
            del self._sessions[session_id]
            self._save()
        # 删除该会话的数据缓存目录
        _clear_session_cache(session_id)
        return True

    def update_context(self, session_id: str, **kwargs):
        """更新会话上下文并持久化（线程安全）"""
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess:
                for key, value in kwargs.items():
                    if hasattr(sess, key):
                        setattr(sess, key, value)
                sess.updated_at = time.time()
                # 只在关键字段变化时保存
                if any(k in kwargs for k in ("last_stock_code", "last_company_name")):
                    self._save()

    def save_session(self, session_id: str):
        """手动触发单个会话的持久化"""
        with self._lock:
            sess = self._sessions.get(session_id)
        if sess:
            self._save()

    @property
    def active_count(self) -> int:
        return len(self._sessions)


# ── 数据缓存（per-session, TTL 7天）─────────────────

def _get_session_cache_dir(session_id: str) -> str:
    return os.path.join(_QA_DATA_DIR, session_id, "cache")


def _make_cache_key(tool_name: str, kwargs: dict) -> str:
    """生成缓存键：工具名 + 参数哈希"""
    raw = tool_name + json.dumps(kwargs, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def get_cached_evidence(session_id: str, tool_name: str, kwargs: dict) -> Optional[str]:
    """读取 per-session 数据缓存，返回时自动标注数据获取时间"""
    cache_dir = _get_session_cache_dir(session_id)
    cache_key = _make_cache_key(tool_name, kwargs)
    cache_path = os.path.join(cache_dir, f"{cache_key}.json")

    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cached_at = data.get("cached_at", 0)
        if time.time() - cached_at > _CACHE_TTL_SECONDS:
            os.remove(cache_path)
            return None
        content = data.get("content", "")
        return _stamp_cached_content(content, cached_at)
    except Exception:
        return None


def _stamp_cached_content(content: str, cached_at: float) -> str:
    """在缓存数据头部插入数据获取时间戳"""
    from datetime import datetime
    fetch_time = datetime.fromtimestamp(cached_at).strftime("%Y-%m-%d %H:%M:%S")
    age_days = (time.time() - cached_at) / 86400
    age_note = f"{age_days:.0f}天前" if age_days >= 1 else "今天"
    return (
        f"[⚠️ 缓存数据 — 数据获取于 {fetch_time}（{age_note}），"
        f"当前分析基准时间可能与此不同，请基于数据时效性判断结论可靠性]\n\n"
        f"{content}"
    )


def set_cached_evidence(session_id: str, tool_name: str, kwargs: dict, content: str):
    """写入 per-session 数据缓存"""
    cache_dir = _get_session_cache_dir(session_id)
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = _make_cache_key(tool_name, kwargs)
    cache_path = os.path.join(cache_dir, f"{cache_key}.json")

    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({
                "tool_name": tool_name,
                "cached_at": time.time(),
                "content": content,
            }, f, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"写入QA缓存失败: {e}")


def _clear_session_cache(session_id: str):
    """删除指定会话的所有数据缓存"""
    cache_dir = _get_session_cache_dir(session_id)
    if os.path.exists(cache_dir):
        import shutil
        try:
            shutil.rmtree(cache_dir)
        except Exception as e:
            logger.warning(f"清理QA缓存失败: {e}")


def clean_expired_cache():
    """清理所有过期数据缓存（可定时调用）"""
    _ensure_dirs()
    now = time.time()
    cleaned = 0
    for cache_file in glob_module.glob(os.path.join(_QA_DATA_DIR, "*", "cache", "*.json")):
        try:
            mtime = os.path.getmtime(cache_file)
            if now - mtime > _CACHE_TTL_SECONDS:
                os.remove(cache_file)
                cleaned += 1
        except Exception:
            pass
    # 同时清理跨会话全局缓存
    for cache_file in glob_module.glob(os.path.join(_QA_DATA_DIR, "global_cache", "*.json")):
        try:
            mtime = os.path.getmtime(cache_file)
            if now - mtime > _CACHE_TTL_SECONDS:
                os.remove(cache_file)
                cleaned += 1
        except Exception:
            pass
    if cleaned:
        logger.info(f"清理了 {cleaned} 个过期QA数据缓存")


# ── 跨会话全局缓存 ──────────────────────────────

_GLOBAL_CACHE_DIR = os.path.join(_QA_DATA_DIR, "global_cache")


def get_global_cached_evidence(tool_name: str, kwargs: dict) -> Optional[str]:
    """跨会话全局缓存（同股票+同工具+同参数复用，统一7天TTL）"""
    cache_key = _make_cache_key(tool_name, kwargs)
    cache_path = os.path.join(_GLOBAL_CACHE_DIR, f"{cache_key}.json")
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if time.time() - data.get("cached_at", 0) > _CACHE_TTL_SECONDS:
            os.remove(cache_path)
            return None
        content = data.get("content", "")
        return _stamp_cached_content(content, data.get("cached_at", 0))
    except Exception:
        return None


def set_global_cached_evidence(tool_name: str, kwargs: dict, content: str):
    """写入跨会话全局缓存"""
    os.makedirs(_GLOBAL_CACHE_DIR, exist_ok=True)
    cache_key = _make_cache_key(tool_name, kwargs)
    cache_path = os.path.join(_GLOBAL_CACHE_DIR, f"{cache_key}.json")
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({
                "tool_name": tool_name,
                "cached_at": time.time(),
                "content": content,
            }, f, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"写入全局缓存失败: {e}")


# ── 上下文压缩 ──────────────────────────────────

def _compress_history(old_messages: list) -> str:
    """将早期对话消息压缩为简短摘要"""
    if not old_messages:
        return ""

    # 提取关键信息
    stocks_mentioned = set()
    topics = []
    questions = []

    import re
    for msg in old_messages:
        content = msg.content[:300]
        # 提取股票代码
        codes = re.findall(r'\b\d{5,6}\b', content)
        stocks_mentioned.update(codes)
        # 提取问题（用户消息）
        if msg.role == "user":
            q = content[:80].strip()
            if q:
                questions.append(q)

    parts = []
    if stocks_mentioned:
        parts.append(f"涉及股票: {', '.join(sorted(stocks_mentioned)[:5])}")
    if questions:
        parts.append(f"主要问题: {'; '.join(questions[-5:])}")
    parts.append(f"共{len(old_messages)}条历史消息")

    return "。".join(parts)


# ── 全局单例 ──────────────────────────────────────

_global_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    global _global_session_manager
    if _global_session_manager is None:
        _global_session_manager = SessionManager()
    return _global_session_manager
