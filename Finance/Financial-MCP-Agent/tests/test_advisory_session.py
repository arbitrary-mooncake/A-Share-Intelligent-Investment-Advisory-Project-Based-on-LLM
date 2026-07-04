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
