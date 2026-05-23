"""
QA 模块单元测试 — 复杂度分析、任务规划、会话管理
"""
import sys
import os
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import src.qa.session_manager as sm
from src.qa.session_manager import SessionManager, QASession, QAMessage
from src.qa.complexity_analyzer import analyze_complexity, ComplexityResult
from src.qa.task_planner import plan_task, extract_stock_from_question


@pytest.fixture(autouse=True)
def isolated_session_storage(monkeypatch):
    """将会话持久化路径重定向到临时目录，避免测试间污染"""
    tmpdir = tempfile.mkdtemp(prefix="qa_test_")
    monkeypatch.setattr(sm, "_QA_DATA_DIR", tmpdir)
    monkeypatch.setattr(sm, "_SESSIONS_FILE", os.path.join(tmpdir, "sessions.json"))
    # 重置全局单例
    monkeypatch.setattr(sm, "_global_session_manager", None)
    yield
    # 清理临时目录
    import shutil
    try:
        shutil.rmtree(tmpdir)
    except Exception:
        pass


# ── SessionManager 测试 ─────────────────────────

class TestSessionManager:
    def test_create_session(self):
        mgr = SessionManager()
        sid = mgr.create_session()
        assert len(sid) == 8
        assert mgr.get_session(sid) is not None

    def test_get_or_create_new(self):
        mgr = SessionManager()
        sess = mgr.get_or_create(None)
        assert sess.session_id is not None
        assert len(mgr._sessions) == 1

    def test_get_or_create_existing(self):
        mgr = SessionManager()
        sid = mgr.create_session()
        sess = mgr.get_or_create(sid)
        assert sess.session_id == sid
        assert len(mgr._sessions) == 1

    def test_delete_session(self):
        mgr = SessionManager()
        sid = mgr.create_session()
        assert mgr.delete_session(sid) is True
        assert mgr.get_session(sid) is None

    def test_delete_nonexistent(self):
        mgr = SessionManager()
        assert mgr.delete_session("nonexist") is False

    def test_session_history(self):
        mgr = SessionManager()
        sid = mgr.create_session()
        sess = mgr.get_session(sid)
        sess.add_message("user", "测试问题")
        sess.add_message("assistant", "测试回答")
        assert len(sess.history) == 2
        assert sess.history[0].role == "user"
        assert sess.history[1].role == "assistant"

    def test_update_context(self):
        mgr = SessionManager()
        sid = mgr.create_session()
        mgr.update_context(sid, last_stock_code="sh.600519", last_company_name="贵州茅台")
        sess = mgr.get_session(sid)
        assert sess.last_stock_code == "sh.600519"
        assert sess.last_company_name == "贵州茅台"

    def test_history_for_llm(self):
        mgr = SessionManager()
        sid = mgr.create_session()
        sess = mgr.get_session(sid)
        for i in range(5):
            sess.add_message("user", f"问题{i}")
            sess.add_message("assistant", f"回答{i}")
        llm_history = sess.get_history_for_llm(max_turns=3)
        assert len(llm_history) == 6  # 3 turns * 2

    def test_create_session_with_name(self):
        mgr = SessionManager()
        sid = mgr.create_session(name="茅台分析")
        sess = mgr.get_session(sid)
        assert sess.name == "茅台分析"

    def test_rename_session(self):
        mgr = SessionManager()
        sid = mgr.create_session(name="旧名称")
        assert mgr.rename_session(sid, "新名称") is True
        sess = mgr.get_session(sid)
        assert sess.name == "新名称"

    def test_rename_empty_name(self):
        mgr = SessionManager()
        sid = mgr.create_session()
        assert mgr.rename_session(sid, "  ") is False

    def test_list_sessions(self):
        mgr = SessionManager()
        mgr.create_session(name="A")
        mgr.create_session(name="B")
        items = mgr.list_sessions()
        assert len(items) == 2
        assert all("name" in it for it in items)
        assert all("message_count" in it for it in items)

    def test_session_persistence(self):
        """验证会话数据可以序列化和反序列化"""
        mgr = SessionManager()
        sid = mgr.create_session(name="持久化测试")
        sess = mgr.get_session(sid)
        sess.add_message("user", "测试")
        d = sess.to_dict()
        restored = QASession.from_dict(d)
        assert restored.name == "持久化测试"
        assert restored.session_id == sid
        assert len(restored.history) == 1
        assert restored.history[0].role == "user"


# ── ComplexityAnalyzer 测试 ─────────────────────

class TestComplexityAnalyzer:
    def test_l1_simple_question(self):
        result = analyze_complexity("贵州茅台今天跌了多少")
        assert result.level in ("L1", "L2")

    def test_l4_comparison_trigger(self):
        result = analyze_complexity("把宁德时代和比亚迪从估值和盈利角度做个比较")
        assert result.level in ("L3", "L4")

    def test_l4_why_trigger(self):
        result = analyze_complexity("为什么这只股票利润增长但股价不涨")
        assert result.level in ("L3", "L4")

    def test_l4_scenario_trigger(self):
        result = analyze_complexity("如果下季度利润继续改善，现在估值算不算便宜")
        assert result.level in ("L3", "L4")

    def test_l4_recommend_react(self):
        result = analyze_complexity("全面深度分析一下茅台的估值、财务、行业地位和未来前景")
        assert result.recommended_react is True
        assert result.recommended_model == "mimo-v2.5-pro"

    def test_l1_recommend_no_react(self):
        result = analyze_complexity("茅台PE多少")
        assert result.recommended_react is False
        assert result.recommended_model == "mimo-v2.5"

    def test_history_depth_bump(self):
        result = analyze_complexity("茅台PE多少", history_depth=4)
        assert result.level in ("L2", "L3", "L4")

    def test_explicit_deep_analysis(self):
        result = analyze_complexity("帮我详细分析一下这只股票")
        assert result.level in ("L2", "L3", "L4")

    def test_output_fields(self):
        result = analyze_complexity("茅台PE多少")
        assert isinstance(result.level, str)
        assert isinstance(result.score, int)
        assert isinstance(result.triggers, list)
        assert isinstance(result.need_clarify, bool)
        assert isinstance(result.recommended_react, bool)
        assert result.recommended_template in ("quick", "standard", "deep")

    def test_l3_range(self):
        """L3区间: 50-69分"""
        result = analyze_complexity("分析一下茅台的估值水平和行业地位对比同行")
        assert result.level in ("L2", "L3", "L4")


# ── TaskPlanner 测试 ────────────────────────────

class TestTaskPlanner:
    def test_identify_single_domain(self):
        plan = plan_task("茅台PE多少", "L1")
        assert len(plan.domains) >= 1

    def test_identify_multi_domain(self):
        plan = plan_task("茅台最近走势怎么样，估值贵不贵，资金在流入还是流出", "L2")
        assert len(plan.domains) >= 2

    def test_l1_no_react(self):
        plan = plan_task("茅台PE多少", "L1")
        assert plan.need_react is False

    def test_l4_needs_react(self):
        plan = plan_task("深度分析茅台", "L4")
        assert plan.need_react is True

    def test_extract_stock_code(self):
        code, name = extract_stock_from_question("600519")
        assert code is not None
        assert "600519" in code.replace("sh.", "").replace("sz.", "")

    def test_extract_paren_code(self):
        code, name = extract_stock_from_question("test(600519)")
        # 括号内提取: name=test前的中文部分, code=600519
        assert code is not None
        assert "600519" in code.replace("sh.", "").replace("sz.", "")

    def test_extract_reference(self):
        code, name = extract_stock_from_question(
            "这只股票最近怎么样",
            session_stock_code="sh.600519",
            session_company_name="贵州茅台"
        )
        assert code == "sh.600519"
        assert name == "贵州茅台"

    def test_tools_are_unique(self):
        plan = plan_task("分析茅台估值和财务", "L2")
        assert len(plan.tools) == len(set(plan.tools))

    def test_all_fields_present(self):
        plan = plan_task("茅台PE多少", "L1")
        assert isinstance(plan.domains, list)
        assert isinstance(plan.tools, list)
        assert isinstance(plan.need_react, bool)
        assert plan.reason
        assert plan.expected_data_volume in ("small", "medium", "large")
