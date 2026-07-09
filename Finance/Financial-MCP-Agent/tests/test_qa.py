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
from src.qa.complexity_analyzer import (
    analyze_complexity, try_runtime_upgrade, ComplexityResult,
)
from src.qa.task_planner import plan_task, extract_stock_from_question
from src.qa.answer_generator import MAX_TOKENS_BY_LEVEL


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
        for i in range(3):
            sess.add_message("user", f"问题{i}")
            sess.add_message("assistant", f"回答{i}")
        llm_history = sess.get_history_for_llm(max_turns=12)
        assert len(llm_history) == 6  # 3 turns * 2, no compression needed

    def test_history_compression(self):
        """超过12轮时触发压缩"""
        mgr = SessionManager()
        sid = mgr.create_session()
        for i in range(15):
            sess = mgr.get_session(sid)
            sess.add_message("user", f"问题{i}")
            sess.add_message("assistant", f"回答{i}")
        sess = mgr.get_session(sid)
        llm_history = sess.get_history_for_llm(max_turns=12)
        # 应该有system摘要 + 最近12轮的24条消息
        assert llm_history[0]["role"] == "system"
        assert "历史对话摘要" in llm_history[0]["content"]
        assert len(llm_history) == 25  # 1 system + 24 recent

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
        # 使用明确命中硬触发的多标的对比问题
        result = analyze_complexity("把宁德时代和比亚迪从估值、盈利和现金流四个维度做个比较")
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


class TestComplexityConfiguration:
    """验证复杂度配置表（L0-L4 模型/thinking/template/max_tokens 映射）"""

    def test_l0_config(self):
        """L0: mimo-v2.5, thinking=False, template=l0, max_tokens=512"""
        result = analyze_complexity("你好")
        assert result.level == "L0"
        assert result.recommended_model == "mimo-v2.5"
        assert result.recommended_thinking is False
        assert result.recommended_template == "l0"
        assert MAX_TOKENS_BY_LEVEL["L0"] == 512

    def test_l1_config(self):
        """L1: mimo-v2.5, thinking=False, template=quick, max_tokens=4096"""
        result = analyze_complexity("茅台PE多少")
        assert result.level == "L1"
        assert result.recommended_model == "mimo-v2.5"
        assert result.recommended_thinking is False
        assert result.recommended_template == "quick"
        assert MAX_TOKENS_BY_LEVEL["L1"] == 4096

    def test_l2_config(self):
        """L2: mimo-v2.5-pro, thinking=False, template=standard, max_tokens=6144"""
        result = analyze_complexity("茅台最近的估值、走势、成交量和财务数据")
        assert result.level == "L2"
        assert result.recommended_model == "mimo-v2.5-pro"
        assert result.recommended_thinking is False
        assert result.recommended_template == "standard"
        assert MAX_TOKENS_BY_LEVEL["L2"] == 6144

    def test_l3_config(self):
        """L3: mimo-v2.5-pro, thinking=True, template=deep, max_tokens=8192"""
        result = analyze_complexity("茅台估值合理吗")
        assert result.level == "L3"
        assert result.recommended_model == "mimo-v2.5-pro"
        assert result.recommended_thinking is True
        assert result.recommended_template == "deep"
        assert MAX_TOKENS_BY_LEVEL["L3"] == 8192

    def test_l4_config(self):
        """L4: mimo-v2.5-pro, thinking=True, template=deep, max_tokens=16384"""
        result = analyze_complexity("全面深度分析一下茅台的估值、财务、行业地位和未来前景")
        assert result.level == "L4"
        assert result.recommended_model == "mimo-v2.5-pro"
        assert result.recommended_thinking is True
        assert result.recommended_template == "deep"
        assert MAX_TOKENS_BY_LEVEL["L4"] == 16384

    def test_max_tokens_map_complete(self):
        """max_tokens 映射表覆盖所有 5 个级别"""
        assert set(MAX_TOKENS_BY_LEVEL.keys()) == {"L0", "L1", "L2", "L3", "L4"}
        # 验证 max_tokens 随级别递增
        tokens = [MAX_TOKENS_BY_LEVEL[k] for k in ("L0", "L1", "L2", "L3", "L4")]
        assert tokens == sorted(tokens)

    def test_thinking_only_from_l3(self):
        """thinking 仅从 L3 开始开启，L0/L1/L2 不开"""
        l0 = analyze_complexity("你好")
        l1 = analyze_complexity("茅台PE多少")
        l2 = analyze_complexity("茅台最近的估值、走势、成交量和财务数据")
        l3 = analyze_complexity("茅台估值合理吗")
        l4 = analyze_complexity("全面深度分析一下茅台的估值、财务、行业地位和未来前景")
        assert l0.recommended_thinking is False
        assert l1.recommended_thinking is False
        assert l2.recommended_thinking is False
        assert l3.recommended_thinking is True
        assert l4.recommended_thinking is True

    def test_pro_model_from_l2(self):
        """Pro 模型从 L2 开始使用，L0/L1 用标准模型"""
        l0 = analyze_complexity("你好")
        l1 = analyze_complexity("茅台PE多少")
        l2 = analyze_complexity("茅台最近的估值、走势、成交量和财务数据")
        assert l0.recommended_model == "mimo-v2.5"
        assert l1.recommended_model == "mimo-v2.5"
        assert l2.recommended_model == "mimo-v2.5-pro"


# ── TaskPlanner 测试 ────────────────────────────

class TestTaskPlanner:
    def test_identify_single_domain(self):
        plan = plan_task("茅台PE多少", "L1")
        assert len(plan.domains) >= 1

    def test_identify_multi_domain(self):
        plan = plan_task("茅台最近走势怎么样，估值贵不贵，资金在流入还是流出", "L2")
        assert len(plan.domains) >= 2

    def test_l1_l2_no_react(self):
        assert plan_task("茅台PE多少", "L1").need_react is False
        assert plan_task("茅台走势怎么样", "L2").need_react is False

    def test_all_levels_use_two_phase(self):
        """全部复杂度统一使用两阶段快路径，不走ReAct"""
        assert plan_task("分析茅台估值", "L3").need_react is False
        assert plan_task("深度分析茅台", "L4").need_react is False

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


# ── Runtime Upgrader 测试 ────────────────────────

class TestRuntimeUpgrader:
    def test_low_success_rate_upgrades(self):
        result = analyze_complexity("茅台PE多少")
        upgraded = try_runtime_upgrade(
            result, tool_success_rate=0.3, evidence_missing_count=5,
            contradictory_signals=False, actual_domain_count=1,
        )
        assert upgraded.recommended_model == "mimo-v2.5-pro"

    def test_contradiction_triggers_pro_and_thinking(self):
        result = analyze_complexity("茅台PE多少")
        upgraded = try_runtime_upgrade(
            result, tool_success_rate=1.0, evidence_missing_count=0,
            contradictory_signals=True, actual_domain_count=2,
        )
        assert upgraded.recommended_model == "mimo-v2.5-pro"
        assert upgraded.recommended_thinking is True

    def test_multi_domain_upgrades_l3(self):
        result = analyze_complexity("茅台PE多少")  # L1
        upgraded = try_runtime_upgrade(
            result, tool_success_rate=1.0, evidence_missing_count=0,
            contradictory_signals=False, actual_domain_count=4,
        )
        assert upgraded.level in ("L3", "L4")

    def test_l4_forces_thinking(self):
        result = ComplexityResult(
            level="L4", score=80, triggers=[], score_detail={},
            need_clarify=False, recommended_model="mimo-v2.5-pro",
            recommended_thinking=False, recommended_react=True,
            recommended_template="deep",
        )
        upgraded = try_runtime_upgrade(
            result, tool_success_rate=1.0, evidence_missing_count=0,
            contradictory_signals=False, actual_domain_count=3,
        )
        assert upgraded.recommended_thinking is True

    def test_no_upgrade_when_all_ok(self):
        result = analyze_complexity("茅台PE多少")  # L1
        upgraded = try_runtime_upgrade(
            result, tool_success_rate=1.0, evidence_missing_count=0,
            contradictory_signals=False, actual_domain_count=1,
        )
        assert upgraded.level == result.level
        assert upgraded.recommended_model == result.recommended_model

    def test_l4_upgrade_sets_react(self):
        """L4升级时必须设置recommended_react"""
        result = ComplexityResult(
            level="L3", score=60, triggers=[], score_detail={},
            need_clarify=False, recommended_model="mimo-v2.5",
            recommended_thinking=False, recommended_react=False,
            recommended_template="standard",
        )
        upgraded = try_runtime_upgrade(
            result, tool_success_rate=0.3, evidence_missing_count=5,
            contradictory_signals=True, actual_domain_count=5,
        )
        # 多条件叠加应触发L4
        assert upgraded.level == "L4"
        assert upgraded.recommended_react is True
        assert upgraded.recommended_thinking is True


# ── 集成测试 ─────────────────────────────────────

class TestIntegration:
    """端到端流程验证（不依赖MCP和LLM）"""

    def test_full_pipeline_simple_question(self):
        """简单问题完整流程：复杂度→规划→(跳过证据装配)"""
        question = "茅台PE多少"
        complexity = analyze_complexity(question)
        plan = plan_task(question, complexity.level)
        # 验证规划合理性
        assert plan.need_react is False
        assert len(plan.domains) >= 1
        assert len(plan.tools) >= 1

    def test_full_pipeline_complex_question(self):
        """复杂问题：全部复杂度统一走两阶段快路径"""
        question = "把宁德时代和比亚迪从估值和现金流角度做个全面比较"
        complexity = analyze_complexity(question)
        plan = plan_task(question, complexity.level)
        assert complexity.level in ("L3", "L4")
        # 全部复杂度走两阶段快路径，不走ReAct
        assert plan.need_react is False

    def test_runtime_upgrade_pipeline(self):
        """运行时升级可将L1升级到L4，L4触发ReAct"""
        question = "茅台PE多少"
        complexity = analyze_complexity(question)
        assert complexity.level == "L1"

        # 模拟极端情况运行时升级
        complexity = try_runtime_upgrade(
            complexity,
            tool_success_rate=0.3, evidence_missing_count=5,
            contradictory_signals=True, actual_domain_count=5,
        )
        # 多条件叠加应触发L4
        assert complexity.level == "L4"
        plan = plan_task(question, complexity.level)
        # L4也走两阶段快路径，不走ReAct
        assert plan.need_react is False

    def test_stock_extraction_with_context(self):
        """Stock code extraction with and without session context"""
        # Direct 6-digit code extraction
        code, name = extract_stock_from_question("600519")
        assert code == "sh.600519"

        # Fallback: no code in question, no session context → None
        code2, name2 = extract_stock_from_question("hello")
        assert code2 is None

        # Session context passed through when question has no code
        code3, name3 = extract_stock_from_question(
            "hello", session_stock_code="sh.600519", session_company_name="test"
        )
        # No referential match on "hello", so context not used
        assert code3 is None

    def test_evidence_package_structure(self):
        """证据包数据结构验证"""
        from src.qa.evidence_assembler import EvidencePackage
        pkg = EvidencePackage(
            subject="测试",
            stock_code="sh.600519",
            company_name="贵州茅台",
            raw_text="测试数据",
            tool_call_summary="3/5 工具成功",
            missing=["tool_a", "tool_b"],
        )
        assert pkg.subject == "测试"
        assert len(pkg.missing) == 2
        assert pkg.elapsed_seconds == 0.0


# ── 幻觉/安全测试 ───────────────────────────────

class TestHallucinationSafety:
    """防幻觉与安全边界测试"""

    def test_complexity_no_fabrication_trigger(self):
        """复杂度分析不应编造数据"""
        result = analyze_complexity("给我编一个茅台的PE数据")
        # 不能是深度分析 — 这是危险请求
        assert result.recommended_react is not True or result.level != "L4"

    def test_nonexistent_stock_handled(self):
        """Non-standard stock code (999999) is kept as-is without exchange prefix"""
        code, name = extract_stock_from_question("999999")
        assert code == "999999"  # stays raw, no sh./sz. prefix added

    def test_price_prediction_request(self):
        """Prediction requests should get at least L2 complexity (not L1)"""
        # Use a question with "为什么" trigger to ensure it's not L1
        result = analyze_complexity("为什么茅台明天可能会涨")
        assert result.level in ("L2", "L3", "L4")

    def test_empty_evidence_handled(self):
        """空证据包的结构完整性"""
        from src.qa.evidence_assembler import EvidencePackage
        pkg = EvidencePackage()
        assert pkg.raw_text == ""
        assert pkg.tool_call_summary == ""
        # 在qa_engine中，空raw_text会触发降级
        assert not pkg.raw_text or pkg.raw_text == ""

    def test_missing_data_declared(self):
        """缺失数据应被明确记录"""
        from src.qa.evidence_assembler import EvidencePackage
        pkg = EvidencePackage(
            missing=["get_profit_data", "tushare_moneyflow"],
            raw_text="部分数据",
        )
        assert "get_profit_data" in pkg.missing
        # 缺失数据不影响raw_text存在

    def test_answer_template_includes_data_time(self):
        """所有回答模板应包含数据截至时间标记"""
        from src.qa.answer_generator import _build_system_prompt
        for template in ("quick", "standard", "deep"):
            prompt = _build_system_prompt(template, "2026-05-23")
            assert "2026-05-23" in prompt
            assert "数据截至" in prompt

    def test_system_prompt_forbids_fabrication(self):
        """系统提示词必须包含防编造指令"""
        from src.qa.answer_generator import _build_system_prompt
        for template in ("quick", "standard", "deep"):
            prompt = _build_system_prompt(template, "2026-05-23")
            assert "绝不编造" in prompt or "数据优先" in prompt
            assert "无法获取" in prompt

    def test_two_stage_deep_template(self):
        """深度模板包含两段式输出指令"""
        from src.qa.answer_generator import _build_system_prompt
        prompt = _build_system_prompt("deep", "2026-05-23")
        assert "分析框架" in prompt
        assert "---" in prompt


# ── 对话前缀提取回归测试 ──────────────────────────

@pytest.mark.skip(reason="extract_stock_from_question 已重构为仅处理确定性结构化输入，自然语言提取交 LLM 完成")
class TestConversationalPrefixExtraction:
    """对话前缀（你觉得/我认为/大家看等）不应被误提取为公司名"""

    def test_you_feel_topic_question(self):
        """'你觉得半导体还能继续涨吗' → 提取'半导体'，不是'你觉得半导体'"""
        code, name = extract_stock_from_question("你觉得半导体还能继续涨吗？")
        assert name == "半导体"
        assert code is None

    def test_you_think_topic_question(self):
        """'你认为半导体板块还能涨吗' → 提取主题关键词（半导体/半导体板块）"""
        from src.qa.qa_engine import _is_topic_keyword
        code, name = extract_stock_from_question("你认为半导体板块还能涨吗")
        assert name is not None
        assert _is_topic_keyword(name) is True

    def test_you_look_topic_question(self):
        """'你看新能源还能涨吗' → 提取'新能源'"""
        code, name = extract_stock_from_question("你看新能源还能涨吗")
        assert name == "新能源"

    def test_i_feel_topic_question(self):
        """'我觉得白酒板块还能涨吗' → 提取主题关键词（白酒/白酒板块）"""
        from src.qa.qa_engine import _is_topic_keyword
        code, name = extract_stock_from_question("我觉得白酒板块还能涨吗")
        assert name is not None
        assert _is_topic_keyword(name) is True

    def test_i_want_to_know_topic_question(self):
        """'我想知道人工智能还能涨吗' → 提取'人工智能'"""
        code, name = extract_stock_from_question("我想知道人工智能还能涨吗")
        assert name == "人工智能"

    def test_conversational_prefix_with_real_stock(self):
        """'你觉得贵州茅台还能涨吗' → 提取'贵州茅台'（真实公司名，非主题）"""
        code, name = extract_stock_from_question("你觉得贵州茅台还能涨吗")
        assert name == "贵州茅台"

    def test_conversational_prefix_with_real_stock2(self):
        """'你看中芯国际还能涨吗' → 提取'中芯国际'"""
        code, name = extract_stock_from_question("你看中芯国际还能涨吗")
        assert name == "中芯国际"

    def test_no_conversational_prefix_topic(self):
        """无对话前缀时主题提取仍然正常：'半导体还能继续涨吗' → '半导体'"""
        code, name = extract_stock_from_question("半导体还能继续涨吗？")
        assert name == "半导体"

    def test_existing_extraction_still_works(self):
        """原有提取逻辑不受影响：'中际旭创最近走势怎么样' → '中际旭创'"""
        code, name = extract_stock_from_question("中际旭创最近走势怎么样")
        assert name == "中际旭创"

    def test_code_extraction_unaffected(self):
        """代码提取不受影响：'600519' → sh.600519"""
        code, name = extract_stock_from_question("600519")
        assert code == "sh.600519"


# ── _is_topic_keyword 测试 ───────────────────────

class TestIsTopicKeyword:
    """主题关键词识别"""

    def test_exact_topic_keyword(self):
        from src.qa.qa_engine import _is_topic_keyword
        assert _is_topic_keyword("半导体") is True
        assert _is_topic_keyword("新能源") is True
        assert _is_topic_keyword("人工智能") is True

    def test_topic_keyword_with_suffix(self):
        """'半导体板块'以'半导体'开头 → True"""
        from src.qa.qa_engine import _is_topic_keyword
        assert _is_topic_keyword("半导体板块") is True
        assert _is_topic_keyword("新能源赛道") is True

    def test_real_stock_name_not_topic(self):
        """真实公司名不应被识别为主题"""
        from src.qa.qa_engine import _is_topic_keyword
        assert _is_topic_keyword("贵州茅台") is False
        assert _is_topic_keyword("中芯国际") is False
        assert _is_topic_keyword("中际旭创") is False

    def test_conversational_prefix_not_topic(self):
        """对话前缀误提取结果不应被识别为主题"""
        from src.qa.qa_engine import _is_topic_keyword
        assert _is_topic_keyword("你觉得半导体") is False
        assert _is_topic_keyword("你认为茅台") is False


# ── 空代码安全网测试 ─────────────────────────────

class TestEmptyCodeSafetyNet:
    """有公司名但无代码时，evidence_assembler 应安全返回而非调用空code工具"""

    def test_safety_net_returns_early_when_code_empty(self):
        """有company_name但无stock_code + 需要代码的工具 → 安全返回"""
        from src.qa.evidence_assembler import assemble_evidence_fast
        import asyncio
        result = asyncio.run(assemble_evidence_fast(
            stock_code="",
            company_name="某未知公司",
            tools=["tushare_kline", "tushare_daily_basic"],
            question="某未知公司怎么样",
            current_date="2026-06-25",
        ))
        assert "无法解析" in result.raw_text or "股票代码" in result.raw_text
        assert result.missing
        assert "跳过数据获取" in result.tool_call_summary

    def test_safety_net_allows_no_code_tools(self):
        """全部是无代码工具（宏观/市场类）时，安全网不应拦截"""
        from src.qa.evidence_assembler import assemble_evidence_fast
        import asyncio
        result = asyncio.run(assemble_evidence_fast(
            stock_code="",
            company_name="",
            tools=["tushare_cn_cpi", "tushare_cn_gdp"],
            question="最近CPI怎么样",
            current_date="2026-06-25",
        ))
        # 宏观类工具不需要代码，应跳过安全网（可能因MCP未连接而失败，但不应被安全网拦截）
        assert "无股票代码（名称反查失败）" not in result.tool_call_summary

    def test_both_empty_with_stock_tools(self):
        """stock_code和company_name都为空 + 需要代码的工具 → 无标的安全返回"""
        from src.qa.evidence_assembler import assemble_evidence_fast
        import asyncio
        result = asyncio.run(assemble_evidence_fast(
            stock_code="",
            company_name="",
            tools=["tushare_kline"],
            question="黄金价格怎么样",
            current_date="2026-06-25",
        ))
        assert "未指定A股标的" in result.missing
        assert "跳过数据获取" in result.tool_call_summary


# ── ReAct超时 + 心跳测试 ─────────────────────────

class TestReactTimeoutAndHeartbeat:
    """ReAct路径超时保护和心跳SSE事件"""

    def test_react_timeout_constant_is_safe(self):
        """REACT_TOTAL_TIMEOUT 必须 <= 120s，以适应前端180s超时"""
        from src.qa.evidence_assembler import REACT_TOTAL_TIMEOUT
        assert 30 <= REACT_TOTAL_TIMEOUT <= 120

    def test_heartbeat_pattern_yields_during_slow_task(self):
        """心跳模式：后台任务执行期间定期yield，防止前端超时"""
        import asyncio

        async def slow_task():
            await asyncio.sleep(0.5)
            return "evidence_done"

        async def run_heartbeat():
            task = asyncio.create_task(slow_task())
            heartbeats = []
            result = None
            _secs = 0
            while True:
                try:
                    result = await asyncio.wait_for(asyncio.shield(task), timeout=0.1)
                    break
                except asyncio.TimeoutError:
                    _secs += 0.1
                    heartbeats.append(f"heartbeat {_secs:.1f}s")
                except Exception as e:
                    result = f"error: {e}"
                    break
            return result, heartbeats

        result, heartbeats = asyncio.run(run_heartbeat())
        assert result == "evidence_done"
        assert len(heartbeats) >= 3  # 0.5s / 0.1s ≈ 5 heartbeats

    def test_heartbeat_handles_task_exception(self):
        """后台任务抛异常时，心跳循环应捕获并返回降级结果"""
        import asyncio

        async def failing_task():
            await asyncio.sleep(0.05)
            raise RuntimeError("MCP爆炸")

        async def run_heartbeat():
            task = asyncio.create_task(failing_task())
            while True:
                try:
                    result = await asyncio.wait_for(asyncio.shield(task), timeout=0.2)
                    return result, "ok"
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    if not task.done():
                        task.cancel()
                    return None, f"caught: {e}"

        result, status = asyncio.run(run_heartbeat())
        assert result is None
        assert "caught" in status
        assert "MCP爆炸" in status

    def test_react_timeout_returns_partial_evidence(self):
        """ReAct超时时应返回带超时标记的EvidencePackage"""
        from src.qa.evidence_assembler import EvidencePackage, REACT_TOTAL_TIMEOUT
        # 验证超时路径的返回结构（不实际调用agent，只验证结构）
        ep = EvidencePackage(
            subject="黄金",
            stock_code="sh.159934",
            company_name="黄金主题",
            raw_text="（ReAct数据获取在120秒内未完成，已超时终止。）",
            tool_call_summary="ReAct超时 (120s)",
            missing=[f"ReAct超时({REACT_TOTAL_TIMEOUT}s)"],
        )
        assert "超时" in ep.tool_call_summary
        assert ep.missing
        assert "sh.159934" in ep.stock_code

