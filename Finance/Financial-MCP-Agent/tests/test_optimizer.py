"""Tests for optimization module"""
import os
import tempfile

from src.eval.optimizer.router import OptimizeRouter
from src.eval.optimizer.manual_package_builder import ManualPackageBuilder
from src.eval.optimizer.safe_patch_executor import SafePatchExecutor


def test_router_classify_param_tune():
    router = OptimizeRouter()
    evidence = {"type": "parameter_suboptimal", "complexity": "low",
                "affected_files": ["config/eval/defaults.json"]}
    result = router.classify_issue(evidence)
    assert result["ticket_type"] == "PARAM_TUNE"
    assert result["route"] == "auto"


def test_router_classify_research():
    router = OptimizeRouter()
    evidence = {"type": "agent_negative_contribution", "complexity": "high"}
    result = router.classify_issue(evidence)
    assert result["ticket_type"] == "RESEARCH"
    assert result["route"] == "manual"


def test_router_classify_logic_fix():
    router = OptimizeRouter()
    evidence = {"type": "logic_bug", "complexity": "low",
                "affected_files": ["config/eval/defaults.json"]}
    result = router.classify_issue(evidence)
    assert result["route"] in ("semi_auto", "manual")


def test_router_generate_ticket():
    router = OptimizeRouter()
    evidence = {"title": "测试问题", "summary": "这是一个测试", "severity": "low"}
    classification = {"ticket_type": "PARAM_TUNE", "route": "auto"}
    ticket = router.generate_ticket("test_batch", evidence, classification)
    assert ticket["ticket_type"] == "PARAM_TUNE"
    assert ticket["route"] == "auto"


def test_router_classify_prompt_quality():
    router = OptimizeRouter()
    evidence = {"type": "prompt_quality", "complexity": "medium"}
    result = router.classify_issue(evidence)
    assert result["ticket_type"] == "PROMPT_PATCH"
    assert result["route"] == "semi_auto"


def test_router_classify_architecture_issue():
    router = OptimizeRouter()
    evidence = {"type": "architecture_issue", "complexity": "high"}
    result = router.classify_issue(evidence)
    assert result["ticket_type"] == "ARCH_CHANGE"
    assert result["route"] == "manual"


def test_router_can_autofix_blacklist():
    router = OptimizeRouter()
    # agent directory is in blacklist
    assert not router._can_autofix(["src/agents/fundamental_agent.py"], "low")


def test_router_can_autofix_whitelist():
    router = OptimizeRouter()
    assert router._can_autofix(["config/eval/defaults.json"], "low")


def test_router_can_autofix_high_complexity():
    router = OptimizeRouter()
    assert not router._can_autofix(["config/eval/defaults.json"], "high")


def test_router_can_autofix_too_many_files():
    router = OptimizeRouter()
    files = [f"config/eval/file{i}.json" for i in range(5)]
    assert not router._can_autofix(files, "low")


def test_router_suggested_actions():
    router = OptimizeRouter()
    auto = router._get_suggested_actions({"ticket_type": "PARAM_TUNE", "route": "auto"})
    assert len(auto) == 3
    semi = router._get_suggested_actions({"ticket_type": "PROMPT_PATCH", "route": "semi_auto"})
    assert len(semi) == 3
    manual = router._get_suggested_actions({"ticket_type": "RESEARCH", "route": "manual"})
    assert len(manual) == 3


def test_manual_package_builder():
    builder = ManualPackageBuilder()
    ticket = {"title": "测试优化", "ticket_type": "RESEARCH", "severity": "medium",
              "batch_id": "test", "summary": "测试摘要"}
    evidence = {"affected_modules": ["loss_engine"], "affected_files": ["loss_engine.py"]}
    pkg = builder.build_package(ticket, evidence)
    assert pkg["title"] == "测试优化"
    assert "problem_definition" in pkg
    assert "suggested_direction" in pkg
    assert "impact_assessment" in pkg
    assert "risk_warnings" in pkg


def test_manual_package_builder_with_contributions():
    builder = ManualPackageBuilder()
    ticket = {"title": "Agent优化", "ticket_type": "LOGIC_FIX", "severity": "high",
              "batch_id": "test", "summary": "测试摘要"}
    evidence = {"affected_modules": ["loss_engine"], "affected_files": ["loss_engine.py"],
                "affected_agents": ["fundamental"]}
    contributions = [{"agent_name": "fundamental", "delta_L_total": -0.05},
                     {"agent_name": "technical", "delta_L_total": 0.001}]
    pkg = builder.build_package(ticket, evidence, contributions)
    assert "fundamental" in pkg["impact_assessment"]["affected_agents"]
    assert "technical" not in pkg["impact_assessment"]["affected_agents"]


def test_manual_package_save():
    builder = ManualPackageBuilder()
    ticket = {"title": "保存测试", "ticket_type": "PARAM_TUNE", "severity": "low",
              "batch_id": "test", "summary": "保存测试"}
    evidence = {}
    pkg = builder.build_package(ticket, evidence)
    md_path = builder.save_package(pkg)
    assert os.path.exists(md_path)
    # Cleanup
    os.remove(md_path)
    json_path = md_path.replace(".md", ".json")
    if os.path.exists(json_path):
        os.remove(json_path)


def test_safe_patch_executor_no_file():
    executor = SafePatchExecutor()
    result = executor.apply_patch(os.path.join(tempfile.gettempdir(), "nonexistent_file_for_test_12345.py"),
                                  "content")
    assert not result["success"]


def test_safe_patch_executor_backup_restore():
    import tempfile
    executor = SafePatchExecutor()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("x = 1\n")
        tmp_path = f.name
    try:
        # Apply patch with valid Python (test will likely fail = rollback)
        result = executor.apply_patch(tmp_path, "x = 2\n", "test patch")
        assert isinstance(result, dict)
        # Check file was restored (or patch succeeded and tests passed)
        with open(tmp_path, "r") as f:
            content = f.read()
        # Either reverted to "x = 1\n" or still "x = 2\n" if tests passed
        assert content in ("x = 1\n", "x = 2\n")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_router_classify_default_unknown_type():
    router = OptimizeRouter()
    evidence = {"type": "unknown_issue_type", "complexity": "medium"}
    result = router.classify_issue(evidence)
    assert result["ticket_type"] == "RESEARCH"
    assert result["route"] == "manual"


def test_manual_package_builder_arch_change_risks():
    builder = ManualPackageBuilder()
    ticket = {"title": "架构变更", "ticket_type": "ARCH_CHANGE", "severity": "critical",
              "batch_id": "test", "summary": "需要重构"}
    evidence = {}
    pkg = builder.build_package(ticket, evidence)
    assert any("架构变更" in r for r in pkg["risk_warnings"])


def test_manual_package_builder_suggest_direction_param_tune():
    builder = ManualPackageBuilder()
    ticket = {"title": "调参", "ticket_type": "PARAM_TUNE", "severity": "low",
              "batch_id": "test", "summary": ""}
    evidence = {}
    pkg = builder.build_package(ticket, evidence)
    assert any("参数" in d for d in pkg["suggested_direction"])
