"""
Tests for ManualPackageBuilder: structured optimization package generation.

Covers:
  - Package creation from manual ticket data
  - Validation of input format completeness
  - Error handling for malformed / missing input
  - Impact assessment with contributions
  - Direction suggestion by ticket type
  - Markdown and JSON serialization
"""
import json
import os
import pytest
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.eval.optimizer.manual_package_builder import ManualPackageBuilder


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def builder():
    """Create a ManualPackageBuilder pointing to a temp directory."""
    return ManualPackageBuilder()


@pytest.fixture
def valid_ticket():
    """A valid ticket dict for testing."""
    return {
        "title": "调整短线买入阈值",
        "ticket_type": "PARAM_TUNE",
        "severity": "medium",
        "batch_id": "batch-001",
        "summary": "当前短线买入阈值75过低，导致买入信号过多，建议调整为78。",
    }


@pytest.fixture
def valid_evidence():
    """A valid evidence dict for testing."""
    return {
        "affected_modules": ["scoring", "strategy"],
        "affected_files": [
            "config/eval/strategy_defaults.json",
            "src/eval/strategies/short_ablation.py",
        ],
        "data_source": ["loss_engine", "contribution_engine"],
    }


# ── Basic Package Creation ───────────────────────────────────────────────────


def test_build_package_basic(builder, valid_ticket, valid_evidence):
    """正常构建完整优化包，所有必要字段存在"""
    pkg = builder.build_package(valid_ticket, valid_evidence)

    assert pkg["title"] == "调整短线买入阈值"
    assert pkg["ticket_type"] == "PARAM_TUNE"
    assert pkg["severity"] == "medium"
    assert "generated_at" in pkg
    assert "batch_id" in pkg

    # Problem definition
    assert "problem_definition" in pkg
    assert pkg["problem_definition"]["summary"] == valid_ticket["summary"]

    # Impact assessment
    assert "impact_assessment" in pkg

    # Direction
    assert "suggested_direction" in pkg
    assert len(pkg["suggested_direction"]) >= 1

    # File tracking
    assert "related_files" in pkg
    assert "affected_modules" in pkg

    # Tests & criteria
    assert "suggested_tests" in pkg
    assert "expected_acceptance_criteria" in pkg

    # Risks
    assert "risk_warnings" in pkg


def test_build_package_minimal_ticket(builder):
    """最小化ticket输入，使用默认值填充"""
    pkg = builder.build_package({}, {})
    assert pkg["title"] == "优化建议"
    assert pkg["ticket_type"] == "RESEARCH"
    assert pkg["severity"] == "medium"


def test_build_package_with_contributions(builder, valid_ticket, valid_evidence):
    """带贡献数据时，impact assessment 反映贡献信息"""
    contributions = [
        {"agent_name": "fundamental", "delta_L_total": 0.042},
        {"agent_name": "news", "delta_L_total": -0.005},
        {"agent_name": "value", "delta_L_total": 0.001},
    ]

    pkg = builder.build_package(valid_ticket, valid_evidence, contributions)
    impact = pkg["impact_assessment"]
    affected = impact["affected_agents"]

    # Only agents with |delta_L_total| > 0.01 should be listed
    assert "fundamental" in affected
    assert "news" not in affected
    assert "value" not in affected


def test_build_package_empty_contributions(builder, valid_ticket, valid_evidence):
    """空贡献列表不应导致错误"""
    pkg = builder.build_package(valid_ticket, valid_evidence, [])
    assert "impact_assessment" in pkg
    assert pkg["impact_assessment"]["affected_agents"] == []


# ── Direction By Ticket Type ─────────────────────────────────────────────────


@pytest.mark.parametrize("ticket_type,expected_dir_keyword", [
    ("PARAM_TUNE", "调整"),
    ("PROMPT_PATCH", "修改"),
    ("LOGIC_FIX", "定位"),
    ("ARCH_CHANGE", "影响面"),
    ("RESEARCH", "分析"),
])
def test_suggest_direction_by_type(builder, valid_evidence, ticket_type, expected_dir_keyword):
    """不同ticket类型应产生不同的建议方向"""
    ticket = {
        "title": f"Test {ticket_type}",
        "ticket_type": ticket_type,
        "severity": "medium",
        "summary": f"Summary for {ticket_type}",
    }
    pkg = builder.build_package(ticket, valid_evidence)
    direction = pkg["suggested_direction"]
    # At least one suggestion should contain the keyword
    assert any(expected_dir_keyword in d for d in direction), \
        f"Expected '{expected_dir_keyword}' in directions: {direction}"


# ── Serialization ────────────────────────────────────────────────────────────


def test_save_package_creates_files(builder, valid_ticket, valid_evidence):
    """保存优化包应创建 .md 和 .json 两个文件"""
    pkg = builder.build_package(valid_ticket, valid_evidence)
    md_path = builder.save_package(pkg)

    assert os.path.exists(md_path)
    assert md_path.endswith(".md")

    json_path = md_path.replace(".md", ".json")
    assert os.path.exists(json_path)

    # Verify JSON is valid
    with open(json_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["title"] == pkg["title"]

    # Cleanup
    os.remove(md_path)
    os.remove(json_path)


def test_save_package_markdown_content(builder, valid_ticket, valid_evidence):
    """生成的Markdown包含必要章节"""
    pkg = builder.build_package(valid_ticket, valid_evidence)
    md_path = builder.save_package(pkg)

    with open(md_path, "r", encoding="utf-8") as f:
        md = f.read()

    required_sections = [
        "# 优化建议",
        "## 问题诊断",
        "## 影响评估",
        "## 建议方向",
        "## 相关文件",
        "## 建议测试",
        "## 验收标准",
        "## 风险提示",
    ]
    for section in required_sections:
        assert section in md, f"Missing section: {section}"

    json_path = md_path.replace(".md", ".json")
    os.remove(md_path)
    os.remove(json_path)


# ── Risk Assessment ──────────────────────────────────────────────────────────


def test_arch_change_adds_extra_risk(builder, valid_evidence):
    """架构变更类型应额外提示风险"""
    ticket = {
        "title": "重构评分引擎",
        "ticket_type": "ARCH_CHANGE",
        "severity": "high",
        "summary": "需要重构scoring engine架构",
    }
    pkg = builder.build_package(ticket, valid_evidence)
    risks = pkg["risk_warnings"]
    assert any("影响面" in r for r in risks)


def test_non_arch_change_default_risks(builder, valid_ticket, valid_evidence):
    """非架构变更类型只有默认风险提示"""
    pkg = builder.build_package(valid_ticket, valid_evidence)
    risks = pkg["risk_warnings"]
    assert any("bug" in r for r in risks)


# ── Impact Scope ─────────────────────────────────────────────────────────────


def test_long_summary_multi_module(builder, valid_evidence):
    """长summary应判定为跨模块"""
    ticket = {
        "title": "Long summary test",
        "ticket_type": "RESEARCH",
        "severity": "medium",
        "summary": "A" * 120,  # > 100 chars
    }
    pkg = builder.build_package(ticket, valid_evidence)
    assert pkg["impact_assessment"]["scope"] == "跨模块"


def test_short_summary_single_module(builder, valid_evidence):
    """短summary应判定为单模块"""
    ticket = {
        "title": "Short summary test",
        "ticket_type": "RESEARCH",
        "severity": "medium",
        "summary": "Short fix",
    }
    pkg = builder.build_package(ticket, valid_evidence)
    assert pkg["impact_assessment"]["scope"] == "单模块"
