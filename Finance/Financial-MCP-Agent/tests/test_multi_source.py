"""
多数据源集成测试 — Web Search + AKShare International + Yahoo Finance
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from src.qa.task_planner import plan_task, DATA_DOMAINS


class TestInternationalDataDomains:
    """验证国际数据域和工具已正确注册"""

    def test_international_domain_exists(self):
        assert "国际" in DATA_DOMAINS
        domain = DATA_DOMAINS["国际"]
        assert "web_search" in domain["tools"]
        assert "get_commodity_price" in domain["tools"]
        assert "get_dollar_index" in domain["tools"]
        assert "get_us_cpi" in domain["tools"]

    def test_macro_domain_has_international_tools(self):
        macro = DATA_DOMAINS["宏观"]
        assert "get_us_cpi" in macro["tools"]
        assert "web_search" in macro["tools"]

    def test_news_domain_has_web_search(self):
        news = DATA_DOMAINS["新闻"]
        assert "web_search" in news["tools"]


class TestGoldTopicTaskPlanning:
    """验证黄金主题的问题规划包含国际工具"""

    def test_gold_topic_includes_international_domains(self):
        plan = plan_task(
            "深度分析一下黄金未来的价格走势",
            "L4",
            topic_name="黄金",
            stock_code="sh.159934",
            company_name="黄金主题(黄金ETF)",
        )
        assert "国际" in plan.domains or "宏观" in plan.domains
        assert "web_search" in plan.tools

    def test_gold_topic_includes_us_macro_tools(self):
        plan = plan_task(
            "黄金最近走势如何",
            "L3",
            topic_name="黄金",
            stock_code="sh.159934",
            company_name="黄金主题(黄金ETF)",
        )
        all_tools = set(plan.tools)
        international_tools = {
            "get_us_cpi", "get_us_pmi", "get_us_non_farm",
            "get_spot_gold_sge", "get_commodity_price",
            "web_search",
        }
        assert len(all_tools & international_tools) >= 2

    def test_silver_topic_includes_international_tools(self):
        plan = plan_task(
            "分析白银走势",
            "L2",
            topic_name="白银",
            stock_code="sh.518880",
            company_name="白银主题(黄金ETF)",
        )
        assert "web_search" in plan.tools or "get_commodity_price" in plan.tools

    def test_crude_oil_topic_includes_international_tools(self):
        plan = plan_task(
            "原油价格未来走势分析",
            "L2",
            topic_name="原油",
            stock_code="sh.510410",
            company_name="原油主题(资源ETF)",
        )
        assert "web_search" in plan.tools or "get_commodity_price" in plan.tools

    def test_a_stock_topic_does_not_include_international(self):
        """普通A股主题不应包含国际工具"""
        plan = plan_task(
            "分析一下半导体板块走势",
            "L2",
            topic_name="半导体",
            stock_code="sh.512480",
            company_name="半导体主题(半导体ETF)",
        )
        assert "国际" not in plan.domains


class TestMCPConfig:
    """验证 MCP 配置正确注册了新服务器"""

    def test_web_search_and_yfinance_in_server_configs(self):
        from src.tools.mcp_config import SERVER_CONFIGS
        assert "web_search" in SERVER_CONFIGS
        assert "yfinance" in SERVER_CONFIGS
        assert SERVER_CONFIGS["web_search"]["transport"] == "stdio"
        assert SERVER_CONFIGS["yfinance"]["transport"] == "stdio"

    def test_server_files_exist(self):
        from src.tools.mcp_config import SERVER_CONFIGS
        import os as _os
        for name in ("web_search", "yfinance"):
            args = SERVER_CONFIGS[name]["args"]
            path = _os.path.join(*args[1:])  # skip "-u"
            assert _os.path.exists(path), f"{name} server not found: {path}"
