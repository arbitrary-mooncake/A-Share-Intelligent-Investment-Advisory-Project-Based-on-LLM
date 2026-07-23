"""
按期限的数据需求声明（4.1 定稿：TermDataSpec）。

内容 1:1 归并自 7 个分析 Agent 源码中实际的工具调用参数（2026-07-22 逐一核对），
保证 DataGateway 预取的 (tool, kwargs) 与 Agent 自身调用完全一致——
预取结果通过共享工具缓存（tool_cache, 相同 MD5 键）被 Agent 命中，
数据等价性由构造保证（同一工具、同一参数、同一缓存路径）。

注意：fina_indicator 在 fundamental 用 years=4，value/quality_risk 为裸参数——
两者是不同数据需求，各自保留条目（去重只针对完全相同的 (tool, kwargs)）。
"""
from typing import Any, Dict, List, Tuple

# 期限 → 该期限打分实际需要的分析 Agent（与 scoring_engine._TERM_AGENTS 对齐）
TERM_AGENT_MAP: Dict[str, List[str]] = {
    "short": ["technical", "news", "event", "moneyflow"],
    "medium": ["fundamental", "technical", "value", "news", "event", "quality_risk", "moneyflow"],
    "long": ["fundamental", "technical", "value", "news", "event", "quality_risk", "moneyflow"],
    "full": ["fundamental", "technical", "value", "news", "event", "quality_risk", "moneyflow"],
}

# ETF 期望缺席的 Agent（fundamental/value 对 ETF early-exit，预取同样跳过）
ETF_EXEMPT_AGENTS = {"fundamental", "value"}

# 占位符：{code} = 纯数字代码；{date} = YYYYMMDD（moneyflow top_list 用）
AgentToolSpec = Tuple[str, Dict[str, Any]]

AGENT_TOOL_SPECS: Dict[str, List[AgentToolSpec]] = {
    "fundamental": [
        ("tushare_stock_info", {"code": "{code}"}),
        ("tushare_income", {"code": "{code}"}),
        ("tushare_balancesheet", {"code": "{code}"}),
        ("tushare_cashflow", {"code": "{code}"}),
        ("tushare_fina_indicator", {"code": "{code}", "years": 4}),
        ("tushare_daily_basic", {"code": "{code}", "days": 250}),
        ("tushare_ev_ebitda", {"code": "{code}"}),
        ("tushare_dividend", {"code": "{code}"}),
        ("tushare_adj_factor", {"code": "{code}", "days": 500}),
        ("tushare_top10_holders", {"code": "{code}"}),
        ("tushare_st_status", {"code": "{code}"}),
    ],
    # technical 为 ReAct Agent：仅预计算阶段的 kline 调用是确定性的，
    # ReAct 自由调用部分不在预取范围（4.6 后续处理）
    "technical": [
        ("tushare_kline", {"code": "{code}", "days": 250}),
    ],
    "value": [
        ("tushare_stock_info", {"code": "{code}"}),
        ("tushare_income", {"code": "{code}"}),
        ("tushare_balancesheet", {"code": "{code}"}),
        ("tushare_cashflow", {"code": "{code}"}),
        ("tushare_fina_indicator", {"code": "{code}"}),
        ("tushare_daily_basic", {"code": "{code}"}),
        ("tushare_pe_percentile", {"code": "{code}"}),
        ("tushare_ev_ebitda", {"code": "{code}"}),
        ("tushare_dividend", {"code": "{code}"}),
        ("tushare_top10_holders", {"code": "{code}"}),
        ("tushare_holder_num", {"code": "{code}"}),
    ],
    "news": [
        ("crawl_news", {"query": "{code}", "top_k": 10}),
    ],
    "event": [
        ("tushare_anns_d", {"code": "{code}", "days": 90}),
        ("tushare_new_share", {"code": "{code}"}),
        ("tushare_pledge_stat", {"code": "{code}"}),
        ("tushare_repurchase", {"code": "{code}"}),
        ("tushare_share_float", {"code": "{code}"}),
        ("tushare_top10_holders", {"code": "{code}"}),
        ("tushare_stk_holdertrade", {"code": "{code}"}),
        ("tushare_dividend", {"code": "{code}"}),
        ("tushare_namechange", {"code": "{code}"}),
        ("tushare_suspend", {"code": "{code}"}),
        ("crawl_news", {"query": "{code}", "top_k": 10}),
        ("tushare_st_status", {"code": "{code}"}),
    ],
    "quality_risk": [
        ("tushare_income", {"code": "{code}"}),
        ("tushare_balancesheet", {"code": "{code}"}),
        ("tushare_cashflow", {"code": "{code}"}),
        ("tushare_fina_indicator", {"code": "{code}"}),
        ("tushare_pledge_stat", {"code": "{code}"}),
        ("tushare_top10_holders", {"code": "{code}"}),
        ("tushare_stk_holdertrade", {"code": "{code}"}),
        ("tushare_st_status", {"code": "{code}"}),
    ],
    "moneyflow": [
        ("tushare_moneyflow", {"code": "{code}", "days": 60}),
        ("tushare_moneyflow_hsgt", {"code": "{code}", "days": 60}),
        ("tushare_margin", {"code": "{code}"}),
        ("tushare_margin_detail", {"code": "{code}", "days": 60}),
        ("tushare_top_list", {"date": "{date}"}),
        ("tushare_block_trade", {"code": "{code}", "days": 60}),
        ("tushare_daily_basic", {"code": "{code}", "days": 250}),
        ("tushare_kline", {"code": "{code}", "days": 250}),
        ("tushare_cyq_chips", {"code": "{code}"}),
    ],
}


def _resolve_kwargs(template: Dict[str, Any], code: str, date_yyyymmdd: str) -> Dict[str, Any]:
    resolved: Dict[str, Any] = {}
    for key, value in template.items():
        if isinstance(value, str):
            resolved[key] = value.replace("{code}", code).replace("{date}", date_yyyymmdd)
        else:
            resolved[key] = value
    return resolved


def build_term_spec(
    term: str, stock_code: str, current_date: str = "", is_etf: bool = False,
) -> List[AgentToolSpec]:
    """生成某期限的统一取数清单（跨 Agent 去重，仅针对完全相同的 (tool, kwargs)）。

    Args:
        term: short/medium/long/full
        stock_code: 带交易所前缀的代码（sh.600000 / sz.300001）
        current_date: YYYY-MM-DD（moneyflow top_list 需要）
        is_etf: ETF 跳过 fundamental/value 的取数项
    """
    clean_code = (
        stock_code.replace("sh.", "").replace("sz.", "").replace("bj.", "")
        .replace(".SH", "").replace(".SZ", "").replace(".BJ", "").strip()
    )
    date_yyyymmdd = current_date.replace("-", "")

    agents = list(TERM_AGENT_MAP.get(term, TERM_AGENT_MAP["full"]))
    if is_etf:
        agents = [a for a in agents if a not in ETF_EXEMPT_AGENTS]

    seen = set()
    spec: List[AgentToolSpec] = []
    for agent in agents:
        for tool_name, kwargs_tmpl in AGENT_TOOL_SPECS.get(agent, []):
            kwargs = _resolve_kwargs(kwargs_tmpl, clean_code, date_yyyymmdd)
            key = (tool_name, tuple(sorted(kwargs.items())))
            if key in seen:
                continue
            seen.add(key)
            spec.append((tool_name, kwargs))
    return spec
