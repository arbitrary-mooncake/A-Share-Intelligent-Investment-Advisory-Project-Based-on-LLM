"""
任务规划器 — 根据问题意图和复杂度确定所需数据域和工具。

快路径(L1/L2): 确定数据域 → 并行拉取全部数据 → LLM直接回答
慢路径(L3/L4): 确定数据域 → 先尝试并行拉取 → 不足则启用ReAct
"""
from dataclasses import dataclass, field
from typing import List, Set
import re


# ── 数据域定义 ────────────────────────────────────

DATA_DOMAINS = {
    "行情": {
        "keywords": ["价格", "涨", "跌", "走势", "行情", "趋势", "K线", "均线",
                     "今天", "最近", "最高", "最低", "收盘", "开盘", "振幅", "换手"],
        "tools": ["get_historical_k_data", "tushare_kline", "tushare_daily_basic",
                  "get_stock_basic_info", "get_latest_trading_date"],
        "description": "价格、涨跌幅、K线、换手率等行情数据",
    },
    "估值": {
        "keywords": ["PE", "PB", "PS", "市盈", "市净", "市销", "估值", "贵", "便宜",
                     "分位", "EV/EBITDA", "股息", "分红"],
        "tools": ["tushare_daily_basic", "tushare_pe_percentile", "tushare_ev_ebitda",
                  "tushare_dividend", "get_dividend_data", "get_stock_basic_info"],
        "description": "市盈率、市净率、估值分位、股息率等估值数据",
    },
    "财务": {
        "keywords": ["ROE", "ROA", "利润", "收入", "毛利", "净利", "现金", "负债",
                     "资产", "财报", "业绩", "盈利", "成长", "增速", "杜邦"],
        "tools": ["get_profit_data", "get_balance_data", "get_cash_flow_data",
                  "get_growth_data", "get_dupont_data", "get_operation_data",
                  "tushare_fina_indicator", "tushare_stock_info"],
        "description": "利润表、资产负债表、现金流量表、杜邦分析等财务数据",
    },
    "资金": {
        "keywords": ["资金", "主力", "北向", "融资", "融券", "流入", "流出",
                     "成交", "量比", "换手", "净买"],
        "tools": ["tushare_moneyflow", "get_market_analysis_timeframe",
                  "tushare_daily_basic"],
        "description": "主力资金流、融资融券、北向资金等资金面数据",
    },
    "行业": {
        "keywords": ["行业", "板块", "赛道", "同行", "竞品", "龙头", "排名",
                     "行业地位", "市场份额", "行业对比"],
        "tools": ["get_stock_industry", "get_stock_basic_info",
                  "tushare_stock_info", "get_market_analysis_timeframe"],
        "description": "行业分类、行业对比、板块强弱等行业数据",
    },
    "新闻": {
        "keywords": ["新闻", "公告", "消息", "事件", "发布", "披露", "分红方案",
                     "回购", "减持", "增持", "业绩预告", "ST", "风险警示"],
        "tools": ["crawl_news", "get_st_risk_data", "tushare_st_status"],
        "description": "新闻公告、ST风险、重大事件等舆情数据",
    },
}


@dataclass
class TaskPlan:
    """任务规划结果"""
    domains: List[str]                # 需要的数据域
    tools: List[str]                  # 需要的工具列表
    need_react: bool                  # 是否使用 ReAct
    reason: str                       # 规划理由
    expected_data_volume: str         # "small" | "medium" | "large"


def plan_task(question: str, complexity_level: str, history_text: str = "") -> TaskPlan:
    """
    根据问题内容和复杂度等级规划数据获取任务。

    L1/L2: 并行拉取全量数据
    L3: 先并行拉取，不足时运行时升级为ReAct
    L4: 直接使用ReAct
    """
    # 结合历史对话提取额外关键词
    augmented_question = question
    if history_text:
        # 提取历史中最近提到的数据域关键词追加到问题中
        recent_kw = _extract_domain_keywords_from_text(history_text)
        if recent_kw:
            augmented_question = question + " " + " ".join(recent_kw)

    domains = _identify_domains(augmented_question)
    tools = _get_tools_for_domains(domains)

    # 确定是否需要 ReAct
    # L1/L2: 永不用ReAct
    # L3: 标注为按需（先用快路径，不足时升级）
    # L4: 直接用ReAct
    need_react = (complexity_level == "L4")

    data_volume = "small"
    if len(domains) >= 4:
        data_volume = "large"
    elif len(domains) >= 2:
        data_volume = "medium"

    reason = f"问题涉及{len(domains)}个数据域({', '.join(domains)})，复杂度={complexity_level}"

    return TaskPlan(
        domains=domains,
        tools=tools,
        need_react=need_react,
        reason=reason,
        expected_data_volume=data_volume,
    )


def _extract_domain_keywords_from_text(text: str) -> List[str]:
    """从历史对话文本中提取数据域关键词"""
    all_kw = []
    for domain_info in DATA_DOMAINS.values():
        for kw in domain_info["keywords"]:
            if kw in text:
                all_kw.append(kw)
    return all_kw[-10:]  # 最多取10个


def _identify_domains(question: str) -> List[str]:
    """识别问题涉及的数据域"""
    matched = []
    for domain_name, domain_info in DATA_DOMAINS.items():
        for kw in domain_info["keywords"]:
            if kw in question:
                matched.append(domain_name)
                break
    # 至少包含行情域（大多数问题都需要）
    if not matched:
        matched = ["行情"]
    return matched


def _get_tools_for_domains(domains: List[str]) -> List[str]:
    """获取数据域对应的工具列表（去重）"""
    tools: Set[str] = set()
    for domain in domains:
        if domain in DATA_DOMAINS:
            tools.update(DATA_DOMAINS[domain]["tools"])
    return sorted(tools)


def extract_stock_from_question(question: str, session_stock_code: str = "",
                                 session_company_name: str = "") -> tuple:
    """
    从问题中提取股票代码和公司名称。
    优先使用问题中的信息，其次使用会话上下文。
    """
    code = None
    name = None

    # 提取股票代码
    code_match = re.search(r'\b(\d{5,6})\b', question)
    if code_match:
        code = code_match.group(1)

    # 括号内提取: 嘉友国际(603871)
    paren_match = re.search(r'([^（(]+?)\s*[（(](\d{5,6})[)）]', question)
    if paren_match:
        name = paren_match.group(1).strip()
        code = paren_match.group(2)

    # 引用前文主语
    ref_pattern = r'(?:它|这个|那个|这只|那家|刚才|上次)(?:股票|公司|股|标的)?'
    if re.search(ref_pattern, question) and not code:
        code = session_stock_code
        name = session_company_name

    # 标准化代码格式（仅限合法A股前缀: 6=沪市, 0/3=深市, 688=科创板, 8=北交所）
    if code:
        if not code.startswith(("sh.", "sz.")):
            if code.startswith(("6", "688", "5", "8")):
                code = f"sh.{code}"
            elif code.startswith(("0", "3", "1", "4")):
                code = f"sz.{code}"
            # 其他前缀（如999）不添加交易所前缀，保持原样

    return code, name
